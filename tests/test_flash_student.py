"""BellowsKV-Flash conversion checks (GPU + Llama weights; one 8K calibration window, block 0, n = 2).

  1. Closed form at full rank (every rope component, full latent rank): the deployment-shaped student must equal
     exact attention (tests the rope split, W_QR, the latent fit and the sink / record / band softmax).
  2. A trained-like block (closed form at the shipped shape): the batched student must match a per-query loop
     written with explicit index sets (exact sink, record [sink, t - recent), exact band [t - recent, t]).

    python tests/test_flash_student.py [--model meta-llama/Llama-3.1-8B-Instruct]
"""
import argparse
import sys

import torch

from bellowskv.calibration import calibration_windows
from bellowskv.flash.closed_form import closed_form_block
from bellowskv.flash.config import FLASH_CODE_SOURCES, FlashConfig
from bellowskv.flash.distill import exact_kv, student_out, teacher_out
from bellowskv.flash.ops import LayerW, cos_sin_table, rope_full, rope_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    model.requires_grad_(False)
    cfg = FlashConfig()
    layers = model.model.layers; d = model.config.hidden_size; dev = next(model.parameters()).device
    S, n = 8192, cfg.group_size
    grp = list(range(n)); cur = {}
    hooks = [layers[l].self_attn.k_proj.register_forward_pre_hook((lambda l: (lambda mo, i: cur.__setitem__(l, i[0])))(l))
             for l in grp]
    w = calibration_windows(tok, S, 2, sources=FLASH_CODE_SOURCES)[0][0]
    with torch.no_grad():
        model.model(input_ids=w[None].to(dev), use_cache=False)
    for h in hooks:
        h.remove()
    xs = [cur[l][0].float() for l in grp]
    H = torch.cat(xs, -1)
    Gj = H.T @ H / S
    Gl = [Gj[i * d:(i + 1) * d, i * d:(i + 1) * d] for i in range(n)]
    ws = [LayerW(layers[l].self_attn) for l in grp]
    kv = layers[0].self_attn.k_proj.weight.shape[0]
    cos, sin = cos_sin_table(model.model.rotary_emb.inv_freq.float().to(dev), S, dev)
    qp = torch.linspace(cfg.sink + cfg.recent + 1, S - 1, 256, device=dev).long()
    ok = True
    with torch.no_grad():
        full = closed_form_block(model, grp, Gj, Gl, R=n * 2 * kv, r=n * kv)
        outs = student_out(full, ws, xs, qp, cos, sin, cfg.sink, cfg.recent)
        for i, wl in enumerate(ws):
            ref = teacher_out(wl, xs[i], qp, cos, sin)
            e = ((outs[i] - ref).norm() / ref.norm()).item()
            ok &= e < 2e-3
            print(f"[check1] layer {i}: full-rank student vs exact attention, rel err {e:.2e}", flush=True)
        del full
        prm = closed_form_block(model, grp, Gj, Gl, cfg.r_latent, cfg.r_rope)
        outs = student_out(prm, ws, xs, qp, cos, sin, cfg.sink, cfg.recent)
        c = H @ prm["Wd"]; srot = rope_pairs(H @ prm["Wr"], torch.arange(S, device=dev), prm["theta"])
        for i, wl in enumerate(ws):
            hd, g, nq = wl.hd, wl.gq, wl.nq; R = c.shape[1]; r = srot.shape[1]
            k, v = exact_kv(wl, xs[i], cos, sin)
            worst = 0.0
            for j in (0, 1, 100, 255):
                t = int(qp[j])
                ex = list(range(cfg.sink)) + list(range(t - cfg.recent, t + 1)); rc = list(range(cfg.sink, t - cfg.recent))
                qh = (xs[i][t] @ wl.wq).view(nq, hd)
                qf = rope_full(qh[:, None], cos[t:t + 1], sin[t:t + 1])[:, 0]
                qr = rope_pairs((xs[i][t] @ prm["WQR"][i]).view(nq, 1, r), torch.tensor([t], device=dev), prm["theta"])[:, 0]
                o = torch.empty(nq, hd, device=dev)
                for hh in range(nq):
                    gg = hh // g
                    se = qf[hh] @ k[gg, ex].T
                    sr = (prm["WuK"][i].view(R, -1, hd)[:, gg] @ qh[hh]) @ c[rc].T + qr[hh] @ srot[rc].T
                    p = torch.softmax(torch.cat([se, sr]) * wl.scaling, 0)
                    o[hh] = p[:len(ex)] @ v[gg, ex] + (p[len(ex):] @ c[rc]) @ prm["WuV"][i].view(R, -1, hd)[:, gg]
                worst = max(worst, ((outs[i][j].view(nq, hd) - o).norm() / o.norm()).item())
            ok &= worst < 1e-4
            print(f"[check2] layer {i}: batched student vs per-query loop, worst rel err {worst:.2e}", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
