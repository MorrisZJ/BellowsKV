"""BellowsKV-Flash-Turbo: on-policy distillation of the record maps, starting from a Lite conversion.

Teacher = the original model (exact attention). Student = the same frozen model whose decode-time attention goes
through the record exactly as deployed: exact sink + exact recent window + record over prompt positions older than
t - recent. Only the record maps (W_d, W_r, W_UK, W_UV, W_QR) are trained.

Per sample: a long instruction prompt (exact prefill, as deployed) + a continuation of Tg <= recent tokens sampled
from the deployed engine (flash8, temperature 1), whose weights are refreshed in place after every step. With
Tg <= recent no continuation token enters the record, so one parallel pass over the continuation equals token-by-token
deployed decoding. Loss: reverse KL(student || teacher) per token, full vocabulary.

Optimizer: Adam with a per-group relative learning rate (group lr = lr x the group's element RMS at init; the groups'
scales differ ~140x, and a uniform lr destabilizes training within ~50 steps), warmup 10 steps, cosine to 10%. The
checkpoint with the lowest held-out KL is kept. Reference recipe (Llama-3.1-8B-Instruct, 2026-11-23): 60 steps x 16
samples, relative lr 1e-3, 17.7 min on 4 x H100 NVL; held-out KL 0.0957 -> 0.0848; Protocol-B 93.5 -> 95.3.

    torchrun --nproc_per_node 4 -m bellowskv.flash.turbo --init <lite params dir> --out <turbo params dir>
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import params as PF
from .deploy import deploy_settings

BF = torch.bfloat16
GROUPS = ("Wd", "Wr", "WuK", "WuV", "WQR")


# ----------------------------------------------------------------------------------------------- model pieces
def rmsnorm(x, norm):
    xf = x.float()
    return norm.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)).to(x.dtype)


def rope_tab(inv_freq, pos):
    ang = pos.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([ang, ang], -1)
    return emb.cos().to(BF), emb.sin().to(BF)


def rope(x, cos, sin):                         # x (N, heads, D); cos/sin (N, D)
    h = x.shape[-1] // 2
    return x * cos[:, None] + torch.cat([-x[..., h:], x[..., :h]], -1) * sin[:, None]


def rope_pairs(x, pos, theta):                 # x (heads, N, r) or (N, r); pos (N,)
    ang = pos.float()[:, None] * theta[None, :]
    c, s = ang.cos(), ang.sin()
    x2 = x.float().unflatten(-1, (-1, 2)); x0, x1 = x2[..., 0], x2[..., 1]
    return torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], -1).flatten(-2)


@torch.no_grad()
def teacher_pass(model, ids, Tp):
    """One exact forward over prompt + continuation: the prompt's attention inputs per layer (Tp, d) bf16 and the
    teacher logits at the continuation positions (Tg, V) fp32."""
    layers = model.model.layers; Xc = [None] * len(layers)
    hooks = [layers[l].self_attn.k_proj.register_forward_pre_hook(
        (lambda l: (lambda m, inp: Xc.__setitem__(l, inp[0][0, :Tp].clone())))(l)) for l in range(len(layers))]
    try:
        out = model(input_ids=ids[None], use_cache=False, logits_to_keep=ids.numel() - Tp)
    finally:
        for h in hooks:
            h.remove()
    return Xc, out.logits[0].float()


def block_record(Xc, g0, n, prm, Tp):
    H = torch.cat([Xc[g0 + i] for i in range(n)], -1)
    c = F.linear(H, prm["Wd"].t().to(BF)).float()
    s = F.linear(H, prm["Wr"].t().to(BF)).float()
    return c, rope_pairs(s, torch.arange(Tp, device=H.device), prm["theta"])


def student_layer(r, l, li, model, prm, rec, Xc_l, Tp, sink, recent, inv_freq):
    """One decoder layer for the continuation rows r (Tg, d), attention as deployed. Returns the new r."""
    layer = model.model.layers[l]; at = layer.self_attn
    Tg, d = r.shape; dev = r.device
    hd = at.head_dim; nq = at.q_proj.out_features // hd; kvh = at.k_proj.out_features // hd; g = nq // kvh
    R = prm["Wd"].shape[1]; rr = prm["Wr"].shape[1]
    c, srot = rec
    h = rmsnorm(r, layer.input_layernorm)
    pos_q = torch.arange(Tp, Tp + Tg, device=dev)
    x_ex = torch.cat([Xc_l[:sink], Xc_l[Tp - recent:Tp], h], 0)              # sink + last `recent` + continuation
    pos_ex = torch.cat([torch.arange(sink, device=dev), torch.arange(Tp - recent, Tp + Tg, device=dev)])
    ce, se = rope_tab(inv_freq, pos_ex)
    k_ex = rope(at.k_proj(x_ex).view(-1, kvh, hd), ce, se).float()
    v_ex = at.v_proj(x_ex).view(-1, kvh, hd).float()
    j = torch.arange(Tg, device=dev)
    eidx = torch.cat([torch.arange(sink, device=dev)[None].expand(Tg, -1),
                      sink + j[:, None] + torch.arange(recent + 1, device=dev)[None]], 1)
    E = eidx.shape[1]
    q = at.q_proj(h).view(Tg, nq, hd)
    cq, sq = rope_tab(inv_freq, pos_q)
    qf = rope(q, cq, sq).float().view(Tg, kvh, g, hd)
    kg = k_ex[eidx]; vg = v_ex[eidx]
    sc_e = torch.einsum("tkgd,tekd->kgte", qf, kg).reshape(nq, Tg, E)
    qa = torch.einsum("tkgd,rkd->kgtr", q.float().view(Tg, kvh, g, hd), prm["WuK"][li].view(R, kvh, hd)).reshape(nq, Tg, R)
    qr = rope_pairs(F.linear(h, prm["WQR"][li].t().to(BF)).float().view(Tg, nq, rr).transpose(0, 1), pos_q, prm["theta"])
    p = torch.arange(Tp, device=dev)
    bad = (p[None, :] < sink) | (p[None, :] >= (pos_q[:, None] - recent))     # record = [sink, t - recent)
    sc_r = (qa @ c.T + qr @ srot.T).masked_fill(bad[None], float("-inf"))
    P = torch.softmax(torch.cat([sc_e, sc_r], -1) * at.scaling, -1)
    o_e = torch.einsum("kgte,tekd->tkgd", P[..., :E].reshape(kvh, g, Tg, E), vg)
    ol = (P[..., E:] @ c).view(kvh, g, Tg, R)
    o_r = torch.einsum("kgtr,rkd->tkgd", ol, prm["WuV"][li].view(R, kvh, hd))
    r = r + at.o_proj((o_e + o_r).reshape(Tg, nq * hd).to(BF))
    h2 = rmsnorm(r, layer.post_attention_layernorm)
    m = layer.mlp
    return r + m.down_proj(F.silu(m.gate_proj(h2)) * m.up_proj(h2))


def student_logits(model, n, P, Xc, ids, Tp, sink, recent, ckpt=True):
    inv_freq = model.model.rotary_emb.inv_freq.float()
    r = model.model.embed_tokens(ids[Tp:])
    for g0 in range(0, len(model.model.layers), n):
        rec = block_record(Xc, g0, n, P[g0], Tp)
        for li in range(n):
            l = g0 + li
            f = lambda r_, l=l, li=li, rec=rec: student_layer(r_, l, li, model, P[g0], rec, Xc[l], Tp, sink, recent, inv_freq)
            r = checkpoint(f, r, use_reentrant=False) if ckpt and torch.is_grad_enabled() else f(r)
    return model.lm_head(rmsnorm(r, model.model.norm)).float()


def reverse_kl(ls, lt):
    """Mean over positions of KL(p_s || p_t), full vocabulary; ls, lt: logits (Tg, V)."""
    ps = F.log_softmax(ls, -1); pt = F.log_softmax(lt, -1)
    return (ps.exp() * (ps - pt)).sum(-1).mean()


# ----------------------------------------------------------------------------------------------- engine sampling
@torch.no_grad()
def refresh_engine(M, P):
    """Copy the current record maps into the engine's folded bf16 tensors in place (captured graphs stay valid)."""
    H, HKV, D, DM, n = M.H, M.HKV, M.D, M.DM, M.n
    for g0, p in P.items():
        blk = M.blocks[g0]; R = p["Wd"].shape[1]; r = p["Wr"].shape[1]; cs, ss = p["cs"], p["ss"]
        Wdr = torch.cat([p["Wd"], p["Wr"]], 1)
        blk["Wdr16"].copy_(Wdr.t()); blk["Wdr8"].copy_((Wdr / torch.cat([cs, ss])[None]).t())
        for li in range(n):
            wk = p["WuK"][li].view(R, HKV, D).permute(1, 2, 0); wv = p["WuV"][li].view(R, HKV, D).permute(1, 0, 2)
            blk["WuK16"][li].copy_(wk); blk["WuK8"][li].copy_(wk * cs)
            blk["WuV16"][li].copy_(wv); blk["WuV8"][li].copy_(wv * cs[None, :, None])
            lw = M.L[g0 + li]
            for sfx in ("16", "8"):
                w = lw["wqkv" + sfx]
                w[M.NQKV:M.NQKV + H * r].copy_(p["WQR"][li].t())
                w[M.NQKV + H * r:].copy_(blk["Wdr" + sfx][:, li * DM:(li + 1) * DM])


def _draw(temp, gen):
    if temp <= 0:
        return lambda lg: int(lg.argmax(-1))
    return lambda lg: int(torch.multinomial(torch.softmax(lg.float() / temp, -1), 1, generator=gen))


# ----------------------------------------------------------------------------------------------- training
def train(a):
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM
    from .engine import Cache, Engine, Model
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))); torch.cuda.set_device(dev)
    log = (lambda m: print(m, flush=True)) if rank == 0 else (lambda m: None)
    if a.samples_per_step % world:
        raise ValueError(f"--samples_per_step {a.samples_per_step} must be a multiple of the world size {world}")
    accum = a.samples_per_step // world
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=BF, attn_implementation="sdpa").to(dev).eval()
    model.requires_grad_(False)
    init = PF.load(a.init, device=dev)
    n, meta = init["n"], init["meta"]
    dep = deploy_settings(meta, a.recent)
    sink, recent = dep["sink"], dep["recent"]
    if a.tg > recent:
        raise ValueError(f"--tg {a.tg} > recent {recent}: continuation tokens would enter the record")
    P = {g0: dict(Wd=b["Wd"].clone(), Wr=b["Wr"].clone(), theta=b["theta"], cs=b["cs"], ss=b["ss"],
                  WuK=[t.clone() for t in b["WuK"]], WuV=[t.clone() for t in b["WuV"]],
                  WQR=[t.clone() for t in b["WQR"]], qscale=b["qscale"]) for g0, b in init["blocks"].items()}
    grp = {k: [] for k in GROUPS}
    for g0 in P:
        for k in GROUPS:
            for t in ([P[g0][k]] if k in ("Wd", "Wr") else P[g0][k]):
                t.requires_grad_(True); grp[k].append(t)
    params = [t for k in GROUPS for t in grp[k]]
    rms = {k: (sum(float(t.detach().pow(2).sum()) for t in grp[k]) / sum(t.numel() for t in grp[k])) ** 0.5 for k in GROUPS}
    glr = {k: a.lr * (rms[k] if a.lr_mode == "relative" else 1.0) for k in GROUPS}
    log(f"[turbo] {sum(t.numel() for t in params) / 1e6:.0f}M record parameters, world {world} x {accum} samples, "
        f"lr " + ", ".join(f"{k} {glr[k]:.2e}" for k in GROUPS))
    opt = torch.optim.Adam([dict(params=grp[k], lr=glr[k]) for k in GROUPS], lr=a.lr, betas=(0.9, a.beta2))
    warm = max(10, a.steps // 25)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, st / a.steps)))))
    M = Model(a.model, a.init, dev, recent=recent)
    pr = torch.load(a.prompts)["ids"]
    Eng = Engine(M, Cache(M, "flash8", 1, max(x.numel() for x in pr) + a.tg + 64)); Eng.keep_logits = True
    eos = set(a.eos)
    t_a = Eng.generate(pr[0].long().to(dev), 32)
    refresh_engine(M, P)                                              # unchanged params: must be a no-op
    if Eng.generate(pr[0].long().to(dev), 32) != t_a:
        raise RuntimeError("engine refresh changed the greedy output")
    held = list(range(a.n_eval))                                      # fixed eval prompts, never trained on
    order = list(range(a.n_eval, len(pr))); random.Random(a.seed).shuffle(order)
    mine = order[rank::world]
    gen = torch.Generator(device=dev).manual_seed(a.seed * 1000 + rank)
    ev = [(pr[i].long().to(dev),) for i in held[rank::world]]
    ev = [(ids, torch.tensor(Eng.generate(ids, a.tg, eos), device=dev)) for (ids,) in ev]   # greedy from init

    def evaluate():
        tot = torch.zeros(2, device=dev)
        with torch.no_grad():
            for ids_p, cont in ev:
                ids = torch.cat([ids_p, cont]); Tp = ids_p.numel()
                Xc, lt = teacher_pass(model, ids, Tp)
                ls = student_logits(model, n, P, Xc, ids, Tp, sink, recent)
                tot += torch.stack([reverse_kl(ls, lt), (ls.argmax(-1) == lt.argmax(-1)).float().mean()])
                del Xc
        if world > 1:
            dist.all_reduce(tot)
        return (tot / a.n_eval).tolist()

    k0, a0 = evaluate(); best = k0; best_step = 0
    log(f"[turbo] step 0: held-out KL(student||teacher) {k0:.4f}, argmax agree {a0:.3f}")
    t0 = time.time(); hist = []; k_next = 0
    draw = _draw(a.temp, gen)
    for step in range(a.steps):
        for _ in range(accum):
            ids_p = pr[mine[k_next % len(mine)]].long().to(dev); Tp = ids_p.numel(); k_next += 1
            cont = torch.tensor(Eng.generate(ids_p, a.tg, eos, sample=draw), device=dev)
            ids = torch.cat([ids_p, cont])
            Xc, lt = teacher_pass(model, ids, Tp)
            loss = reverse_kl(student_logits(model, n, P, Xc, ids, Tp, sink, recent), lt)
            (loss / accum).backward()
            hist.append(loss.item())
            del Xc, lt
        if world > 1:
            for t in params:
                dist.all_reduce(t.grad); t.grad.div_(world)
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        refresh_engine(M, P)
        if step % 10 == 0 or step == a.steps - 1:
            log(f"[turbo] step {step}: train KL {sum(hist[-40:]) / len(hist[-40:]):.4f}, grad norm {float(gn):.2f}, "
                f"lr x{sched.get_last_lr()[0] / glr['Wd']:.2f}, {(time.time() - t0) / (step + 1):.1f}s/step")
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            k1, a1 = evaluate()
            log(f"[turbo] step {step + 1}: held-out KL {k1:.4f} (start {k0:.4f}), argmax agree {a1:.3f}")
            if k1 < best:
                best, best_step = k1, step + 1
                if rank == 0:
                    _save(a, P, n, meta, recent, dict(step=step + 1, heldout_kl=k1, heldout_agree=a1, heldout_kl_start=k0,
                                                      train_minutes=round((time.time() - t0) / 60, 1), world=world))
            if k1 > a.stop_ratio * k0:
                log(f"[turbo] held-out KL above {a.stop_ratio} x start: stopping")
                break
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
    if best_step == 0:
        log(f"[turbo] done: no step beat the init's held-out KL {k0:.4f}; nothing written (keep using {a.init})")
    else:
        log(f"[turbo] done: best held-out KL {best:.4f} at step {best_step} (start {k0:.4f}) -> {a.out}")


def _save(a, P, n, meta, recent, info):
    os.makedirs(a.out, exist_ok=True)
    recs = [dict(g0=g0, n=n, c_scale=p["cs"].detach().cpu(), s_scale=p["ss"].detach().cpu(), q_scale=p["qscale"],
                 params=dict(Wd=p["Wd"].detach().cpu(), Wr=p["Wr"].detach().cpu(), theta=p["theta"].cpu(),
                             WuK=[t.detach().cpu() for t in p["WuK"]], WuV=[t.detach().cpu() for t in p["WuV"]],
                             WQR=[t.detach().cpu() for t in p["WQR"]])) for g0, p in P.items()]
    m = {k: v for k, v in meta.items() if k not in ("format", "version")}
    m["deploy"] = dict(meta.get("deploy", {}), recent=recent)
    m["turbo"] = dict(init=a.init, prompts=a.prompts, steps=a.steps, samples_per_step=a.samples_per_step, lr=a.lr,
                      lr_mode=a.lr_mode, beta2=a.beta2, temp=a.temp, tg=a.tg, clip=a.clip, seed=a.seed, **info)
    PF.save(a.out, recs, m)


def main(argv=None):
    ap = argparse.ArgumentParser(description="BellowsKV-Flash-Turbo: on-policy distillation from a Lite conversion")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--init", required=True, help="Lite params dir (python -m bellowskv.flash.convert)")
    ap.add_argument("--prompts", required=True, help="prompt set (python -m bellowskv.flash.data)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recent", type=int, default=None, help="deployed exact window (default: the params' deploy)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--samples_per_step", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_mode", choices=("relative", "uniform"), default="relative")
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tg", type=int, default=123, help="sampled continuation length (<= recent)")
    ap.add_argument("--n_eval", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--stop_ratio", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eos", type=int, nargs="*", default=[128001, 128008, 128009])
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
