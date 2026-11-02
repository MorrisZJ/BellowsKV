"""Latent quantization, INT4 packing and the latent-side rotation."""
from __future__ import annotations

import torch


def quantize(c: torch.Tensor, bits: int = 4, qgroup: int = 128, clip: float = 1.0):
    """Per-token, per-128-channel asymmetric min-max quantization with range clipping (the paper's `_quant`).

    Returns (q, scale, zero): q integer-valued in [0, 2^bits - 1] shaped like c; scale/zero (..., R/qgroup) fp16.
    (scale, zero) are rounded to fp16 before quantizing, so dequantization with the stored metadata is unbiased.
    """
    *pre, R = c.shape
    nb = R // qgroup
    x = c.reshape(-1, nb, qgroup).float()
    lo = x.amin(-1, keepdim=True); hi = x.amax(-1, keepdim=True)
    if clip < 1.0:
        mid = (hi + lo) / 2; half = (hi - lo) / 2 * clip
        lo, hi = mid - half, mid + half
    qmax = (1 << bits) - 1
    scale = ((hi - lo).clamp_min(1e-8) / qmax).half().float()
    lo = lo.half().float()
    q = torch.round((x - lo) / scale).clamp_(0, qmax)
    return q.reshape(*pre, R), scale.reshape(*pre, nb).half(), lo.reshape(*pre, nb).half()


def dequantize(q, scale, zero, qgroup: int = 128, dtype=torch.float16):
    *pre, R = q.shape
    x = q.reshape(*pre, R // qgroup, qgroup).float()
    return (x * scale.float().unsqueeze(-1) + zero.float().unsqueeze(-1)).reshape(*pre, R).to(dtype)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """(..., R) INT4 values -> (..., R/2) uint8; channel 2j in the low nibble of byte j, 2j+1 in the high one."""
    *pre, R = q.shape
    v = q.reshape(*pre, R // 2, 2).to(torch.uint8)
    return (v[..., 0] | (v[..., 1] << 4)).contiguous()


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    lo = (p & 0x0F).float(); hi = ((p >> 4) & 0x0F).float()
    return torch.stack((lo, hi), -1).reshape(*p.shape[:-1], 2 * p.shape[-1])


class BandRotation:
    """Rotates a key latent by the token's RoPE angles, band by band.

    The latent holds, per rotary band i (frequency i of the model's RoPE), r_i complex coordinates stored as
    [Re(r_i) | Im(r_i)]. Rotating by position t multiplies each by e^{j t theta_i}; with the equivariant
    read-head, Wu^T rot_t(c) == RoPE_t(Wu^T c).
    """

    def __init__(self, ranks, device):
        ranks = [int(r) for r in ranks]
        R = 2 * sum(ranks)
        band = torch.empty(R, dtype=torch.long); is_re = torch.zeros(R, dtype=torch.bool)
        partner = torch.empty(R, dtype=torch.long)
        col = 0
        for i, r in enumerate(ranks):
            band[col:col + 2 * r] = i; is_re[col:col + r] = True
            partner[col:col + r] = torch.arange(col + r, col + 2 * r)
            partner[col + r:col + 2 * r] = torch.arange(col, col + r)
            col += 2 * r
        self.band, self.is_re, self.partner = band.to(device), is_re.to(device), partner.to(device)

    def __call__(self, c: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """c (..., T, R) fp32; cos/sin (..., T, hd) from the model's rotary embedding (cos[i] == cos[i + hd/2])."""
        ct = cos.float().index_select(-1, self.band); st = sin.float().index_select(-1, self.band)
        other = c.index_select(-1, self.partner)
        return torch.where(self.is_re, c * ct - other * st, c * ct + other * st)
