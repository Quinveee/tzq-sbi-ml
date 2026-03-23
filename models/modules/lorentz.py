"""
Lorentz / Minkowski utilities and LLoCa frame construction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_MINK_SIGN = torch.tensor([1.0, -1.0, -1.0, -1.0])

_MINK_SIGN = torch.tensor([1.0, -1.0, -1.0, -1.0])


def minkowski_dot(a, b):
    sign = _MINK_SIGN.to(a.device, a.dtype)
    return (a * b * sign).sum(-1)


def safe_norm_sq(x, eps=1e-6):
    n = minkowski_dot(x, x)
    return torch.where(torch.isfinite(n), n, torch.zeros_like(n)).clamp(min=eps)


def safe_sqrt(x, eps=1e-6):
    return torch.sqrt(torch.clamp(x, min=eps))


def build_lloca_frames(particles, ptr, K, eps=1e-6):
    """
    Batched version of LLoCa frame construction.

    particles: (N, 4)
    ptr:       (B+1,)
    returns:   (N, K, 4)
    """

    device, dtype = particles.device, particles.dtype
    ptr = ptr.long()

    B = len(ptr) - 1
    lengths = ptr[1:] - ptr[:-1]
    max_n = int(lengths.max().item())
    N = particles.shape[0]

    # ── 1. pack into padded tensor ────────────────────────
    padded = torch.zeros(B, max_n, 4, device=device, dtype=dtype)
    mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)

    for b in range(B):
        s, e = int(ptr[b]), int(ptr[b + 1])
        padded[b, :e - s] = particles[s:e]
        mask[b, :e - s] = True

    # ── 2. build frames tensor ────────────────────────────
    frames = torch.zeros(B, max_n, K, 4, device=device, dtype=dtype)

    # ── 3. anchor (vectorized) ────────────────────────────
    anchor = padded.clone()

    spatial_norm = torch.sqrt(
        (anchor[..., 1:] ** 2).sum(-1).clamp(min=eps)
    ).unsqueeze(-1)

    anchor[..., 1:] = anchor[..., 1:] / spatial_norm
    anchor[..., 0] = 1.0

    frames[..., 0, :] = anchor

    if K > 1:
        # ── 4. pairwise spatial distances ─────────────────
        diff = padded.unsqueeze(2) - padded.unsqueeze(1)  # (B, max_n, max_n, 4)
        d_ij = (diff[..., 1:] ** 2).sum(-1)              # (B, max_n, max_n)

        # mask invalid pairs
        mask_ij = mask.unsqueeze(2) & mask.unsqueeze(1)
        eye = torch.eye(max_n, device=device).bool().unsqueeze(0)

        d_ij = torch.where(mask_ij & ~eye, d_ij, float("inf"))

        # ── 5. nearest neighbours ─────────────────────────
        k_use = min(K - 1, max_n - 1)
        _, nn_idx = torch.topk(d_ij, k_use, dim=2, largest=False)

        # ── 6. gather neighbours ──────────────────────────
        nn_idx_exp = nn_idx.unsqueeze(-1).expand(-1, -1, -1, 4)
        nn = torch.gather(
            padded.unsqueeze(1).expand(-1, max_n, -1, -1),
            dim=2,
            index=nn_idx_exp,
        )  # (B, max_n, k_use, 4)

        center = padded.unsqueeze(2)
        rel = nn - center

        # ── 7. spatial directions only ────────────────────
        rel[..., 0] = 0.0

        norm = torch.sqrt(
            (rel[..., 1:] ** 2).sum(-1).clamp(min=eps)
        ).unsqueeze(-1)

        rel = rel / norm

        frames[..., 1:1 + k_use, :] = rel

        if k_use < K - 1:
            frames[..., 1 + k_use:, :] = rel[..., :1, :].expand(
                -1, -1, K - 1 - k_use, -1
            )

    # ── 8. remove padding → back to (N, K, 4) ─────────────
    out = torch.zeros(N, K, 4, device=device, dtype=dtype)

    for b in range(B):
        s, e = int(ptr[b]), int(ptr[b + 1])
        out[s:e] = frames[b, :e - s]

    return torch.nan_to_num(out)


def lloca_dot_product_attention(
    query, key, value,
    frames,
    n_scalars,
    n_vectors,
    attn_mask=None
):
    H, N, d_head = query.shape
    K = frames.shape[1]
    d_vec = n_vectors * 4

    sign = _MINK_SIGN.to(query.device, query.dtype)

    # ── Scalars ─────────────────────────────
    q_s = torch.cat([query[..., :n_scalars], query[..., n_scalars + d_vec:]], dim=-1)
    k_s = torch.cat([key[..., :n_scalars], key[..., n_scalars + d_vec:]], dim=-1)

    scale_s = max(q_s.shape[-1], 1) ** -0.5
    attn = torch.bmm(q_s, k_s.transpose(-1, -2)) * scale_s

    # ── Vector term ─────────────────────────
    if n_vectors > 0:
        q_v = query[..., n_scalars:n_scalars + d_vec].reshape(H, N, n_vectors, 4)
        k_v = key[..., n_scalars:n_scalars + d_vec].reshape(H, N, n_vectors, 4)

        frames_exp = frames.unsqueeze(0).unsqueeze(2)

        q_proj = (q_v.unsqueeze(-2) * frames_exp * sign).sum(-1)
        k_proj = (k_v.unsqueeze(-2) * frames_exp * sign).sum(-1)

        # VERY strong scaling
        scale_v = 1.0 / max(n_vectors * K * 4, 1)

        vec_term = torch.bmm(
            q_proj.reshape(H, N, -1),
            k_proj.reshape(H, N, -1).transpose(-1, -2),
        )

        attn = attn + vec_term * scale_v

    # ── FULL SAFETY BLOCK ───────────────────

    attn = torch.nan_to_num(attn, nan=0.0, posinf=50.0, neginf=-50.0)
    attn = attn.clamp(-50, 50)

    if attn_mask is not None:
        attn = attn.masked_fill(~attn_mask, -1e9)

    # prevent all -inf rows
    row_max = attn.max(dim=-1, keepdim=True).values
    attn = attn - row_max

    attn_weights = torch.softmax(attn, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # ── Scalar output ───────────────────────
    v_s = torch.cat([value[..., :n_scalars], value[..., n_scalars + d_vec:]], dim=-1)
    out_scalar = torch.bmm(attn_weights, v_s)

    if n_vectors == 0:
        return out_scalar

    # ── Vector output ───────────────────────
    v_v = value[..., n_scalars:n_scalars + d_vec].reshape(H, N, n_vectors, 4)

    Fi_g = frames * sign
    v_local = torch.einsum("jkm,hjlm->hjlk", Fi_g, v_v)
    v_local = torch.nan_to_num(v_local)

    out_local = torch.bmm(
        attn_weights,
        v_local.reshape(H, N, -1),
    ).reshape(H, N, n_vectors, K)

    out_v = torch.einsum("hilk,ikm->hilm", out_local, frames)
    out_v = torch.nan_to_num(out_v)

    out_v = out_v.reshape(H, N, d_vec)

    return torch.cat([out_scalar[..., :n_scalars], out_v, out_scalar[..., n_scalars:]], dim=-1)