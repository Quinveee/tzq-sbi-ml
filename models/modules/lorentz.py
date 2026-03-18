"""
Lorentz / Minkowski utilities and LLoCa frame construction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Minkowski metric (+, -, -, -)
_MINK_SIGN = torch.tensor([1.0, -1.0, -1.0, -1.0])


def minkowski_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sign = _MINK_SIGN.to(device=a.device, dtype=a.dtype)
    return (a * b * sign).sum(-1)


def minkowski_norm_sq(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = minkowski_dot(a, a)
    return torch.where(norm.abs() < eps, torch.sign(norm) * eps, norm)


def pseudorapidity(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pz = p[..., 3]
    pt = p[..., 1:3].norm(dim=-1).clamp(min=eps)
    p_mag = (pt**2 + pz**2 + eps).sqrt()
    return 0.5 * torch.log((p_mag + pz + eps) / (p_mag - pz + eps))


def azimuthal_angle(p: torch.Tensor) -> torch.Tensor:
    return torch.atan2(p[..., 2], p[..., 1])


def _gram_schmidt_minkowski(vecs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    vecs : (N, K, 4)
    returns : (N, K, 4) — Minkowski-orthonormal basis vectors
    """
    ortho: list[torch.Tensor] = []
    for k in range(vecs.shape[1]):
        v = vecs[:, k].clone()
        for u in ortho:
            denom = minkowski_dot(u, u).unsqueeze(-1).abs().clamp(min=eps)
            proj = minkowski_dot(v, u).unsqueeze(-1) / denom
            v = v - proj * u
        norm_sq = minkowski_norm_sq(v, eps).unsqueeze(-1)
        v = v / torch.sqrt(norm_sq.abs() + eps)
        ortho.append(v)
    return torch.stack(ortho, dim=1)


def pairwise_minkowski_distance_sq(p: torch.Tensor) -> torch.Tensor:
    diff = p.unsqueeze(1) - p.unsqueeze(0)
    return -minkowski_dot(diff, diff)


def build_lloca_frames(
    particles: torch.Tensor,
    ptr: torch.Tensor,
    K: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build K local canonical Lorentz frames per particle using K nearest
    neighbours by Lorentz-invariant distance within the same event.

    particles : (N, 4)
    ptr       : (B+1,)
    returns   : (N, K, 4)
    """
    N = particles.shape[0]
    frames = torch.zeros(N, K, 4, device=particles.device, dtype=particles.dtype)
    ptr = ptr.long()

    for b in range(len(ptr) - 1):
        s, e = int(ptr[b]), int(ptr[b + 1])
        ep = particles[s:e]
        n = e - s

        if n == 0:
            continue
        if n == 1:
            frames[s:e] = ep.unsqueeze(1).expand(-1, K, -1)
            continue

        d_ij = pairwise_minkowski_distance_sq(ep)
        d_ij.fill_diagonal_(float("inf"))

        k_use = min(K, n - 1)
        _, nn_idx = d_ij.topk(k_use, dim=1, largest=False)
        nn_momenta = ep[nn_idx]  # (n, k_use, 4)

        if k_use < K:
            pad = ep.unsqueeze(1).expand(-1, K - k_use, -1)
            nn_momenta = torch.cat([nn_momenta, pad], dim=1)

        frames[s:e] = _gram_schmidt_minkowski(nn_momenta, eps)

    return frames  # (N, K, 4)


def lloca_dot_product_attention(
    query: torch.Tensor,          # (H, N, d_head)
    key: torch.Tensor,            # (H, N, d_head)
    value: torch.Tensor,          # (H, N, d_head)
    frames: torch.Tensor,         # (N, K, 4)
    n_scalars: int,
    n_vectors: int,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """
    LLoCa attention with equivariant value aggregation.

    Attention score (Lorentz invariant):
        a_ij = (1/√d_s)  q_s_i · k_s_j
             + (1/√(n_v·K))  Σ_{l,k}  (f_i^k · q_i^{v,l}) (f_i^k · k_j^{v,l})

    Value aggregation (Lorentz equivariant):
        Scalar channels : out_i = Σ_j w_ij v_j^s          (standard)
        Vector channels : out_i = Σ_j w_ij (F_i g F_j^T) v_j^v
                        = F_i [ Σ_j w_ij (F_j g v_j^v) ]  (factored form)

    The factored form avoids materialising N×N×4×4 rotation matrices and
    preserves d_head exactly.
    """
    H, N, d_head = query.shape
    K = frames.shape[1]
    d_vec = n_vectors * 4
    d_scalar_total = n_scalars + max(0, d_head - n_scalars - d_vec)

    sign = _MINK_SIGN.to(device=query.device, dtype=query.dtype)

    # ── Attention scores ─────────────────────────────────────────────────────

    # Scalar channels (standard dot product, includes remainder)
    q_s = torch.cat([query[..., :n_scalars], query[..., n_scalars + d_vec:]], dim=-1)
    k_s = torch.cat([key[..., :n_scalars], key[..., n_scalars + d_vec:]], dim=-1)
    scale_s = d_scalar_total ** -0.5 if d_scalar_total > 0 else 1.0
    attn = torch.bmm(q_s, k_s.transpose(-1, -2)) * scale_s  # (H, N, N)

    # Vector channels (Lorentz-invariant frame projections)
    if n_vectors > 0 and K > 0:
        q_v = query[..., n_scalars:n_scalars + d_vec].reshape(H, N, n_vectors, 4)
        k_v = key[..., n_scalars:n_scalars + d_vec].reshape(H, N, n_vectors, 4)

        frames_exp = frames.unsqueeze(0).unsqueeze(2)           # (1, N, 1, K, 4)
        q_proj = (q_v.unsqueeze(-2) * frames_exp * sign).sum(-1)  # (H, N, n_v, K)
        k_proj = (k_v.unsqueeze(-2) * frames_exp * sign).sum(-1)

        scale_v = (n_vectors * K) ** -0.5
        attn = attn + torch.bmm(
            q_proj.reshape(H, N, n_vectors * K),
            k_proj.reshape(H, N, n_vectors * K).transpose(-1, -2),
        ) * scale_v

    # ── Mask, softmax, dropout ───────────────────────────────────────────────
    if attn_mask is not None:
        attn = attn.masked_fill(~attn_mask, float("-inf"))

    attn_weights = torch.softmax(attn, dim=-1)
    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # ── Equivariant value aggregation ────────────────────────────────────────

    # Scalar / remainder channels: standard weighted sum
    v_s = torch.cat([value[..., :n_scalars], value[..., n_scalars + d_vec:]], dim=-1)
    out_scalar = torch.bmm(attn_weights, v_s)  # (H, N, n_scalars + remainder)

    if n_vectors == 0:
        return out_scalar  # d_head == n_scalars when n_vectors == 0

    # Vector channels: factored frame-to-frame rotation
    #   F_i g F_j^T v_j  ==  F_i [ (F_j * sign) @ v_j ]
    #                           ↑ un-project from j    ↑ re-project into i
    v_v = value[..., n_scalars:n_scalars + d_vec].reshape(H, N, n_vectors, 4)

    # Step 1 — project v_j onto its own frame (gives Lorentz-scalar local coords)
    #   v_local[h, j, l, k] = Σ_μ  (frames[j,k,μ] * sign_μ) * v_v[h,j,l,μ]
    Fi_g = frames * sign                              # (N, K, 4)
    v_local = torch.einsum("jkm,hjlm->hjlk", Fi_g, v_v)  # (H, N, n_v, K)

    # Step 2 — weighted sum of local coords (scalars, so frame-independent)
    out_local = torch.bmm(
        attn_weights,
        v_local.reshape(H, N, n_vectors * K),
    ).reshape(H, N, n_vectors, K)                    # (H, N, n_v, K)

    # Step 3 — reconstruct in frame i (un-project into global representation)
    #   out_v[h, i, l, μ] = Σ_k  out_local[h,i,l,k] * frames[i,k,μ]
    out_v = torch.einsum("hilk,ikm->hilm", out_local, frames)  # (H, N, n_v, 4)
    out_v = out_v.reshape(H, N, d_vec)               # (H, N, n_v * 4)

    # ── Reconstruct in original channel order → (H, N, d_head) ──────────────
    out_s1 = out_scalar[..., :n_scalars]
    out_s2 = out_scalar[..., n_scalars:]
    return torch.cat([out_s1, out_v, out_s2], dim=-1)