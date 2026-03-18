"""
Lorentz / Minkowski utilities and LLoCa frame construction.
Note: this was made using Claude AI so it may contain errors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Minkowski metric (+, -, -, -)
_MINK_SIGN = torch.tensor([1.0, -1.0, -1.0, -1.0])


def minkowski_dot(a: torch.Tensor, b: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """
    Lorentz inner product  g_μν a^μ b^ν  with signature (+,-,-,-).

    a, b : (..., 4)
    returns : (...)
    """
    sign = _MINK_SIGN.to(device=a.device, dtype=a.dtype)
    return (a * b * sign).sum(-1)


def minkowski_norm_sq(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Returns signed norm squared.
    """
    norm = minkowski_dot(a, a)
    return torch.where(norm.abs() < eps, torch.sign(norm) * eps, norm)


def pseudorapidity(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """η  from four-momentum (E, px, py, pz)."""
    pz = p[..., 3]
    pt = p[..., 1:3].norm(dim=-1).clamp(min=eps)
    p_mag = (pt**2 + pz**2 + eps).sqrt()
    return 0.5 * torch.log((p_mag + pz + eps) / (p_mag - pz + eps))


def azimuthal_angle(p: torch.Tensor) -> torch.Tensor:
    """φ  from four-momentum (E, px, py, pz)."""
    return torch.atan2(p[..., 2], p[..., 1])


def _gram_schmidt_minkowski(vecs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Gram-Schmidt orthogonalisation under the Minkowski inner product.

    vecs : (N, K, 4)
    returns : (N, K, 4)  — orthonormal Lorentz basis vectors
    """
    ortho: list[torch.Tensor] = []

    for k in range(vecs.shape[1]):
        v = vecs[:, k].clone()  # (N, 4)

        for u in ortho:
            denom = minkowski_dot(u, u).unsqueeze(-1).abs().clamp(min=eps)
            proj = minkowski_dot(v, u).unsqueeze(-1) / denom
            v = v - proj * u

        norm_sq = minkowski_norm_sq(v, eps).unsqueeze(-1)

        # Normalize while preserving sign structure
        v = v / torch.sqrt(norm_sq.abs() + eps)  
        
        ortho.append(v)

    return torch.stack(ortho, dim=1)  # (N, K, 4)


def pairwise_minkowski_distance_sq(p: torch.Tensor) -> torch.Tensor:
    """
    Computes invariant distance:
        d_ij = -(p_i - p_j)^2

    p: (n, 4)
    returns: (n, n)
    """
    diff = p.unsqueeze(1) - p.unsqueeze(0)
    return -minkowski_dot(diff, diff)


def build_lloca_frames(
    particles: torch.Tensor,
    ptr: torch.Tensor,
    K: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build K local canonical Lorentz frames for every particle using its K
    nearest neighbours (by ΔR) within the same event.

    particles : (N, 4)   four-momenta [E, px, py, pz], all events flattened
    ptr       : (B+1,)   event boundaries  (same as the rest of the codebase)
    K         : number of frame 4-vectors per particle
    returns   : (N, K, 4)
    """
    N = particles.shape[0]
    frames = torch.zeros(N, K, 4, device=particles.device, dtype=particles.dtype)
    ptr = ptr.long()

    for b in range(len(ptr) - 1):
        s, e = int(ptr[b]), int(ptr[b + 1])
        ep = particles[s:e]  # (n, 4)
        n = e - s

        if n == 0:
            continue

        # Degenerate case: a single particle has no neighbours
        if n == 1:
            frames[s:e] = ep.unsqueeze(1).expand(-1, K, -1)
            continue

        # Lorentz-invariant distance matrix:
        d_ij = pairwise_minkowski_distance_sq(ep)
        d_ij.fill_diagonal_(float("inf"))

        k_use = min(K, n - 1)
        _, nn_idx = d_ij.topk(k_use, dim=1, largest=False)
        nn_momenta = ep[nn_idx] # (n, k_use, 4)

        # Pad to exactly K vectors when the event is small
        if k_use < K:
            pad = ep.unsqueeze(1).expand(-1, K - k_use, -1)
            nn_momenta = torch.cat([nn_momenta, pad], dim=1)

        frames[s:e] = _gram_schmidt_minkowski(nn_momenta, eps)

    return frames  # (N, K, 4)


def lloca_dot_product_attention(
    query: torch.Tensor,         # (H, N, d_head)
    key: torch.Tensor,           # (H, N, d_head)
    value: torch.Tensor,         # (H, N, d_head)
    frames: torch.Tensor,        # (N, K, 4)
    n_scalars: int,
    n_vectors: int,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = True,
    project_values: bool = False
) -> torch.Tensor:
    """
    LLoCa attention score (Eq. from the paper):

        a_ij = (1/√d_s)  q_s_i · k_s_j
             + (1/√(n_v·K))  Σ_{l,k}  (f_i^k · q_i^{v,l}) (f_i^k · k_j^{v,l})

    The second term is a sum of products of Minkowski inner products, which are
    Lorentz scalars, making the attention score Lorentz invariant.

    Any remaining head channels (beyond n_scalars + n_vectors*4) are treated as
    additional standard scalar channels.
    """
    H, N, d_head = query.shape
    K = frames.shape[1]
    d_vec = n_vectors * 4
    d_scalar_total = n_scalars + max(0, d_head - n_scalars - d_vec)  # include remainder

    sign = _MINK_SIGN.to(device=query.device, dtype=query.dtype)

    # ── Scalar / remainder channels (standard dot product) ──────────────────
    q_s = torch.cat([query[..., :n_scalars], query[..., n_scalars + d_vec:]], dim=-1)
    k_s = torch.cat([key[..., :n_scalars], key[..., n_scalars + d_vec:]], dim=-1)

    scale_s = d_scalar_total**-0.5 if d_scalar_total > 0 else 1.0
    attn = torch.bmm(q_s, k_s.transpose(-1, -2)) * scale_s  # (H, N, N)

    # ── Vector channels (Lorentz-invariant frame projections) ────────────────
    if n_vectors > 0 and K > 0:
        q_v = query[..., n_scalars:n_scalars + d_vec].reshape(H, N, n_vectors, 4)
        k_v = key[..., n_scalars:n_scalars + d_vec].reshape(H, N, n_vectors, 4)

        # frames_exp : (1, N, 1, K, 4)
        frames_exp = frames.unsqueeze(0).unsqueeze(2)

        # Minkowski projection:  proj[h, i, l, k] = g_μν f_i^k^ν  q_i^{v,l,μ}
        # q_v.unsqueeze(-2) : (H, N, n_v, 1, 4)
        q_proj = (q_v.unsqueeze(-2) * frames_exp * sign).sum(-1)  # (H, N, n_v, K)
        k_proj = (k_v.unsqueeze(-2) * frames_exp * sign).sum(-1)  # (H, N, n_v, K)

        q_proj_flat = q_proj.reshape(H, N, n_vectors * K)
        k_proj_flat = k_proj.reshape(H, N, n_vectors * K)

        scale_v = (n_vectors * K) ** -0.5
        attn = attn + torch.bmm(q_proj_flat, k_proj_flat.transpose(-1, -2)) * scale_v

    # ── Mask, softmax, dropout ───────────────────────────────────────────────
    if attn_mask is not None:
        attn = attn.masked_fill(~attn_mask, float("-inf"))

    attn_weights = torch.softmax(attn, dim=-1)

    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # value projection -> turns value vectors into Lorentz scalars by projecting onto the same frames
    if project_values and n_vectors > 0:
        v_v = value[..., n_scalars:n_scalars + d_vec].reshape(H, N, n_vectors, 4)
        frames_exp = frames.unsqueeze(0).unsqueeze(2)

        v_proj = (v_v.unsqueeze(-2) * frames_exp * sign).sum(-1)
        v_proj = v_proj.reshape(H, N, -1)

        v_s = torch.cat([value[..., :n_scalars], value[..., n_scalars + d_vec:]], dim=-1)

        value = torch.cat([v_s, v_proj], dim=-1)

    return torch.bmm(attn_weights, value)  # (H, N, d_head)