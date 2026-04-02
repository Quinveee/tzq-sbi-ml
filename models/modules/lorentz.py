"""
Lorentz / Minkowski utilities and LLoCa modules.

This module provides:
1) Learnable local frame prediction (Frames-Net style)
2) Canonicalization helpers for four-momenta tokens
3) LLoCa attention transport: local -> global SDPA -> local
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_MINK_SIGN = torch.tensor([1.0, -1.0, -1.0, -1.0])


def minkowski_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sign = _MINK_SIGN.to(a.device, a.dtype)
    return (a * b * sign).sum(-1)


def _normalize_by_minkowski_norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mink2 = minkowski_dot(v, v)
    norm = torch.sqrt(torch.clamp(mink2.abs(), min=eps))
    out = v / norm.unsqueeze(-1)

    # Near light-like vectors can explode under Minkowski normalization; use a
    # Euclidean fallback there to keep the frame network stable.
    near_null = mink2.abs() < (10.0 * eps)
    if near_null.any():
        euclid = v.norm(dim=-1, keepdim=True).clamp_min(eps)
        out_euclid = v / euclid
        out = torch.where(near_null.unsqueeze(-1), out_euclid, out)

    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _enforce_timelike(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Project four-vectors to a future-like timelike cone to avoid unstable boosts.
    """
    energy = v[..., :1].abs()
    spatial = v[..., 1:]
    spatial_norm = spatial.norm(dim=-1, keepdim=True)
    energy = torch.maximum(energy, spatial_norm + eps)
    return torch.cat((energy, spatial), dim=-1)


def _pad_by_ptr(values: torch.Tensor, ptr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad flattened per-particle tensors to (batch, max_particles, ...).

    Returns:
        padded: padded tensor with zeros on invalid slots
        valid:  boolean mask of valid particle slots with shape (batch, max_particles)
    """
    ptr = ptr.long()
    lengths = ptr[1:] - ptr[:-1]
    batch_size = int(lengths.numel())

    if batch_size == 0:
        shape = (0, 0) + tuple(values.shape[1:])
        return values.new_zeros(shape), values.new_zeros((0, 0), dtype=torch.bool)

    max_len = int(lengths.max().item())
    if max_len == 0:
        shape = (batch_size, 0) + tuple(values.shape[1:])
        return values.new_zeros(shape), values.new_zeros((batch_size, 0), dtype=torch.bool)

    local_idx = torch.arange(max_len, device=values.device)
    valid = local_idx.unsqueeze(0) < lengths.unsqueeze(1)

    gather_idx = ptr[:-1].unsqueeze(1) + local_idx.unsqueeze(0)
    max_index = max(int(values.shape[0]) - 1, 0)
    gather_idx = gather_idx.clamp_max(max_index)

    padded = values[gather_idx.reshape(-1)].reshape(batch_size, max_len, *values.shape[1:])

    if values.ndim == 2:
        padded = padded.masked_fill(~valid.unsqueeze(-1), 0.0)
    elif values.ndim == 3:
        padded = padded.masked_fill(~valid.unsqueeze(-1).unsqueeze(-1), 0.0)
    else:
        view_shape = valid.shape + (1,) * (values.ndim - 1)
        padded = padded.masked_fill(~valid.view(view_shape), 0.0)

    return padded, valid


def restframe_boost(fourmomenta: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Construct Lorentz boosts that map each four-vector into its rest frame.

    Input shape: (..., 4)
    Output shape: (..., 4, 4)
    """
    t0 = fourmomenta[..., :1]
    beta = fourmomenta[..., 1:] / t0.clamp_min(eps)
    beta2 = beta.square().sum(dim=-1, keepdim=True)
    # Limit ultra-relativistic boosts to keep gamma in a numerically safe range.
    beta2 = torch.clamp(beta2, max=1.0 - 1e-4)

    gamma = torch.rsqrt((1.0 - beta2).clamp_min(eps))
    boost = -gamma * beta

    eye3 = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye3 = eye3.reshape(*(1,) * len(fourmomenta.shape[:-1]), 3, 3)
    eye3 = eye3.expand(*fourmomenta.shape[:-1], 3, 3)

    scale = (gamma - 1.0) / beta2.clamp_min(eps)
    outer = beta.unsqueeze(-1) * beta.unsqueeze(-2)
    rot = eye3 + scale.unsqueeze(-1) * outer

    row0 = torch.cat((gamma, boost), dim=-1)
    lower = torch.cat((boost.unsqueeze(-1), rot), dim=-1)
    return torch.cat((row0.unsqueeze(-2), lower), dim=-2)


def _orthogonalize_3d(references: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    GS3 orthonormalization in the boosted rest frame.

    references shape: (..., 2, 3)
    returns spatial rotation block with shape (..., 3, 3), stored by rows.
    """
    w1 = references[..., 0, :]
    w2 = references[..., 1, :]

    def norm3(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    e1 = norm3(w1)

    proj = (w2 * e1).sum(dim=-1, keepdim=True)
    u2 = w2 - proj * e1
    u2_norm = u2.norm(dim=-1, keepdim=True)

    # Deterministic fallback if vectors become nearly collinear.
    abs_e1 = e1.abs()
    idx = abs_e1.argmin(dim=-1, keepdim=True)
    axis = torch.zeros_like(e1)
    axis.scatter_(-1, idx, 1.0)
    u2_fallback = axis - (axis * e1).sum(dim=-1, keepdim=True) * e1
    u2 = torch.where(u2_norm > eps, u2, u2_fallback)

    e2 = norm3(u2)
    e3 = norm3(torch.linalg.cross(e1, e2, dim=-1))

    return torch.stack((e1, e2, e3), dim=-2)


def polar_decomposition_frames(
    boost_vectors: torch.Tensor,
    references: torch.Tensor,
    eps: float = 1e-10,
    use_float64: bool = True,
) -> torch.Tensor:
    """
    Construct local Lorentz frames from one boost vector and two references.

    boost_vectors shape: (..., 4)
    references shape: (..., 2, 4)
    returns frames L with shape (..., 4, 4), where local vectors are x_L = x @ L^T.
    """
    if use_float64:
        in_dtype = boost_vectors.dtype
        boost_vectors = boost_vectors.to(torch.float64)
        references = references.to(torch.float64)

    boost_vectors = _enforce_timelike(boost_vectors, eps=eps)
    boost_vectors = _normalize_by_minkowski_norm(boost_vectors, eps=eps)
    references = torch.nan_to_num(references, nan=0.0, posinf=0.0, neginf=0.0)
    boost = restframe_boost(boost_vectors, eps=eps)

    references_rest = torch.matmul(references, boost.transpose(-1, -2))
    rotation_block = _orthogonalize_3d(references_rest[..., 1:], eps=eps)

    rotation = torch.zeros_like(boost)
    rotation[..., 0, 0] = 1.0
    rotation[..., 1:, 1:] = rotation_block

    frames = torch.matmul(rotation, boost)

    if use_float64:
        frames = frames.to(in_dtype)

    return frames


class LLoCaFramePredictor(nn.Module):
    """
    Minimal Frames-Net inspired by Eq. (13) of the LLoCa paper.

    A small MLP predicts per-pair weights from Lorentz scalars; weighted sums of
    normalized (p_i + p_j) vectors are then orthonormalized via polar decomposition.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        eps: float = 1e-8,
        use_float64: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.use_float64 = use_float64

        layers: list[nn.Module] = []
        if num_layers <= 0:
            layers.append(nn.LazyLinear(3))
        else:
            layers.append(nn.LazyLinear(hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, 3))

        self.phi = nn.Sequential(*layers)

    def _predict_vectors(
        self,
        particles: torch.Tensor,
        ptr: torch.Tensor,
        scalars: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ptr = ptr.long()
        if particles.shape[0] == 0:
            return particles.new_zeros((0, 3, 4))

        p_pad, valid = _pad_by_ptr(particles, ptr)  # (B, M, 4), (B, M)
        pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, M, M)

        pi = p_pad.unsqueeze(2)  # (B, M, 1, 4)
        pj = p_pad.unsqueeze(1)  # (B, 1, M, 4)

        pij = _normalize_by_minkowski_norm(pi + pj, eps=self.eps)  # (B, M, M, 4)
        mij = minkowski_dot(pi, pj).unsqueeze(-1)  # (B, M, M, 1)

        if scalars is not None:
            s_pad, _ = _pad_by_ptr(scalars, ptr)  # (B, M, S)
            si = s_pad.unsqueeze(2).expand(-1, -1, p_pad.shape[1], -1)
            sj = s_pad.unsqueeze(1).expand(-1, p_pad.shape[1], -1, -1)
            phi_in = torch.cat((mij, si, sj), dim=-1)
        else:
            phi_in = mij

        phi_in = torch.nan_to_num(phi_in, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.phi(phi_in)  # (B, M, M, 3)

        min_value = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~pair_mask.unsqueeze(-1), min_value)
        weights = torch.softmax(logits, dim=2)
        weights = torch.where(pair_mask.unsqueeze(-1), weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(self.eps)

        vecs = torch.einsum("bijk,bijm->bikm", weights, pij)  # (B, M, 3, 4)

        # Event-level normalization for numerical stability.
        sq = minkowski_dot(vecs, vecs).abs().sum(dim=1, keepdim=True).clamp_min(self.eps)
        vecs = vecs / sq.sqrt().unsqueeze(-1)
        vecs = torch.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)

        # Unpad back to flattened particle order.
        return vecs[valid]

    def forward(
        self,
        particles: torch.Tensor,
        ptr: torch.Tensor,
        scalars: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vecs = self._predict_vectors(particles, ptr, scalars=scalars)
        boost = vecs[:, 0, :]
        references = vecs[:, 1:, :]
        return polar_decomposition_frames(
            boost,
            references,
            eps=self.eps,
            use_float64=self.use_float64,
        )


def _fallback_frames_from_particles(particles: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """
    Deterministic fallback when no learnable Frames-Net is provided.
    """
    device, dtype = particles.device, particles.dtype
    ptr = ptr.long()

    boost = _normalize_by_minkowski_norm(particles)
    references = torch.zeros(
        particles.shape[0],
        2,
        4,
        device=device,
        dtype=dtype,
    )

    for b in range(len(ptr) - 1):
        s, e = int(ptr[b]), int(ptr[b + 1])
        if e <= s:
            continue

        p = particles[s:e]
        n = p.shape[0]

        if n == 1:
            references[s:e, 0, 1] = 1.0
            references[s:e, 1, 2] = 1.0
            continue

        diff = p.unsqueeze(1) - p.unsqueeze(0)
        d2 = (diff[..., 1:] ** 2).sum(-1)
        d2.fill_diagonal_(float("inf"))

        k = min(2, n - 1)
        _, idx = torch.topk(d2, k=k, dim=1, largest=False)
        refs = p[idx]

        if k == 1:
            refs = torch.cat((refs, refs), dim=1)

        references[s:e] = refs[:, :2, :]

    return polar_decomposition_frames(boost, references)


def _valid_frame_mask(frames: torch.Tensor, det_eps: float = 1e-10) -> torch.Tensor:
    """
    Return a per-frame mask for finite and non-singular 4x4 transforms.
    """
    frames64 = frames.to(torch.float64)
    finite = torch.isfinite(frames64).all(dim=-1).all(dim=-1)
    det = torch.linalg.det(frames64)
    nonsingular = det.abs() > det_eps
    return finite & nonsingular


def safe_inverse_frames(frames: torch.Tensor, det_eps: float = 1e-10) -> torch.Tensor:
    """
    Robust batched inverse with pseudo-inverse fallback for singular frames.
    """
    flat = frames.reshape(-1, 4, 4)
    flat64 = flat.to(torch.float64)

    good = _valid_frame_mask(flat64, det_eps=det_eps)
    inv64 = torch.empty_like(flat64)

    if good.any():
        inv64[good] = torch.linalg.inv(flat64[good])
    if (~good).any():
        inv64[~good] = torch.linalg.pinv(flat64[~good], rtol=det_eps)

    inv = inv64.to(flat.dtype)
    return inv.reshape_as(frames)


def _replace_invalid_frames(
    frames: torch.Tensor,
    particles: torch.Tensor,
    ptr: torch.Tensor,
    det_eps: float = 1e-10,
) -> torch.Tensor:
    """
    Replace invalid predicted frames with deterministic particle-based frames.
    """
    valid = _valid_frame_mask(frames, det_eps=det_eps)
    if bool(valid.all()):
        return frames

    fallback = _fallback_frames_from_particles(particles, ptr)
    return torch.where(valid.unsqueeze(-1).unsqueeze(-1), frames, fallback)


def build_lloca_frames(
    particles: torch.Tensor,
    ptr: torch.Tensor,
    K: int | None = None,
    frame_predictor: LLoCaFramePredictor | None = None,
    scalars: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build per-particle local Lorentz frames.

    Kept compatible with previous call sites through the optional `K` argument.
    """
    _ = K  # kept for backwards compatibility with older configs
    if frame_predictor is not None:
        frames = frame_predictor(particles, ptr, scalars=scalars)
        return _replace_invalid_frames(frames, particles, ptr)
    return _fallback_frames_from_particles(particles, ptr)


def canonicalize_input_fourmomenta(tokens: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize the final four channels of token features using local frames.
    """
    if tokens.shape[-1] < 4:
        return tokens

    vec = tokens[..., -4:]
    vec_local = torch.einsum("nm,nam->na", vec, frames)
    vec_local = torch.nan_to_num(vec_local, nan=0.0, posinf=1e6, neginf=-1e6)
    if tokens.shape[-1] == 4:
        return vec_local
    return torch.cat((tokens[..., :-4], vec_local), dim=-1)


def _transform_vector_channels(
    tensor: torch.Tensor,
    frames: torch.Tensor,
    n_scalars: int,
    n_vectors: int,
    *,
    to_global: bool,
    inv_frames: torch.Tensor | None = None,
    lower_key: bool = False,
) -> torch.Tensor:
    if n_vectors == 0:
        return tensor

    d_vec = n_vectors * 4
    if n_scalars + d_vec > tensor.shape[-1]:
        raise ValueError(
            f"Invalid scalar/vector split: n_scalars={n_scalars}, n_vectors={n_vectors}, "
            f"head_dim={tensor.shape[-1]}"
        )

    start = n_scalars
    stop = start + d_vec

    mats = inv_frames if to_global and inv_frames is not None else None
    if mats is None:
        mats = safe_inverse_frames(frames) if to_global else frames

    vec = tensor[..., start:stop].reshape(*tensor.shape[:-1], n_vectors, 4)
    vec = torch.einsum("...nvm,nam->...nva", vec, mats)
    vec = torch.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)

    if lower_key:
        sign = _MINK_SIGN.to(vec.device, vec.dtype)
        vec = vec * sign

    vec_flat = vec.reshape(*tensor.shape[:-1], d_vec)
    before = tensor[..., :start] if start > 0 else None
    after = tensor[..., stop:] if stop < tensor.shape[-1] else None

    if before is None and after is None:
        return vec_flat
    if before is None:
        return torch.cat((vec_flat, after), dim=-1)
    if after is None:
        return torch.cat((before, vec_flat), dim=-1)
    return torch.cat((before, vec_flat, after), dim=-1)


def lloca_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    frames: torch.Tensor,
    n_scalars: int,
    n_vectors: int,
    attn_mask: torch.Tensor | None = None,
    inv_frames: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    LLoCa attention transport in the style of Eq. (12):
    local q/k/v -> global attention -> local output.
    """
    q_global = _transform_vector_channels(
        query,
        frames,
        n_scalars,
        n_vectors,
        to_global=True,
        inv_frames=inv_frames,
        lower_key=False,
    )
    k_global = _transform_vector_channels(
        key,
        frames,
        n_scalars,
        n_vectors,
        to_global=True,
        inv_frames=inv_frames,
        lower_key=True,
    )
    v_global = _transform_vector_channels(
        value,
        frames,
        n_scalars,
        n_vectors,
        to_global=True,
        inv_frames=inv_frames,
        lower_key=False,
    )

    out_global = F.scaled_dot_product_attention(
        q_global,
        k_global,
        v_global,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
    )

    return _transform_vector_channels(
        out_global,
        frames,
        n_scalars,
        n_vectors,
        to_global=False,
        lower_key=False,
    )