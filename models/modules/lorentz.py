"""
Lorentz / Minkowski utilities and LLoCa modules.

Ported from https://github.com/heidelberg-hepml/lloca, adapted for
flattened particle tensors of shape (N_total, 4) accompanied by a ptr
tensor of shape (B+1,). The structure and nomenclature mirror the
original as closely as practical:

  1. Minkowski utilities (lorentz_inner, lorentz_squarednorm, lorentz_eye)
  2. 3D orthogonalization (Gram-Schmidt) with collinearity regularization
  3. Polar decomposition: boost + rotation, with lightlike regularization
  4. Clamp-boost regularization (gamma_max)
  5. Frames / InverseFrames / LowerIndicesFrames bookkeeping (analytic inverse)
  6. Edge-convolution Frames-Net (MLPVectors-style)
  7. LLoCa attention transport (Eq. 12 of the paper)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


# ------------------------------------------------------------
# Minkowski utilities
# ------------------------------------------------------------
def lorentz_inner(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """v1^T g v2 with g = diag(1, -1, -1, -1). Last dim must be 4."""
    return v1[..., 0] * v2[..., 0] - (v1[..., 1:] * v2[..., 1:]).sum(dim=-1)


def lorentz_squarednorm(v: torch.Tensor) -> torch.Tensor:
    return lorentz_inner(v, v)


def lorentz_eye(dims, device=None, dtype=torch.float32) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    base = torch.eye(4, dtype=dtype, device=device)
    return base.view((1,) * len(dims) + (4, 4)).expand(*dims, 4, 4)


# ------------------------------------------------------------
# 3D orthogonalization
# ------------------------------------------------------------
def regularize_collinear(vecs: torch.Tensor, eps_reg: float | None = None):
    eps_reg = torch.finfo(vecs.dtype).eps if eps_reg is None else eps_reg
    v0, v1 = vecs.unbind(dim=-2)
    cross = torch.linalg.cross(v0, v1, dim=-1)
    mask = (cross ** 2).sum(dim=-1) < eps_reg
    v0_reg = torch.where(mask.unsqueeze(-1), v0 + eps_reg * torch.randn_like(v0), v0)
    v1_reg = torch.where(mask.unsqueeze(-1), v1 + eps_reg * torch.randn_like(v1), v1)
    return torch.stack([v0_reg, v1_reg], dim=-2), mask.sum()


def orthogonalize_gramschmidt_3d(vecs: torch.Tensor, eps_norm: float) -> torch.Tensor:
    vecs = F.normalize(vecs, dim=-1, eps=eps_norm)
    e0, v1 = vecs.unbind(dim=-2)
    u1 = v1 - (v1 * e0).sum(dim=-1, keepdim=True) * e0
    e1 = F.normalize(u1, dim=-1, eps=eps_norm)
    e2 = torch.linalg.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-2)


def orthogonalize_3d(
    vecs: torch.Tensor,
    eps_norm: float | None = None,
    eps_reg: float | None = None,
    return_reg: bool = False,
):
    eps_norm = torch.finfo(vecs.dtype).eps if eps_norm is None else eps_norm
    vecs, reg_collinear = regularize_collinear(vecs, eps_reg)
    trafo = orthogonalize_gramschmidt_3d(vecs, eps_norm)
    return (trafo, reg_collinear) if return_reg else trafo


# ------------------------------------------------------------
# Lightlike regularization and rest-frame boost
# ------------------------------------------------------------
def regularize_lightlike(vecs: torch.Tensor, eps_reg_lightlike: float | None = None):
    """Nudge near-lightlike vectors onto a timelike direction, preserving gradients."""
    eps_reg_lightlike = (
        torch.finfo(vecs.dtype).eps if eps_reg_lightlike is None else eps_reg_lightlike
    )
    inners = lorentz_squarednorm(vecs)
    mask = inners.abs() < eps_reg_lightlike
    randn = torch.randn_like(vecs).abs()
    randn_3sq = (randn[..., 1:] ** 2).sum(dim=-1)
    randn[..., 0] = (2 * randn_3sq).sqrt()  # heuristic factor of 2 to stay timelike
    vecs_reg = vecs + mask.unsqueeze(-1) * eps_reg_lightlike * randn
    return vecs_reg, mask.sum()


def restframe_boost(fourmomenta: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Lorentz boost into the rest frame of each four-vector."""
    t0 = fourmomenta.narrow(-1, 0, 1)
    beta = fourmomenta[..., 1:] / t0.clamp_min(eps)
    beta2 = beta.square().sum(dim=-1, keepdim=True)
    one_minus_beta2 = torch.clamp_min(1 - beta2, min=eps)
    gamma = torch.rsqrt(one_minus_beta2)
    boost = -gamma * beta

    eye3 = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye3 = eye3.reshape(*(1,) * len(fourmomenta.shape[:-1]), 3, 3).expand(
        *fourmomenta.shape[:-1], 3, 3
    )
    scale = (gamma - 1) / torch.clamp_min(beta2, min=eps)
    outer = beta.unsqueeze(-1) * beta.unsqueeze(-2)
    rot = eye3 + scale.unsqueeze(-1) * outer

    row0 = torch.cat((gamma, boost), dim=-1)
    lower = torch.cat((boost.unsqueeze(-1), rot), dim=-1)
    return torch.cat((row0.unsqueeze(-2), lower), dim=-2)


def polar_decomposition(
    fourmomenta: torch.Tensor,
    references: torch.Tensor,
    use_float64: bool = True,
    eps_norm: float | None = None,
    eps_reg: float | None = None,
    eps_reg_lightlike: float | None = None,
    return_reg: bool = False,
) -> torch.Tensor:
    """Lorentz transformation as a polar decomposition: boost then rotation."""
    assert fourmomenta.shape[:-1] == references.shape[:-2]

    if use_float64:
        orig_dtype = fourmomenta.dtype
        fourmomenta = fourmomenta.to(torch.float64)
        references = references.to(torch.float64)

    fourmomenta, reg_lightlike = regularize_lightlike(fourmomenta, eps_reg_lightlike)
    boost = restframe_boost(fourmomenta)
    ref_rest = torch.matmul(references, boost.transpose(-1, -2))

    out = orthogonalize_3d(
        ref_rest[..., 1:], eps_norm=eps_norm, eps_reg=eps_reg, return_reg=return_reg
    )
    if return_reg:
        ortho_3d, reg_collinear = out
    else:
        ortho_3d = out

    rotation = torch.zeros_like(boost)
    rotation[..., 0, 0] = 1
    rotation[..., 1:, 1:] = ortho_3d
    trafo = torch.matmul(rotation, boost)

    if use_float64:
        trafo = trafo.to(orig_dtype)
    return (trafo, reg_lightlike, reg_collinear) if return_reg else trafo


# ------------------------------------------------------------
# Gamma-factor regularization on predicted boost vectors
# ------------------------------------------------------------
def _soft_clamp(x, min=None, max=None, hardness=None):
    if hardness is None:
        return x.clamp(min=min, max=max)
    out = max - F.softplus(max - x, beta=hardness)
    return out.clamp(min=min)


def clamp_boost(x: torch.Tensor, gamma_max: float | None, gamma_hardness: float | None = None) -> torch.Tensor:
    """Cap gamma at gamma_max; rescale beta so the Minkowski norm is preserved."""
    if gamma_max is None:
        return x
    mass = lorentz_squarednorm(x).clamp(min=0).sqrt().unsqueeze(-1)
    t0 = x.narrow(-1, 0, 1)
    beta = x[..., 1:] / t0.clamp_min(1e-10)
    gamma = t0 / mass.clamp_min(1e-10)
    gamma_reg = _soft_clamp(gamma, min=1, max=gamma_max, hardness=gamma_hardness)
    beta_scaling = (
        torch.sqrt(torch.clamp(1 - 1 / gamma_reg.clamp(min=1e-10).square(), min=1e-10))
        / (beta ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-10).sqrt()
    )
    beta_reg = beta * beta_scaling
    return mass * torch.cat((gamma_reg, gamma_reg * beta_reg), dim=-1)


# ------------------------------------------------------------
# Frames bookkeeping with analytic inverse (g L^T g)
# ------------------------------------------------------------
class Frames:
    """Collection of Lorentz transformations (..., 4, 4) with cached inverse/det.

    The inverse is computed analytically as g L^T g (exploiting
    Lorentz-orthogonality), which is exact and cheap compared to a
    numerical matrix inverse.
    """

    def __init__(
        self,
        matrices: torch.Tensor | None = None,
        is_global: bool = False,
        det: torch.Tensor | None = None,
        inv: torch.Tensor | None = None,
        is_identity: bool = False,
        shape=None,
        device=None,
        dtype=None,
    ):
        self.is_identity = is_identity
        if is_identity:
            if matrices is None:
                assert shape is not None and device is not None and dtype is not None
            else:
                shape = matrices.shape[:-2]
                device = matrices.device
                dtype = matrices.dtype
            self.matrices = lorentz_eye(shape, device=device, dtype=dtype)
            self.is_global = True
            self.det = torch.ones(self.matrices.shape[:-2], dtype=dtype, device=device)
            self.inv = self.matrices
            return

        assert matrices is not None
        assert matrices.shape[-2:] == (4, 4)
        self.matrices = matrices
        self.is_global = is_global
        self.det = det
        self.inv = inv

        if self.det is None:
            self.det = torch.linalg.det(self.matrices.to(torch.float64)).to(self.matrices.dtype)
        if self.inv is None:
            inv_mat = self.matrices.transpose(-1, -2).clone()
            inv_mat[..., 1:, :] *= -1
            inv_mat[..., :, 1:] *= -1
            self.inv = inv_mat

    @property
    def device(self):
        return self.matrices.device

    @property
    def dtype(self):
        return self.matrices.dtype

    @property
    def shape(self):
        return self.matrices.shape


class InverseFrames(Frames):
    def __init__(self, frames: Frames):
        super().__init__(
            matrices=frames.inv,
            is_global=frames.is_global,
            inv=frames.matrices,
            det=frames.det,
            is_identity=frames.is_identity,
        )


class LowerIndicesFrames(Frames):
    """Frames with the first index lowered by the metric: g L.

    Used in LLoCa attention to transport keys into the global frame.
    """

    def __init__(self, frames: Frames):
        matrices = frames.matrices.clone()
        matrices[..., 1:, :] *= -1
        inv = frames.inv.clone()
        inv[..., :, 1:] *= -1
        det = -frames.det if frames.det is not None else None
        super().__init__(
            matrices=matrices,
            inv=inv,
            det=det,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
        )


# ------------------------------------------------------------
# Edge index for flattened particles (fully connected per event)
# ------------------------------------------------------------
def _edge_index_from_ptr(ptr: torch.Tensor, remove_self_loops: bool = True) -> torch.Tensor:
    ptr = ptr.long()
    lengths = ptr[1:] - ptr[:-1]
    device = ptr.device
    n_edges = lengths * lengths
    total = int(n_edges.sum().item())
    if total == 0:
        return ptr.new_zeros((2, 0))

    starts = torch.cat([ptr.new_zeros(1), n_edges.cumsum(0)])[:-1]
    edge_local = torch.arange(total, device=device) - starts.repeat_interleave(n_edges)
    n_per_event = lengths.repeat_interleave(n_edges)
    i_local = torch.div(edge_local, n_per_event, rounding_mode="floor")
    j_local = edge_local - i_local * n_per_event
    event_start = ptr[:-1].repeat_interleave(n_edges)
    src = event_start + i_local
    dst = event_start + j_local
    if remove_self_loops:
        keep = src != dst
        src, dst = src[keep], dst[keep]
    return torch.stack((src, dst), dim=0)


def _scatter_softmax(logits: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    logit_max = scatter(logits.detach(), index, dim=0, dim_size=dim_size, reduce="max")
    exp = (logits - logit_max[index]).exp()
    denom = scatter(exp, index, dim=0, dim_size=dim_size, reduce="sum").clamp_min(1e-16)
    return exp / denom[index]


# ------------------------------------------------------------
# Equivariant edge convolution (Frames-Net backbone)
# Mirrors lloca.equivectors.mlp.EquiEdgeConv / MLPVectors.
# ------------------------------------------------------------
class _EquiEdgeConv(nn.Module):
    """Predict ``n_vectors`` equivariant four-vectors per particle.

    For every directed edge (i, j) within the same event:
      logits_ij = MLP([s_i, s_j, edge_attr_ij])                 with edge_attr_ij a Lorentz scalar
      w_ij      = softmax_j( logits_ij )  (per receiver i)      ensures positive, sum-to-one weights
      fm_rel_ij = (p_i + p_j) / |p_i + p_j|_Mink
      v_i       = sum_j w_ij * fm_rel_ij                         (per-node equivariant vectors)
      v_i       /= sqrt(|sum_v ||v||^2|)                          (per-particle Minkowski layer-norm)
    """

    def __init__(
        self,
        n_vectors: int,
        hidden_channels: int,
        num_layers: int,
        include_edges: bool = True,
        fm_norm: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.n_vectors = n_vectors
        self.include_edges = include_edges
        self.fm_norm = fm_norm
        self.layer_norm = layer_norm

        layers: list[nn.Module] = [nn.LazyLinear(hidden_channels), nn.ReLU()]
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(hidden_channels, hidden_channels), nn.ReLU()]
        layers += [nn.Linear(hidden_channels, n_vectors)]
        self.mlp = nn.Sequential(*layers)

        if include_edges:
            self.register_buffer("edge_inited", torch.tensor(False))
            self.register_buffer("edge_mean", torch.tensor(0.0))
            self.register_buffer("edge_std", torch.tensor(1.0))

    def _maybe_init_standardization(self, edge_attr: torch.Tensor) -> None:
        if not bool(self.edge_inited):
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            self.edge_inited.fill_(True)

    def forward(
        self,
        fourmomenta: torch.Tensor,
        scalars: torch.Tensor | None,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Padded implementation. Inputs:
            fourmomenta: (B, L, 4)
            scalars:     (B, L, S) or None
            valid_mask:  (B, L) bool — True for real particles.

        Returns (B, L, n_vectors, 4) with pad rows zeroed.
        """
        B, L, _ = fourmomenta.shape
        if B == 0 or L == 0:
            return fourmomenta.new_zeros((B, L, self.n_vectors, 4))

        # Pairwise Lorentz inner: edge_attr[b, i, j] = <p_i, p_j>
        p_i = fourmomenta.unsqueeze(2)  # (B, L, 1, 4)
        p_j = fourmomenta.unsqueeze(1)  # (B, 1, L, 4)
        edge_attr = None
        if self.include_edges:
            edge_attr = lorentz_inner(p_i, p_j).unsqueeze(-1)  # (B, L, L, 1)
            # Standardize using only valid pairs to avoid pad contamination.
            valid_pair = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
            self._maybe_init_standardization(edge_attr[valid_pair])
            edge_attr = (edge_attr - self.edge_mean) / self.edge_std

        if scalars is None:
            scalars = fourmomenta.new_zeros((B, L, 0))
        s_i = scalars.unsqueeze(2).expand(-1, -1, L, -1)  # (B, L, L, S)
        s_j = scalars.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, L, S)
        mlp_in_parts = [s_i, s_j]
        if edge_attr is not None:
            mlp_in_parts.append(edge_attr.to(s_i.dtype))
        logits = self.mlp(torch.cat(mlp_in_parts, dim=-1))  # (B, L, L, n_vectors)

        # Mask out self-loops and any pair touching padding. Per-receiver
        # softmax over the j axis gives valid weights only over real keys.
        eye = torch.eye(L, device=fourmomenta.device, dtype=torch.bool)
        valid_pair = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1) & ~eye
        logits = logits.masked_fill(~valid_pair.unsqueeze(-1), float("-inf"))

        # Softmax over j (dim 2). A row with no valid keys (i.e. event of size
        # 1, padded row, etc.) becomes all-NaN; nan_to_num below clears it.
        weights = torch.nn.functional.softmax(logits, dim=2)
        weights = torch.nan_to_num(weights, nan=0.0)

        fm_rel = p_i + p_j  # (B, L, L, 4)
        if self.fm_norm:
            norm = lorentz_squarednorm(fm_rel).abs().sqrt().clamp_min(1e-6).unsqueeze(-1)
            fm_rel = fm_rel / norm

        # Weighted sum over j → (B, L, n_vectors, 4)
        contribs = weights.unsqueeze(-1) * fm_rel.unsqueeze(-2)
        vecs = contribs.sum(dim=2)

        if self.layer_norm:
            lnorm = lorentz_squarednorm(vecs).sum(dim=-1, keepdim=True)
            vecs = vecs / lnorm.abs().sqrt().clamp_min(1e-5).unsqueeze(-1)

        # Zero pad-rows so downstream code (polar_decomposition, einsums)
        # doesn't see undefined values.
        vecs = vecs * valid_mask[..., None, None].to(vecs.dtype)
        return vecs


# ------------------------------------------------------------
# Learnable frame predictor (polar decomposition)
# Mirrors lloca.framesnet.equi_frames.LearnedPDFrames.
# ------------------------------------------------------------
class LLoCaFramePredictor(nn.Module):
    """Learned polar-decomposition frames on flattened particle tensors.

    Predicts three equivariant vectors per particle (one boost, two
    rotation references) and composes them into a Lorentz frame via
    polar decomposition.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        eps: float = 1e-8,
        use_float64: bool = True,
        gamma_max: float | None = None,
        gamma_hardness: float | None = None,
        include_edges: bool = True,
    ):
        super().__init__()
        self.eps = eps  # kept for backward-compat with existing configs
        self.use_float64 = use_float64
        self.gamma_max = gamma_max
        self.gamma_hardness = gamma_hardness
        self.equivectors = _EquiEdgeConv(
            n_vectors=3,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            include_edges=include_edges,
        )

    def forward(
        self,
        fourmomenta: torch.Tensor,
        valid_mask: torch.Tensor,
        scalars: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """fourmomenta: (B, L, 4); valid_mask: (B, L). Returns (B, L, 4, 4)."""
        vecs = self.equivectors(fourmomenta, scalars=scalars, valid_mask=valid_mask)
        boost = vecs[..., 0, :]
        references = vecs[..., 1:, :]
        boost = clamp_boost(boost, self.gamma_max, self.gamma_hardness)
        return polar_decomposition(boost, references, use_float64=self.use_float64)


# ------------------------------------------------------------
# Deterministic fallback and validity checks
# ------------------------------------------------------------
def _fallback_frames_from_particles(
    particles: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Deterministic fallback frames on padded particles.

    particles: (B, L, 4); valid_mask: (B, L). Returns (B, L, 4, 4).

    For each event we pick, per particle, the two real other particles with
    smallest spatial distance as reference vectors. Events with a single
    particle fall back to canonical x/y unit vectors. Pad rows produce
    identity-like frames (boost=zero, refs=x,y unit) so downstream einsums
    never see undefined data.
    """
    device, dtype = particles.device, particles.dtype
    B, L, _ = particles.shape
    boost = particles.clone()
    references = torch.zeros(B, L, 2, 4, device=device, dtype=dtype)

    # Default refs: x and y unit spatial directions. These get overwritten
    # below for events with at least 2 valid particles.
    references[..., 0, 1] = 1.0
    references[..., 1, 2] = 1.0

    if L < 2:
        return polar_decomposition(boost, references)

    # Pairwise spatial distances; set distances involving padding to +inf so
    # topk never selects them. Self-loops also masked.
    diff = particles.unsqueeze(2) - particles.unsqueeze(1)  # (B, L, L, 4)
    d2 = (diff[..., 1:] ** 2).sum(-1)  # (B, L, L)
    pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    eye = torch.eye(L, device=device, dtype=torch.bool)
    d2 = d2.masked_fill(~pair_valid | eye, float("inf"))

    # Lengths and "events with at least 2 valid particles" mask
    lengths = valid_mask.sum(dim=1)  # (B,)
    has_pair = lengths >= 2

    if has_pair.any():
        # For events with valid pairs, pick top-2 closest in each row.
        # For rows that are themselves padding, the picked values are
        # meaningless but we'll mask their references back to defaults.
        _, idx = torch.topk(d2, k=2, dim=2, largest=False)  # (B, L, 2)
        # Gather: refs[b, l, k, :] = particles[b, idx[b, l, k], :]
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, 4)  # (B, L, 2, 4)
        gathered = torch.gather(
            particles.unsqueeze(2).expand(-1, -1, 2, -1),  # (B, L, 2, 4) of self
            dim=1,
            index=idx_expanded,
        )
        # Use gathered refs only where the event has ≥2 particles AND the row
        # itself is a valid particle.
        use_gathered = (
            has_pair.unsqueeze(-1) & valid_mask
        ).unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        references = torch.where(use_gathered, gathered, references)

    return polar_decomposition(boost, references)


def _valid_frame_mask(trafo: torch.Tensor, det_eps: float = 1e-10) -> torch.Tensor:
    finite = torch.isfinite(trafo).all(dim=-1).all(dim=-1)
    det = torch.linalg.det(trafo.to(torch.float64))
    return finite & (det.abs() > det_eps)


def _replace_invalid_frames(
    trafo: torch.Tensor,
    particles: torch.Tensor,
    valid_mask: torch.Tensor,
    det_eps: float = 1e-10,
) -> torch.Tensor:
    valid = _valid_frame_mask(trafo, det_eps=det_eps)
    if bool(valid.all()):
        return trafo
    fallback = _fallback_frames_from_particles(particles, valid_mask)
    return torch.where(valid.unsqueeze(-1).unsqueeze(-1), trafo, fallback)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def build_lloca_frames(
    particles: torch.Tensor,
    valid_mask: torch.Tensor,
    frame_predictor: LLoCaFramePredictor | None = None,
    scalars: torch.Tensor | None = None,
) -> Frames:
    """Build per-particle local Lorentz frames on padded particles.

    particles: (B, L, 4); valid_mask: (B, L). Returns a Frames whose matrices
    have shape (B, L, 4, 4). The returned object caches the analytic Lorentz
    inverse (``g L^T g``) and the determinant.
    """
    if frame_predictor is not None:
        trafo = frame_predictor(particles, valid_mask=valid_mask, scalars=scalars)
        trafo = _replace_invalid_frames(trafo, particles, valid_mask)
    else:
        trafo = _fallback_frames_from_particles(particles, valid_mask)
    return Frames(matrices=trafo, is_global=False)


def _as_matrices(frames) -> torch.Tensor:
    return frames.matrices if isinstance(frames, Frames) else frames


def _analytic_lorentz_inverse(mat: torch.Tensor) -> torch.Tensor:
    inv = mat.transpose(-1, -2).clone()
    inv[..., 1:, :] *= -1
    inv[..., :, 1:] *= -1
    return inv


def safe_inverse_frames(frames, det_eps: float = 1e-10) -> torch.Tensor:
    """Return the Lorentz inverse ``g L^T g`` with a pinv fallback for singular frames."""
    if isinstance(frames, Frames):
        return frames.inv

    analytic = _analytic_lorentz_inverse(frames)
    flat = frames.reshape(-1, 4, 4).to(torch.float64)
    finite = torch.isfinite(flat).all(dim=-1).all(dim=-1)
    det_ok = torch.linalg.det(flat).abs() > det_eps
    good = finite & det_ok
    if bool(good.all()):
        return analytic

    inv_flat = analytic.reshape(-1, 4, 4).to(torch.float64).clone()
    bad = ~good
    inv_flat[bad] = torch.linalg.pinv(flat[bad], rtol=det_eps)
    return inv_flat.to(frames.dtype).reshape_as(frames)


def canonicalize_input_fourmomenta(tokens: torch.Tensor, frames) -> torch.Tensor:
    """Canonicalize the last four channels of ``tokens``: ``p_local = L · p_global``.

    Shape-agnostic: tokens of shape (..., D) line up with mats of shape
    (..., 4, 4) on the leading dims (the per-particle axis must match).
    """
    if tokens.shape[-1] < 4:
        return tokens
    mats = _as_matrices(frames).to(tokens.dtype)
    vec = tokens[..., -4:]
    vec_local = torch.einsum("...m,...am->...a", vec, mats)
    vec_local = torch.nan_to_num(vec_local, nan=0.0, posinf=1e6, neginf=-1e6)
    if tokens.shape[-1] == 4:
        return vec_local
    return torch.cat((tokens[..., :-4], vec_local), dim=-1)


def _apply_frame_to_vector_channels(
    tensor: torch.Tensor,
    mats: torch.Tensor,
    n_scalars: int,
    n_vectors: int,
) -> torch.Tensor:
    """Apply a per-particle 4×4 transform to the vector channels of ``tensor``.

    Generalized for both flat attention shapes (``tensor=(H, N, D)``, ``mats=(N, 4, 4)``)
    and padded attention (``tensor=(B, H, L, D)``, ``mats=(B, L, 4, 4)``) by
    left-padding ``mats`` with singleton head dims until ranks align.
    """
    if n_vectors == 0:
        return tensor
    d_vec = n_vectors * 4
    if n_scalars + d_vec > tensor.shape[-1]:
        raise ValueError(
            f"Invalid scalar/vector split: n_scalars={n_scalars}, n_vectors={n_vectors}, "
            f"last_dim={tensor.shape[-1]}"
        )
    start, stop = n_scalars, n_scalars + d_vec
    mats = mats.to(tensor.dtype)
    vec = tensor[..., start:stop].reshape(*tensor.shape[:-1], n_vectors, 4)
    # Insert singleton head dim(s) into mats so its leading dims broadcast
    # against vec's. The per-particle axis is at position -3 of mats; the new
    # singleton goes just before it, i.e. at positive index ``mats.dim() - 3``
    # of the current tensor (pushing the per-particle axis one step right).
    while mats.dim() < vec.dim():
        mats = mats.unsqueeze(mats.dim() - 3)
    vec = torch.einsum("...vm,...am->...va", vec, mats)
    vec = torch.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
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
    frames,
    n_scalars: int,
    n_vectors: int,
    attn_mask: torch.Tensor | None = None,
    inv_frames=None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """LLoCa attention: local q/k/v → global → SDPA → local.

    Follows Eq. (12) of the LLoCa paper:
      q_global = L⁻¹ q_local        (using InverseFrames)
      k_global = (g L⁻¹) k_local    (using LowerIndicesFrames(InverseFrames))
      v_global = L⁻¹ v_local        (using InverseFrames)
      out_local = L out_global      (using Frames)

    Works on both flat ``(H, N, D)`` and padded ``(B, H, L, D)`` Q/K/V; the
    frame matrices are expected to have a matching per-particle axis (``(N, 4, 4)``
    or ``(B, L, 4, 4)`` respectively). ``_apply_frame_to_vector_channels``
    handles the rank alignment by inserting singleton head dims.
    """
    if isinstance(frames, Frames):
        frames_obj = frames
    else:
        frames_obj = Frames(matrices=frames, inv=inv_frames, is_global=False)

    fwd_mats = frames_obj.matrices
    inv_obj = InverseFrames(frames_obj)
    lower_inv_obj = LowerIndicesFrames(inv_obj)

    q_global = _apply_frame_to_vector_channels(query, inv_obj.matrices, n_scalars, n_vectors)
    k_global = _apply_frame_to_vector_channels(key, lower_inv_obj.matrices, n_scalars, n_vectors)
    v_global = _apply_frame_to_vector_channels(value, inv_obj.matrices, n_scalars, n_vectors)

    out_global = F.scaled_dot_product_attention(
        q_global,
        k_global,
        v_global,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
    )

    return _apply_frame_to_vector_channels(out_global, fwd_mats, n_scalars, n_vectors)
