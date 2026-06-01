"""
Transormer architecture wrappers
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel
from torch_geometric.utils import scatter

from .base_wrapper import BaseWrapper
from .utils import (
    att_mask,
    derive_valid_mask,
    flat_to_padded,
    get_backends,
    padded_to_flat,
    ptr2index,
)
from models.modules.lorentz import (
    LLoCaFramePredictor,
    build_lloca_frames,
    canonicalize_input_fourmomenta,
    safe_inverse_frames,
)


class BaseTransformerWrapper(BaseWrapper, ABC):
    """
    Base Transormer architecture wrapper
    """

    def __init__(self, *args, **kwds):
        # `mode` lives on cfg.model so the experiment can forward it via
        # embedding_kwargs at call time; absorb it here so it doesn't reach
        # BaseWrapper (which only accepts net + key). Stored on `self` after
        # super().__init__() to satisfy nn.Module's setattr ordering.
        mode = kwds.pop("mode", "channels")
        # `theta_encoder` is shared in the transformer base config but only
        # consumed by ParametrizedTransformerWrapper (which pops it before
        # calling super). Absorb it here so non-parametrized subclasses don't
        # forward it to BaseWrapper.
        kwds.pop("theta_encoder", None)
        # Extract LLoCa configuration before passing to parent
        lloca_config = kwds.pop("LLoCa", {})
        lloca = lloca_config.get("active", None)
        lloca_frames = lloca_config.get("LLoCa_frames", None)
        lloca_num_scalars = lloca_config.get("LLoCa_num_scalars", None)
        lloca_num_vectors = lloca_config.get("LLoCa_num_vectors", None)
        lloca_frame_hidden = int(lloca_config.get("LLoCa_frame_hidden", 128))
        lloca_frame_layers = int(lloca_config.get("LLoCa_frame_layers", 2))
        lloca_eps = float(lloca_config.get("LLoCa_eps", 1e-8))
        lloca_use_float64 = bool(lloca_config.get("LLoCa_use_float64", True))
        # Bound the Lorentz factor of the predicted boost. Without this the
        # frame predictor can output arbitrarily-boosted frames for near-
        # lightlike events, causing large loss spikes. Recommended in the
        # LLoCa paper as a regularizer for stable training.
        _gamma_max_raw = lloca_config.get("LLoCa_gamma_max", None)
        lloca_gamma_max = float(_gamma_max_raw) if _gamma_max_raw is not None else None
        _gamma_hardness_raw = lloca_config.get("LLoCa_gamma_hardness", None)
        lloca_gamma_hardness = (
            float(_gamma_hardness_raw) if _gamma_hardness_raw is not None else None
        )

        kwds["key"] = "transformer_lloca" if lloca else "Transformer"
        super().__init__(*args, **kwds)

        self.mode = mode
        self.lloca = lloca
        self.lloca_frames = lloca_frames
        self.lloca_num_scalars = lloca_num_scalars
        self.lloca_num_vectors = lloca_num_vectors
        self.lloca_frame_hidden = lloca_frame_hidden
        self.lloca_frame_layers = lloca_frame_layers
        self.lloca_eps = lloca_eps
        self.lloca_use_float64 = lloca_use_float64
        self.lloca_gamma_max = lloca_gamma_max
        self.lloca_gamma_hardness = lloca_gamma_hardness

        if self.lloca:
            n_scalars = int(self.lloca_num_scalars or 0)
            n_vectors = int(self.lloca_num_vectors or 0)
            if n_vectors > 0:
                sa_config = self.net.te_blocks[0].sa.config
                emb_head = int(sa_config.emb_head)
                emb_size = int(sa_config.emb_size)
                num_heads = int(sa_config.num_heads)
                required = n_scalars + n_vectors * 4

                if emb_head < required:
                    dim_in = int(self.net.linear_in.in_features)
                    min_emb_size = required * num_heads
                    min_emb_factor = (min_emb_size + dim_in - 1) // dim_in
                    raise ValueError(
                        "Invalid LLoCa configuration for Transformer: "
                        f"emb_head={emb_head}, required={required} "
                        f"(n_scalars={n_scalars}, n_vectors={n_vectors}, "
                        f"emb_size={emb_size}, num_heads={num_heads}). "
                        f"For dim_in={dim_in}, emb_factor must be at least {min_emb_factor} "
                        "(or reduce num_heads / LLoCa channels)."
                    )

        self.lloca_frame_predictor = None
        if self.lloca:
            self.lloca_frame_predictor = LLoCaFramePredictor(
                hidden_dim=self.lloca_frame_hidden,
                num_layers=self.lloca_frame_layers,
                eps=self.lloca_eps,
                use_float64=self.lloca_use_float64,
                gamma_max=self.lloca_gamma_max,
                gamma_hardness=self.lloca_gamma_hardness,
            )

    @abstractmethod
    def embed(self, *args, **kwds):
        pass

    @staticmethod
    def _broadcast_event_features(
        features: torch.Tensor, L_max: int
    ) -> torch.Tensor:
        """Broadcast event-level features (B, F) to per-token (B, L_max, F)."""
        if features.ndim == 1:
            features = features.unsqueeze(-1)
        if features.ndim != 2:
            raise ValueError(
                f"Expected event-level features with shape (batch, dim), got {features.shape}"
            )
        return features.unsqueeze(1).expand(-1, L_max, -1)

    def _to_token_features(
        self,
        features: torch.Tensor | None,
        L_max: int,
        n_events: int,
    ) -> torch.Tensor | None:
        """
        Align an event-level (B, F) or already-tokenized (B, L_max, F) tensor
        to the per-token shape (B, L_max, F). Returns None if features is None
        or has zero last-dim.
        """
        if features is None:
            return None
        if features.ndim == 1:
            features = features.unsqueeze(-1)
        if features.shape[-1] == 0:
            return None
        if features.ndim == 3:
            return features
        if features.shape[0] == n_events:
            return self._broadcast_event_features(features, L_max)
        raise ValueError(
            "Unable to align frame scalar features with tokens: "
            f"features shape={features.shape}, n_events={n_events}, L_max={L_max}"
        )

    def _collect_frame_scalars(
        self,
        L_max: int,
        n_events: int,
        embedding_kwargs: dict,
    ) -> torch.Tensor | None:
        candidates = [
            embedding_kwargs.get("theta", None),
            embedding_kwargs.get("preprocessed", None),
            embedding_kwargs.get("met", None),
        ]
        aligned = [
            self._to_token_features(feat, L_max, n_events) for feat in candidates
        ]
        aligned = [feat for feat in aligned if feat is not None]
        if not aligned:
            return None
        return torch.cat(aligned, dim=-1)

    def forward(
        self,
        particles: torch.Tensor,
        ptr: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        force_math: bool = False,
        embedding_kwargs={},
    ) -> torch.Tensor:
        """
        Forward wrapper for the Transformer. Operates on padded inputs:

            particles: (B, L_max, 4)
            valid_mask: (B, L_max) bool — True for real tokens, False for pad.

        `ptr` is accepted for compatibility with call sites that still hand it
        in; if `valid_mask` is None we derive it from `ptr`. The transformer
        body runs as batched attention with the padding mask, giving
        O(B * L_max^2) compute instead of O((B*L_avg)^2).
        """
        backends = get_backends(force_math)

        mode = embedding_kwargs.get("mode", "channels")
        if mode != "channels":
            raise ValueError(
                "Padded transformer wrapper currently supports mode='channels' "
                "only; tokens-mode theta conditioning has not been ported."
            )

        if particles.ndim != 3:
            raise ValueError(
                "Padded transformer wrapper expects particles of shape "
                f"(B, L_max, 4); got {tuple(particles.shape)}."
            )

        B, L_max = particles.shape[0], particles.shape[1]
        if valid_mask is None:
            valid_mask = derive_valid_mask(ptr, L_max).to(particles.device)
        else:
            valid_mask = valid_mask.to(torch.bool)

        # Delegate embedding to subclasses. Subclasses return padded
        # tokens (B, L_max, dim_in).
        tokens = self.embed(particles, valid_mask=valid_mask, **embedding_kwargs)

        attn_kwargs = {}
        if self.lloca is not None:
            attn_kwargs["lloca"] = self.lloca

            if self.lloca:
                raw_p = (
                    particles[..., -4:]
                    if particles.shape[-1] > 4
                    else particles
                )
                if raw_p.shape[-1] != 4:
                    raise ValueError(
                        "LLoCa requires particle-level fourmomenta with last dimension 4"
                    )

                frame_scalars = self._collect_frame_scalars(
                    L_max=L_max,
                    n_events=B,
                    embedding_kwargs=embedding_kwargs,
                )

                frames = build_lloca_frames(
                    raw_p,
                    valid_mask=valid_mask,
                    frame_predictor=self.lloca_frame_predictor,
                    scalars=frame_scalars,
                )

                attn_kwargs["frames"] = frames
                attn_kwargs["inv_frames"] = safe_inverse_frames(frames)

                tokens = canonicalize_input_fourmomenta(tokens, frames)

        if self.lloca_num_scalars is not None:
            attn_kwargs["lloca_num_scalars"] = self.lloca_num_scalars
        if self.lloca_num_vectors is not None:
            attn_kwargs["lloca_num_vectors"] = self.lloca_num_vectors

        with sdpa_kernel(backends):
            out_padded = self.net(
                tokens,
                attn_mask=valid_mask,
                attn_kwargs=attn_kwargs,
            )

        # Masked mean over real tokens per event -> (B, dim_out)
        weight = valid_mask.unsqueeze(-1).to(out_padded.dtype)
        counts = weight.sum(dim=1).clamp(min=1)
        return (out_padded * weight).sum(dim=1) / counts


class LocalTransformerWrapper(BaseTransformerWrapper):
    def embed(
        self,
        tokens: torch.Tensor,
        preprocessed: torch.Tensor | None = None,
        met: torch.Tensor | None = None,
        ptr: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        **kwds,
    ) -> torch.Tensor:
        """
        Append preprocessed event-level features and/or MET to every particle
        token. Operates on padded tokens of shape (B, L_max, 4); event-level
        features (B, F) are broadcast across the token axis.
        """
        del ptr, valid_mask
        L_max = tokens.shape[1]
        extra = []
        if preprocessed is not None and preprocessed.shape[-1] > 0:
            extra.append(self._broadcast_event_features(preprocessed, L_max))
        if met is not None and met.shape[-1] > 0:
            extra.append(self._broadcast_event_features(met, L_max))
        if not extra:
            return tokens
        return torch.cat(extra + [tokens], dim=-1)


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _build_theta_encoder(
    in_dim: int, hidden: int, out_dim: int, layers: int, activation: str
) -> nn.Module:
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown theta_encoder activation {activation!r}; "
            f"choose from {list(_ACTIVATIONS)}"
        )
    Act = _ACTIVATIONS[activation]
    layers = max(int(layers), 1)
    if layers == 1:
        return nn.Linear(in_dim, out_dim)
    modules: list[nn.Module] = [nn.Linear(in_dim, hidden), Act()]
    for _ in range(layers - 2):
        modules += [nn.Linear(hidden, hidden), Act()]
    modules.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*modules)


class ParametrizedTransformerWrapper(BaseTransformerWrapper):
    def __init__(self, *args, **kwds):
        # Pop the encoder config before BaseTransformerWrapper sees the kwargs.
        # When disabled this is a no-op; when enabled we build a small MLP that
        # lifts θ to a wider embedding before concat / frame conditioning.
        encoder_cfg = kwds.pop("theta_encoder", None) or {}
        super().__init__(*args, **kwds)

        if encoder_cfg.get("enabled", False):
            if self.mode != "channels":
                raise ValueError(
                    "theta_encoder is only supported with mode='channels'; "
                    "tokens mode places θ components in dedicated slots and "
                    "is incompatible with a learned θ embedding."
                )
            in_dim = int(encoder_cfg.get("in_dim", 0) or 0)
            out_dim = int(encoder_cfg.get("out", 0) or 0)
            if in_dim <= 0 or out_dim <= 0:
                raise ValueError(
                    "theta_encoder requires in_dim>0 and out>0, got "
                    f"in_dim={in_dim}, out={out_dim}. Make sure derive_config "
                    "populated theta_encoder.in_dim from dataset.theta_dim."
                )
            self.theta_encoder = _build_theta_encoder(
                in_dim=in_dim,
                hidden=int(encoder_cfg.get("hidden", 64)),
                out_dim=out_dim,
                layers=int(encoder_cfg.get("layers", 2)),
                activation=str(encoder_cfg.get("activation", "relu")),
            )
        else:
            self.theta_encoder = None

    def forward(
        self,
        particles: torch.Tensor,
        ptr: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        force_math: bool = False,
        embedding_kwargs={},
    ) -> torch.Tensor:
        # Encode θ once at the top so the frame predictor and the embed step
        # see the same lifted representation.
        if self.theta_encoder is not None:
            theta = embedding_kwargs.get("theta", None)
            if theta is None:
                raise ValueError(
                    "theta_encoder is enabled but embedding_kwargs has no 'theta'"
                )
            embedding_kwargs = {**embedding_kwargs, "theta": self.theta_encoder(theta)}
        return super().forward(
            particles=particles,
            ptr=ptr,
            valid_mask=valid_mask,
            force_math=force_math,
            embedding_kwargs=embedding_kwargs,
        )

    def embed(
        self,
        particles: torch.Tensor,
        theta: torch.Tensor,
        ptr: torch.Tensor | None = None,
        preprocessed: torch.Tensor | None = None,
        met: torch.Tensor | None = None,
        mode: str = "channels",
        valid_mask: torch.Tensor | None = None,
        **kwds,
    ) -> torch.Tensor:
        """
        Build padded input tokens (B, L_max, dim_in) for the transformer.

        ``mode="channels"`` concatenates θ (and optional preprocessed / MET
        features) onto every particle token — θ is a constant additive signal
        shared across all tokens of an event. Tokens-mode embedding (theta
        as dedicated prepended tokens) is currently not supported in the
        padded path.
        """
        del ptr, valid_mask
        if mode != "channels":
            raise ValueError(
                "Padded ParametrizedTransformerWrapper supports mode='channels' only."
            )
        return self._embed_channels(particles, theta, preprocessed, met)

    def _embed_channels(
        self,
        particles: torch.Tensor,
        theta: torch.Tensor,
        preprocessed: torch.Tensor | None,
        met: torch.Tensor | None,
    ) -> torch.Tensor:
        """particles: (B, L_max, p_dim); theta/preprocessed/met: (B, F_*)."""
        B, L_max, _ = particles.shape

        conditioning = [self._broadcast_event_features(theta, L_max)]
        if preprocessed is not None and preprocessed.shape[-1] > 0:
            conditioning.append(self._broadcast_event_features(preprocessed, L_max))
        if met is not None and met.shape[-1] > 0:
            conditioning.append(self._broadcast_event_features(met, L_max))
        return torch.cat(conditioning + [particles], dim=-1)

    # `_embed_tokens` (theta as dedicated prepended tokens, flat layout) has
    # not been ported to the padded path; it would need (B, theta_dim + L_max,
    # dim_in) tokens with valid_mask extended across the theta slots. The
    # transformer wrapper now rejects mode='tokens' so this is not called.


class LocalTransformerFeaturesWrapper(LocalTransformerWrapper):
    """Transformer wrapper for feature-level local score regression."""

    @staticmethod
    def _to_feature_tokens(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One scalar token per feature, no padding (all tokens valid)."""
        B, n_features = x.shape
        tokens = x.unsqueeze(-1)  # (B, n_features, 1)
        valid_mask = x.new_ones(B, n_features, dtype=torch.bool)
        ptr = torch.arange(
            0,
            (B + 1) * n_features,
            n_features,
            dtype=torch.long,
            device=x.device,
        )
        return tokens, ptr, valid_mask

    def forward(self, x: torch.Tensor, force_math: bool = False, embedding_kwargs={}):
        tokens, ptr, valid_mask = self._to_feature_tokens(x)
        return super().forward(
            particles=tokens,
            ptr=ptr,
            valid_mask=valid_mask,
            force_math=force_math,
            embedding_kwargs=embedding_kwargs,
        )


class ParametrizedTransformerFeaturesWrapper(ParametrizedTransformerWrapper):
    """Transformer wrapper for feature-level ratio regression."""

    @staticmethod
    def _to_feature_tokens(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, n_features = x.shape
        tokens = x.unsqueeze(-1)  # (B, n_features, 1)
        valid_mask = x.new_ones(B, n_features, dtype=torch.bool)
        ptr = torch.arange(
            0,
            (B + 1) * n_features,
            n_features,
            dtype=torch.long,
            device=x.device,
        )
        return tokens, ptr, valid_mask

    def forward(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        force_math: bool = False,
        embedding_kwargs={},
    ):
        tokens, ptr, valid_mask = self._to_feature_tokens(x)
        embedding_kwargs = {"theta": theta, "ptr": ptr, **embedding_kwargs}
        return super().forward(
            particles=tokens,
            ptr=ptr,
            valid_mask=valid_mask,
            force_math=force_math,
            embedding_kwargs=embedding_kwargs,
        )
