"""
Transormer architecture wrappers
"""

from abc import ABC, abstractmethod

import torch
from torch.nn.attention import sdpa_kernel
from torch_geometric.utils import scatter

from .base_wrapper import BaseWrapper
from .utils import att_mask, get_backends, ptr2index
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
    def _repeat_event_features(features: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
        """Broadcast event-level features to particle-level tokens."""
        if features.ndim == 1:
            features = features.unsqueeze(-1)
        if features.ndim != 2:
            raise ValueError(
                f"Expected event-level features with shape (batch, dim), got {features.shape}"
            )

        ptr = ptr.to(dtype=torch.long)
        return features.repeat_interleave(ptr[1:] - ptr[:-1], dim=0)

    def _to_particle_features(
        self,
        features: torch.Tensor | None,
        ptr: torch.Tensor,
        n_particles: int,
    ) -> torch.Tensor | None:
        if features is None:
            return None
        if features.ndim == 1:
            features = features.unsqueeze(-1)
        if features.shape[-1] == 0:
            return None

        if features.shape[0] == n_particles:
            return features

        n_events = int(ptr.shape[0] - 1)
        if features.shape[0] == n_events:
            return self._repeat_event_features(features, ptr)

        raise ValueError(
            "Unable to align frame scalar features with particles: "
            f"features shape={features.shape}, n_particles={n_particles}, n_events={n_events}"
        )

    def _collect_frame_scalars(
        self,
        ptr: torch.Tensor,
        n_particles: int,
        embedding_kwargs: dict,
    ) -> torch.Tensor | None:
        candidates = [
            embedding_kwargs.get("theta", None),
            embedding_kwargs.get("preprocessed", None),
            embedding_kwargs.get("met", None),
        ]
        aligned = [
            self._to_particle_features(feat, ptr, n_particles)
            for feat in candidates
        ]
        aligned = [feat for feat in aligned if feat is not None]
        if not aligned:
            return None
        return torch.cat(aligned, dim=-1)

    def forward(
        self,
        particles: torch.Tensor,
        ptr: torch.Tensor,
        force_math: bool = False,
        embedding_kwargs={},
    ) -> torch.Tensor:
        """
        Forward wrapper for the Transformer

        :param particles: Particles four momenta
        :type particles: torch.Tensor
        :param ptr: Event pointer
        :type ptr: torch.Tensor
        :param force_math: Whether to force non-efficient SA backends
        :type force_math: bool
        :param embedding_kwargs: Additional embedding keyword arguments
        :return: Forwarder tensor
        :rtype: Tensor

        """
        backends = get_backends(force_math)

        mode = embedding_kwargs.get("mode", "channels")
        theta_dim = int(embedding_kwargs.get("theta_dim", 0) or 0)

        if mode == "tokens" and self.lloca:
            raise ValueError(
                "LLoCa requires particle-level tokens and is incompatible with "
                "mode='tokens'. Use mode='channels' when LLoCa is active."
            )

        index = ptr2index(ptr, mode=mode, theta_dim=theta_dim if mode == "tokens" else 0)
        attention_mask = att_mask(index)

        # Delegate embedding to subclasses
        tokens = self.embed(particles, **embedding_kwargs)

        # Build attn_kwargs
        attn_kwargs = {}
        if self.lloca is not None:
            attn_kwargs["lloca"] = self.lloca

            if self.lloca:
                raw_p = particles[:, -4:] if particles.shape[-1] > 4 else particles
                if raw_p.shape[-1] != 4:
                    raise ValueError(
                        "LLoCa requires particle-level fourmomenta with last dimension 4"
                    )

                frame_scalars = self._collect_frame_scalars(
                    ptr=ptr,
                    n_particles=raw_p.shape[0],
                    embedding_kwargs=embedding_kwargs,
                )

                frames = build_lloca_frames(
                    raw_p,
                    ptr,
                    K=self.lloca_frames,
                    frame_predictor=self.lloca_frame_predictor,
                    scalars=frame_scalars,
                )

                attn_kwargs["frames"] = frames
                attn_kwargs["inv_frames"] = safe_inverse_frames(frames)

                # Canonicalize fourmomenta channels before entering the backbone.
                tokens = canonicalize_input_fourmomenta(tokens, frames)

                # TODO: consider normalizing the tokens here, but it may be better to let the model learn its own optimal normalization in the canonical frame.
                # tokens = torch.nn.functional.layer_norm(tokens, tokens.shape[-1:])
                # tokens = tokens / (tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6))

        if self.lloca_num_scalars is not None:
            attn_kwargs["lloca_num_scalars"] = self.lloca_num_scalars
        if self.lloca_num_vectors is not None:
            attn_kwargs["lloca_num_vectors"] = self.lloca_num_vectors

        # Just use allowed self-attention backends
        with sdpa_kernel(backends):
            out = self.net(tokens, attn_mask=attention_mask, attn_kwargs=attn_kwargs)

        # Here `dim=0` represents the particles dimension
        # We "scatter" the resulting batch using the event pointer
        # and assume a properly set linear output head to match the
        # desired output dimensions
        return scatter(src=out, index=index, dim=0, reduce="mean")


class LocalTransformerWrapper(BaseTransformerWrapper):
    def embed(
        self,
        tokens: torch.Tensor,
        preprocessed: torch.Tensor | None = None,
        met: torch.Tensor | None = None,
        ptr: torch.Tensor | None = None,
        **kwds,
    ) -> torch.Tensor:
        """
        Optionally append preprocessed event-level features and/or MET
        to every particle token.

        :param tokens: Particle four momenta
        :type tokens: torch.Tensor
        :param preprocessed: Event-level preprocessed features
        :type preprocessed: torch.Tensor | None
        :param met: Event-level MET features (pt, phi)
        :type met: torch.Tensor | None
        :param ptr: Event pointer
        :type ptr: torch.Tensor | None
        :return: Particle tokens with optional extra conditioning channels
        :rtype: torch.Tensor
        """
        extra = []

        if preprocessed is not None and preprocessed.shape[-1] > 0:
            if ptr is None:
                raise ValueError("ptr is required when using preprocessed features")
            extra.append(self._repeat_event_features(preprocessed, ptr))

        if met is not None and met.shape[-1] > 0:
            if ptr is None:
                raise ValueError("ptr is required when using MET features")
            extra.append(self._repeat_event_features(met, ptr))

        if not extra:
            return tokens

        return torch.cat(extra + [tokens], dim=-1)


class ParametrizedTransformerWrapper(BaseTransformerWrapper):
    def embed(
        self,
        particles: torch.Tensor,
        theta: torch.Tensor,
        ptr: torch.Tensor,
        preprocessed: torch.Tensor | None = None,
        met: torch.Tensor | None = None,
        mode: str = "channels",
        **kwds,
    ) -> torch.Tensor:
        """
        Build input tokens for the transformer.

        ``mode="channels"`` concatenates θ (and optional preprocessed / MET
        features) onto every particle token — θ is a constant additive signal
        shared across all tokens of an event.

        ``mode="tokens"`` prepends ``theta_dim`` dedicated θ tokens per event,
        mirroring the LGATr tokens convention. Each θ token places θ[i] in
        the i-th of the first ``theta_dim`` input slots; remaining slots are
        zero. Particle tokens zero-out the first ``theta_dim`` slots and
        carry [particle, preprocessed, met] in the trailing slots. Total
        input width matches ``mode="channels"``.
        """
        if mode == "channels":
            return self._embed_channels(particles, theta, ptr, preprocessed, met)
        if mode == "tokens":
            return self._embed_tokens(particles, theta, ptr, preprocessed, met)
        raise ValueError(f"Invalid transformer embed mode '{mode}'")

    def _embed_channels(
        self,
        particles: torch.Tensor,
        theta: torch.Tensor,
        ptr: torch.Tensor,
        preprocessed: torch.Tensor | None,
        met: torch.Tensor | None,
    ) -> torch.Tensor:
        n, e = particles.shape
        theta_dim = theta.shape[-1]

        ptr = ptr.to(dtype=torch.long)

        theta = theta.repeat_interleave(ptr[1:] - ptr[:-1], dim=0)

        assert theta.size() == (n, theta_dim)

        conditioning = [theta]
        conditioning_dim = theta_dim

        if preprocessed is not None and preprocessed.shape[-1] > 0:
            preprocessed = self._repeat_event_features(preprocessed, ptr)
            conditioning.append(preprocessed)
            conditioning_dim += preprocessed.shape[-1]

        if met is not None and met.shape[-1] > 0:
            met = self._repeat_event_features(met, ptr)
            conditioning.append(met)
            conditioning_dim += met.shape[-1]

        tokens = torch.cat(conditioning + [particles], dim=-1)

        assert tokens.size() == (n, e + conditioning_dim)

        return tokens

    def _embed_tokens(
        self,
        particles: torch.Tensor,
        theta: torch.Tensor,
        ptr: torch.Tensor,
        preprocessed: torch.Tensor | None,
        met: torch.Tensor | None,
    ) -> torch.Tensor:
        ptr = ptr.to(dtype=torch.long)
        n_particles, p_dim = particles.shape
        n_events = int(ptr.shape[0] - 1)
        theta_dim = int(theta.shape[-1])
        counts = ptr[1:] - ptr[:-1]
        device = particles.device
        dtype = particles.dtype

        pre_dim = (
            preprocessed.shape[-1]
            if preprocessed is not None and preprocessed.shape[-1] > 0
            else 0
        )
        met_dim = met.shape[-1] if met is not None and met.shape[-1] > 0 else 0

        trailing_dim = p_dim + pre_dim + met_dim
        dim_in = theta_dim + trailing_dim

        tokens_per_event = counts + theta_dim
        tokens_ptr = torch.zeros(n_events + 1, dtype=torch.long, device=device)
        tokens_ptr[1:] = tokens_per_event.cumsum(dim=0)
        n_total = int(tokens_ptr[-1])

        out = particles.new_zeros(n_total, dim_in)

        # θ tokens: one per theta component per event. θ[e, i] sits in column i
        # of the i-th θ token; particle/extra slots stay zero.
        theta_offsets = tokens_ptr[:-1]
        dim_idx = torch.arange(theta_dim, device=device)
        theta_rows = theta_offsets.unsqueeze(1) + dim_idx.unsqueeze(0)
        theta_cols = dim_idx.unsqueeze(0).expand(n_events, -1)
        out[theta_rows, theta_cols] = theta.to(dtype)

        # Particle tokens: placed after this event's θ tokens. First theta_dim
        # columns stay zero; particle 4-vector + optional preprocessed / MET
        # occupy the trailing columns.
        particle_offsets = tokens_ptr[:-1] + theta_dim
        event_of_particle = torch.arange(n_events, device=device).repeat_interleave(counts)
        within_event = (
            torch.arange(n_particles, device=device)
            - ptr[:-1].repeat_interleave(counts)
        )
        particle_rows = particle_offsets[event_of_particle] + within_event

        cursor = theta_dim
        out[particle_rows, cursor : cursor + p_dim] = particles
        cursor += p_dim
        if pre_dim > 0:
            pre_repeated = self._repeat_event_features(preprocessed, ptr)
            out[particle_rows, cursor : cursor + pre_dim] = pre_repeated
            cursor += pre_dim
        if met_dim > 0:
            met_repeated = self._repeat_event_features(met, ptr)
            out[particle_rows, cursor : cursor + met_dim] = met_repeated

        return out


class LocalTransformerFeaturesWrapper(LocalTransformerWrapper):
    """Transformer wrapper for feature-level local score regression."""

    @staticmethod
    def _to_feature_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split each event feature vector into one scalar token per feature."""
        batch_size, n_features = x.shape
        ptr = torch.arange(
            0,
            (batch_size + 1) * n_features,
            n_features,
            dtype=torch.long,
            device=x.device,
        )
        tokens = x.reshape(-1, 1)
        return tokens, ptr

    def forward(self, x: torch.Tensor, force_math: bool = False, embedding_kwargs={}):
        tokens, ptr = self._to_feature_tokens(x)
        return super().forward(
            particles=tokens,
            ptr=ptr,
            force_math=force_math,
            embedding_kwargs=embedding_kwargs,
        )


class ParametrizedTransformerFeaturesWrapper(ParametrizedTransformerWrapper):
    """Transformer wrapper for feature-level ratio regression."""

    @staticmethod
    def _to_feature_tokens(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_features = x.shape
        ptr = torch.arange(
            0,
            (batch_size + 1) * n_features,
            n_features,
            dtype=torch.long,
            device=x.device,
        )
        tokens = x.reshape(-1, 1)
        return tokens, ptr

    def forward(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        force_math: bool = False,
        embedding_kwargs={},
    ):
        tokens, ptr = self._to_feature_tokens(x)
        embedding_kwargs = {"theta": theta, "ptr": ptr, **embedding_kwargs}
        return super().forward(
            particles=tokens,
            ptr=ptr,
            force_math=force_math,
            embedding_kwargs=embedding_kwargs,
        )
