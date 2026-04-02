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

        kwds["key"] = "Transformer"
        super().__init__(*args, **kwds)

        self.lloca = lloca
        self.lloca_frames = lloca_frames
        self.lloca_num_scalars = lloca_num_scalars
        self.lloca_num_vectors = lloca_num_vectors
        self.lloca_frame_hidden = lloca_frame_hidden
        self.lloca_frame_layers = lloca_frame_layers
        self.lloca_eps = lloca_eps
        self.lloca_use_float64 = lloca_use_float64

        self.lloca_frame_predictor = None
        if self.lloca:
            self.lloca_frame_predictor = LLoCaFramePredictor(
                hidden_dim=self.lloca_frame_hidden,
                num_layers=self.lloca_frame_layers,
                eps=self.lloca_eps,
                use_float64=self.lloca_use_float64,
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

        # Transfomer already implements a version akin to tokens, so we use 'channels' as default mode 
        index = ptr2index(ptr, mode="channels", theta_dim=embedding_kwargs.get("theta_dim", 0))
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
        **kwds,
    ) -> torch.Tensor:
        """
        Concatenate particles fourmomenta with theory parameters and optional
        preprocessed event-level features / MET. Event-level conditioning vectors
        are repeated for each particle in the corresponding event.

        :param particles: Particles fourmomenta with size: (num particles, 4)
        :type particles: torch.Tensor
        :param theta: Theory parameters vector with size: (batch size, theta dim)
        :type theta: torch.Tensor
        :param ptr: Event pointer with size: (batch size + 1,)
        :type ptr: torch.Tensor
        :param preprocessed: Event-level preprocessed features
        :type preprocessed: torch.Tensor | None
        :param met: Event-level MET features (pt, phi)
        :type met: torch.Tensor | None
        :return: Concatenated tensors with size: (num particles, 4 + theta dim + extras)
        :rtype: Tensor

        """
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
