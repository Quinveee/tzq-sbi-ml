"""
Collate functions for the 'particles' experiments
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional

import torch

from .schemas import ParametrizedParticleBatch, ParticleBatch

if TYPE_CHECKING:
    from .schemas import ParametrizedParticlesEvent, ParticlesEvent


def _collate_particles_common(
    batch: Iterable[ParticlesEvent], extra_attrs: Optional[List[str]] = None
):
    """
    Batch variable-length per-event particle fourmomenta into a padded
    tensor of shape (B, L_max, 4) plus a (B, L_max) bool mask marking real
    tokens. A CSR-style `ptr` is also returned (derived from the per-event
    lengths) so callers that still expect the flattened layout keep working.
    """
    particles_list, lengths_list, scores_list, preprocessed_list, met_list = (
        [],
        [],
        [],
        [],
        [],
    )
    extra_lists = {attr: [] for attr in (extra_attrs or [])}

    for event in batch:

        # Only add non-null particles to batch, so only `event.lenght` first
        particles_list.append(torch.from_numpy(event.fourmomenta[: event.length]))

        lengths_list.append(event.length)
        scores_list.append(torch.from_numpy(event.score))
        preprocessed_list.append(torch.from_numpy(event.preprocessed))
        met_list.append(torch.from_numpy(event.met))
        for attr in extra_lists:
            extra_lists[attr].append(torch.from_numpy(getattr(event, attr)))

    lengths = torch.tensor(lengths_list, dtype=torch.long)

    # Pad to (B, L_max, 4). pad_sequence handles the variable length copy.
    particles = torch.nn.utils.rnn.pad_sequence(
        particles_list, batch_first=True
    )
    L_max = particles.shape[1]
    valid_mask = (
        torch.arange(L_max).unsqueeze(0) < lengths.unsqueeze(1)
    )  # (B, L_max), True for real tokens

    # CSR-style ptr derived from the per-event lengths, kept for downstream
    # consumers (lorentznet, lgatr, the existing flat code paths). Float dtype
    # to match the previous schema.
    ptr = torch.zeros(len(batch) + 1)
    ptr[1:] = torch.cumsum(lengths, dim=0)

    scores = torch.stack(scores_list, dim=0)
    preprocessed = torch.stack(preprocessed_list, dim=0)
    met = torch.stack(met_list, dim=0)

    extras = {attr: torch.stack(lst, dim=0) for attr, lst in extra_lists.items()}

    return particles, ptr, valid_mask, scores, preprocessed, met, extras


def collate_particles_fn(batch: Iterable[ParticlesEvent], lloca: bool = False) -> ParticleBatch:
    """
    Batch particle fourmomenta, score and pointer objects.

    `lloca` is accepted for backwards compatibility but no longer changes the
    collation path — the padded representation is what downstream code uses.
    """
    del lloca
    particles, ptr, valid_mask, score, preprocessed, met, _ = _collate_particles_common(batch)
    return ParticleBatch(
        particles=particles,
        ptr=ptr,
        valid_mask=valid_mask,
        score=score,
        preprocessed=preprocessed,
        met=met,
    )


def parametrized_collate_particles_fn(
    batch: Iterable[ParametrizedParticlesEvent], lloca: bool = False
):
    """
    Batch particle fourmomenta, pointer object, score, theory parameters,
    likelihood ratios and labels in each event.

    `lloca` is accepted for backwards compatibility but no longer changes the
    collation path.
    """
    del lloca
    particles, ptr, valid_mask, score, preprocessed, met, extras = (
        _collate_particles_common(batch, extra_attrs=["theta", "ratio", "label"])
    )
    return ParametrizedParticleBatch(
        particles=particles,
        ptr=ptr,
        valid_mask=valid_mask,
        score=score,
        preprocessed=preprocessed,
        met=met,
        theta=extras["theta"],
        ratio=extras["ratio"],
        label=extras["label"],
    )
