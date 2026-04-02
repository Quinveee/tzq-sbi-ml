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
    Returns a batch of particles with the batch and event dimensions
    flattened into one. A pointer is also return to later divide the particles
    into events
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

    # Pointer object for each event
    lengths = torch.tensor(lengths_list)
    ptr = torch.zeros(len(batch) + 1)
    ptr[1:] = torch.cumsum(lengths, dim=0)

    particles = torch.cat(particles_list, dim=0)
    scores = torch.stack(scores_list, dim=0)
    preprocessed = torch.stack(preprocessed_list, dim=0)
    met = torch.stack(met_list, dim=0)

    extras = {attr: torch.stack(lst, dim=0) for attr, lst in extra_lists.items()}

    return particles, ptr, scores, preprocessed, met, extras


# TODO: old collate function, to be removed once LLoCa-specific collate path is fully implemented
def _collate_particles_lloca(
    batch: Iterable[ParticlesEvent], extra_attrs: Optional[List[str]] = None
):
    """
    LLoCa-specific particle collation path.

    For compatibility with the current wrapper/model interfaces, this currently returns
    the same flattened representation as the default collate function.
    """
    return _collate_particles_common(batch, extra_attrs=extra_attrs)


def collate_particles_fn(batch: Iterable[ParticlesEvent], lloca: bool = False) -> ParticleBatch:
    """
    Batch particle fourmomenta, score and pointer objects

    :param batch: Description
    :type batch: Iterable[ParticlesEvent]
    :return: Description
    :rtype: ParticleBatch
    """
    if lloca:
        particles, ptr, score, preprocessed, met, _ = _collate_particles_lloca(batch)
    else:
        particles, ptr, score, preprocessed, met, _ = _collate_particles_common(batch)
    return ParticleBatch(
        particles=particles,
        ptr=ptr,
        score=score,
        preprocessed=preprocessed,
        met=met,
    )


def parametrized_collate_particles_fn(
    batch: Iterable[ParametrizedParticlesEvent], lloca: bool = False
):
    """
    Batch particle fourmomenta, pointer object, score, theory parameters,
    likelihood ratios and labels in each event

    :param batch: Description
    :type batch: Iterable[ParametrizedParticlesEvent]
    """
    if lloca:
        particles, ptr, score, preprocessed, met, extras = _collate_particles_lloca(
            batch, extra_attrs=["theta", "ratio", "label"]
        )
    else:
        particles, ptr, score, preprocessed, met, extras = _collate_particles_common(
            batch, extra_attrs=["theta", "ratio", "label"]
        )
    return ParametrizedParticleBatch(
        particles=particles,
        ptr=ptr,
        score=score,
        preprocessed=preprocessed,
        met=met,
        theta=extras["theta"],
        ratio=extras["ratio"],
        label=extras["label"],
    )
