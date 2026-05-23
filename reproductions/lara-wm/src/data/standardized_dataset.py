# pyright: reportMissingImports=false, reportExplicitAny=false, reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

"""Standardized dataset wrapper for RoboTwin data.

Provides a raw-preserving dual-track interface:
- raw access: original ``RoboTwinEpisode`` arrays as loaded from disk
- transformed access: copied + optionally transformed standardized views

Standardized samples always expose the top-level keys:
``images``, ``states``, ``actions``, ``rewards``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .robottwin_reader import RoboTwinDataset, RoboTwinEpisode

Array = Any
StandardizedMapping = dict[str, Array]
StandardizedSample = dict[str, Any]
EpisodeTransform = Callable[[StandardizedSample], StandardizedSample]
BatchTransform = Callable[[StandardizedSample], StandardizedSample]


def _copy_mapping(mapping: Mapping[str, Array]) -> StandardizedMapping:
    return {key: np.array(value, copy=True) for key, value in mapping.items()}


def _copy_array(array: Array) -> Array:
    return np.array(array, copy=True)


def _stack_or_object(arrays: Sequence[Array], *, pad_value: int | float = 0) -> Array:
    if not arrays:
        return np.empty((0,), dtype=np.float32)

    first_shape = arrays[0].shape
    first_dtype = arrays[0].dtype
    if all(array.shape == first_shape and array.dtype == first_dtype for array in arrays):
        return np.stack(arrays, axis=0)

    if not arrays[0].shape:
        return np.asarray(arrays)

    trailing_shape = arrays[0].shape[1:]
    if all(array.shape[1:] == trailing_shape for array in arrays):
        max_steps = max(array.shape[0] for array in arrays)
        batched = np.full(
            (len(arrays), max_steps, *trailing_shape),
            fill_value=pad_value,
            dtype=np.result_type(*[array.dtype for array in arrays]),
        )
        for index, array in enumerate(arrays):
            batched[index, : array.shape[0]] = array
        return batched

    return np.asarray(arrays, dtype=object)


def _collate_mapping(items: Sequence[StandardizedMapping], *, pad_value: int | float = 0) -> StandardizedMapping:
    keys = sorted({key for item in items for key in item})
    collated: StandardizedMapping = {}
    for key in keys:
        arrays = [item[key] for item in items if key in item]
        collated[key] = _stack_or_object(arrays, pad_value=pad_value)
    return collated


@dataclass(frozen=True, slots=True)
class TransformPipeline:
    """Transforms applied to copied standardized data only."""

    image_transform: Callable[[Array], Array] | None = None
    state_transform: Callable[[Array], Array] | None = None
    action_transform: Callable[[Array], Array] | None = None
    reward_transform: Callable[[Array], Array] | None = None
    episode_transform: EpisodeTransform | None = None
    batch_transform: BatchTransform | None = None

    def transform_episode(self, sample: StandardizedSample) -> StandardizedSample:
        transformed: StandardizedSample = {
            "images": {
                key: self._apply(array, self.image_transform)
                for key, array in sample["images"].items()
            },
            "states": {
                key: self._apply(array, self.state_transform)
                for key, array in sample["states"].items()
            },
            "actions": {
                key: self._apply(array, self.action_transform)
                for key, array in sample["actions"].items()
            },
            "rewards": self._apply(sample["rewards"], self.reward_transform),
        }
        if self.episode_transform is not None:
            transformed = self.episode_transform(transformed)
        return transformed

    def transform_batch(self, batch: StandardizedSample) -> StandardizedSample:
        if self.batch_transform is not None:
            return self.batch_transform(batch)
        return batch

    @staticmethod
    def _apply(array: Array, transform: Callable[[Array], Array] | None) -> Array:
        copied = _copy_array(array)
        if transform is None:
            return copied
        return np.array(transform(copied), copy=True)


@dataclass(frozen=True, slots=True)
class StandardizedEpisode:
    """Episode wrapper with raw and transformed access."""

    raw_episode: RoboTwinEpisode
    transformed: StandardizedSample

    @property
    def episode_id(self) -> str:
        return self.raw_episode.episode_id

    @property
    def episode_length(self) -> int:
        return self.raw_episode.episode_length

    @property
    def raw(self) -> RoboTwinEpisode:
        return self.raw_episode

    def as_dict(self, *, include_raw: bool = False) -> StandardizedSample:
        sample: StandardizedSample = {
            "images": _copy_mapping(self.transformed["images"]),
            "states": _copy_mapping(self.transformed["states"]),
            "actions": _copy_mapping(self.transformed["actions"]),
            "rewards": _copy_array(self.transformed["rewards"]),
        }
        if include_raw:
            sample["raw"] = self.raw_episode
        return sample


@dataclass
class StandardizedDataset:
    """Standardized wrapper around :class:`RoboTwinDataset`.

    The wrapper never mutates raw data. Standardized outputs are always built from
    copied arrays, then optional transforms are applied to those copies.
    """

    source: RoboTwinDataset | str | Path
    batch_size: int | None = None
    transform_pipeline: TransformPipeline | None = None
    include_raw_in_batch: bool = False
    reward_pad_value: float = 0.0

    _dataset: RoboTwinDataset = field(init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.source, RoboTwinDataset):
            if self.source.transform_obs is not None or self.source.transform_action is not None:
                raise ValueError(
                    "StandardizedDataset requires an untransformed RoboTwinDataset to preserve raw data."
                )
            self._dataset = self.source
        else:
            self._dataset = RoboTwinDataset(data_path=Path(self.source))

        if self.batch_size is None:
            self.batch_size = self._dataset.batch_size
        if self.transform_pipeline is None:
            self.transform_pipeline = TransformPipeline()

    @property
    def dataset(self) -> RoboTwinDataset:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[StandardizedEpisode]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> StandardizedEpisode:
        raw_episode = self._dataset[index]
        transformed = self._transform_episode(raw_episode)
        return StandardizedEpisode(raw_episode=raw_episode, transformed=transformed)

    def iter_episodes(self) -> Iterator[StandardizedEpisode]:
        return iter(self)

    def get_episode(self, index: int) -> StandardizedEpisode:
        return self[index]

    def get_batch(self, indices: Sequence[int] | None = None) -> StandardizedSample:
        """Return a standardized batch.

        If ``indices`` is omitted, the next sequential batch is returned using the
        configured ``batch_size`` and internal cursor.
        """

        if indices is None:
            indices = self._next_indices()

        episodes = [self[index] for index in indices]
        batch: StandardizedSample = {
            "images": _collate_mapping([episode.transformed["images"] for episode in episodes]),
            "states": _collate_mapping([episode.transformed["states"] for episode in episodes]),
            "actions": _collate_mapping([episode.transformed["actions"] for episode in episodes]),
            "rewards": _stack_or_object(
                [episode.transformed["rewards"] for episode in episodes],
                pad_value=self.reward_pad_value,
            ),
        }
        if self.include_raw_in_batch:
            batch["raw"] = [episode.raw for episode in episodes]
        pipeline = self.transform_pipeline or TransformPipeline()
        return pipeline.transform_batch(batch)

    def reset(self) -> None:
        self._index = 0

    def close(self) -> None:
        self._dataset.close()

    def _next_indices(self) -> list[int]:
        if len(self) == 0:
            return []

        batch_size = self.batch_size or self._dataset.batch_size or 1
        start = self._index
        stop = min(start + batch_size, len(self))
        self._index = 0 if stop >= len(self) else stop
        return list(range(start, stop))

    def _transform_episode(self, episode: RoboTwinEpisode) -> StandardizedSample:
        standardized: StandardizedSample = {
            "images": _copy_mapping(episode.image_obs),
            "states": _copy_mapping(episode.state_obs),
            "actions": _copy_mapping(episode.actions),
            "rewards": _copy_array(episode.rewards),
        }
        pipeline = self.transform_pipeline or TransformPipeline()
        return pipeline.transform_episode(standardized)


def create_standardized_dataset(
    source: RoboTwinDataset | str | Path,
    *,
    batch_size: int | None = None,
    transform_pipeline: TransformPipeline | None = None,
    include_raw_in_batch: bool = False,
) -> StandardizedDataset:
    """Factory for the standardized dataset wrapper."""

    return StandardizedDataset(
        source=source,
        batch_size=batch_size,
        transform_pipeline=transform_pipeline,
        include_raw_in_batch=include_raw_in_batch,
    )
