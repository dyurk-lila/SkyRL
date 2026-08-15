from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataTrace,
)

RoutedExpertIndices: TypeAlias = np.ndarray
ROUTED_EXPERT_DTYPES = frozenset({np.dtype(np.uint8), np.dtype(np.int16), np.dtype(np.int32)})

# Global transformer-layer positions are carried as int32-ranged integers. No stack comes close
# to that depth, so an index that does not fit names no layer of any real model.
ROUTED_EXPERT_LAYER_INDEX_DTYPE = np.dtype(np.int32)

# ``TrainingInputBatch.metadata`` key naming the global transformer layer that each slot of
# ``rollout_expert_indices``' layer dimension holds. Consumers map their own layers onto that
# dimension by looking them up here, never by assuming it spans every layer in order.
ROUTED_EXPERT_LAYER_INDICES_KEY = "rollout_expert_layer_indices"


def validate_moe_layer_indices(layer_indices: Sequence[int]) -> tuple[int, ...]:
    """Return ``layer_indices`` as a canonical, strictly increasing tuple of layer positions."""
    canonical = tuple(int(layer_index) for layer_index in layer_indices)
    if not canonical:
        raise ValueError("routed-expert layer indices must name at least one MoE layer")
    if any(later <= earlier for earlier, later in zip(canonical, canonical[1:])):
        raise ValueError(f"routed-expert layer indices must be strictly increasing, got {canonical}")
    if canonical[0] < 0:
        raise ValueError(f"routed-expert layer indices must be non-negative, got {canonical}")
    if canonical[-1] > np.iinfo(ROUTED_EXPERT_LAYER_INDEX_DTYPE).max:
        raise ValueError(f"routed-expert layer index {canonical[-1]} is deeper than any model")
    return canonical


def _validate_routed_expert_shape(indices: RoutedExpertIndices) -> None:
    if not isinstance(indices, np.ndarray):
        raise TypeError("routed expert indices must be a NumPy array")
    if indices.ndim != 3:
        raise ValueError(f"routed expert indices must be a [tokens, layers, topk] array, got shape {indices.shape}")


# eq=False: the generated __eq__ would compare `indices` with `==`, whose array result has no
# truth value, so every comparison would raise instead of answering.
@dataclass(frozen=True, eq=False)
class RoutedExpertRoutes:
    """Captured MoE routes plus the global transformer layer each captured layer came from.

    ``indices`` is ``[tokens, len(layer_indices), topk]``. Its layer dimension covers only the
    layers the inference server captured, so ``layer_indices`` is the single source of truth
    for which transformer layer a slot belongs to: every consumer maps by lookup rather than
    by assuming a contiguous zero-based ordering.

    Expert-id dtype and range remain the business of ``compact_routed_expert_indices`` and the
    wire decoder, so pairing routes with their layers never rescans the payload.
    """

    indices: RoutedExpertIndices
    layer_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_routed_expert_shape(self.indices)
        object.__setattr__(self, "layer_indices", validate_moe_layer_indices(self.layer_indices))
        if self.indices.shape[1] != len(self.layer_indices):
            raise ValueError(
                f"routed experts cover {self.indices.shape[1]} layers but "
                f"{len(self.layer_indices)} layer indices were carried alongside them"
            )

    @classmethod
    def covering_all_layers(cls, indices: RoutedExpertIndices) -> "RoutedExpertRoutes":
        """Pair a capture spanning the whole transformer stack with the identity layer mapping."""
        _validate_routed_expert_shape(indices)
        return cls(indices, tuple(range(indices.shape[1])))

    @property
    def num_tokens(self) -> int:
        return self.indices.shape[0]

    def truncate(self, token_count: int) -> "RoutedExpertRoutes":
        """Return the first ``token_count`` token rows, keeping the layer mapping intact."""
        return RoutedExpertRoutes(self.indices[:token_count], self.layer_indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoutedExpertRoutes):
            return False
        return self.layer_indices == other.layer_indices and np.array_equal(self.indices, other.indices)


class RoutedExpertTrace:
    """Accumulate routed experts across incremental generation calls."""

    def __init__(self) -> None:
        self._metadata = TokenMetadataTrace()
        self._layer_indices: tuple[int, ...] | None = None

    @property
    def prompt_start(self) -> int:
        return self._metadata.num_rows

    def record_generation(
        self,
        *,
        prompt_token_count: int,
        generated_token_count: int,
        routed_experts: RoutedExpertRoutes,
    ) -> None:
        if prompt_token_count < self.prompt_start:
            raise ValueError("routed-expert prompt start exceeds prompt length")
        if generated_token_count < 1:
            raise ValueError("routed-expert generation must produce at least one token")

        expected_rows = prompt_token_count - self.prompt_start + generated_token_count - 1
        if self._layer_indices is None:
            self._layer_indices = routed_experts.layer_indices
        elif self._layer_indices != routed_experts.layer_indices:
            raise ValueError(
                f"routed-expert layer indices changed mid-trajectory from {self._layer_indices} "
                f"to {routed_experts.layer_indices}"
            )
        self._metadata.append(compact_routed_expert_indices(routed_experts.indices), expected_rows=expected_rows)

    def finalize(self, *, token_count: int, loss_mask: Sequence[int]) -> RoutedExpertRoutes:
        """Return the captured route prefix and layer identities without fabricating tail rows."""
        if len(loss_mask) != token_count:
            raise ValueError(f"loss mask has {len(loss_mask)} entries, expected {token_count}")
        if self.prompt_start > token_count:
            raise ValueError(f"routed-expert trace has {self.prompt_start} rows for {token_count} tokens")

        if any(loss_mask[self.prompt_start + 1 : token_count]):
            for source_index in range(self.prompt_start, token_count - 1):
                if loss_mask[source_index + 1] != 0:
                    raise ValueError(f"missing routed-expert row for loss-active target at token {source_index + 1}")

        if self._layer_indices is None:
            raise ValueError("cannot finalize a routed-expert trace before any routes are captured")
        return RoutedExpertRoutes(self._metadata.finalize(expected_rows=self.prompt_start), self._layer_indices)


def compact_routed_expert_indices(routed_experts: RoutedExpertIndices) -> RoutedExpertIndices:
    """Validate and compact a routed-expert array to the canonical integer dtype."""
    if not isinstance(routed_experts, np.ndarray):
        raise TypeError("routed expert indices must be a NumPy array")
    if routed_experts.ndim != 3 or not np.issubdtype(routed_experts.dtype, np.integer):
        raise ValueError(
            "routed expert indices must be an integer [tokens, layers, topk] array, "
            f"got shape {routed_experts.shape} and dtype {routed_experts.dtype}"
        )
    if int(routed_experts.min(initial=0)) < 0:
        raise ValueError("routed expert indices must be non-negative")

    max_expert_id = int(routed_experts.max(initial=0))
    if max_expert_id < 2**8:
        dtype = np.dtype(np.uint8)
    elif max_expert_id < 2**15:
        dtype = np.dtype(np.int16)
    elif max_expert_id < 2**31:
        dtype = np.dtype(np.int32)
    else:
        raise ValueError(f"routed expert index exceeds signed int32: {max_expert_id}")

    compact = np.asarray(routed_experts, dtype=dtype, order="C")
    if not compact.flags.writeable:
        compact = compact.copy(order="C")
    return compact
