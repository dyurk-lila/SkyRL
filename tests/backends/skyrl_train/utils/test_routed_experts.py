import numpy as np
import pytest

from skyrl.backends.skyrl_train.utils.routed_experts import (
    RoutedExpertRoutes,
    RoutedExpertTrace,
    compact_routed_expert_indices,
    validate_moe_layer_indices,
)

MOE_LAYERS = (1, 3)


def _trace_routes(rows: int, layer_indices: tuple[int, ...] = MOE_LAYERS) -> RoutedExpertRoutes:
    indices = np.arange(rows * 4, dtype=np.int32).reshape(rows, 2, 2) % 8
    return RoutedExpertRoutes(indices, layer_indices)


@pytest.mark.parametrize(
    "routes,expected_dtype",
    [
        (np.arange(12).reshape(3, 2, 2), np.uint8),
        (np.array([[[2**8 - 1]]]), np.uint8),
        (np.array([[[0, 2**8]]]), np.int16),
        (np.array([[[0, 2**15 - 1]]]), np.int16),
        (np.array([[[0, 2**15]]]), np.int32),
        (np.array([[[0, 2**31 - 1]]], dtype=np.int64), np.int32),
        (np.empty((0, 2, 2), dtype=np.int64), np.uint8),
    ],
)
def test_compaction_picks_smallest_safe_dtype(routes, expected_dtype):
    compact = compact_routed_expert_indices(routes)

    assert compact.dtype == expected_dtype
    assert compact.flags.c_contiguous
    assert np.array_equal(compact, routes)


def test_compaction_makes_read_only_arrays_writable():
    routes = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    routes.flags.writeable = False

    compact = compact_routed_expert_indices(routes)

    assert compact.dtype == np.uint8
    assert compact.flags.c_contiguous
    assert compact.flags.writeable


def test_compaction_copies_non_contiguous_input():
    compact = compact_routed_expert_indices(np.arange(24).reshape(6, 2, 2)[::2])

    assert compact.flags.c_contiguous
    assert np.array_equal(compact, np.arange(24).reshape(6, 2, 2)[::2])


def test_compaction_rejects_nested_lists():
    with pytest.raises(TypeError, match="NumPy array"):
        compact_routed_expert_indices([[[1, 2]]])


@pytest.mark.parametrize(
    "routes",
    [
        np.array([1, 2]),  # not 3-D
        np.array([[[1.0]]]),  # not integral
        np.array([[[-1]]]),  # negative expert id
        np.array([[[2**31]]], dtype=np.uint64),  # exceeds int32
    ],
)
def test_compaction_rejects_invalid_routes(routes):
    with pytest.raises(ValueError):
        compact_routed_expert_indices(routes)


def test_trace_returns_only_the_rows_it_captured():
    """The row count identifies where capture stops."""
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=_trace_routes(4))
    trace.record_generation(prompt_token_count=7, generated_token_count=2, routed_experts=_trace_routes(4))

    routes = trace.finalize(token_count=10, loss_mask=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0])

    assert routes.indices.shape == (8, 2, 2)
    expected = np.concatenate((_trace_routes(4).indices, _trace_routes(4).indices))
    assert np.array_equal(routes.indices, expected)


def test_trace_keeps_full_coverage_when_every_token_has_a_route():
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=_trace_routes(4))

    routes = trace.finalize(token_count=4, loss_mask=[0, 0, 0, 1])

    assert routes.indices.shape == (4, 2, 2)


def test_trace_rejects_an_uncaptured_loss_active_target():
    """Every loss-active target must have a captured route."""
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=_trace_routes(4))

    with pytest.raises(ValueError, match="missing routed-expert row for loss-active target at token 5"):
        trace.finalize(token_count=6, loss_mask=[0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize(
    "layer_indices,expected",
    [
        ([0, 1, 2], (0, 1, 2)),
        (range(3), (0, 1, 2)),
        ((1, 3, 5, 7), (1, 3, 5, 7)),  # a hybrid stack's MoE layers are not contiguous
        (np.array([2, 4]), (2, 4)),
    ],
)
def test_validate_layer_indices_canonicalizes(layer_indices, expected):
    assert validate_moe_layer_indices(layer_indices) == expected


@pytest.mark.parametrize(
    "layer_indices,match",
    [
        ((), "at least one"),
        ((1, 1), "strictly increasing"),
        ((3, 1), "strictly increasing"),
        ((-1, 2), "non-negative"),
        ((0, 2**31), "deeper than any model"),
    ],
)
def test_validate_layer_indices_rejects_unusable_lists(layer_indices, match):
    with pytest.raises(ValueError, match=match):
        validate_moe_layer_indices(layer_indices)


def test_routes_keep_the_captured_layer_mapping():
    routes = RoutedExpertRoutes(np.zeros((5, 2, 3), dtype=np.uint8), MOE_LAYERS)

    assert routes.layer_indices == MOE_LAYERS
    assert routes.num_tokens == 5
    assert routes.truncate(2).num_tokens == 2
    assert routes.truncate(2).layer_indices == MOE_LAYERS


def test_routes_covering_all_layers_names_every_slot():
    routes = RoutedExpertRoutes.covering_all_layers(np.zeros((2, 4, 2), dtype=np.uint8))

    assert routes.layer_indices == (0, 1, 2, 3)


def test_routes_reject_a_layer_count_that_misses_a_slot():
    with pytest.raises(ValueError, match="cover 3 layers"):
        RoutedExpertRoutes(np.zeros((1, 3, 2), dtype=np.uint8), (0, 1))


@pytest.mark.parametrize("indices", [[[[1, 2]]], np.array([1, 2])])
def test_routes_reject_non_3d_indices(indices):
    with pytest.raises((TypeError, ValueError)):
        RoutedExpertRoutes(indices, (0,))


def test_routes_compare_by_value():
    left = RoutedExpertRoutes(np.zeros((1, 2, 2), dtype=np.uint8), MOE_LAYERS)

    assert left == RoutedExpertRoutes(np.zeros((1, 2, 2), dtype=np.uint8), MOE_LAYERS)
    assert left != RoutedExpertRoutes(np.zeros((1, 2, 2), dtype=np.uint8), (2, 5))
    assert left != RoutedExpertRoutes(np.ones((1, 2, 2), dtype=np.uint8), MOE_LAYERS)
    assert left != np.zeros((1, 2, 2), dtype=np.uint8)


def test_trace_carries_the_captured_layers_to_the_finalized_routes():
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=_trace_routes(4))
    trace.record_generation(prompt_token_count=7, generated_token_count=2, routed_experts=_trace_routes(4))

    result = trace.finalize(token_count=9, loss_mask=[0, 0, 0, 1, 1, 0, 0, 1, 1])

    assert result.indices.shape == (8, 2, 2) and result.indices.dtype == np.uint8
    assert result.layer_indices == MOE_LAYERS


def test_trace_rejects_layer_indices_changing_midtrajectory():
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=_trace_routes(4))

    with pytest.raises(ValueError, match="layer indices changed mid-trajectory"):
        trace.record_generation(
            prompt_token_count=7,
            generated_token_count=2,
            routed_experts=_trace_routes(4, (2, 5)),
        )
