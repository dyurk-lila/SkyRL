import numpy as np
import pytest

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataTrace,
)
from skyrl.utils.routed_experts import RoutedExpertTrace


def test_token_metadata_trace_chunks_and_independent_schema() -> None:
    trace, other = TokenMetadataTrace(), TokenMetadataTrace()
    trace.append(np.ones((2, 3), dtype=np.int32), expected_rows=2)
    trace.append(np.zeros((1, 3), dtype=np.int32), expected_rows=1)
    other.append(np.empty((0, 4), dtype=np.float32), expected_rows=0)

    with pytest.raises(ValueError, match="expected 4"):
        trace.finalize(expected_rows=4)
    result = trace.finalize(expected_rows=3)
    assert result.shape == (3, 3)
    assert other.finalize(expected_rows=0).shape == (0, 4)
    with pytest.raises(RuntimeError, match="already finalized"):
        trace.finalize(expected_rows=3)


@pytest.mark.parametrize(
    ("rows", "expected", "match"),
    [
        (np.ones((2, 2), dtype=np.int32), 1, "has 2 rows"),
        (np.ones((2, 2), dtype=np.int32)[:, ::2], 2, "contiguous"),
        (np.ones((1, 3), dtype=np.int32), 1, "schema changed"),
        (np.ones((1, 2), dtype=np.int16), 1, "schema changed"),
    ],
)
def test_token_metadata_trace_rejects_invalid_chunks(rows, expected, match) -> None:
    trace = TokenMetadataTrace()
    if rows.shape[0] == 1:
        trace.append(np.ones((1, 2), dtype=np.int32), expected_rows=1)
    with pytest.raises(ValueError, match=match):
        trace.append(rows, expected_rows=expected)


def routes(rows: int) -> np.ndarray:
    return np.arange(rows * 4, dtype=np.int32).reshape(rows, 2, 2) % 8


def test_routed_expert_trace_tracks_multiturn_suffix_and_terminal_gap() -> None:
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=routes(4))
    assert trace.prompt_start == 4
    trace.record_generation(prompt_token_count=7, generated_token_count=2, routed_experts=routes(4))

    result = trace.finalize(token_count=9, loss_mask=[0, 0, 0, 1, 1, 0, 0, 1, 1])
    assert result.shape == (9, 2, 2) and result.dtype == np.uint8
    assert np.array_equal(result[-1, 0], [0, 1])


@pytest.mark.parametrize("active", [False, True])
def test_routed_expert_trace_only_pads_masked_suffix(active: bool) -> None:
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=1, routed_experts=routes(3))
    mask = [0, 0, 0, 0, int(active)]
    if active:
        with pytest.raises(ValueError, match="loss-active target"):
            trace.finalize(token_count=5, loss_mask=mask)
    else:
        result = trace.finalize(token_count=5, loss_mask=mask)
        assert np.array_equal(result[-2:, 0], [[0, 1], [0, 1]])
