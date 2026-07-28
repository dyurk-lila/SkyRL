import math
from dataclasses import dataclass

import orjson
import pytest

from skyrl.backends.skyrl_train.inference_servers.logprobs_wire import (
    CLAMPED_LOGPROB,
    build_logprobs_content,
)


@dataclass
class _Logprob:
    """Stand-in for vLLM's ``Logprob``, which the endpoint reads as ``.logprob``."""

    logprob: float


@pytest.mark.parametrize(
    "bad_value",
    [float("-inf"), float("inf"), float("nan")],
)
def test_non_finite_logprob_is_clamped(bad_value):
    content, num_clamped = build_logprobs_content([7], [{7: _Logprob(bad_value)}])
    assert content == [{"logprob": CLAMPED_LOGPROB}]
    assert num_clamped == 1


@pytest.mark.parametrize(
    "resp_logprobs",
    [
        [None],
        [{}],
        [{99: _Logprob(-0.5)}],  # entry present, but not for the sampled token
    ],
    ids=["none_entry", "empty_entry", "missing_sampled_token"],
)
def test_missing_logprob_entry_is_clamped(resp_logprobs):
    content, num_clamped = build_logprobs_content([7], resp_logprobs)
    assert content == [{"logprob": CLAMPED_LOGPROB}]
    assert num_clamped == 1


def test_finite_logprobs_pass_through_bit_exact():
    values = [-0.5, -1e-9, 0.0, -12.3456789]
    token_ids = [1, 2, 3, 4]
    content, num_clamped = build_logprobs_content(token_ids, [{tid: _Logprob(v)} for tid, v in zip(token_ids, values)])
    assert [entry["logprob"] for entry in content] == values
    assert num_clamped == 0


def test_length_matches_token_ids_and_counts_only_bad_tokens():
    # ml.train's TITO client asserts len(logprobs) == len(response_ids).
    token_ids = [10, 11, 12, 13]
    resp_logprobs = [
        {10: _Logprob(-0.25)},
        {11: _Logprob(float("-inf"))},
        None,
        {13: _Logprob(-0.75)},
    ]
    content, num_clamped = build_logprobs_content(token_ids, resp_logprobs)
    assert len(content) == len(token_ids)
    assert num_clamped == 2
    assert [entry["logprob"] for entry in content] == [
        -0.25,
        CLAMPED_LOGPROB,
        CLAMPED_LOGPROB,
        -0.75,
    ]


def test_payload_round_trips_through_orjson():
    # The bug this fixes: orjson serializes non-finite floats to JSON `null`, and
    # orjson.loads then rejects bare Infinity/NaN -- so a non-finite logprob must
    # never reach the wire.
    content, _ = build_logprobs_content([7], [{7: _Logprob(float("-inf"))}])
    decoded = orjson.loads(orjson.dumps({"logprobs": {"content": content}}))
    logprob = decoded["logprobs"]["content"][0]["logprob"]
    assert logprob is not None
    assert math.isfinite(logprob)
    assert logprob == CLAMPED_LOGPROB


def test_unclamped_non_finite_would_serialize_to_null():
    # Documents why the clamp is required rather than defensive: orjson does not
    # raise on non-finite input, it silently emits null.
    assert orjson.dumps({"logprob": float("-inf")}) == b'{"logprob":null}'
    assert orjson.dumps({"logprob": float("nan")}) == b'{"logprob":null}'
    with pytest.raises(orjson.JSONDecodeError):
        orjson.loads("[-Infinity]")


def test_empty_logprobs_yields_empty_content():
    content, num_clamped = build_logprobs_content([], [])
    assert content == []
    assert num_clamped == 0
