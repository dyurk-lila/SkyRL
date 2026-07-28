"""Sampled-token logprob HTTP payloads."""

import math
from typing import Any, Iterable, Mapping, Optional, Tuple

# vLLM floors non-finite logprobs to -9999.0 at every serving boundary (see
# `clamp_prompt_logprobs` and the disagg serving path), and uses it as the
# `ChatCompletionLogProb.logprob` default for tokens it reports no logprob for.
CLAMPED_LOGPROB = -9999.0


def build_logprobs_content(
    token_ids: Iterable[int],
    resp_logprobs: Iterable[Optional[Mapping[int, Any]]],
) -> Tuple[list[dict[str, float]], int]:
    """Build the per-token ``logprobs.content`` wire payload for sampled tokens.

    vLLM occasionally reports a non-finite logprob for a token it just sampled
    (``-inf`` from top-k/top-p masking that sampling then selects anyway), and
    omits the entry entirely for others. Both are floored to ``CLAMPED_LOGPROB``
    so the response stays JSON-serializable and one bad token cannot fail the
    whole request.

    Note ``math.isfinite`` also catches NaN, which vLLM's own ``max(logprob,
    -9999.0)`` floor lets through -- ``max`` returns its first argument when the
    comparison is False. NaN is the more damaging value downstream, since
    ``reduce_loss`` multiplies by the loss mask and ``0.0 * nan`` is ``nan``.

    Caveat: under ``off_policy_correction.tis_ratio_type="sequence"`` the
    per-token log-ratios are summed before exponentiating, so a single clamped
    token pins its whole trajectory at the importance-sampling cap. Under
    ``"token"`` mode the effect is bounded to that one token.

    Returns:
        The ``content`` list (one entry per token id) and the number of tokens
        whose logprob had to be clamped.
    """
    content: list[dict[str, float]] = []
    num_clamped = 0
    for tid, lp_dict in zip(token_ids, resp_logprobs):
        logprob = lp_dict[tid].logprob if (lp_dict and tid in lp_dict) else None
        if logprob is None or not math.isfinite(logprob):
            num_clamped += 1
            logprob = CLAMPED_LOGPROB
        content.append({"logprob": logprob})
    return content, num_clamped
