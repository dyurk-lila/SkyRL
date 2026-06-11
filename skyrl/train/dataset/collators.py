"""Collators that turn tokenized SFT examples into a :class:`TrainingInputBatch`.

Two callables cover the two SFT data paths:

- :class:`DefaultCollator` left-pads sequences to the batch maximum and applies
  the per-non-pad-token loss normalization.
- :class:`PackedDataCollator` performs controller-level FFD bin-packing
  (Megatron-only): once per training step it packs sequences into bins of
  capacity ``max_tokens_per_microbatch``, rounds the bin count up to a multiple
  of ``dp_size`` (so every DP rank gets the same number of micro-batches), and
  emits one row per bin. On the eval path (when the batch size differs from the
  configured training ``batch_size``) it falls back to the un-packed
  :class:`DefaultCollator` behavior.

Both reuse the shared :func:`skyrl.train.sft_trainer.collate_sft_batch` free
function for the un-packed layout.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from loguru import logger

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch

from .bin_packing import make_seq_packer


class DefaultCollator:
    """Left-pad examples into a batch and apply loss normalization.

    Normalizes the ``loss_mask`` so that the sum-reduction in
    ``cross_entropy_loss`` produces a per-non-pad-token mean: the scale is
    ``batch_size / (micro_train_batch_size_per_gpu * total_nonpad)`` where
    ``total_nonpad`` is the count of loss-contributing tokens in the batch.
    This accounts for the ``microbatch_weight`` (FSDP) or ``1/num_microbatches``
    (Megatron) applied during gradient accumulation so the effective gradient
    equals ``d[sum(-log_probs_on_nonpad) / total_nonpad]``.
    """

    def __init__(self, tokenizer, micro_train_batch_size_per_gpu: int):
        self.tokenizer = tokenizer
        self.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu

    def __call__(self, examples: list, batch_size: int) -> TrainingInputBatch:
        """Collate ``examples`` and scale the loss mask.

        Args:
            examples: Tokenized examples to collate.
            batch_size: Global batch dimension used in the loss-mask scaling
                factor. The train path passes ``sft_cfg.batch_size`` and the
                eval path passes its per-dispatch chunk size.
        """
        # Imported lazily to avoid a circular import: ``sft_trainer`` imports
        # this module to select a collator at construction time.
        from skyrl.train.sft_trainer import collate_sft_batch

        batch = collate_sft_batch(examples, self.tokenizer)
        micro_batch_size = self.micro_train_batch_size_per_gpu
        total_nonpad = max(batch["loss_mask"].sum().item(), 1)
        batch["loss_mask"] = batch["loss_mask"].float() * (batch_size / (micro_batch_size * total_nonpad))
        return batch


class PackedDataCollator:
    """Pack examples into bin rows via FFD and return a :class:`TrainingInputBatch`.

    Activates on the training-step batch (``batch_size == self.batch_size``).
    Flow:

    1. Compute per-example sequence lengths.
    2. FFD-pack with ``bin_capacity = max_tokens_per_microbatch``,
       ``min_bin_count = dp_size``, ``bin_count_multiple = dp_size``.
    3. Round-robin assign bins to DP shards (this happens implicitly inside
       ``MeshDispatch.dispatch`` because the rows are laid out in shard-major
       order: shard 0 rows first, then shard 1, etc).
    4. Build the per-bin packed row tensors and the per-row ``sub_seq_lengths``
       data field (a :class:`TensorList`).

    On the eval path (``batch_size != self.batch_size``) it delegates to a
    :class:`DefaultCollator` so eval always uses the un-packed layout; packing
    only fires on the training-step batch.
    """

    def __init__(
        self,
        tokenizer,
        max_tokens_per_microbatch: int,
        tp_size: int,
        pp_size: int,
        cp_size: int,
        dp_size: int,
        batch_size: int,
        micro_train_batch_size_per_gpu: int,
        graph_seqlen: Optional[int] = None,
        max_subseqs_per_bin: Optional[int] = None,
    ):
        if max_tokens_per_microbatch is None:
            raise ValueError("PackedDataCollator requires max_tokens_per_microbatch to be set explicitly.")
        self.max_tokens_per_microbatch = max_tokens_per_microbatch
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.batch_size = batch_size
        self._default_collator = DefaultCollator(tokenizer, micro_train_batch_size_per_gpu)
        self._tokenizer = tokenizer

        # ------------------------------------------------------------------
        # CUDA-graph static-shape padding (opt-in).
        # ------------------------------------------------------------------
        # When ``graph_seqlen`` is set, every packed bin is padded with
        # synthetic pad sub-sequences so that, ACROSS microbatches/steps:
        #   (a) the rmpad token count == graph_seqlen (constant T),
        #   (b) the number of sub-sequences per bin == max_subseqs_per_bin
        #       (constant cu_seqlens shape [K+1]),
        #   (c) max_seqlen is pinned to a constant (worker-side, via env).
        # These are the three things mcore's CUDA-graph replay value-checks,
        # and they let hybrid Mamba2/MoE SFT capture+replay graphs.
        #
        # ``graph_seqlen is None`` (default) => byte-identical to the legacy
        # path: the synthetic-pad branch below is fully skipped.
        self.graph_seqlen = graph_seqlen
        if graph_seqlen is not None:
            # align_size factors (tp/cp) are recomputed in __call__; here we
            # only have access to tp/cp, so reproduce the same expression to
            # validate the graph_seqlen divisibility up front.
            align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
            if graph_seqlen % align_size != 0:
                raise ValueError(
                    f"graph_seqlen ({graph_seqlen}) must be divisible by align_size ({align_size}) "
                    f"= tp_size*cp_size*2 if cp>1 else tp_size (tp={tp_size}, cp={cp_size}); "
                    f"otherwise the synthetic pad sub-seq slack would not be an align multiple."
                )
            if graph_seqlen < max_tokens_per_microbatch:
                raise ValueError(
                    f"graph_seqlen ({graph_seqlen}) must be >= max_tokens_per_microbatch "
                    f"(bin capacity = {max_tokens_per_microbatch}); a bin can pack up to "
                    f"bin_capacity real tokens and must still fit inside graph_seqlen."
                )
            # ``max_subseqs_per_bin`` fixes the cu_seqlens shape [K+1]. The max
            # sub-seq count actually observed in a batch is NOT static across
            # steps, so we never derive it from the batch. If the caller does
            # not pin it, fall back to the absolute, always-static upper bound:
            # a bin of graph_seqlen tokens can hold at most graph_seqlen //
            # align_size sub-seqs (each real sub-seq costs >= align_size padded
            # tokens). This bound is conservative (large K => more zero-length
            # cu_seqlens segments) but guarantees a fixed shape every step.
            if max_subseqs_per_bin is None:
                max_subseqs_per_bin = graph_seqlen // align_size
            self.max_subseqs_per_bin = max_subseqs_per_bin
        else:
            self.max_subseqs_per_bin = max_subseqs_per_bin

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        # The eval fall-through reuses the inner DefaultCollator, so keep both
        # tokenizers in sync.
        self._tokenizer = value
        self._default_collator.tokenizer = value

    def __call__(self, examples: list, batch_size: int) -> TrainingInputBatch:
        # When eval calls the collator with a chunk of the eval set, fall back
        # to the un-packed collate path. Packing only fires on the
        # training-step batch (== self.batch_size).
        if batch_size != self.batch_size:
            return self._default_collator(examples, batch_size=batch_size)

        bin_capacity = self.max_tokens_per_microbatch

        tp_size = self.tp_size
        pp_size = self.pp_size
        cp_size = self.cp_size
        # Each sub-seq's padded length must satisfy two divisibility
        # constraints, which is why ``align_size`` carries both factors:
        #   - Sequence Parallelism (auto-on when tp>1) shards along the seq
        #     dim, so each segment must be divisible by ``tp_size``.
        #   - Context Parallelism splits each segment into ``2*cp_size`` equal
        #     load-balanced causal chunks, so each segment must be divisible by
        #     ``2*cp_size``.
        # This MUST stay in lockstep with the worker's preprocess_packed_seqs
        # (megatron_utils.py): if the divisors drift, the per-rank CP/SP
        # gather/scatter offsets silently corrupt loss/grads (no crash).
        align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

        dp_size = self.dp_size

        # ------------------------------------------------------------------
        # 1. Sequence lengths and full-sequence loss masks
        # ------------------------------------------------------------------
        # We need the *full-sequence* loss mask (one entry per token, not
        # just over the response window) so the packed bin row can have a
        # per-position mask with correct boundary zeros.
        #
        # We keep per-example NumPy arrays for ``input_ids`` and the
        # reconstructed full loss mask so the row construction in section 4 can
        # write them with vectorized slice assignments (one C-level memcpy per
        # sub-seq) instead of a per-token Python loop.
        seq_lengths: List[int] = []
        full_input_ids: List[np.ndarray] = []
        full_loss_masks: List[np.ndarray] = []
        for ex in examples:
            s = len(ex["input_ids"])
            seq_lengths.append(s)
            n_pad = s - ex["num_actions"]
            # full loss mask = [0]*n_pad (prompt prefix) then the per-response
            # token mask. Built as float32 so it can be sliced straight into the
            # float loss_mask row without a per-element cast.
            full_mask = np.empty(s, dtype=np.float32)
            full_mask[:n_pad] = 0.0
            full_mask[n_pad:] = np.asarray(ex["loss_mask"], dtype=np.float32)
            assert full_mask.shape[0] == s, (
                f"Reconstructed full loss_mask length {full_mask.shape[0]} != seq length {s}"
            )
            full_loss_masks.append(full_mask)
            full_input_ids.append(np.asarray(ex["input_ids"], dtype=np.int64))

        # ------------------------------------------------------------------
        # 2. FFD pack with DP-symmetry constraints
        # ------------------------------------------------------------------
        # Each bin row is one worker micro-batch. Megatron's
        # ``forward_backward_func`` runs one micro-batch per bin on each DP
        # rank, and its pipeline schedule requires every DP rank to issue the
        # same number of micro-batches. Forcing the global bin count to a
        # multiple of ``dp_size`` makes the per-DP-rank bin count (and thus
        # ``num_microbatches``) identical across ranks.
        bin_count_multiple = dp_size
        packer = make_seq_packer(
            "first_fit_decreasing",
            bin_capacity=bin_capacity,
            min_bin_count=bin_count_multiple,
            bin_count_multiple=bin_count_multiple,
        )
        bins: List[List[int]] = packer.pack(seq_lengths)

        # Assign bins to DP shards via round-robin (bin_idx % shards).
        # Concretely we want the resulting layout to be shard-major:
        # shard 0's bins occupy rows [0, K/dp), shard 1's bins occupy
        # [K/dp, 2K/dp), etc. MeshDispatch.dispatch chunks the batch
        # by dp_size and sends contiguous slabs, so we lay out the rows
        # already in shard-major order.
        shard_bins: List[List[List[int]]] = [[] for _ in range(dp_size)]
        for bin_idx, bin_indices in enumerate(bins):
            shard_idx = bin_idx % dp_size
            shard_bins[shard_idx].append(bin_indices)
        flat_bins: List[List[int]] = []
        for shard_idx in range(dp_size):
            flat_bins.extend(shard_bins[shard_idx])

        # ------------------------------------------------------------------
        # 3. Compute packed-row lengths (with tp_size alignment per sub-seq)
        #    and the global max packed length (for PP > 1 uniform padding).
        # ------------------------------------------------------------------
        def _round_up(x: int, m: int) -> int:
            return ((x + m - 1) // m) * m

        bin_packed_lengths: List[int] = []
        bin_subseq_lengths: List[List[int]] = []  # one list per bin row
        for bin_indices in flat_bins:
            subseq_lens = [seq_lengths[idx] for idx in bin_indices]
            # Each sub-seq's length is independently aligned to tp_size
            # (matches preprocess_packed_seqs behavior).
            packed_len = sum(_round_up(s, align_size) for s in subseq_lens)
            bin_packed_lengths.append(packed_len)
            bin_subseq_lengths.append(subseq_lens)

        graph_seqlen = self.graph_seqlen
        if graph_seqlen is not None:
            # --------------------------------------------------------------
            # CUDA-graph static-shape padding (opt-in; see __init__ docstring).
            # --------------------------------------------------------------
            # Make BOTH the rmpad token count (T) and the sub-seq count (K)
            # constant across every bin and every step:
            #
            #   (a) Token slack: append ONE synthetic pad sub-seq of length
            #       (graph_seqlen - packed_len) to each bin so the sum of
            #       round_up(len, align_size) == graph_seqlen EXACTLY. Both
            #       graph_seqlen and packed_len are align multiples, so the
            #       slack is too => round_up(slack, align_size) == slack and
            #       the bin's padded total lands precisely on graph_seqlen.
            #       This single pad sub-seq carries ALL the slack so the dense
            #       row width and the rmpad T are static. (We validated in
            #       __init__ that graph_seqlen >= bin_capacity >= packed_len,
            #       so the slack is always >= 0.)
            #
            #   (b) Sub-seq count: after the slack sub-seq, append ZERO-LENGTH
            #       sub-seqs until the bin holds exactly max_subseqs_per_bin
            #       entries, so cu_seqlens has a constant shape [K+1] every
            #       step. Zero-length entries are structurally safe: cu_seqlens
            #       is built from this SAME flattened list (preprocess_packed_
            #       seqs), so a 0-length entry produces a zero-WIDTH cu_seqlens
            #       segment (a duplicated cumsum value) and a matching no-op in
            #       every downstream gather/scatter loop (_build_packed_targets,
            #       _packed_subseq_row_indices_offsets_and_lens), which advance
            #       row_offset by the (zero) padded segment width. The exact
            #       seg-count invariants in those consumers therefore still hold.
            #       Mamba/flash-attn varlen tolerate zero-length segments.
            #
            # The synthetic sub-seqs reference NO real tokens, so the row build
            # loop in section 4 never writes them; their token positions stay at
            # the pad fill (sequences=pad_token_id, attention_mask=0,
            # loss_mask=0) from the np.full/np.zeros init. total_nonpad counts
            # only real loss_mask==1 positions and is unaffected.
            #
            # CAVEAT (MoE expert-bias): these synthetic pad tokens carry no
            # attention_mask, but the MoE router's expert-bias accumulator
            # (moe_router_enable_expert_bias) currently has NO padding_mask, so
            # the pad tokens leak into the per-expert token-count used to update
            # the bias. A padding_mask must be plumbed through the router before
            # this mode is used with moe_router_enable_expert_bias=true. That
            # correctness fix is intentionally out of scope for this PR.
            max_subseqs_per_bin = self.max_subseqs_per_bin
            for row_idx, packed_len in enumerate(bin_packed_lengths):
                slack = graph_seqlen - packed_len
                # (a) ALWAYS append exactly one slack sub-seq absorbing the
                #     (align-multiple) slack, even when slack == 0. Appending it
                #     unconditionally keeps the per-bin segment layout identical
                #     across steps: [n_real real segments] + [1 slack segment] +
                #     [zero-length K-pad]. Skipping it when slack==0 (the prior
                #     behavior) made the position of the first zero-pad segment
                #     depend on the data, which left ``cu_seqlens`` shape constant
                #     (the zero-pad backfills) but its *values* data-dependent. A
                #     fixed, exactly-once slack segment makes both the shape and
                #     the segment boundaries deterministic for graph replay.
                #     A zero-length slack segment is structurally identical to a
                #     zero-length K-pad segment (a duplicated cumsum value), so
                #     this never changes the real tokens' boundaries.
                bin_subseq_lengths[row_idx].append(slack)
                # (b) zero-length sub-seqs to reach the fixed K.
                n_real_and_slack = len(bin_subseq_lengths[row_idx])
                if n_real_and_slack > max_subseqs_per_bin:
                    raise ValueError(
                        f"bin {row_idx} has {n_real_and_slack} sub-seqs (incl. slack pad) which exceeds "
                        f"max_subseqs_per_bin ({max_subseqs_per_bin}); raise max_subseqs_per_bin or lower "
                        f"the per-bin sequence count."
                    )
                bin_subseq_lengths[row_idx].extend([0] * (max_subseqs_per_bin - n_real_and_slack))
                # Hard invariant for CUDA-graph replay: every bin row MUST hold
                # EXACTLY max_subseqs_per_bin sub-seqs so the worker's cu_seqlens
                # shape is the constant [max_subseqs_per_bin + 1] on every step.
                assert len(bin_subseq_lengths[row_idx]) == max_subseqs_per_bin, (
                    f"bin {row_idx} has {len(bin_subseq_lengths[row_idx])} sub-seqs after padding, "
                    f"expected exactly max_subseqs_per_bin ({max_subseqs_per_bin}); cu_seqlens shape "
                    f"would not be static across steps."
                )
                # Every bin now sums (with align round-up) to exactly graph_seqlen.
                bin_packed_lengths[row_idx] = graph_seqlen
            # Dense row width is constant graph_seqlen too.
            max_packed_len = graph_seqlen
        elif pp_size > 1:
            # Pad all packed rows to the global max so Megatron's
            # pipeline schedule sees uniform shapes.
            max_packed_len = max(bin_packed_lengths) if bin_packed_lengths else 0
            # Also align the global max to tp_size to keep TP/SP happy.
            max_packed_len = _round_up(max_packed_len, align_size)
        else:
            max_packed_len = max(bin_packed_lengths) if bin_packed_lengths else 0

        # Guard against degenerate rows (e.g. an empty bin from
        # _adjust_bin_count) — empty bins must not be produced in practice
        # because the redistribution moves one sub-seq into every empty
        # bin. If we ever see one, we widen this assertion.
        for bin_indices in flat_bins:
            assert bin_indices, "FFD produced an empty bin; _adjust_bin_count should prevent this"

        # ------------------------------------------------------------------
        # 4. Build per-row tensors: sequences, attention_mask, loss_mask
        # ------------------------------------------------------------------
        pad_token_id = self.tokenizer.pad_token_id
        num_bins = len(flat_bins)

        n_samples = len(examples)
        logger.info(
            f"sequence packing | packed {n_samples} samples into {num_bins} bins "
            f"(~{num_bins // dp_size}/DP rank, bin_capacity={bin_capacity} tokens)"
        )

        # Build the row tensors in NumPy and convert once at the end. The
        # per-position writes are vectorized slice assignments (one per
        # sub-seq), replacing the former per-token Python loop that dominated
        # the controller-side collate wall-time (~97% of it). Each sub-seq
        # touches O(s) elements but in a single C-level memcpy, so the total
        # cost is O(sum of sub-seq lengths) of memory traffic instead of that
        # many Python bytecode iterations. The produced tensors are
        # bit-identical to the previous implementation.
        sequences_np = np.full((num_bins, max_packed_len), pad_token_id, dtype=np.int64)
        attention_mask_np = np.zeros((num_bins, max_packed_len), dtype=np.int64)
        # loss_mask is one position shorter than the row to match
        # `token_logprobs[:, :-1]` semantics inside the loss function.
        loss_mask_np = np.zeros((num_bins, max_packed_len - 1), dtype=np.float32)
        loss_mask_width = max_packed_len - 1

        for row_idx, bin_indices in enumerate(flat_bins):
            row_offset = 0
            for ex_idx in bin_indices:
                ex = examples[ex_idx]
                s = seq_lengths[ex_idx]
                # Write the sub-seq tokens into the row (vectorized memcpy).
                sequences_np[row_idx, row_offset : row_offset + s] = full_input_ids[ex_idx]
                attention_mask_np[row_idx, row_offset : row_offset + s] = 1

                # Build the per-position loss mask for this sub-seq.
                # Position p (in row coords, p in [row_offset, row_offset + s))
                # predicts token at p+1. The loss_mask at p (in the [B, S-1]
                # action_log_probs slot) is 1 iff p+1 is a response/assistant
                # token AND p+1 is in the same sub-seq.
                #   For p_local in [0, s - 1): mask[row_offset + p_local] =
                #       full_mask[p_local + 1]  (== full_mask[1:s]).
                #   For p_local == s - 1: 0 (sub-seq boundary / row end).
                # We clamp the write window to ``loss_mask_width`` to reproduce
                # the original ``row_p < max_packed_len - 1`` guard.
                if s > 1:
                    write_end = min(row_offset + s - 1, loss_mask_width)
                    n_write = write_end - row_offset
                    if n_write > 0:
                        loss_mask_np[row_idx, row_offset:write_end] = full_loss_masks[ex_idx][1 : 1 + n_write]

                # Advance row_offset, padding sub-seq to tp_size multiple.
                row_offset += _round_up(s, align_size)

        # ``total_nonpad`` (sum of 1s BEFORE scaling) is the float-exact sum of
        # the binary loss mask; compute it in one vectorized reduction.
        total_nonpad = int(loss_mask_np.sum())

        sequences = torch.from_numpy(sequences_np)
        attention_mask = torch.from_numpy(attention_mask_np)
        loss_mask = torch.from_numpy(loss_mask_np)

        # ------------------------------------------------------------------
        # 5. Loss normalization
        # ------------------------------------------------------------------
        # The realized gradient is sum(loss * loss_mask) / (num_microbatches
        # * dp_size). Each bin row is one micro-batch, so num_microbatches *
        # dp_size = num_bins. So loss_mask *= num_bins / total_nonpad yields
        # mean_over_nonpad.
        scale = num_bins / max(total_nonpad, 1)
        loss_mask.mul_(scale)

        # ------------------------------------------------------------------
        # 6. Pack into TrainingInputBatch with sub_seq_lengths data field
        # ------------------------------------------------------------------
        # ``sub_seq_lengths`` is genuinely per-sample data: after FFD the
        # batch's "sample" *is* a bin, so ``len(bin_subseq_lengths) == num_bins
        # == batch_size``, co-indexed with ``sequences[r]``. We store it as a
        # ``TensorList`` (one 1-D int tensor per bin, ragged across bins — same
        # pattern as ``image_grid_thw``) so ``MeshDispatch`` shards it per-DP
        # rank automatically alongside ``sequences``/``attention_mask``,
        # eliminating the worker-side per-rank slice. ``preprocess_packed_seqs``
        # and the Megatron packed-logprob scatter want ``list[list[int]]``, so a
        # ``.tolist()`` happens at the ``forward_step`` boundary.
        sub_seq_lengths = TensorList([torch.tensor(lens, dtype=torch.long) for lens in bin_subseq_lengths])
        batch = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "sub_seq_lengths": sub_seq_lengths,
            }
        )
        batch.metadata = {
            "response_length": max_packed_len - 1,
        }
        return batch
