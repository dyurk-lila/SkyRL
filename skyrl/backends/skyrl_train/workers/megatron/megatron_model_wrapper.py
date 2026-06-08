from dataclasses import asdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import megatron.core.parallel_state as mpu
import torch
import torch.nn as nn
from megatron.core.distributed import finalize_model_grads
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf

from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    get_model_config,
    make_batch_generator,
    preprocess_packed_seqs,
    recover_left_padding,
    remove_left_padding,
)
from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
    vocab_parallel_entropy,
    vocab_parallel_entropy_packed_sequences,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    compute_approx_kl,
)
from skyrl.backends.skyrl_train.utils.replay_utils import (
    setup_per_microbatch_replay_backward,
    setup_per_microbatch_replay_forward,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.train.config import TrainerConfig


def _install_fused_lm_head_capture(actor_module: List[nn.Module], holder: dict) -> bool:
    """Wrap each last-pipeline-stage model's ``output_layer.forward`` to return the
    pre-projection hidden state (skipping the vocab GEMM) so the fused log-prob can apply the
    head per sequence-chunk and never materialize the full ``[*, seq, vocab // TP]`` logits.

    This is the SkyRL-side, megatron-source-free realisation of "return hidden instead of
    logits" (cf. verl's ``model_forward_fused``): we replace only the ``output_layer.forward``
    so the surrounding model forward (its ``_scale_logits`` + ``transpose`` to ``[b, s, h]``)
    is unchanged. The replacement:
      * resolves the LM-head weight from the ``weight=`` kwarg the model passes — which is
        ``shared_embedding_or_output_weight()`` — so **tied embeddings are handled for free**
        (no ``output_layer.weight is None`` crash); falls back to ``self.weight`` otherwise;
      * when the layer is sequence-parallel, gathers the hidden across TP with
        ``tensor_parallel_output_grad=True`` (its backward is a reduce-scatter — exactly what
        stock ``ColumnParallelLinear`` does for its input gradient, so the fused function's
        per-shard ``grad_hidden`` is reduced correctly without any extra all-reduce);
      * stashes the resolved weight in ``holder["weight"]`` for ``loss_func`` to read, and
        returns the (gathered) hidden in the logits position with a ``None`` bias.

    Returns True if the capture was installed on at least one stage. Returns False (caller
    falls back to the stock logits path) when the model uses MuP output scaling — the fused
    path bypasses ``_scale_logits`` so it would be silently wrong — or has no ``output_layer``.
    Raises on a genuinely unexpected megatron layout (fail loud, never silently mis-train).
    """
    from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
    from megatron.core.utils import unwrap_model

    installed = False
    for vp_model in actor_module:
        model = unwrap_model(vp_model)
        output_layer = getattr(model, "output_layer", None)
        if output_layer is None:
            # Not the last pipeline stage (or no head here) — nothing to capture.
            continue
        # MuP / post-projection logit scaling is applied AFTER output_layer in the model
        # forward; the fused path skips it, so it is unsupported — fall back to stock.
        model_config = getattr(model, "config", None)
        if model_config is not None and getattr(model_config, "use_mup", False):
            return False
        if getattr(output_layer, "_skyrl_fused_lm_head_wrapped", False):
            installed = True
            continue

        orig_forward = output_layer.forward
        tp_group = getattr(output_layer, "tp_group", None) or mpu.get_tensor_model_parallel_group()

        def make_capture(layer, original, tpg):
            def fused_output_layer_forward(input_, weight=None, runtime_gather_output=None):
                w = weight if weight is not None else getattr(layer, "weight", None)
                if w is None:
                    # Tied-embedding stage with no allocated head weight AND none passed in:
                    # we cannot fuse safely — defer to the original projection.
                    return original(input_, weight=weight, runtime_gather_output=runtime_gather_output)
                hidden = input_
                if getattr(layer, "sequence_parallel", False):
                    # Gather the sequence-parallel-scattered hidden across TP. Its backward is a
                    # reduce-scatter (tensor_parallel_output_grad=True), which reduces the fused
                    # function's per-shard grad_hidden exactly as ColumnParallelLinear would.
                    hidden = gather_from_sequence_parallel_region(
                        hidden, tensor_parallel_output_grad=True, group=tpg
                    )
                holder["weight"] = w
                # Return hidden in the logits position (+ no bias). The model's _scale_logits is
                # identity here (MuP excluded above) and its transpose yields [b, s, h].
                return hidden, None

            return fused_output_layer_forward

        output_layer.forward = make_capture(output_layer, orig_forward, tp_group)
        output_layer._skyrl_fused_lm_head_wrapped = True
        output_layer._skyrl_fused_lm_head_orig_forward = orig_forward
        installed = True

    return installed


def _build_packed_targets(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_params,
    sub_seq_lengths: Optional[list[list[int]]] = None,
) -> torch.Tensor:
    """Pack full target token IDs without context-parallel sharding."""
    cu_padded = packed_seq_params.cu_seqlens_q_padded.to(device=sequences.device, dtype=torch.long)
    total_padded_tokens = int(cu_padded[-1].item())

    targets = torch.zeros((total_padded_tokens,), dtype=sequences.dtype, device=sequences.device)
    if sub_seq_lengths is not None:
        cu_padded_cpu = cu_padded.detach().cpu().tolist()
        seg_idx = 0
        for row_idx, row_lens in enumerate(sub_seq_lengths):
            row_offset = 0
            for seq_len in row_lens:
                seq_len = int(seq_len)
                if seg_idx + 1 >= len(cu_padded_cpu):
                    raise ValueError("sub_seq_lengths contains more sub-sequences than packed_seq_params")
                packed_start = cu_padded_cpu[seg_idx]
                targets[packed_start : packed_start + seq_len] = sequences[row_idx, row_offset : row_offset + seq_len]
                row_offset += cu_padded_cpu[seg_idx + 1] - cu_padded_cpu[seg_idx]
                seg_idx += 1
        if seg_idx != len(cu_padded_cpu) - 1:
            raise ValueError(
                f"sub_seq_lengths describes {seg_idx} sub-sequences, "
                f"but packed_seq_params describes {len(cu_padded_cpu) - 1}"
            )
        return targets.unsqueeze(0)

    attention_mask = attention_mask.to(device=sequences.device, dtype=torch.bool)
    token_offsets = attention_mask.to(torch.long).cumsum(dim=1) - 1
    packed_indices = cu_padded[:-1].unsqueeze(1) + token_offsets
    targets[packed_indices[attention_mask]] = sequences[attention_mask]
    return targets.unsqueeze(0)


class MegatronModelWrapper:
    def __init__(
        self,
        config: TrainerConfig,
        actor_module: List[nn.Module],
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_loss_fn: Optional[Callable] = None,
    ):
        self.cfg = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.policy_loss_fn = policy_loss_fn
        self.remove_microbatch_padding = self.cfg.remove_microbatch_padding

        # Fused LM-head log-prob (optional): when enabled, replace output_layer.forward so the
        # model returns its pre-projection hidden state and the head GEMM is fused into the
        # chunked log-prob (the full [*, seq, vocab // TP] logits never materialize). The
        # captured LM-head weight shard for the current microbatch lands in
        # ``self._fused_lm_head["weight"]``; it is None when fusion is off or unsupported
        # (e.g. MuP), in which case every path uses the stock logits computation unchanged.
        self.fused_linear_logprob = bool(getattr(self.cfg, "fused_linear_logprob", False))
        self.fused_linear_logprob_backend = getattr(self.cfg, "fused_linear_logprob_backend", "torch")
        self._fused_lm_head: dict = {"weight": None}
        if self.fused_linear_logprob:
            self.fused_linear_logprob = _install_fused_lm_head_capture(self.actor_module, self._fused_lm_head)

        config = get_model_config(self.actor_module[0])
        # This is set to None by default: https://github.com/NVIDIA/Megatron-LM/blob/07b22a05136a3cb08ece05f7de38cf6aeeb165fb/megatron/core/model_parallel_config.py#L95
        # use the build in finalize_model_grads function to all reduce gradients across parallelism dimensions
        config.finalize_model_grads_func = finalize_model_grads
        # Wire up the optimizer's loss scaler so Megatron's pipeline schedule can scale
        # the loss before backward (critical for fp16 dynamic loss scaling, MoE aux loss
        # scaling, and any explicit loss_scale configuration).
        if actor_optimizer is not None:
            config.grad_scale_func = actor_optimizer.scale_loss

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward-only inference to compute log-probs over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: List of micro-batch dicts with keys: "sequences", "attention_mask", "position_ids",
                           and "num_actions".
            seq_len: Padded sequence length per sample.
            micro_batch_size: Per-micro-batch size.
            temperature: Optional temperature scaling for logits.

        Returns:
            torch.Tensor of concatenated log-probs across micro-batches (valid on pipeline last stage only).
        """
        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, data):
            sequences = data["sequences"]
            packed_seq_params = data.get("packed_seq_params")
            packed_targets = data.get("packed_targets")
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # Fused LM-head path: ``logits`` is actually the pre-projection hidden state and the
            # head GEMM is fused into the chunked log-prob. The vocab range is derived from the
            # captured weight shard (NOT logits.shape[-1], which is hidden_size here). Applying
            # temperature to the hidden state is exact (the head is linear: (h/τ)·W = (h·W)/τ).
            fused_w = self._fused_lm_head["weight"] if self.fused_linear_logprob else None
            fused_backend = self.fused_linear_logprob_backend
            shard_vocab = fused_w.shape[0] if fused_w is not None else logits.shape[-1]

            if temperature != 1.0:
                logits.div_(temperature)

            if packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=tp_rank * shard_vocab,
                    vocab_end_index=(tp_rank + 1) * shard_vocab,
                    group=tp_grp,
                    inference_only=True,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                    lm_head_weight=fused_w,
                    fused_backend=fused_backend,
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    sequences,
                    vocab_start_index=tp_rank * shard_vocab,
                    vocab_end_index=(tp_rank + 1) * shard_vocab,
                    tp_group=tp_grp,
                    inference_only=True,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
                    lm_head_weight=fused_w,
                    fused_backend=fused_backend,
                )
            return torch.tensor(0.0, device=token_logprobs.device), {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=get_model_config(model),
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            sub_seq_lengths_field = batch.get("sub_seq_lengths")
            sub_seq_lengths = [t.tolist() for t in sub_seq_lengths_field] if sub_seq_lengths_field is not None else None
            batch["sub_seq_lengths_list"] = sub_seq_lengths

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                    sub_seq_lengths=sub_seq_lengths,
                )
                batch["packed_seq_params"] = packed_seq_params
                batch["packed_targets"] = _build_packed_targets(
                    sequences, attention_mask, packed_seq_params, sub_seq_lengths=sub_seq_lengths
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            outputs = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
                packed_seq_params=packed_seq_params,
            )

            if not self.remove_microbatch_padding:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

            return outputs, partial(collection_func, data=batch)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=True,
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            log_probs = [o["log_probs"] for o in output]
            log_probs = torch.cat(log_probs, dim=0)
            # take last num_actions tokens per micro; concatenate later
            # Assume all micros have same num_actions
            num_actions = micro_batches[0]["num_actions"]
            log_probs = log_probs[:, -num_actions:]
        else:
            # return dummy tensor for non-last pp stages
            device = micro_batches[0]["sequences"].device
            log_probs = torch.zeros(size=(1, 1), dtype=torch.bfloat16, device=device)
        return log_probs

    def forward_backward_mini_batch(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
        forward_only: bool = False,
    ) -> List[dict]:
        """
        Run forward-backward over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: A list of micro-batch dicts. Each dict must contain keys:
                "sequences", "attention_mask", "position_ids", "num_actions",
                "old_action_log_probs", "base_action_log_probs", "advantages",
                "loss_mask", "rollout_action_logprobs".
            seq_len: Sequence length (tokens) per sample (assumed same across micros after padding).
            micro_batch_size: Micro-batch size per forward pass.
            temperature: Optional temperature for logits scaling.
            loss_fn: Optional loss function name (e.g., "cross_entropy", "ppo").
                     If provided, overrides the config's policy_loss_type.
            loss_fn_config: Optional config overrides for the loss function.
            forward_only: If True, run the forward pass without backward (no gradients).
                          Useful for evaluation / loss-only inference paths (e.g., SFT
                          ``forward(loss_fn=...)`` codepath).

        Returns:
            List[dict]: one metrics dict per micro-batch in order.
        """
        forward_backward_func = get_forward_backward_func()

        # Resolve loss function
        resolved_loss_name = loss_fn if loss_fn is not None else self.cfg.algorithm.policy_loss_type
        if loss_fn is not None:
            current_loss_fn = PolicyLossRegistry.get(loss_fn)
        else:
            current_loss_fn = self.policy_loss_fn

        # Build config for loss function, applying any overrides
        loss_config = self.cfg.algorithm
        if loss_fn_config is not None:

            new_loss_config = OmegaConf.merge(OmegaConf.create(asdict(loss_config)), OmegaConf.create(loss_fn_config))
            # NOTE: users can provide a custom loss config class, so we need to use the same class after applying overrides
            loss_config = type(loss_config).from_dict_config(new_loss_config)

        def loss_func(logits, data):
            sequences = data["sequences"]
            packed_seq_params = data.get("packed_seq_params")
            packed_targets = data.get("packed_targets")
            num_actions = data["num_actions"]
            old_action_log_probs = data["old_action_log_probs"]
            base_action_log_probs = data["base_action_log_probs"]
            advantages = data["advantages"]
            loss_mask = data["loss_mask"]
            rollout_action_logprobs = data["rollout_action_logprobs"]
            action_mask = data.get("action_mask")
            num_microbatches = data.get("num_microbatches")

            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # Fused LM-head path: ``logits`` is the pre-projection hidden state; the head GEMM
            # is fused into the chunked log-prob and the vocab range comes from the captured
            # weight shard. The RL entropy/KL terms below need the full per-position logits,
            # which the fused path deliberately never materializes — so fusion serves ONLY the
            # cross_entropy (SFT) loss here. ``_install_fused_lm_head_capture`` already returns
            # hidden in the logits slot, so a non-CE loss with fusion on would be wrong; guard
            # it. (RL fusion that also fuses entropy is a possible follow-up.)
            fused_w = self._fused_lm_head["weight"] if self.fused_linear_logprob else None
            if fused_w is not None and resolved_loss_name != "cross_entropy":
                raise RuntimeError(
                    "fused_linear_logprob is only supported for the cross_entropy (SFT) loss; "
                    f"got loss={resolved_loss_name!r}. Disable fused_linear_logprob for this loss."
                )
            fused_backend = self.fused_linear_logprob_backend
            shard_vocab = fused_w.shape[0] if fused_w is not None else logits.shape[-1]

            # temperature normalization (exact on hidden: the head is linear)
            if temperature != 1.0:
                logits.div_(temperature)

            if packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=tp_rank * shard_vocab,
                    vocab_end_index=(tp_rank + 1) * shard_vocab,
                    group=tp_grp,
                    inference_only=False,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                    lm_head_weight=fused_w,
                    fused_backend=fused_backend,
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    sequences,
                    vocab_start_index=tp_rank * shard_vocab,
                    vocab_end_index=(tp_rank + 1) * shard_vocab,
                    tp_group=tp_grp,
                    inference_only=False,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
                    lm_head_weight=fused_w,
                    fused_backend=fused_backend,
                )

            action_log_probs = token_logprobs[:, -num_actions:]

            # policy loss should be calculated based on the selected token logprobs
            policy_loss, loss_metrics = current_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=loss_config,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )

            # SFT path: cross_entropy loss (negative log likelihood)
            if resolved_loss_name == "cross_entropy":
                loss = policy_loss

                # Compute elementwise loss for Tinker API (per-token NLL)
                with torch.no_grad():
                    elementwise_loss = -action_log_probs
                    if loss_mask is not None:
                        elementwise_loss = elementwise_loss * loss_mask

                # Build per-sequence loss_fn_outputs.
                # Compute valid_lens vectorized on GPU, then move tensors to CPU
                # exactly once before iterating in Python — avoids ~3N GPU->CPU
                # syncs per micro-batch (item()/cpu()/tolist() inside the loop).
                batch_size = action_log_probs.shape[0]
                seq_len = action_log_probs.shape[1]
                if action_mask is not None:
                    valid_lens_t = action_mask.sum(dim=-1).long()
                elif loss_mask is not None:
                    valid_lens_t = loss_mask.sum(dim=-1).long()
                else:
                    valid_lens_t = torch.full((batch_size,), seq_len, device=action_log_probs.device, dtype=torch.long)

                # Bulk GPU->CPU sync: one transfer for logprobs, elementwise_loss, and valid_lens.
                action_log_probs_cpu = action_log_probs.detach().cpu()
                elementwise_loss_cpu = elementwise_loss.detach().cpu()
                valid_lens = valid_lens_t.cpu().tolist()

                loss_fn_outputs = []
                for i in range(batch_size):
                    valid_len = valid_lens[i]
                    loss_fn_outputs.append(
                        {
                            "logprobs": (action_log_probs_cpu[i, -valid_len:].tolist() if valid_len > 0 else []),
                            "elementwise_loss": (
                                elementwise_loss_cpu[i, -valid_len:].tolist() if valid_len > 0 else []
                            ),
                        }
                    )

                metrics = {
                    "loss": loss.item(),
                    "response_length": num_actions,
                    "loss_fn_outputs": loss_fn_outputs,
                }
                return loss, metrics

            # RL path: add optional KL/entropy terms
            with torch.set_grad_enabled(loss_config.use_entropy_loss):
                if packed_seq_params is not None and packed_targets is not None:
                    entropy, entropy_for_loss = vocab_parallel_entropy_packed_sequences(
                        logits,
                        packed_seq_params.cu_seqlens_q_padded,
                        sequences.shape[1],
                        num_actions,
                        data["attention_mask"],
                        loss_mask,
                        mpu.get_context_parallel_group(),
                        sub_seq_lengths=data.get("sub_seq_lengths_list"),
                    )
                else:
                    action_logits = logits[:, -num_actions - 1 : -1, :]
                    entropy_BS = vocab_parallel_entropy(action_logits)
                    entropy = masked_mean(entropy_BS, loss_mask)
                    entropy_for_loss = entropy

            if loss_config.use_entropy_loss:
                entropy_loss_term = entropy_for_loss * loss_config.entropy_loss_coef
            else:
                entropy_loss_term = torch.tensor(0.0, device=logits.device)

            if loss_config.use_kl_loss:
                kl_loss = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    loss_mask=loss_mask,
                    kl_estimator_type=loss_config.kl_estimator_type,
                )
                kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
            else:
                kl_loss = torch.tensor(0.0, device=logits.device)
            kl_loss_term = kl_loss * loss_config.kl_loss_coef

            # Policy losses are pre-scaled to achieve the correct loss_reduction
            # when summing across the entire minibatch (see `apply_loss_reduction_to_advantages_minibatch`).
            # Megatron divides loss by num_microbatches
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/pipeline_parallel/schedules.py#L248)
            # and the data parallel all-reduce averages gradients across dp_size.
            # Megatron's schedule separately multiplies loss by the CP size for two-output loss funcs,
            # so CP ranks are not included in this correction factor.
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/distributed/distributed_data_parallel.py#L285)
            # so we multiply by both factors to recover the correct sum reduction.
            grad_sum_correction_factor = num_microbatches * dp_size

            # NOTE: The KL and entropy loss terms are not pre-scaled,
            # so we just average them across microbatches and DP workers.
            # KL and entropy use Megatron's existing microbatch and CP schedule scaling.
            loss = policy_loss * grad_sum_correction_factor + (kl_loss_term - entropy_loss_term)
            unscaled_loss = loss / grad_sum_correction_factor

            # Build per-sequence loss_fn_outputs with logprobs.
            batch_size = action_log_probs.shape[0]
            seq_len = action_log_probs.shape[1]

            if action_mask is not None:
                valid_lens = action_mask.sum(dim=1).int().tolist()
            elif loss_mask is not None:
                valid_lens = loss_mask.sum(dim=1).int().tolist()
            else:
                valid_lens = [seq_len] * batch_size

            detached_log_probs = action_log_probs.detach().cpu()
            loss_fn_outputs = []
            for i, valid_len in enumerate(valid_lens):
                loss_fn_outputs.append(
                    {
                        "logprobs": detached_log_probs[i, -valid_len:].tolist() if valid_len > 0 else [],
                    }
                )

            metrics = {
                "final_loss": unscaled_loss.detach().item(),
                "policy_loss": policy_loss.detach().item(),
                "policy_entropy": entropy.detach().item(),
                "policy_kl": kl_loss.detach().item(),
                "loss_fn_outputs": loss_fn_outputs,
            }
            for k, v in loss_metrics.items():
                metrics["loss_metrics/" + k] = v
            return loss, metrics

        def forward_step(batch_iter, model):
            # NOTE(Charlie): despite the name, methods like `remove_left_padding()` are padding-agnostic
            # (can be left, or right) as it uses attention_mask to locate real tokens. Same thing
            # for recover_left_padding and setup_per_microbatch_replay_forward. Especially relevant
            # after this PR https://github.com/NovaSky-AI/SkyRL/pull/1285.
            batch = next(batch_iter)

            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=get_model_config(model),
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            # When present, sub_seq_lengths enumerates every sub-sequence
            # inside every row of the micro-batch (controller-side mini-batch
            # packing). preprocess_packed_seqs uses it to emit cu_seqlens
            # entries covering all sub-seqs, not one per row.
            #
            # It arrives as a ``TensorList`` data field.
            # ``preprocess_packed_seqs`` and the packed-logprob scatter use
            # ``list[list[int]]``, so convert tensors -> python lists here.
            sub_seq_lengths_field = batch.get("sub_seq_lengths")
            sub_seq_lengths = [t.tolist() for t in sub_seq_lengths_field] if sub_seq_lengths_field is not None else None
            batch["sub_seq_lengths_list"] = sub_seq_lengths

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                    sub_seq_lengths=sub_seq_lengths,
                )
                batch["packed_seq_params"] = packed_seq_params
                batch["packed_targets"] = _build_packed_targets(
                    sequences, attention_mask, packed_seq_params, sub_seq_lengths=sub_seq_lengths
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            outputs = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
                packed_seq_params=packed_seq_params,
            )

            if not self.remove_microbatch_padding:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_backward()

            return outputs, partial(loss_func, data=batch)

        # batch should be a list of micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        metrics_list = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=forward_only,
        )

        # broadcast metrics to all pp ranks
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            metrics_list = [None] * len(micro_batches)
        with torch.no_grad():
            torch.distributed.broadcast_object_list(
                metrics_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        return metrics_list
