"""Fused router replay on real sharded Megatron meshes, through SkyRL's own worker.

Everything the single-GPU test cannot reach runs here: ``setup_per_microbatch_replay_forward``'s
context-parallel ``2*cp_size`` front/back chunk split, its tensor-parallel sequence slice, the
pipeline layer-offset mapping, ``scatter_router_padding_mask_for_model``'s sequence-parallel
branch, the alltoall dispatcher under expert parallelism, and the backward-replay FIFO under a
genuine multi-stage pipeline schedule. The full patch set the worker installs is live --
``patch_topk_router_layer_number``, ``patch_topk_router_expert_bias_padding_mask`` and
``patch_topk_router_fused_replay`` -- and the test asserts all three are present in every
worker process.

Three independent properties per mesh, because none alone is sufficient:

* **100% index overlap** between the experts the router actually dispatched to and the replay
  indices installed on that rank, per layer, per microbatch, in both REPLAY_FORWARD and the
  REPLAY_BACKWARD recompute. A layout bug shows up here and nowhere else -- probabilities stay
  perfectly plausible while the wrong tokens get the wrong routes, and enabling
  ``moe_router_fusion`` alongside replay lands at chance-level overlap (measured 4.408% at
  E=512/topk=22) with no other symptom at all.
* **Layer provenance.** Overlap alone compares the router against whatever routes were
  installed in *its* ``RouterReplay``, so it cannot see routes installed into the wrong
  layer's router. Each captured layer therefore draws its experts from a disjoint block of
  the expert space, and every dispatched expert must fall in its own layer's block. This
  model's MoE layers happen to be contiguous and zero-based; the non-contiguous case (Moonlight,
  whose layer 0 is dense so the capture names layers 1..26) is covered by test_router_replay.py.
* **Kernel-on vs kernel-off equivalence** of routing probabilities, logprobs and the
  optimizer's grad norm, with the layout code shared between the two runs so it cancels out
  and only the kernel is under test.

Needs 8 GPUs. Not wired into ci/gpu_ci_run_h100.sh, which provisions 4.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s \
  tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_fused_router_replay_sharded.py
"""

import pytest
import ray
import torch
from transformers import AutoConfig, AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.replay_utils import replay_padding_row
from skyrl.backends.skyrl_train.utils.routed_experts import (
    ROUTED_EXPERT_LAYER_INDICES_KEY,
    validate_moe_layer_indices,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import init_worker_with_type

# Tiny Qwen3-MoE: 2 layers, both MoE (decoder_sparse_step=1), 8 experts, topk=2.
# Already exercised at pp2_cp2 and tp2_ep2 by test_megatron_models.py, so the layout
# machinery is known to work for it -- which is what lets a failure here be read as a
# replay/kernel problem rather than a model problem.
MOE_MODEL_NAME = "eatang/qwen3-moe-tiny-random"
NUM_EXPERTS = 8
TOPK = 2
NUM_SAMPLES = 8

# The router itself is fp32, so kernel-on and kernel-off routing probabilities must agree to
# fp32 noise wherever the router's *input* is bit-identical between the two runs -- which is
# the case on the first pipeline stage, whose logits come straight from the embedding. This is
# the assertion that actually pins the kernel.
PROB_ATOL = 1e-6

# Downstream is a different matter: every activation is bf16, so a sub-1e-7 change in a
# routing probability can flip a bf16 rounding and then propagate. The right yardstick is
# therefore the bf16 unit-in-last-place at the logprob magnitude, not an absolute constant --
# bf16 keeps 8 mantissa bits, so one ULP at |logprob| ~ 10 is already 0.0625. The bound below
# allows a few composed roundings across layers; for reference, test_megatron_models.py
# accepts 2e-1 for this model against vLLM.
LOGPROB_ULP_TOLERANCE = 4.0
GRAD_NORM_RTOL = 5e-2

pytestmark = [
    pytest.mark.h100,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 8,
        reason="needs 8 GPUs to build tp2/cp2/pp2 and ep8 meshes",
    ),
]

# (tp, pp, cp, ep, etp). expert_model_parallel_size * expert_tensor_parallel_size must
# divide data_parallel_size * tensor_model_parallel_size.
MESHES = [
    pytest.param(2, 2, 2, 2, 1, id="tp2_pp2_cp2_ep2"),
    pytest.param(4, 1, 2, 4, 1, id="tp4_cp2_ep4"),
    pytest.param(1, 1, 1, 8, 1, id="ep8"),
]


def captured_moe_layer_indices() -> tuple[int, ...]:
    """The global transformer layers this checkpoint actually routes with.

    Derived from the HF config the way Qwen3-MoE builds its decoder rather than hardcoded:
    the layer dimension of ``rollout_expert_indices`` covers captured MoE layers only, and
    ``rollout_expert_layer_indices`` is the sole record of which layer each slot holds. A
    checkpoint change that made the MoE layers sparse must move this test with it, not
    silently re-point slot 0 at another layer.
    """
    config = AutoConfig.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)
    mlp_only_layers = set(getattr(config, "mlp_only_layers", None) or ())
    sparse_step = config.decoder_sparse_step
    layer_indices = [
        layer_index
        for layer_index in range(config.num_hidden_layers)
        if layer_index not in mlp_only_layers and config.num_experts > 0 and (layer_index + 1) % sparse_step == 0
    ]
    assert config.num_experts == NUM_EXPERTS, config.num_experts
    assert config.num_experts_per_tok == TOPK, config.num_experts_per_tok
    return validate_moe_layer_indices(layer_indices)


def expert_block(slot: int, num_captured_layers: int) -> range:
    """The disjoint slice of the expert space that captured-layer ``slot`` routes into."""
    block_size = NUM_EXPERTS // num_captured_layers
    assert block_size >= TOPK, "each layer's expert block must be able to hold topk distinct experts"
    return range(slot * block_size, (slot + 1) * block_size)


def allowed_experts_by_layer(captured_layer_indices: tuple[int, ...]) -> dict[int, set[int]]:
    """Experts each *global* layer may legitimately dispatch to.

    The padding row is ``arange(topk)`` -- topk distinct experts, as Megatron's dropless
    ``tokens * topk`` dispatcher requires -- so it is allowed everywhere regardless of block.
    """
    padding_experts = set(replay_padding_row(TOPK, dtype=torch.int32).tolist())
    return {
        layer_index: set(expert_block(slot, len(captured_layer_indices))) | padding_experts
        for slot, layer_index in enumerate(captured_layer_indices)
    }


def get_test_actor_config(tp, pp, cp, ep, etp, *, fused: bool, score_function: str) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.policy.model.path = MOE_MODEL_NAME
    # One sample per microbatch: with dp=1 that is NUM_SAMPLES microbatches through the
    # pipeline, which is what makes the backward FIFO interleaving real.
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    # Required for context parallelism.
    cfg.trainer.remove_microbatch_padding = True
    cfg.generator.inference_engine.enable_return_routed_experts = True
    # validate_inference_engine_cfg requires this pairing; no engine is started here.
    cfg.generator.inference_engine.distributed_executor_backend = "mp"

    megatron_config = cfg.trainer.policy.megatron_config
    megatron_config.moe_enable_routing_replay = True
    megatron_config.moe_fused_routing_replay = fused
    # The tiny model routes with softmax natively. The fused kernel implements Megatron's
    # sigmoid contract, so force sigmoid to put the kernel on the fast path; this test never
    # compares against vLLM, so overriding the score function is sound. The native softmax
    # case is covered by test_fallback_on_sharded_mesh below.
    megatron_config.moe_router_score_function = score_function
    # Left at the model's native value: expert bias has no HF counterpart in the Qwen3-MoE
    # bridge, and the kernel ignores it by construction (it only perturbs top-k selection,
    # which replay replaces). The production expert-bias path is covered by
    # test_router_replay.py on Moonlight.
    megatron_config.moe_router_enable_expert_bias = None
    megatron_config.moe_router_dtype = "fp32"
    # Recompute the MoE layers so backward re-runs the routers and drains the replay FIFO.
    # Megatron rejects recompute_method/recompute_num_layers with selective granularity, and
    # SkyRL's DEFAULT_TRANSFORMER_CONFIG_KWARGS set both for the "full" default.
    megatron_config.transformer_config_kwargs.update(
        {
            "recompute_granularity": "selective",
            "recompute_modules": ["moe"],
            "recompute_method": None,
            "recompute_num_layers": None,
        }
    )
    validate_cfg(cfg)

    cfg.trainer.placement.policy_num_gpus_per_node = 8
    megatron_config.tensor_model_parallel_size = tp
    megatron_config.pipeline_model_parallel_size = pp
    megatron_config.context_parallel_size = cp
    megatron_config.expert_model_parallel_size = ep
    megatron_config.expert_tensor_parallel_size = etp
    return cfg


def packed_layer_blocked_routes(
    attention_mask: torch.Tensor,
    num_captured_layers: int,
) -> PackedTensor:
    """Routes packed to real tokens, with every captured layer in its own expert block.

    Routes vary per (sample, token, layer) so a cross-token or cross-layer mixup changes the
    dispatched experts rather than landing on a coincidentally identical set, and the
    per-layer blocks are disjoint so a slot-to-layer mismatch is visible from the experts
    alone -- which is the one thing an overlap check against the installed indices cannot see.
    """
    route_offsets = torch.arange(TOPK, dtype=torch.int32)
    segments = []
    for real_tokens in attention_mask.sum(dim=1).tolist():
        token_index = torch.arange(real_tokens, dtype=torch.int32).view(real_tokens, 1, 1)
        blocks = torch.tensor(
            [expert_block(slot, num_captured_layers).start for slot in range(num_captured_layers)],
            dtype=torch.int32,
        ).view(1, num_captured_layers, 1)
        block_size = NUM_EXPERTS // num_captured_layers
        segments.append(blocks + (token_index + route_offsets) % block_size)
    return PackedTensor.from_segments(segments)


def build_training_input(tokenizer, captured_layer_indices: tuple[int, ...]) -> TrainingInputBatch:
    """Variable-length samples with per-(sample, token, layer) distinguishable routes.

    Lengths deliberately differ so a mispaired replay changes the token count instead of
    silently reusing a same-shaped tensor.
    """
    prompts, responses, rewards, loss_masks = [], [], [], []
    for i, filler in enumerate([1, 5, 2, 9, 3, 13, 4, 7][:NUM_SAMPLES]):
        prompt_ids = tokenizer.encode("Question: " + ("token " * filler) + f"what is {i}+{i}?")
        response_ids = tokenizer.encode(("because " * filler) + f"the answer is {i + i}.")
        if tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != tokenizer.eos_token_id):
            response_ids.append(tokenizer.eos_token_id)
        prompts.append(prompt_ids)
        responses.append(response_ids)
        rewards.append([1.0] * len(response_ids))
        loss_masks.append([1] * len(response_ids))

    sequences, attention_mask, response_mask, rewards_t, loss_mask_t, _, _, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            pad_token_id=tokenizer.pad_token_id,
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            loss_masks=loss_masks,
        )
    )
    assert attention_mask.sum(dim=-1).unique().numel() > 1, "variable-length premise broken"

    batch_size, _ = sequences.shape
    num_actions = response_mask.shape[1]

    generator = torch.Generator().manual_seed(42)
    training_input = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rewards": rewards_t,
            "loss_mask": loss_mask_t,
            "rollout_logprobs": -torch.rand((batch_size, num_actions), generator=generator) * 2.0,
            "rollout_expert_indices": packed_layer_blocked_routes(attention_mask, len(captured_layer_indices)),
            "router_padding_mask": ~attention_mask.bool(),
            "action_log_probs": -torch.rand((batch_size, num_actions), generator=generator) * 2.0,
            "base_action_log_probs": -torch.rand((batch_size, num_actions), generator=generator) * 2.0,
            "advantages": torch.randn((batch_size, num_actions), generator=generator),
            "action_mask": response_mask.to(dtype=torch.int64),
        }
    )
    training_input.metadata = {
        "response_length": num_actions,
        ROUTED_EXPERT_LAYER_INDICES_KEY: captured_layer_indices,
    }
    return training_input


def _install_router_probe(worker):
    """Run inside each worker process: record dispatched-vs-replayed overlap per routing call.

    Wraps ``TopKRouter.routing``, so it sees the routing map the dispatcher will actually
    consume -- after the patched ``topk_routing_with_score_function``, after token dropping
    and after expert-bias accounting.
    """
    import torch as torch_
    from megatron.core.transformer.moe import moe_utils
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    records = []
    worker._router_probe_records = records
    if getattr(TopKRouter, "_overlap_probe_installed", False):
        return
    original_routing = TopKRouter.routing

    # Signature-agnostic: TopKRouter.routing's optional arguments differ across
    # megatron-core versions.
    def probed_routing(router, logits, *args, **kwargs):
        replay = router.router_replay
        action = replay.router_replay_action if replay is not None else None
        # Peek, never pop: REPLAY_BACKWARD's pop belongs to the routing call itself.
        if action == RouterReplayAction.REPLAY_FORWARD:
            expected = replay.target_topk_idx
        elif action == RouterReplayAction.REPLAY_BACKWARD and replay.replay_backward_list:
            expected = replay.replay_backward_list[0]
        else:
            expected = None

        probs, routing_map = original_routing(router, logits, *args, **kwargs)

        if expected is not None:
            expected = expected.to(routing_map.device).long()
            replayed = torch_.zeros_like(routing_map, dtype=torch_.int8).scatter(1, expected, 1).bool()
            intersection = int((routing_map & replayed).sum())
            records.append(
                {
                    "action": action.value,
                    "layer_number": router.layer_number,
                    "num_tokens": int(routing_map.shape[0]),
                    "expected_tokens": int(expected.shape[0]),
                    "overlap": intersection / max(1, int(replayed.sum())),
                    "routed_count": int(routing_map.sum()),
                    "replayed_count": int(replayed.sum()),
                    # Which experts this layer dispatched to at all, for the provenance
                    # check: routes installed into the wrong layer's router still show 100%
                    # overlap against themselves.
                    "routed_experts": sorted({int(expert) for expert in routing_map.nonzero()[:, 1].unique()}),
                    # Kept so the kernel-on and kernel-off runs can be compared at the
                    # router itself, before bf16 activations carry a rounding downstream.
                    "probs": probs.detach().float().cpu(),
                }
            )
        return probs, routing_map

    TopKRouter.routing = probed_routing
    TopKRouter._overlap_probe_installed = True
    worker._patch_state = {
        "layer_number": bool(getattr(TopKRouter, "_set_layer_number_patched", False)),
        "expert_bias_padding_mask": bool(getattr(TopKRouter, "_expert_bias_padding_mask_patched", False)),
        "fused_replay": bool(getattr(moe_utils, "_fused_replay_patched", False)),
    }


def _collect_probe(worker):
    import megatron.core

    from skyrl.backends.skyrl_train.kernels import replay_router
    from skyrl.backends.skyrl_train.utils import replay_utils

    return {
        "records": getattr(worker, "_router_probe_records", []),
        "patches": getattr(worker, "_patch_state", {}),
        "fallback_reasons": sorted(replay_utils._logged_fallback_reasons),
        "replayed_layer_count": replay_utils._replayed_layer_count,
        "kernel_available": replay_router.is_available(),
        "kernel_reason": replay_router.unavailable_reason(),
        "megatron_version": megatron.core.__version__,
    }


def _run_mesh(cfg, training_input):
    """Build the worker group, probe it, run forward + forward_backward + optim_step.

    One Ray session per run: ``init_worker_with_type`` creates a placement group it never
    removes, so a second 8-GPU group in the same session cannot be scheduled.
    """
    with ray_init():
        return _run_mesh_in_session(cfg, training_input)


def _run_mesh_in_session(cfg, training_input):
    actor_group = init_worker_with_type("policy", num_gpus_per_node=8, cfg=cfg)
    try:
        # __ray_call__ runs a callable inside the actor process with the actor as its
        # first argument -- the only way to instrument code that lives behind Ray.
        ray.get([actor.__ray_call__.remote(_install_router_probe) for actor in actor_group._actor_handlers])

        forward_out = WorkerOutput.cat(
            actor_group.actor_infos,
            ray.get(actor_group.async_run_ray_method("mesh", "forward", data=training_input)),
        )
        logprobs = loss_fn_outputs_to_tensor(forward_out.loss_fn_outputs, key="logprobs")

        backward_results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))
        grad_norm = ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))[0]

        probes = ray.get([actor.__ray_call__.remote(_collect_probe) for actor in actor_group._actor_handlers])
        return {
            "logprobs": logprobs,
            "policy_loss": backward_results[0].metrics["policy_loss"],
            "grad_norm": grad_norm,
            "probes": probes,
        }
    finally:
        for actor in actor_group._actor_handlers:
            ray.kill(actor)


def _assert_total_overlap(probes, allowed_experts, mesh_id, expect_backward_replay=True):
    forward_calls = backward_calls = 0
    layers_seen = set()
    for rank, probe in enumerate(probes):
        patches = probe["patches"]
        assert patches.get("layer_number"), f"{mesh_id} rank{rank}: layer_number patch missing"
        assert patches.get("expert_bias_padding_mask"), f"{mesh_id} rank{rank}: expert-bias patch missing"
        assert patches.get("fused_replay"), f"{mesh_id} rank{rank}: fused-replay patch missing"
        for record in probe["records"]:
            assert record["overlap"] == 1.0, (
                f"{mesh_id} rank{rank} layer{record['layer_number']} {record['action']}: "
                f"dispatched/replayed index overlap {record['overlap']:.4%} != 100% "
                f"({record['routed_count']} routed vs {record['replayed_count']} replayed, "
                f"{record['num_tokens']} tokens)"
            )
            assert record["num_tokens"] == record["expected_tokens"], (
                f"{mesh_id} rank{rank}: router saw {record['num_tokens']} tokens but the "
                f"replay slice holds {record['expected_tokens']} -- layout desync"
            )
            # layer_number is Megatron's 1-based global position; the capture names layers
            # 0-based, and nothing here may assume the two sets coincide.
            layer_index = record["layer_number"] - 1
            assert layer_index in allowed_experts, (
                f"{mesh_id} rank{rank}: replay ran on layer {layer_index}, which the rollout "
                f"never captured (captured {sorted(allowed_experts)})"
            )
            unexpected = sorted(set(record["routed_experts"]) - allowed_experts[layer_index])
            assert not unexpected, (
                f"{mesh_id} rank{rank} layer{layer_index} {record['action']}: dispatched to experts "
                f"{unexpected}, which belong to another captured layer's block -- replay routes "
                f"were installed into the wrong layer's router"
            )
            layers_seen.add(layer_index)
            if record["action"] == "replay_forward":
                forward_calls += 1
            else:
                backward_calls += 1
    assert forward_calls > 0, f"{mesh_id}: no REPLAY_FORWARD routing calls were observed"
    assert layers_seen == set(allowed_experts), (
        f"{mesh_id}: replay only ran on layers {sorted(layers_seen)} of the captured "
        f"{sorted(allowed_experts)}, so the provenance check is partly vacuous"
    )
    if expect_backward_replay:
        assert backward_calls > 0, (
            f"{mesh_id}: no REPLAY_BACKWARD routing calls -- MoE recompute never ran, so the FIFO " "was not exercised"
        )
    return forward_calls, backward_calls


def _first_stage_prob_delta(probes_off, probes_on):
    """max|delta| of routing probabilities where the router's input is run-invariant.

    Only the first pipeline stage qualifies: its router logits come from the embedding, so
    they are bit-identical in both runs and any difference in the output is the router's own.
    Later stages consume bf16 activations that have already absorbed a rounding difference,
    so a difference there is propagation, not a router discrepancy.
    """
    first_layer = min(
        (record["layer_number"] for probe in probes_off for record in probe["records"]),
        default=None,
    )
    delta = 0.0
    compared = 0
    for probe_off, probe_on in zip(probes_off, probes_on, strict=True):
        assert len(probe_off["records"]) == len(probe_on["records"]), "routing call counts differ"
        for record_off, record_on in zip(probe_off["records"], probe_on["records"], strict=True):
            if record_off["layer_number"] != first_layer:
                continue
            assert record_off["action"] == record_on["action"], "routing call order differs"
            delta = max(delta, (record_on["probs"] - record_off["probs"]).abs().max().item())
            compared += 1
    return delta, compared


def _tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.mark.parametrize("tp,pp,cp,ep,etp", MESHES)
def test_fused_replay_on_sharded_mesh(tp, pp, cp, ep, etp):
    mesh_id = f"tp{tp}_pp{pp}_cp{cp}_ep{ep}"
    captured_layer_indices = captured_moe_layer_indices()
    allowed_experts = allowed_experts_by_layer(captured_layer_indices)
    training_input = build_training_input(_tokenizer(), captured_layer_indices)

    results = {}
    for fused in (False, True):
        cfg = get_test_actor_config(tp, pp, cp, ep, etp, fused=fused, score_function="sigmoid")
        results[fused] = _run_mesh(cfg, training_input)

    print(f"\n{mesh_id}: megatron-core {results[True]['probes'][0]['megatron_version']}")
    for fused, result in results.items():
        forward_calls, backward_calls = _assert_total_overlap(
            result["probes"], allowed_experts, f"{mesh_id} fused={fused}"
        )
        print(
            f"{mesh_id} fused={fused}: {forward_calls} replay-forward and {backward_calls} "
            f"replay-backward routing calls, all at 100% index overlap, each layer inside its "
            f"own expert block"
        )

    # The flag must actually change which path served the routing.
    on_reasons = {reason for probe in results[True]["probes"] for reason in probe["fallback_reasons"]}
    off_reasons = {reason for probe in results[False]["probes"] for reason in probe["fallback_reasons"]}
    assert all(probe["kernel_available"] for probe in results[True]["probes"]), [
        probe["kernel_reason"] for probe in results[True]["probes"]
    ]
    assert not on_reasons, f"{mesh_id}: fused kernel fell back: {sorted(on_reasons)}"
    assert off_reasons == {"moe_fused_routing_replay=False"}, f"{mesh_id}: unexpected reasons {off_reasons}"

    # Every rank reports the layer count the path log would have named.
    for fused, result in results.items():
        for rank, probe in enumerate(result["probes"]):
            assert probe["replayed_layer_count"], f"{mesh_id} fused={fused} rank{rank}: no layers replayed"

    # The decisive numerical check: routing probabilities on the pipeline stage whose router
    # input is bit-identical between the two runs must agree to fp32 noise.
    prob_delta, compared = _first_stage_prob_delta(results[False]["probes"], results[True]["probes"])
    print(f"{mesh_id}: max|delta routing probs| = {prob_delta:.3e} over {compared} first-stage routing calls")
    assert compared > 0, f"{mesh_id}: no first-stage routing calls to compare"
    assert prob_delta < PROB_ATOL

    logprobs_off, logprobs_on = results[False]["logprobs"], results[True]["logprobs"]
    logprob_delta = (logprobs_on - logprobs_off).abs().max().item()
    # bf16 keeps 8 mantissa bits; one ULP at the largest |logprob| present.
    scale = max(logprobs_off.abs().max().item(), 1e-6)
    bf16_ulp = 2.0 ** (torch.tensor(scale).log2().floor().item() - 7)
    grad_norm_on, grad_norm_off = results[True]["grad_norm"], results[False]["grad_norm"]
    print(
        f"{mesh_id}: max|delta logprobs| = {logprob_delta:.3e} "
        f"({logprob_delta / bf16_ulp:.2f} bf16 ULP at |logprob|<={scale:.3f}), "
        f"grad_norm fused={grad_norm_on:.6f} unfused={grad_norm_off:.6f}, "
        f"policy_loss fused={results[True]['policy_loss']:.6f} unfused={results[False]['policy_loss']:.6f}"
    )
    assert logprob_delta <= LOGPROB_ULP_TOLERANCE * bf16_ulp
    assert grad_norm_on == pytest.approx(grad_norm_off, rel=GRAD_NORM_RTOL)


def test_fallback_on_sharded_mesh():
    """The tiny model's native softmax routing must fall back, still at 100% overlap.

    Proves the guard/fallback path is correct on a sharded mesh too, not just that the
    kernel is.
    """
    tp, pp, cp, ep, etp = 2, 2, 2, 2, 1
    captured_layer_indices = captured_moe_layer_indices()
    allowed_experts = allowed_experts_by_layer(captured_layer_indices)
    training_input = build_training_input(_tokenizer(), captured_layer_indices)

    cfg = get_test_actor_config(tp, pp, cp, ep, etp, fused=True, score_function="softmax")
    result = _run_mesh(cfg, training_input)

    forward_calls, backward_calls = _assert_total_overlap(result["probes"], allowed_experts, "softmax_tp2_pp2_cp2_ep2")
    reasons = {reason for probe in result["probes"] for reason in probe["fallback_reasons"]}
    print(
        f"\nsoftmax_tp2_pp2_cp2_ep2: {forward_calls} replay-forward and {backward_calls} "
        f"replay-backward routing calls at 100% overlap; fallback reasons {sorted(reasons)}"
    )
    assert any("score_function" in reason for reason in reasons)
