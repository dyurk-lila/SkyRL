"""Resolve which transformer layers of a served model own an MoE router.

vLLM's routed-expert capturer sizes its buffer by ``num_hidden_layers`` and writes only the
layers that own a router, so hybrid architectures (Nemotron-3's interleaved mamba / attention
/ MoE stack) leave the majority of that layer dimension untouched, and DeepSeek V3-style
leading dense layers leave the head of it untouched. Resolving the MoE positions from the live
model instance -- the same ``static_forward_context`` vLLM binds its capture hooks to -- lets
the server drop those slots before packing, and gives the trainer an authoritative layer
mapping to match its own routers against.
"""

import asyncio
from typing import Any


def collect_moe_layer_indices(worker: Any) -> list[int]:
    """Return the global layer indices of this worker's routed-MoE layers, ascending.

    Runs inside a vLLM worker process via ``collective_rpc``. ``MoERunner`` is what
    ``register_layer_for_moe_forward_op`` puts in ``static_forward_context``, and its
    ``layer_id`` is the transformer-layer index parsed from the registered layer name --
    exactly the ``layer_id`` the capturer indexes its buffer with. ``FusedMoE`` is not the
    type to test: as of vLLM 0.26 it is a factory *function* returning ``MoERunner``, and
    ``isinstance`` against a function raises ``TypeError`` -- from inside a worker, on the
    first R3 request, long after the import that would have named the problem succeeded.

    The router must be a ``BaseRouter``: that is the layer of the hierarchy which owns the
    concrete ``set_capture_fn``, so a runner whose router is some other ``FusedMoERouter``
    cannot be captured and its slots are never written.

    ``isinstance(module, MoERunner) and isinstance(module.router, BaseRouter)`` is not a
    guess at the capture set -- it is character for character the predicate
    ``GPUModelRunner._bind_routed_experts_capturer`` selects on, reading ``module.layer_id``
    for the same purpose. The set is therefore the capture set by construction, and
    ``MoELayerIndexResolver.crosscheck_against_capture`` re-derives it from one real capture
    to check that reasoning rather than trust it.
    """
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    static_forward_context = worker.vllm_config.compilation_config.static_forward_context
    return sorted(
        {
            module.layer_id
            for module in static_forward_context.values()
            if isinstance(module, MoERunner) and isinstance(module.router, BaseRouter)
        }
    )


class MoELayerIndexResolver:
    """Fetch the served model's MoE layer indices once and reuse them for every request.

    The layer structure is fixed at model load, so weight sync cannot change it.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._layer_indices: tuple[int, ...] | None = None
        self._lock = asyncio.Lock()
        self._crosschecked = False

    async def get(self) -> tuple[int, ...]:
        if self._layer_indices is not None:
            return self._layer_indices
        async with self._lock:
            if self._layer_indices is None:
                self._layer_indices = await self._fetch()
        return self._layer_indices

    def crosscheck_against_capture(self, capture: Any) -> None:
        """Confirm the resolved layers are the ones vLLM actually wrote, once per server.

        ``collect_moe_layer_indices`` reads the same registry vLLM binds its capture hooks to,
        so the two cannot disagree by construction. This re-derives the answer from the data as
        an independent check on that reasoning: the capturer zeroes its buffer every step and
        writes only routed layers, so an unwritten layer is entirely zero.

        A layer we selected may still be all-zero for legitimate reasons -- a router that masks
        slots to expert 0, or a single short capture -- so only the reverse direction is an
        error: a layer we dropped that carries nonzero routes was real data. That asymmetry is
        also why the all-zero pattern cannot be used to derive the layer set in the first place.

        Costs one scan of one capture (measured 71 ms on a 120B/32k buffer) and nothing
        thereafter, so it stays off the per-request path.
        """
        if self._crosschecked or self._layer_indices is None:
            return
        self._crosschecked = True
        written = {int(layer) for layer in capture.any(axis=(0, 2)).nonzero()[0]}
        dropped_but_written = sorted(written - set(self._layer_indices))
        if dropped_but_written:
            raise RuntimeError(
                f"vLLM captured routes for layers {dropped_but_written}, which were not reported as "
                f"routed-MoE layers (reported: {list(self._layer_indices)}). Dropping them would "
                "discard real routing data."
            )

    async def _fetch(self) -> tuple[int, ...]:
        per_worker = await self._engine.collective_rpc(collect_moe_layer_indices)
        if not per_worker:
            raise RuntimeError("no vLLM worker reported MoE layer indices for routed-expert capture")
        distinct = {tuple(layer_indices) for layer_indices in per_worker}
        if len(distinct) != 1:
            raise RuntimeError(f"vLLM workers disagree on which layers are MoE: {sorted(distinct)}")
        layer_indices = distinct.pop()
        if not layer_indices:
            raise RuntimeError(
                "routed-expert capture is enabled but the served model has no routed-MoE layers; "
                "R3 requires an MoE model"
            )
        return layer_indices
