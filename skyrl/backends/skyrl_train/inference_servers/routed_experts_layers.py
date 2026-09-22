"""Resolve the routed-MoE layers of a served model."""

import asyncio
from typing import Any


def collect_moe_layer_indices(worker: Any) -> list[int]:
    """Return the global layer indices of this worker's routed-MoE layers, ascending.

    The predicate mirrors vLLM's routed-expert capture binding: ``MoERunner`` identifies
    MoE layers and ``BaseRouter`` identifies routers that support capture hooks.
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
        """Check once that no layer omitted by the resolver contains captured routes.

        Selected layers may legitimately contain only expert zero, so the check is
        intentionally one-way.
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
