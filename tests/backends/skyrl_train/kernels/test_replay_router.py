"""CPU tests for the fused router-replay build probe and argument marshalling."""

import pathlib
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.kernels import replay_router


@pytest.fixture
def fresh_probe(monkeypatch):
    """Reset the memoized build state so each test drives the probe from scratch."""
    monkeypatch.setattr(replay_router, "_extension", None)
    monkeypatch.setattr(replay_router, "_unavailable_reason", None)
    return replay_router


def test_cuda_source_ships_beside_the_module():
    """The JIT source must be package data, or the probe fails only inside the image."""
    assert replay_router._SOURCE.exists(), replay_router._SOURCE
    assert replay_router._SOURCE.parent.name == "csrc"
    assert replay_router._SOURCE.read_text().lstrip().startswith("//")


def test_build_directory_is_node_local_and_torch_keyed():
    """Concurrent ranks share the directory, so a torch bump must not reuse its objects."""
    build_dir = replay_router._default_build_dir()

    assert build_dir.is_absolute()
    assert build_dir.parts[1] == "tmp", build_dir
    assert torch.__version__ in build_dir.name


def test_probe_reports_missing_cuda_instead_of_raising(fresh_probe, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert fresh_probe.warm_compile() == "CUDA is not available"
    assert fresh_probe.is_available() is False
    assert fresh_probe.unavailable_reason() == "CUDA is not available"


def test_probe_reports_missing_source_instead_of_raising(fresh_probe, monkeypatch):
    """A wheel built without the package-data declaration lands here."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(fresh_probe, "_SOURCE", pathlib.Path("/nonexistent/replay_router.cu"))

    reason = fresh_probe.warm_compile()

    assert reason is not None and "CUDA source missing" in reason


def test_probe_captures_build_failure_as_a_one_line_reason(fresh_probe, monkeypatch):
    """A broken toolchain must degrade, not raise inside the pipeline schedule."""
    import torch.utils.cpp_extension as cpp_extension

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    captured = {}

    def exploding_load(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("nvcc: command not found\nwith a second line")

    monkeypatch.setattr(cpp_extension, "load", exploding_load)

    reason = fresh_probe.warm_compile()

    assert reason.startswith("extension build failed (RuntimeError)")
    assert "\n" not in reason
    assert fresh_probe.is_available() is False
    assert "-std=c++20" in captured["extra_cflags"]
    assert "-std=c++20" in captured["extra_cuda_cflags"]
    assert "-gencode=arch=compute_90,code=sm_90" in captured["extra_cuda_cflags"]
    assert captured["build_directory"] == str(fresh_probe._default_build_dir())


def test_dense_entry_point_refuses_when_unavailable(fresh_probe, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="fused router-replay kernel unavailable"):
        fresh_probe.fused_replay_routing_dense(torch.zeros(2, 4), torch.zeros(2, 2, dtype=torch.int32))


@pytest.fixture
def recording_extension(monkeypatch):
    """Stand in for the compiled extension to check what the wrapper hands it."""
    calls = []

    def replay_fwd_dense(logits, indices, scaling):
        calls.append({"logits": logits, "indices": indices, "scaling": scaling})
        num_tokens, num_experts = logits.shape
        topk = indices.shape[1]
        return (
            torch.zeros(num_tokens, num_experts, dtype=logits.dtype),
            torch.zeros(num_tokens, num_experts, dtype=torch.bool),
            torch.zeros(num_tokens, topk),
            torch.ones(num_tokens),
        )

    monkeypatch.setattr(replay_router, "_extension", SimpleNamespace(replay_fwd_dense=replay_fwd_dense))
    return calls


@pytest.mark.parametrize(
    ("scaling", "expected"),
    [
        (None, 1.0),
        (0.0, 1.0),
        (2.5, 2.5),
    ],
)
def test_scaling_factor_matches_megatrons_falsy_gate(recording_extension, scaling, expected):
    replay_router.fused_replay_routing_dense(
        torch.zeros(3, 8),
        torch.zeros(3, 2, dtype=torch.int32),
        scaling,
    )

    assert recording_extension[0]["scaling"] == expected


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int64])
def test_compact_replay_dtypes_reach_the_kernel_as_int32(recording_extension, dtype):
    """Routes travel in the rollout's compact dtype; the kernel only accepts int32."""
    indices = torch.zeros(3, 2, dtype=dtype)

    replay_router.fused_replay_routing_dense(torch.zeros(3, 8), indices)

    passed = recording_extension[0]["indices"]
    assert passed.dtype == torch.int32
    assert passed.is_contiguous()


def test_index_validation_is_opt_in(recording_extension, monkeypatch):
    """Off by default: the check is a device-to-host sync on every routing call."""
    indices = torch.tensor([[0, 99]], dtype=torch.int32)
    replay_router.fused_replay_routing_dense(torch.zeros(1, 8), indices)
    assert len(recording_extension) == 1

    monkeypatch.setenv(replay_router.VALIDATE_INDICES_ENV_VAR, "1")
    with pytest.raises(ValueError, match="out of range for num_experts=8"):
        replay_router.fused_replay_routing_dense(torch.zeros(1, 8), indices)
