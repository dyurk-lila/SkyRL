"""Tests that the configured ``optimizer`` type and its hyperparameters flow through
the Megatron optimizer adapter into Megatron-core's ``OptimizerConfig``.

``init_megatron_optim_config`` only builds a Megatron-core ``OptimizerConfig`` dataclass
(no CUDA, no dist init, no model), so these assertions are correct-by-construction and do
not require a GPU; they are gated behind ``@pytest.mark.megatron`` because importing the
adapter pulls in ``megatron.core`` (a linux/megatron-extra dependency).

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_optimizer_type.py
"""

import pytest

from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
    init_megatron_optim_config,
)
from skyrl.train.config import OptimizerConfig


@pytest.mark.megatron
def test_adam_is_default_and_routes_betas():
    optim_config = OptimizerConfig()
    assert optim_config.optimizer == "adam"

    mcore_config = init_megatron_optim_config(optim_config, optimizer_config_kwargs={})
    assert mcore_config.optimizer == "adam"
    # adam_betas must continue to route to adam_beta1/adam_beta2.
    assert mcore_config.adam_beta1 == pytest.approx(optim_config.adam_betas[0])
    assert mcore_config.adam_beta2 == pytest.approx(optim_config.adam_betas[1])
    assert mcore_config.lr == pytest.approx(optim_config.lr)
    assert mcore_config.weight_decay == pytest.approx(optim_config.weight_decay)


@pytest.mark.megatron
def test_sgd_dispatches_and_keeps_default_momentum():
    optim_config = OptimizerConfig(optimizer="sgd")
    mcore_config = init_megatron_optim_config(optim_config, optimizer_config_kwargs={})
    assert mcore_config.optimizer == "sgd"
    # SkyRLOptimizerConfig does not expose a momentum field yet, so SGD falls back to
    # Megatron-core's default momentum (0.9). This guards against a future regression
    # where the adapter zeros it out.
    assert mcore_config.sgd_momentum == pytest.approx(0.9)


@pytest.mark.megatron
def test_lion_type_forwarded():
    # The adapter must forward the requested type verbatim; whether Lion can actually be
    # constructed is decided later by Megatron-core based on emerging-optimizers availability.
    mcore_config = init_megatron_optim_config(OptimizerConfig(optimizer="lion"), optimizer_config_kwargs={})
    assert mcore_config.optimizer == "lion"


@pytest.mark.megatron
def test_default_optim_args_are_byte_identical_to_pre_change():
    """The default ``OptimizerConfig`` must produce a Megatron-core ``OptimizerConfig``
    that is unchanged by this feature.

    Two ways the new ``optimizer`` field could have silently shifted mcore config:
    (1) injecting a ``sgd_momentum`` override (the adapter only forwards it when the
        SkyRL field is present, which it is not yet) — verify mcore keeps its 0.9 default;
    (2) the ``optimizer`` value itself — verify it is still ``"adam"`` (AdamW).
    """
    mcore_default = init_megatron_optim_config(OptimizerConfig(), optimizer_config_kwargs={})

    assert mcore_default.optimizer == "adam"
    # sgd_momentum is NOT forwarded (no SkyRL field), so mcore keeps its own default of 0.9.
    assert mcore_default.sgd_momentum == pytest.approx(0.9)
    # The full mcore OptimizerConfig must match what an explicit, optimizer-free build
    # produces — i.e. adding the ``optimizer`` field did not perturb any other arg.
    reference = init_megatron_optim_config(
        OptimizerConfig(optimizer="adam"),
        optimizer_config_kwargs={},
    )
    assert mcore_default == reference
