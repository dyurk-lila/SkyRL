# Utils ported from Verl
# https://github.com/volcengine/verl/blob/e1603dc97f3c20c58feed1f5be34acd5c72a830c/verl/utils/megatron/optimizer.py#L4
# The original copyright is reproduced below:

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Tuple, Union

import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import (
    get_megatron_optimizer as get_megatron_optimizer_native,
)
# Megatron internals for per-parameter optimizer routing (e.g. forcing the 2D
# MoE router/gate to Adam while the rest of the 2D matrices use Muon). The
# public get_megatron_optimizer runs check_config_overrides_consistency, which
# REJECTS any override whose ``optimizer`` field differs from config.optimizer.
# Megatron's own non-linear/embedding -> Adam routing dodges that check because
# its default_param_overrides are merged in *after* it, inside
# _get_megatron_emerging_optimizer. We replicate that exact seam: seed the
# standard overrides, add our own, and call the emerging factory directly.
# Pinned megatron-core rev 71e418ea.
from megatron.core.optimizer import (
    _get_megatron_emerging_optimizer,
    get_standard_config_overrides,
)
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from omegaconf import DictConfig

from skyrl.backends.skyrl_train.distributed.megatron.optimizer_dtype import (
    coerce_optimizer_dtype_kwargs,
)
from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig

__all__ = [
    "coerce_optimizer_dtype_kwargs",
    "init_megatron_optim_config",
    "get_megatron_optimizer",
    "get_megatron_optimizer_param_scheduler",
    "get_megatron_last_lr",
]

# Custom (non-Megatron) key understood by SkyRL inside ``optimizer_config_kwargs``:
# a list of fnmatch globs of parameter NAMES that must be optimized by Adam even
# though they are 2D (so Megatron's emerging-optimizer path would otherwise hand
# them to Muon). The motivating case is the MoE router/gate weight, which is 2D
# and therefore NOT caught by the built-in non-linear/embedding -> Adam predicate;
# orthogonalizing routing logits destroys the relative row magnitudes the
# expert-softmax relies on. Stripped out before building OptimizerConfig (which
# would reject the unknown field) and consumed by get_megatron_optimizer.
FORCE_ADAM_PARAM_GLOBS_KEY = "force_adam_param_globs"


def _pop_force_adam_globs(optimizer_config_kwargs) -> Tuple[dict, List[str]]:
    """Split ``optimizer_config_kwargs`` into (megatron-safe kwargs, glob list).

    Returns a NEW dict without ``FORCE_ADAM_PARAM_GLOBS_KEY`` plus the parsed
    list of name globs (empty if the key is absent/empty). Does not mutate input.
    """
    globs: List[str] = []
    clean: dict = {}
    for key, value in dict(optimizer_config_kwargs or {}).items():
        if key == FORCE_ADAM_PARAM_GLOBS_KEY:
            if value:
                globs = [str(g) for g in value]
        else:
            clean[key] = value
    return clean, globs


def init_megatron_optim_config(
    optim_config: Union[SkyRLOptimizerConfig, DictConfig], optimizer_config_kwargs: dict
) -> OptimizerConfig:
    adam_betas = getattr(optim_config, "adam_betas", (0.9, 0.999))
    optim_args = {
        "optimizer": getattr(optim_config, "optimizer", "adam"),
        "lr": getattr(optim_config, "lr", 1e-6),
        "min_lr": getattr(optim_config, "min_lr", 0.0),
        "clip_grad": getattr(optim_config, "max_grad_norm", 1.0),
        "weight_decay": getattr(optim_config, "weight_decay", 1e-2),
        "adam_beta1": float(adam_betas[0]),
        "adam_beta2": float(adam_betas[1]),
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_distributed_optimizer": True,
    }
    # Strip SkyRL-only keys (e.g. force_adam_param_globs) before they reach
    # Megatron's OptimizerConfig, which would reject the unknown field. The globs
    # are re-parsed at the get_megatron_optimizer call site (worker passes the
    # same raw kwargs), so nothing is lost here.
    clean_kwargs, _ = _pop_force_adam_globs(optimizer_config_kwargs)
    # Coerce any ``*_dtype`` string (e.g. "bf16" from YAML) into a real torch.dtype
    # before it reaches Megatron's OptimizerConfig / FusedAdam, which require dtypes.
    optim_args.update(coerce_optimizer_dtype_kwargs(clean_kwargs))

    config = OptimizerConfig(**optim_args)
    return config


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
    optimizer_config_kwargs=None,
):
    """Build the Megatron optimizer, optionally routing named 2D params to Adam.

    When ``optimizer_config_kwargs`` carries ``force_adam_param_globs`` AND the
    optimizer is an emerging optimizer (e.g. Muon), the named params are forced
    onto the Adam sub-optimizer. We must NOT route this through the public
    ``get_megatron_optimizer``: it runs ``check_config_overrides_consistency``,
    which raises on any override whose ``optimizer`` field differs from
    ``config.optimizer``. Megatron's own non-linear/embedding -> Adam routing
    avoids that check because its ``default_param_overrides`` are merged in
    *after* it, inside ``_get_megatron_emerging_optimizer``. We mirror that seam:
    seed ``get_standard_config_overrides`` (the wd-skip / decoupled-LR rules that
    a non-None overrides dict would otherwise REPLACE), add our router globs, and
    call the emerging factory directly. The embedding/output -> Adam defaults are
    still applied downstream (line ~801 of megatron's optimizer __init__), so
    those remain intact regardless.
    """
    _, force_adam_globs = _pop_force_adam_globs(optimizer_config_kwargs)

    if force_adam_globs and config.optimizer not in ("adam", "sgd"):
        config_overrides = get_standard_config_overrides(config)
        for glob in force_adam_globs:
            config_overrides[ParamKey(name=glob)] = {"optimizer": "adam"}
        return _get_megatron_emerging_optimizer(
            config=config,
            model_chunks=model,
            config_overrides=config_overrides,
        )

    # Base optimizer (standard path; also covers Muon with no router override).
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config: Union[SkyRLOptimizerConfig, DictConfig],
    num_training_steps: int = 1e9,  # default to a large number for constant lr/wd
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    # TODO: support other schedulers for Megatron
    if getattr(config, "scheduler", "constant_with_warmup") != "constant_with_warmup":
        raise ValueError("Only constant_with_warmup scheduler is supported for Megatron")

    lr_warmup_steps = config.num_warmup_steps
    if getattr(config, "lr_decay_steps", None) is None:
        lr_decay_steps = num_training_steps
    if getattr(config, "lr_warmup_steps_ratio", None) is not None and (
        getattr(config, "lr_warmup_steps", None) is None or getattr(config, "lr_warmup_steps", None) <= 0
    ):
        lr_warmup_steps = int(config.lr_warmup_steps_ratio * lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=getattr(config, "lr_warmup_init", 0.0),
        max_lr=getattr(config, "lr", 1e-6),
        min_lr=getattr(config, "min_lr", 0.0),
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style="constant",
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=num_training_steps,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
        wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]
