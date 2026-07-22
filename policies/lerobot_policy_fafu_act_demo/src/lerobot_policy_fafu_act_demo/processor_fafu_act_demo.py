"""Reuse ACT's official normalization processors for the residual-head demo."""

from __future__ import annotations

from typing import Any

import torch
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_fafu_act_demo import FafuActDemoConfig


def make_fafu_act_demo_pre_post_processors(
    config: FafuActDemoConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Keep training and inference normalization identical to official ACT."""

    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
