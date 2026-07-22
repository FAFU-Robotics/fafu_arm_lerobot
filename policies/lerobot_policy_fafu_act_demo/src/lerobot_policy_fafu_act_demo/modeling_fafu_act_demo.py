"""A minimal, trainable ACT network modification example."""

from __future__ import annotations

from typing import Any

from lerobot.policies.act.modeling_act import ACTPolicy
from torch import Tensor, nn

from .configuration_fafu_act_demo import FafuActDemoConfig


class ResidualActionHead(nn.Module):
    """Add a small zero-initialized residual MLP beside ACT's official linear head."""

    def __init__(
        self,
        base_head: nn.Module,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.base_head = base_head
        self.residual_scale = residual_scale
        self.residual = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )
        final = self.residual[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("residual head must end with nn.Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.base_head(features) + self.residual_scale * self.residual(features)


class FafuActDemoPolicy(ACTPolicy):
    """Official ACT with a learnable residual correction on every predicted action."""

    config_class = FafuActDemoConfig
    name = "fafu_act_demo"

    def __init__(self, config: FafuActDemoConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        action_feature = config.action_feature
        if action_feature is None:
            raise ValueError("fafu_act_demo requires an action output feature")
        self.model.action_head = ResidualActionHead(
            base_head=self.model.action_head,
            feature_dim=config.dim_model,
            action_dim=action_feature.shape[0],
            hidden_dim=config.residual_hidden_dim,
            dropout=config.residual_dropout,
            residual_scale=config.residual_scale,
        )

    @property
    def residual_parameter_count(self) -> int:
        """Expose the demo's added parameter count for experiment logs and tests."""

        return sum(parameter.numel() for parameter in self.model.action_head.residual.parameters())
