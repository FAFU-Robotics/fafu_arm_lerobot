"""Configuration for the FAFU ACT residual-head example."""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("fafu_act_demo")
@dataclass
class FafuActDemoConfig(ACTConfig):
    """Official ACT configuration plus parameters for a residual action MLP."""

    residual_hidden_dim: int = 256
    residual_dropout: float = 0.1
    residual_scale: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.residual_hidden_dim <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        if not 0.0 <= self.residual_dropout < 1.0:
            raise ValueError("residual_dropout must be in [0, 1)")
        if self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
