"""FAFU ACT architecture modification example for LeRobot."""

from .configuration_fafu_act_demo import FafuActDemoConfig
from .modeling_fafu_act_demo import FafuActDemoPolicy, ResidualActionHead
from .processor_fafu_act_demo import make_fafu_act_demo_pre_post_processors

__all__ = [
    "FafuActDemoConfig",
    "FafuActDemoPolicy",
    "ResidualActionHead",
    "make_fafu_act_demo_pre_post_processors",
]
