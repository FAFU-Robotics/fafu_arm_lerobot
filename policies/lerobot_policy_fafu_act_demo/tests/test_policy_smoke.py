"""End-to-end smoke tests for the out-of-tree ACT policy example."""

from __future__ import annotations

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot_policy_fafu_act_demo import FafuActDemoConfig, FafuActDemoPolicy


def make_tiny_config() -> FafuActDemoConfig:
    return FafuActDemoConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=4,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        residual_hidden_dim=16,
    )


def make_batch() -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(2, 7),
        OBS_ENV_STATE: torch.randn(2, 3),
        ACTION: torch.randn(2, 4, 7),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }


def test_policy_forward_and_backward() -> None:
    policy = FafuActDemoPolicy(make_tiny_config())

    loss, metrics = policy.forward(make_batch())
    loss.backward()

    assert torch.isfinite(loss)
    assert "l1_loss" in metrics
    assert policy.model.action_head.residual[-1].weight.grad is not None


def test_policy_save_and_load_round_trip(tmp_path) -> None:
    policy = FafuActDemoPolicy(make_tiny_config()).eval()
    observation = {
        OBS_STATE: torch.randn(2, 7),
        OBS_ENV_STATE: torch.randn(2, 3),
    }
    with torch.no_grad():
        expected = policy.predict_action_chunk(observation)
    policy.save_pretrained(tmp_path)

    loaded = FafuActDemoPolicy.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        actual = loaded.predict_action_chunk(observation)

    torch.testing.assert_close(actual, expected)
