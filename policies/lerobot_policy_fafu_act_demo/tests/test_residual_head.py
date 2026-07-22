"""Run these tests in an environment with the complete LeRobot 0.6 training dependencies."""

import torch
from lerobot_policy_fafu_act_demo.modeling_fafu_act_demo import ResidualActionHead
from torch import nn


def test_residual_head_starts_identical_to_official_linear_head():
    base = nn.Linear(16, 7)
    head = ResidualActionHead(base, 16, 7, 32, 0.0, 0.1)
    features = torch.randn(2, 5, 16)

    torch.testing.assert_close(head(features), base(features))


def test_residual_branch_receives_gradients():
    head = ResidualActionHead(nn.Linear(16, 7), 16, 7, 32, 0.0, 0.1)
    loss = head(torch.randn(2, 5, 16)).square().mean()

    loss.backward()

    assert head.residual[-1].weight.grad is not None
