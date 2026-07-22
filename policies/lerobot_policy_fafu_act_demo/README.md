# FAFU ACT network demo

This example policy targets Python 3.12 and LeRobot `>=0.6,<0.7`. It keeps the official ACT policy and adds a
zero-initialized residual action head.

From the `fafu_arm_lerobot` repository root, install the plugin and validate the training command without starting a
GPU job:

```bash
python -m pip install -e ./policies/lerobot_policy_fafu_act_demo
fafu-arm-train act --config configs/train/fafu_act_demo.yaml
```

See the [FAFU Arm Policy Training guide](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/TRAINING.md)
for training, checkpoint compatibility, evaluation, and network modification instructions.
