# FAFU Arm × LeRobot

`fafu_arm_lerobot` 是 FAFU 六轴机械臂的 LeRobot 第三方插件。硬件通信由
[fafu_arm_sdk](https://github.com/FAFU-Robotics/fafu_arm_sdk) 提供，本项目负责 LeRobot 设备接口、
运动学、动作表示、数据工具和训练入口。

## 功能

- `fafu_follower`：6 关节 + 夹爪、相机、动作限幅和流式控制。
- `fafu_leader`：读取人工拖动产生的关节与夹爪目标，用于主从遥操作。
- action：`joint`、`ee_pose`、`ee_delta` 和归档用 `all`。
- observation：可同时保存关节位置/速度、绝对 EE 位姿、相邻帧 EE 增量和相机图像。
- `FafuArmKinematics`：使用 pytracik 完成 FK/IK。
- `fafu-arm-dataset`、`fafu-arm-wrs-view`：检查、读取、导出和查看本地数据。
- `fafu-arm-train`：ACT 数据预检、YAML 配置和官方 LeRobot 训练入口。

## 推荐运行档案

| 用途 | Python | LeRobot | FAFU SDK |
|---|---|---|---|
| 完整采集、ACT 训练与 rollout（推荐） | 3.12 | 0.6.x；复现基线固定为 0.6.0 | 用同一 Python 重新编译 |
| 复用已有 Windows `cp310` SDK | 3.10 | 0.4.3 | 可使用匹配 ABI 的已有二进制 |

不要混用不同 Python ABI 生成的 `fafu_motor`。完整版本说明、平台构建和验收步骤见
[部署指南](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/DEPLOYMENT.md)。

## 最短安装路径

获取包含 SDK submodule 的完整仓库：

```bash
git clone --recurse-submodules https://github.com/FAFU-Robotics/fafu_arm_lerobot.git
cd fafu_arm_lerobot
```

创建并激活 Python 3.12 环境：

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Ubuntu
python3.12 -m venv .venv
source .venv/bin/activate
```

安装固定基线和本项目：

```bash
python -m pip install --upgrade pip
python -m pip install "lerobot[core-scripts]==0.6.0"
python -m pip install -e .
```

然后按[部署指南](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/DEPLOYMENT.md)为当前 Python 构建 FAFU SDK，再执行：

```bash
fafu-arm-check
```

通过标准：输出包含 `[OK] URDF / pytracik`、`[OK] FAFU SDK` 和
`[OK] software checks passed`；此命令不会打开串口。

## 第一次连接

先确认 follower 的实际端口。Ubuntu 主从设备统一使用稳定且彼此不同的
`/dev/serial/by-id/...` 路径，不使用 `/dev/ttyUSB0` 的枚举顺序：

```bash
ls -l /dev/serial/by-id/
fafu-arm-check --connect --port /dev/serial/by-id/FOLLOWER_DEVICE
```

Windows 使用设备管理器确认端口后运行：

```powershell
fafu-arm-check --connect --port COM14
```

`--connect` 只读一次状态，不使能电机、不发送运动命令。看到
`[OK] hardware check passed; no motion command was sent` 后，才进入小步遥操作、相机和采集验收。

## 数据与训练

完整的数据采集命令只在 [Data Collection 指南](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/DATA_COLLECTION.md)维护，包括：

- follower/leader 串口与相机配置；
- `joint`、`ee_delta`、`ee_pose`、`all` 的字段和适用范围；
- 本地录制、续录、读取、数据检查、LeRobot/WRS 查看和安全回放；
- 故障恢复与数据集发布。

首次采集必须显式设置 `--dataset.push_to_hub=false`。确认相机画面和元数据可发布后，使用
`--dataset.push_to_hub=true --dataset.private=true`；只有明确需要公开时才设置
`--dataset.private=false`。

ACT 训练从版本化 YAML 启动。第一条命令只检查数据并打印最终 LeRobot 命令，第二条才开始训练：

```bash
fafu-arm-train act --config configs/train/act_baseline.yaml
fafu-arm-train act --config configs/train/act_baseline.yaml --run
```

action 表示选择、调参、断点续训、策略评估及 ACT 网络修改 demo 见
[Policy Training 指南](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/TRAINING.md)。

## FK / IK

```python
import numpy as np
from lerobot_robot_fafu_arm import FafuArmKinematics

kin = FafuArmKinematics()
q = np.array([0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
pose = kin.forward(q)
solution = kin.inverse(position=pose.position, rotation=pose.rotation, seed=q)

if solution is None:
    raise RuntimeError("target is unreachable")
```

自定义模型可传入 URDF 路径：

```python
kin = FafuArmKinematics("/path/to/custom_fafu.urdf")
```

TCP、关节轴和模型替换方法见 [URDF 说明](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/URDF.md)。

## 文档导航

| 目标 | 文档 |
|---|---|
| 选择版本、构建 SDK、端口分配和分阶段验收 | [部署指南](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/DEPLOYMENT.md) |
| 采集、保存、读取、查看、发布、回放和恢复 | [Data Collection 指南](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/DATA_COLLECTION.md) |
| ACT 训练、调参、评估和网络修改 | [Policy Training 指南](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/TRAINING.md) |
| URDF、TCP 和坐标轴 | [URDF 说明](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/docs/URDF.md) |

## 安全要求

- 上电前确认急停可用、工作空间内无人，并托住可能因释放而下落的重力轴。
- 首次运动使用 `robot.max_relative_target=0.03`，逐关节核对方向、软限位和释放行为。
- 未经实机标定，不修改 TCP、关节方向、SDK 软限位或启用 MIT 控制。
- 保持严格 action 字段校验、限幅和 servo watchdog；校验失败时修复输入，不绕过保护。
- 真机回放和策略评估前执行 Data Collection 指南中的预检，并从低速短时运行开始。

## 开发验证

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
```

## License

[MIT](https://github.com/FAFU-Robotics/fafu_arm_lerobot/blob/main/LICENSE)
