# FAFU Arm × LeRobot

`fafu_arm_lerobot` 是 FAFU 六轴机械臂的 LeRobot 第三方插件，提供数据采集、主从遥操作、
数据回放、策略评估以及基于 pytracik 的 FK/IK。项目参考了
[Panthera-HT_lerobot](https://github.com/HighTorque-Robotics/Panthera-HT_lerobot) 的设备划分，
但硬件通信全部使用 [fafu_arm_sdk](https://github.com/FAFU-Robotics/fafu_arm_sdk)，不会修改或
复制 LeRobot 源码。

## 已实现

- `fafu_follower`：LeRobot `Robot`，支持 6 关节 + 夹爪、相机、动作限幅和 30 Hz 流式控制。
- `fafu_leader`：LeRobot `Teleoperator`，释放电机后读取人工拖动的关节和夹爪位置。
- `FafuArmKinematics`：使用 [pytracik](https://pypi.org/project/pytracik/) 完成 FK/IK。
- LeRobot 0.4.3–0.6.x 第三方插件自动发现，无需修改 `lerobot` 包。
- Windows 和 Ubuntu SDK 构建流程、离线软件检查、无动作硬件连通检查。

## URDF 与 TCP

`src/lerobot_robot_fafu_arm/resources/fafu_arm.urdf` 是只包含运动学链的可部署 URDF：

- 六个关节的 origin、axis 和 limit 来自 WRS `PantheraHT` Python 模型，并与提供的
  `fafu_follower.urdf` 交叉核对。
- `tool_link` 从 `link6` 沿 URDF X 轴偏移 **0.175 m**。
- 175 mm 来自当前夹爪代码中的 5 mm coupling + 170 mm acting center；相较原
  `fafu_follower.urdf` 的 165 mm，TCP 向前更新了 10 mm。

详细推导见 [docs/URDF.md](docs/URDF.md)。

## 兼容环境

| Python | LeRobot | FAFU SDK |
|---|---|---|
| 3.10 / 3.11 | 0.4.3 系列 | Windows 可直接使用匹配 ABI 的现有 `.pyd`，否则重编 |
| 3.12 / 3.13 | 0.5.x / 0.6.x | 需要为当前 Python 重编 `fafu_motor` |

推荐第一次部署使用 Python 3.10。无论使用哪个版本，`fafu_motor.cpXY...` 的 `XY` 必须和
当前 Python ABI 一致。

## 安装

```bash
git clone --recurse-submodules https://github.com/FAFU-Robotics/fafu_arm_lerobot.git
cd fafu_arm_lerobot
python -m venv .venv
```

激活虚拟环境后安装插件：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

如果第一次 clone 时没有拉 submodule：

```bash
git submodule update --init --recursive
```

### 编译 FAFU SDK（Windows）

在已经激活的 Python 环境中：

```powershell
python -m pip install pybind11
cd third_party/fafu_arm_sdk/fafu_robot_cpp
./build.bat
cd ../../..
```

脚本会把匹配当前 Python ABI 的 `fafu_motor` 和 `serial_cmake.dll` 复制到
`third_party/fafu_arm_sdk/fafu_robot_python/`。

### 编译 FAFU SDK（Ubuntu）

```bash
cd third_party/fafu_arm_sdk/fafu_robot_cpp
bash linux/install_deps.sh
bash linux/setup_udev.sh
bash linux/build.sh
cd ../../..
```

完整系统配置和故障排查见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

## 部署前检查

只检查软件、URDF、pytracik 和 SDK 导入，不会打开串口：

```bash
fafu-arm-check
```

打开串口并读取一次状态，但不使能电机、不发送运动指令：

```bash
fafu-arm-check --connect --port COM14
# Ubuntu: fafu-arm-check --connect --port /dev/fafu_debug_board
```

也可让插件查找外部 SDK，而不使用 submodule：

```bash
# Windows PowerShell
$env:FAFU_ARM_SDK_PATH = "D:\code\fafu_arm_sdk"

# Linux / macOS
export FAFU_ARM_SDK_PATH=/opt/fafu_arm_sdk
```

## LeRobot 使用

### 主从遥操作

两台机械臂需要使用不同串口：

```bash
lerobot-teleoperate \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/ttyUSB0 \
  --teleop.type=fafu_leader \
  --teleop.id=fafu_leader \
  --teleop.port=/dev/ttyUSB1
```

### 数据采集

```bash
lerobot-record \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/ttyUSB0 \
  --teleop.type=fafu_leader \
  --teleop.id=fafu_leader \
  --teleop.port=/dev/ttyUSB1 \
  --dataset.repo_id=FAFU-Robotics/fafu_demo \
  --dataset.single_task="pick and place" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=30
```

相机使用 LeRobot 标准 camera config 传给 `--robot.cameras`。未配置相机时，插件只记录关节和
夹爪状态。

### 回放

```bash
lerobot-replay \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --dataset.repo_id=FAFU-Robotics/fafu_demo \
  --dataset.episode=0
```

### 策略评估/采集 rollout

```bash
lerobot-record \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --policy.path=outputs/train/fafu_act/checkpoints/last/pretrained_model \
  --dataset.repo_id=FAFU-Robotics/fafu_eval \
  --dataset.single_task="evaluate pick and place" \
  --dataset.num_episodes=5
```

LeRobot 的训练命令与其他机器人相同；本插件只负责硬件输入输出。

## FK / IK

```python
import numpy as np
from lerobot_robot_fafu_arm import FafuArmKinematics

kin = FafuArmKinematics()
q = np.array([0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
pose = kin.forward(q)

solution = kin.inverse(
    position=pose.position,
    rotation=pose.rotation,
    seed=q,
)
if solution is None:
    raise RuntimeError("target is unreachable")
```

自定义 URDF 或 TCP：

```python
kin = FafuArmKinematics("/path/to/custom_fafu.urdf")
```

LeRobot CLI 中可传 `--robot.urdf_path=/path/to/custom_fafu.urdf`。

## 关键配置

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `robot.use_servo` | `true` | 使用 SDK `servo_j` 连续控制 |
| `robot.servo_watchdog_ms` | `250` | 主机失联时固件刹车超时 |
| `robot.servo_use_mit` | `false` | 默认走稳定的位置通道；调好动力学后可启用 MIT |
| `robot.max_relative_target` | `0.15` rad | 每帧最大关节目标变化 |
| `robot.enforce_urdf_limits` | `true` | 在 SDK 软限位之前再做一次 URDF 限幅 |
| `robot.gripper_effort` | `300` raw | 夹爪最大力矩上限 |
| `robot.joint_release` | `stop` | 断开后关节释放策略 |
| `robot.gripper_release` | `brake` | 断开后夹爪短路制动，减少掉落风险 |

如需修改 SDK 电机软限位，请复制包内 `fafu_arm.cfg` 后通过
`--robot.sdk_config_path=/path/to/robot.cfg` 使用，不要直接修改 site-packages。

## 安全

- 第一次连接先运行 SDK 的 `01_smoke` 或 `fafu-arm-check --connect`。
- 上真机前确认急停可用、工作空间内无人、机械臂下方有支撑。
- 先用低速和小 `max_relative_target` 验证关节方向，再采集数据或运行策略。
- `fafu_leader` 会释放关节；未托住重力轴时机械臂可能下落。
- 未经实机标定不要修改 TCP、关节方向、软限位或开启 MIT 控制。

## 开发验证

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
```

## License

[MIT](LICENSE)
