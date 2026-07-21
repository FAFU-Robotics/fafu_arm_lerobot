# FAFU Arm × LeRobot

`fafu_arm_lerobot` 是 FAFU 六轴机械臂的 LeRobot 第三方插件，提供数据采集、主从遥操作、
数据回放、策略评估以及基于 pytracik 的 FK/IK。硬件通信使用
[fafu_arm_sdk](https://github.com/FAFU-Robotics/fafu_arm_sdk)，运动学模型、控制安全限制和
LeRobot 设备接口均由本项目直接维护。

## 已实现

- `fafu_follower`：LeRobot `Robot`，支持 6 关节 + 夹爪、相机、动作限幅和 30 Hz 流式控制。
- `fafu_leader`：LeRobot `Teleoperator`，释放电机后读取人工拖动的关节和夹爪位置。
- 动作支持关节角、绝对 EE 位姿、EE 增量和同步归档四种模式。
- 默认观测同时保留关节位置/速度、绝对 EE 位姿和相邻帧 EE 增量。
- `FafuArmKinematics`：使用 [pytracik](https://pypi.org/project/pytracik/) 完成 FK/IK。
- LeRobot 0.4.3–0.6.x 第三方插件自动发现，无需修改 `lerobot` 包。
- Windows 和 Ubuntu SDK 构建流程、离线软件检查、无动作硬件连通检查。

## URDF 与 TCP

`src/lerobot_robot_fafu_arm/resources/fafu_arm.urdf` 是只包含运动学链的可部署 URDF：

- 六个关节的 origin、axis 和 limit 按 FAFU Arm 的机械尺寸与 `fafu_follower.urdf`
  维护。
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

串口发现、双相机配置、数据字段检查、安全回放和故障恢复的完整流程见
[采集、检查与故障恢复示例](docs/OPERATIONS.md)。

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
  --robot.action_mode=joint \
  --robot.observation_mode=all \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=fafu_leader \
  --teleop.id=fafu_leader \
  --teleop.port=/dev/ttyUSB1 \
  --teleop.action_mode=joint \
  --dataset.repo_id=FAFU-Robotics/fafu_demo \
  --dataset.root=./datasets/fafu_demo \
  --dataset.single_task="pick and place" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=30 \
  --dataset.push_to_hub=false
```

相机使用 LeRobot 标准 camera config 传给 `--robot.cameras`。默认的 `observation_mode=all`
会同时记录关节位置/速度、绝对 EE 位姿、相邻帧 EE 增量和夹爪状态；action 默认是关节目标。

> 隐私默认：LeRobot 当前 `push_to_hub` 默认值为 `true`，因此本项目所有首次采集示例都
> 明确使用 `--dataset.push_to_hub=false`。确认相机画面和元数据不含实验室敏感信息后，发布时必须
> 显式设置 `--dataset.push_to_hub=true --dataset.private=true`；只有确实要公开时才改为
> `--dataset.private=false`。

`action_mode` 可在 `joint`、`ee_pose`、`ee_delta`、`all` 中选择，robot 与 teleop 必须一致。
通常应在一个训练数据集中选择一种 action 表示；`all` 适合归档后再筛选，不建议直接让策略同时学习
三套冗余动作。字段、单位、推荐配置及 pytracik 适用范围见
[数据表示与采集模式](docs/DATA_FORMAT.md)。

默认严格要求每帧 action 字段完整且没有未知字段，并在安全裁剪成功后将最终下发值写回动作字典。
因此 LeRobot 默认录制流水线保存的是限幅后的命令，不是 leader 的越界请求值。

### 回放

先在不连接机械臂的情况下核对字段、action mode 和 episode：

```bash
fafu-arm-dataset-check --root ./datasets/fafu_demo --action-mode joint --episode 0
```

```bash
lerobot-replay \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/ttyUSB0 \
  --robot.action_mode=joint \
  --robot.max_relative_target=0.03 \
  --dataset.repo_id=FAFU-Robotics/fafu_demo \
  --dataset.root=./datasets/fafu_demo \
  --dataset.episode=0
```

回放使用的 `robot.action_mode` 必须与数据集 action 字段一致。完整回放检查清单见
[采集、检查与故障恢复示例](docs/OPERATIONS.md#5-回放前校验和低速回放)。

### 策略评估/采集 rollout

```bash
lerobot-rollout \
  --strategy.type=base \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/ttyUSB0 \
  --robot.action_mode=joint \
  --policy.path=outputs/train/fafu_act/checkpoints/last/pretrained_model \
  --task="evaluate pick and place" \
  --duration=60
```

上面是 LeRobot 0.6 的只运行策略示例。需要保存 rollout 时应再次明确设置本地数据目录、
`--dataset.push_to_hub=false`，或在确实发布时使用 `--dataset.private=true`。

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
| `robot.action_mode` | `joint` | 数据集 action 与控制表示；也可选 `ee_pose`、`ee_delta`、`all` |
| `teleop.action_mode` | `joint` | leader 输出表示，必须与 robot 一致 |
| `robot.observation_mode` | `all` | 同时保存关节、绝对 EE 与 EE 增量状态 |
| `robot.record_joint_velocity` | `true` | 保存 SDK 关节/夹爪速度 |
| `robot.record_motor_effort` | `false` | 保存未标定的 SDK torque raw 值 |
| `robot.max_ee_translation_step_m` | `0.03` m | 每帧最大 TCP 平移 |
| `robot.max_ee_rotation_step_rad` | `0.20` rad | 每帧最大 TCP 转角 |
| `robot.use_servo` | `true` | 使用 SDK `servo_j` 连续控制 |
| `robot.servo_watchdog_ms` | `250` | 主机失联时固件刹车超时 |
| `robot.servo_use_mit` | `false` | 默认走稳定的位置通道；调好动力学后可启用 MIT |
| `robot.max_relative_target` | `0.15` rad | 每帧最大关节目标变化 |
| `robot.enforce_urdf_limits` | `true` | 在 SDK 软限位之前再做一次 URDF 限幅 |
| `robot.strict_action_features` | `true` | 缺字段、未知字段或非有限数值时整帧拒绝，不发送运动命令 |
| `robot.write_sent_action_back` | `true` | 将安全限幅后的动作原地写回，供默认 LeRobot recorder 保存 |
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
- 当前 EE 已经位于配置的工作空间之外时，笛卡尔动作会被拒绝；先人工确认零位/TCP 后低速恢复。
- 动作字段或数值校验失败时不要关闭严格校验，应修复 teleop、策略或数据集字段。

## 开发验证

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
```

## License

[MIT](LICENSE)
