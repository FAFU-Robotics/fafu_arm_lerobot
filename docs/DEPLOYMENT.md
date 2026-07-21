# 部署指南

## 1. 选择 Python / LeRobot 版本

- 希望直接复用 SDK 现有 Windows `cp310` 二进制：使用 Python 3.10，pip 会选择兼容的
  LeRobot 0.4.3 系列。
- 使用 LeRobot 0.5.x / 0.6.x：使用 Python 3.12 或 3.13，并为同一个 Python 重编 FAFU SDK。
- 不要把不同虚拟环境生成的 `fafu_motor` 混用。

检查 ABI：

```bash
python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
```

## 2. 获取完整仓库

```bash
git clone --recurse-submodules https://github.com/FAFU-Robotics/fafu_arm_lerobot.git
cd fafu_arm_lerobot
```

SDK 固定为 submodule commit，便于复现实验。如果需要测试 SDK 新版本，先在
`third_party/fafu_arm_sdk` 内切换并验证，再更新主仓库的 gitlink。

## 3. Windows

要求 Visual Studio 2022 的“使用 C++ 的桌面开发”、CMake 和 64 位 Python。

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip pybind11
cd third_party/fafu_arm_sdk/fafu_robot_cpp
./build.bat
cd ../../..
python -m pip install -e .
fafu-arm-check
```

若 `pytracik` 报 DLL load failed，先强制重装与当前 ABI 对应的 wheel：

```powershell
python -m pip install --force-reinstall --no-cache-dir pytracik==0.0.3
```

若 `fafu_motor` 报 DLL load failed，确认 `serial_cmake.dll` 与 `.pyd` 同在
`third_party/fafu_arm_sdk/fafu_robot_python/`，并重新运行 SDK `build.bat`。

## 4. Ubuntu 22.04 / 24.04

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

cd third_party/fafu_arm_sdk/fafu_robot_cpp
bash linux/install_deps.sh
bash linux/setup_udev.sh
bash linux/build.sh --python "$(command -v python)"
cd ../../..

python -m pip install -e .
fafu-arm-check
```

重新登录后 `dialout` 用户组才会生效。使用 udev 规则时建议将 follower 和 leader 分别绑定到
稳定的设备名，或在 CLI 中显式传 `/dev/ttyUSB0`、`/dev/ttyUSB1`。

## 5. 分阶段验收

1. `fafu-arm-check`：只验证 Python 软件栈。
2. SDK `01_smoke`：只读底层通信。
3. `fafu-arm-check --connect`：通过 Python SDK 只读关节状态。
4. 用 `robot.max_relative_target=0.03` 做小步动作测试。
5. 检查 6 个关节和夹爪方向、软限位、断开释放行为。
6. 再接相机、leader、数据采集和策略。

## 6. 双臂/主从注意事项

- follower 和 leader 必须明确使用不同串口。
- 每台机械臂可以使用独立的 SDK cfg，通过 `--robot.sdk_config_path` 和
  `--teleop.sdk_config_path` 指定。
- 默认 follower 断开时 joints=`stop`、gripper=`brake`；leader 全部 `stop`。
- 默认 follower servo watchdog 为 250 ms，适合 30 Hz LeRobot loop。生产环境应根据实测
  jitter 调整，但不建议关闭 watchdog。

## 7. 发布/安装形态

开发和现场部署推荐 editable install，因为 SDK submodule 中包含平台相关二进制：

```bash
python -m pip install -e .
```

Python wheel 只包含 LeRobot adapter、URDF 和默认 cfg，不包含 SDK C++ 二进制。安装 wheel 时，
需要把 SDK 单独放在机器上并设置：

```bash
export FAFU_ARM_SDK_PATH=/opt/fafu_arm_sdk
```

## 8. 现场操作手册

以下现场步骤集中在 [采集、检查与故障恢复示例](OPERATIONS.md)：

- follower/leader 串口发现与只读连接；
- OpenCV 双相机枚举、配置和画面确认；
- 默认不上传的本地采集，以及私有/公开发布选择；
- 数据字段检查、回放前校验、低速回放和故障恢复。
