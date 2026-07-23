# 部署指南

本文是 Python/LeRobot 版本、SDK ABI、平台构建、SDK 定位、端口角色和首次验收的唯一部署说明。
数据采集与回放见 [Data Collection 指南](DATA_COLLECTION.md)，训练见 [Policy Training 指南](TRAINING.md)。

## 1. 固定运行档案

| 档案 | Python | LeRobot | 适用场景 | SDK 要求 |
|---|---|---|---|---|
| 完整功能（推荐） | 3.12 | 0.6.x；复现基线为 `0.6.0` | 采集、ACT 训练、rollout | 必须用该 Python 重编 |
| Windows cp310 兼容 | 3.10 | `0.4.3` | 复用仓库内已有 `cp310` 二进制 | `.pyd` 必须为 cp310 |

项目依赖允许 LeRobot 0.4.3–0.6.x，但部署时应显式固定版本。`fafu_motor` 文件名中的
`cp310`、`cp312` 必须与当前解释器一致，不能跨 ABI 混用。

## 2. 获取仓库

```bash
git clone --recurse-submodules https://github.com/FAFU-Robotics/fafu_arm_lerobot.git
cd fafu_arm_lerobot
git submodule update --init --recursive
git submodule status --recursive
```

submodule 状态行不应以 `-` 开头。SDK commit 由主仓库固定；升级 SDK 时应先验证，再提交新 gitlink。

## 3. Windows

要求：64 位 Python、CMake 3.18+，以及 Visual Studio 2022 的“使用 C++ 的桌面开发”。

### 3.1 Python 3.12 + LeRobot 0.6.0（推荐）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip pybind11
python -m pip install "lerobot[core-scripts]==0.6.0"
cd third_party\fafu_arm_sdk\fafu_robot_cpp
.\build.bat
cd ..\..\..
python -m pip install -e .
fafu-arm-check
```

`build.bat` 必须在目标环境中运行。输出的 `fafu_motor.cp312-win_amd64.pyd` 应与
`serial_cmake.dll` 同在 `third_party/fafu_arm_sdk/fafu_robot_python/`。

### 3.2 Python 3.10 + LeRobot 0.4.3

仓库已包含 Windows cp310 二进制；只有在二进制缺失、SDK 变更或导入失败时才需重编。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "lerobot[core-scripts]==0.4.3"
python -m pip install -e .
fafu-arm-check
```

## 4. Ubuntu

Ubuntu 24.04 推荐 Python 3.12。先安装对应的 `venv/dev` 包，再在虚拟环境显式安装 pybind11：

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev patchelf
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip pybind11
```

依赖安装和构建脚本必须显式使用同一个虚拟环境解释器：

```bash
cd third_party/fafu_arm_sdk/fafu_robot_cpp
bash linux/install_deps.sh --python "$(command -v python)"
bash linux/build.sh --python "$(command -v python)"
cd ../../..
python -m pip install "lerobot[core-scripts]==0.6.0"
python -m pip install -e .
fafu-arm-check
```

预装 pybind11 可避免依赖脚本在 venv 中尝试 `pip --user`。Ubuntu 22.04 可使用 Python 3.10 +
LeRobot 0.4.3，但 Linux SDK 仍须用该 Python 重编；完整档案需从认可的软件源安装 Python 3.12
及对应 `venv/dev`，不能用 3.10 headers 编译 cp312 模块。

构建通过标准：`fafu-arm-check` 能定位 SDK；SDK Python 目录内同时存在 `fafu_motor*.so` 和
`libserial_cmake.so`，模块 RPATH 为 `$ORIGIN`。缺少 `patchelf` 时不要忽略构建警告。

## 5. SDK 定位与安装形态

源码部署默认加载 `third_party/fafu_arm_sdk`。外部 SDK 可通过环境变量指定：

```powershell
# Windows PowerShell
$env:FAFU_ARM_SDK_PATH = "D:\code\fafu_arm_sdk"
```

```bash
# Ubuntu
export FAFU_ARM_SDK_PATH=/opt/fafu_arm_sdk
```

路径可指向 SDK 根目录或 `fafu_robot_python`。外部 SDK 仍须匹配当前 ABI，并用
`fafu-arm-check` 确认实际加载路径。项目 wheel 不包含平台相关 SDK 二进制；硬件部署推荐
editable install，安装 wheel 时必须另外部署 SDK。

## 6. 端口与权限

### Windows

```powershell
Get-CimInstance Win32_SerialPort | Select-Object DeviceID, Name, PNPDeviceID
```

记录 follower/leader 的物理设备和 COM 号。两者必须不同；重新插拔后再次核对角色。

### Ubuntu

使用稳定路径，并确认两个路径指向不同设备：

```bash
ls -l /dev/serial/by-id/
readlink -f /dev/serial/by-id/FOLLOWER_DEVICE
readlink -f /dev/serial/by-id/LEADER_DEVICE
```

生产和双臂部署优先使用 `dialout` 组；组变更后重新登录：

```bash
sudo usermod -aG dialout "$USER"
```

SDK 的 `linux/99-fafu-debug-board.rules` 仅适合单设备临时调试：

- 规则按 USB vendor ID 而不是设备序列号匹配；
- 多设备会竞争同一个 `/dev/fafu_debug_board` 软链接；
- `MODE=0666` 允许所有本机用户读写串口，权限过宽。

双臂必须使用各自的 `/dev/serial/by-id/...`。没有唯一 by-id 时，应按序列号创建站点专用规则，
并使用最小权限（如 `MODE=0660, GROUP=dialout`），不要使用当前通用规则。

## 7. 分阶段验收

每阶段通过后再继续；失败时断开机械臂并排查，不跳过保护。

### 7.1 软件与 ABI（不打开串口）

```bash
python -c "import sys; from importlib.metadata import version; print(sys.executable, sys.version); print('LeRobot', version('lerobot'))"
fafu-arm-check
```

通过标准：版本符合所选档案，并出现 `[OK] URDF / pytracik`、`[OK] FAFU SDK` 和
`[OK] software checks passed (hardware was not opened)`。

### 7.2 follower 只读连接（不发送运动命令）

```powershell
# Windows
fafu-arm-check --connect --port COM14
```

```bash
# Ubuntu；替换为实际稳定路径
fafu-arm-check --connect --port /dev/serial/by-id/FOLLOWER_DEVICE
```

通过标准：输出有限的关节弧度值和 `[OK] hardware check passed; no motion command was sent`；
没有权限错误、端口占用、异常跳变或错误的设备角色。

### 7.3 首次小步运动

确认急停有效、工作空间无人，并托住 leader 可能下落的重力轴。Ubuntu 示例：

```bash
lerobot-teleoperate \
  --fps=30 \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/serial/by-id/FOLLOWER_DEVICE \
  --robot.action_mode=joint \
  --robot.servo_rate_hz=30 \
  --robot.max_relative_target=0.03 \
  --teleop.type=fafu_leader \
  --teleop.id=fafu_leader \
  --teleop.port=/dev/serial/by-id/LEADER_DEVICE \
  --teleop.action_mode=joint
```

通过标准：逐关节和夹爪缓慢测试时方向正确、没有突跳、不能越过软限位；停止后 follower 为
joints=`stop`、gripper=`brake`，servo watchdog 保持启用。方向错误、快速运动、持续跟随误差或
watchdog 触发时立即停止验收。

### 7.4 相机、短录与回放

按 [Data Collection 指南](DATA_COLLECTION.md)完成相机预览、1 个短 episode、数据验收和低速回放。
首次连接与回放使用 `robot.max_relative_target=0.03`；验证稳定后日常采集可按指南升至 `0.05`。

## 8. 常见故障

| 症状 | 处理 |
|---|---|
| Windows `fafu_motor`：`DLL load failed` | 检查 `.pyd` ABI、64 位架构及同目录 `serial_cmake.dll`，在目标 venv 重跑 `build.bat` |
| Windows pytracik：`DLL load failed` | 重装当前 ABI wheel：`python -m pip install --force-reinstall --no-cache-dir pytracik==0.0.3` |
| Ubuntu 找不到 `libserial_cmake.so` | 检查同目录 `.so` 和 `$ORIGIN` RPATH，安装 `patchelf` 后重跑 `build.sh --python ...` |
| `undefined symbol: _Py...` | SDK 与解释器 ABI 不一致，使用当前 venv Python 重编 |
| 串口无权限或主从接反 | 重新登录使 `dialout` 生效，并核对 COM/PNPDeviceID 或两个 by-id 的实际指向 |
| 加载了错误 SDK | 清除错误的 `FAFU_ARM_SDK_PATH`，检查 submodule commit 和检查命令打印的路径 |
