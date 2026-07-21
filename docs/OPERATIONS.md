# 采集、检查与故障恢复示例

本文给出 FAFU Arm 从设备发现到安全回放的一套可复制流程。示例以 LeRobot 0.6 为主；使用
LeRobot 0.4/0.5 时，先通过对应版本的 `--help` 确认参数名称。

## 1. 查找 follower、leader 串口

LeRobot 的交互式工具会先列出串口，然后要求拔下目标设备，从端口差异中识别它：

```bash
lerobot-find-port
```

也可以先直接列出系统串口。

Windows PowerShell：

```powershell
Get-CimInstance Win32_SerialPort | Select-Object DeviceID, Name, PNPDeviceID
```

Ubuntu：

```bash
ls -l /dev/serial/by-id/
```

分别拔插 follower 和 leader，记录两个稳定设备名。不要让两台机械臂使用同一端口。正式运行前做
只读检查：

```powershell
fafu-arm-check --connect --port COM14
```

```bash
fafu-arm-check --connect --port /dev/serial/by-id/<follower-device>
```

## 2. 查找和验证相机

安装 LeRobot 相机依赖后，让工具枚举 OpenCV 相机并保存几秒测试图像：

```bash
lerobot-find-cameras opencv --output-dir outputs/camera_test --record-time-s 3
```

双相机配置示例：

```text
{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}
```

Ubuntu 上若相机编号会变化，优先使用 `/dev/v4l/by-id/...` 作为 `index_or_path`。正式采集前运行一次
短时遥操作并加上 `--display_data=true`，确认 front/wrist 没有接反、颜色正常且实际帧率稳定。

## 3. 默认只在本地采集

首次采集明确关闭 Hub 上传。以下示例同时保存关节、EE 位姿、EE 增量观察，但使用关节角作为动作标签：

```bash
lerobot-record \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/serial/by-id/<follower-device> \
  --robot.action_mode=joint \
  --robot.observation_mode=all \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=fafu_leader \
  --teleop.id=fafu_leader \
  --teleop.port=/dev/serial/by-id/<leader-device> \
  --teleop.action_mode=joint \
  --dataset.repo_id=FAFU-Robotics/fafu_demo \
  --dataset.root=./datasets/fafu_demo \
  --dataset.single_task="pick and place" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=30 \
  --dataset.push_to_hub=false \
  --display_data=true
```

LeRobot 当前 `push_to_hub` 默认值为 `true`，因此 FAFU 示例始终显式传 `--dataset.push_to_hub=false`。本地检查
相机画面、任务文本、动作范围和 episode 后，确实需要发布时再明确选择：

```text
--dataset.push_to_hub=true --dataset.private=true
```

建议先使用 `private=true`。确认画面不包含人脸、屏幕、工牌、地址、密钥或其他实验室敏感信息后，
才主动改为 `private=false` 发布公开数据。不要依赖组织账号的默认可见性。

把已经完成本地检查的数据集上传为私有仓库：

```bash
python -c "from lerobot.datasets import LeRobotDataset; ds = LeRobotDataset('FAFU-Robotics/fafu_demo', root='./datasets/fafu_demo'); ds.push_to_hub(private=True)"
```

只有明确决定公开时才把上面的 `private=True` 改成 `private=False`。

## 4. 检查数据字段

先查看 LeRobot 的完整数据集信息：

```bash
lerobot-edit-dataset \
  --repo_id=FAFU-Robotics/fafu_demo \
  --root=./datasets/fafu_demo \
  --operation.type=info \
  --operation.show_features=true
```

再使用本项目的只读检查命令核对 FAFU action 字段顺序、episode 范围、FPS 和相机字段：

```bash
fafu-arm-dataset-check \
  --root ./datasets/fafu_demo \
  --action-mode joint \
  --episode 0
```

自动化环境可以追加 `--json`。检查失败时不要连接或移动机械臂。`--action-mode` 必须和采集时的
`robot.action_mode` 完全相同。

## 5. 回放前校验和低速回放

按顺序执行：

1. 支撑机械臂并清空工作空间，确认急停可用。
2. `fafu-arm-check` 检查软件、URDF 和 SDK。
3. `fafu-arm-check --connect --port ...` 只读真实关节状态。
4. `fafu-arm-dataset-check` 校验数据字段与 episode。
5. 第一次回放将每帧关节步长降到 `0.03 rad`。

```bash
lerobot-replay \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/serial/by-id/<follower-device> \
  --robot.action_mode=joint \
  --robot.max_relative_target=0.03 \
  --dataset.repo_id=FAFU-Robotics/fafu_demo \
  --dataset.root=./datasets/fafu_demo \
  --dataset.episode=0
```

回放 `ee_pose`、`ee_delta` 或 `all` 数据时，必须把 `--robot.action_mode` 改为同名模式；`all` 还要
设置正确的 `--robot.all_control_source`。严格字段校验会在模式不匹配时停止，而不是复用旧目标。

## 6. 常见故障与恢复

### 动作字段、NaN 或 IK 错误

这些错误在硬件写入前发生，不会发送本帧动作。停止采集，检查 robot/teleop action mode、数据字段和
上游策略输出；不要通过关闭严格校验来掩盖模型输出错误。

### 当前 EE 位于工作空间外

控制器会拒绝自动移动，因为直接投影回边界可能形成一个很大的意外位移。保持急停可用，人工确认
URDF/TCP、零位和工作空间配置；支撑机械臂后使用经过验证的关节模式低速恢复，或在重新测量后调整边界。

### SDK 拒绝 servo、看门狗、串口断开或电机 fault

1. 立即停止采集/回放；若运动异常则使用急停。
2. 支撑机械臂，避免 `stop` 释放后下坠；不要在故障状态循环自动重试。
3. 关闭程序和机械臂电源，检查串口、供电、急停和机械干涉。
4. 重新上电后先运行 SDK `01_smoke`，再运行 `fafu-arm-check --connect`。
5. 只用小 `max_relative_target` 做单关节方向测试；全部正常后再恢复采集。

### 相机丢帧或断开

终止当前 episode，不要把缺帧 episode 用于训练。检查 USB 带宽、相机 FPS/分辨率和视频编码负载；
重新运行 `lerobot-find-cameras`，确认设备编号和画面后再开始新的 episode。
