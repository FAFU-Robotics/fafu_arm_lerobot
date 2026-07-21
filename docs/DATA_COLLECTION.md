# Data Collection：录制、读取、查看与回放

这篇文档是 FAFU Arm 数据集工作的唯一主入口，覆盖设备发现、相机检查、动作/观测表示、
本地录制、数据保存、读取、可视化、WRS 轨迹检查、续录、发布和安全回放。安装和 SDK 编译见
[部署指南](DEPLOYMENT.md)，运动学尺寸和 TCP 见 [URDF 与 TCP](URDF.md)。

训练数据准备完成后转到 [Policy Training 指南](TRAINING.md)。本文命令以 LeRobot 0.6 为主。项目同时兼容 LeRobot 0.4.3–0.6.x；不同版本先运行相应命令的
`--help`，确认参数名后再连接真机。

## 1. 数据流和推荐选择

每个采样周期按以下顺序运行：

1. follower 读取关节、夹爪和相机观测；
2. leader 产生动作；
3. follower 校验、限幅并下发动作；
4. LeRobot 把当前 observation 和 action 写入同一数据帧。

因此同一行中的 observation 描述动作执行前的状态，action 描述接下来要执行的目标。下一行
observation 才能用于判断动作的实际执行结果。

### 动作模式

`robot.action_mode` 与 `teleop.action_mode` 必须相同。

| 模式 | Action 字段 | 含义 | 建议用途 |
|---|---|---|---|
| `joint` | `joint1.pos` … `joint6.pos` | 绝对目标关节角，rad | 默认，最容易验证和安全回放 |
| `ee_pose` | `ee.x/y/z/wx/wy/wz` | 基座系绝对 TCP 位姿 | 固定工作空间、绝对位姿策略 |
| `ee_delta` | `ee_delta.x/y/z/wx/wy/wz` | 相邻控制帧 TCP 增量 | 局部操作、增量控制策略 |
| `all` | 上述三组同时保存 | 同步归档三种表示 | 离线分析；训练前应筛选字段 |

所有模式都包含绝对 `gripper.pos`，单位为 rad。位置和位置增量单位为 m，旋转采用旋转向量
（axis-angle / rotvec），单位为 rad。

EE delta 的定义是：

```text
delta_position = current_position - previous_position
delta_rotation = Log(previous_rotation.T @ current_rotation)
```

平移在 `base_link` 坐标系表达，旋转在上一帧 `tool_link` 局部坐标系表达。它是每个控制帧的
位移，不是速度；若要得到速度，需要再除以实测时间间隔。

第一次计算 delta 或两次采样间隔超过 `delta_reset_timeout_s` 时会输出零。LeRobot 的 episode
重置阶段仍会连续采样，所以后续 episode 的首帧可能相对于重置阶段最后一个未保存采样计算，
不保证一定为零。绝对 EE 位姿会同时保存在默认 observation 中，不影响位姿恢复。

### 观测模式

| `robot.observation_mode` | 保存内容 |
|---|---|
| `joint` | 关节/夹爪位置，可选速度和 effort |
| `ee_pose` | 绝对 EE 位姿和夹爪位置 |
| `all`（默认） | 关节、速度、绝对 EE 位姿、相邻观测 EE delta、夹爪 |

推荐通用数据集使用 `action_mode=joint`、`observation_mode=all`。这样 action 标签简单稳定，
同时保留以后离线研究 EE 表示需要的信息。一个训练数据集通常只选择一种主要 action 表示。

`record_joint_velocity=true` 保存 SDK 的 rad/s。`record_motor_effort=false` 默认关闭；打开后保存的是
SDK `MotorState.torque` raw 值，不是经过标定的腕部六维力/力矩。

### 请求动作与保存动作

follower 默认启用：

- `strict_action_features=true`：字段缺失、未知字段、NaN 或 Inf 会使整帧在硬件写入前失败；
- `write_sent_action_back=true`：完成关节/EE 步长、工作空间、URDF 和夹爪限位后，将最终命令写回
  默认 LeRobot 录制流水线。

因此默认流水线保存安全限幅后的命令。例如 leader 请求移动 0.20 m、单步限制为 0.03 m，action
保存 0.03 m。它仍是 SDK 接收到的目标，不等于机械臂已精确到达；实际结果看下一帧 observation。
若自行添加会复制 action 字典的 processor 或 IPC，应通过故意触发限幅的集成测试重新确认写回行为。

## 2. 采集前检查

### 2.1 软件和串口

```bash
fafu-arm-check
lerobot-find-port
```

分别拔插 follower 和 leader，记录两个稳定端口；两台机械臂不能使用同一端口。只读硬件检查不会
使能电机或发送运动命令：

```powershell
fafu-arm-check --connect --port COM14
```

```bash
fafu-arm-check --connect --port /dev/serial/by-id/<follower-device>
```

Windows 也可查看：

```powershell
Get-CimInstance Win32_SerialPort | Select-Object DeviceID, Name, PNPDeviceID
```

Ubuntu 建议使用不会随重启变化的名称：

```bash
ls -l /dev/serial/by-id/
```

### 2.2 相机

```bash
lerobot-find-cameras opencv --output-dir outputs/camera_test --record-time-s 3
```

双相机配置：

```text
{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}
```

Ubuntu 优先使用 `/dev/v4l/by-id/...`。正式录制前短时遥操作并设置 `--display_data=true`，确认
front/wrist 没有接反、颜色正常、视野不包含敏感屏幕或人员信息，实际 FPS 能稳定达到目标。

### 2.3 安全清单

- 支撑未制动的 leader 重力轴，确认急停可用；
- 清空 follower 工作空间，检查零位、关节方向、软限位和 TCP；
- 首次使用 `robot.max_relative_target=0.03` 做低速小步验证；
- EE 控制同时设置经过实测的 `ee_workspace_min/max`，从很小步长开始；
- 确认本地磁盘空间足以保存全部相机视频和临时编码文件。

## 3. 本地录制和保存

以下命令把 action 保存为关节目标，把 joint、EE pose 和 EE delta 同时保存为 observation，并明确
禁止自动上传：

```bash
lerobot-record \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/serial/by-id/<follower-device> \
  --robot.action_mode=joint \
  --robot.observation_mode=all \
  --robot.max_relative_target=0.05 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=fafu_leader \
  --teleop.id=fafu_leader \
  --teleop.port=/dev/serial/by-id/<leader-device> \
  --teleop.action_mode=joint \
  --dataset.repo_id=FAFU-Robotics/fafu_demo \
  --dataset.root=./datasets/fafu_demo \
  --dataset.single_task="pick and place" \
  --dataset.fps=30 \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false \
  --display_data=true
```

录制中使用 LeRobot 快捷键：

- `→`：提前保存当前 episode 并进入下一条；
- `←`：丢弃当前 episode 并重新录制；
- `Esc`：结束录制，让 LeRobot完成 Parquet/视频写入和数据集 finalize。

不要直接关闭电源、结束 Python 进程或在录制中读取仍在写入的 Parquet 文件。正常退出后再运行检查工具。
LeRobot v3 会将多条 episode 合并到数据和视频分片中；不能根据单个 Parquet 文件名推断 episode，
应通过 metadata 或本项目的读取函数筛选 `episode_index`。

典型目录：

```text
datasets/fafu_demo/
├── meta/info.json          # FPS、字段、总帧数和路径模板
├── meta/stats.json         # 归一化统计
├── meta/episodes/          # episode 边界和任务元数据
├── data/                   # state/action/timestamp Parquet 分片
└── videos/                 # 各相机 MP4 分片
```

EE 增量 action 采集只需把两端 action mode 同时改为 `ee_delta`，并使用更保守的步长：

```text
--robot.action_mode=ee_delta
--robot.max_ee_translation_step_m=0.02
--robot.max_ee_rotation_step_rad=0.15
--teleop.action_mode=ee_delta
```

### 续录

使用同一个 `repo_id` 和同一个 `root`，增加 `--resume=true`。`dataset.num_episodes` 表示本次新增
episode 数量，不是续录后的总数量：

```text
--dataset.repo_id=FAFU-Robotics/fafu_demo
--dataset.root=./datasets/fafu_demo
--dataset.num_episodes=5
--resume=true
```

续录前先备份数据目录并执行下一节的 `info` 和 `check`。字段、相机名、FPS 或 action mode 不一致时，
新建数据集，不要强行合并。

## 4. 检查、读取和导出

### 4.1 命令行检查

查看数据集概要和字段：

```bash
fafu-arm-dataset info --root ./datasets/fafu_demo
```

回放前严格检查 action 字段顺序、模式、FPS、episode 和相机声明：

```bash
fafu-arm-dataset check \
  --root ./datasets/fafu_demo \
  --action-mode joint \
  --episode 0
```

原有兼容命令仍可使用：

```bash
fafu-arm-dataset-check --root ./datasets/fafu_demo --action-mode joint --episode 0
```

查看一条 episode 的前 3 行低维数据，不解码视频：

```bash
fafu-arm-dataset preview --root ./datasets/fafu_demo --episode 0 --rows 3
```

自动化环境可给 `info`、`check` 或 `preview` 添加 `--json`。导出 state/action/timestamp 到扁平 CSV：

```bash
fafu-arm-dataset export \
  --root ./datasets/fafu_demo \
  --episode 0 \
  --output ./exports/fafu_demo_ep000.csv
```

默认拒绝覆盖已有 CSV；确认后显式加 `--force`。视频不写入 CSV，保留在原数据集的 `videos/` 中。

### 4.2 Python 读取接口

本项目读取器直接按 `episode_index` 筛选 LeRobot v2.1/v3 Parquet，既不连接机械臂，也不解码视频：

```python
from lerobot_robot_fafu_arm.local_dataset import (
    export_episode_csv,
    load_dataset_info,
    load_episode,
)

info = load_dataset_info("./datasets/fafu_demo")
print(info.fps, info.total_episodes, list(info.features))

episode = load_episode("./datasets/fafu_demo", 0)
print(len(episode), episode.columns.keys())

measured_joints = episode.joint_trajectory("observation")  # [frames, 6], rad
joint_targets = episode.joint_trajectory("action")          # 仅 joint/all action 可用
first_rows = episode.records(limit=3)

export_episode_csv(episode, "./exports/fafu_demo_ep000.csv")
```

`load_episode(..., columns=[...])` 可只读指定 Parquet 列。若 action 数据是 `ee_pose` 或 `ee_delta`，
`joint_trajectory("action")` 会明确报出缺少关节字段；此时 WRS 播放应使用默认 observation。

如需相机帧、时间同步解码、PyTorch Dataset 或训练 DataLoader，使用 LeRobot 官方接口：

```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    "FAFU-Robotics/fafu_demo",
    root="./datasets/fafu_demo",
    episodes=[0],
)
frame = dataset[0]
print(frame.keys())
```

## 5. 数据查看

### 5.1 录制时查看

`lerobot-record --display_data=true` 用于确认相机、状态和动作是否更新。它不是录制完成后的质量审查；
每条 episode 保存后仍应抽查视频、动作曲线和异常帧。

### 5.2 LeRobot 官方本地查看器

该查看器同步显示相机、robot state 和 action：

```bash
lerobot-dataset-viz \
  --repo-id FAFU-Robotics/fafu_demo \
  --root ./datasets/fafu_demo \
  --mode local \
  --episode-index 0
```

若提示缺少 dataset 或可视化依赖：

```bash
python -m pip install "lerobot[dataset-viz]>=0.6,<0.7"
```

使用 LeRobot 0.4/0.5 时不要强行安装 0.6 的 extra，改为给当前已锁定版本安装其可视化依赖。

### 5.3 WRS 三维运动学查看

`fafu-arm-wrs-view` 从 observation 或 joint action 中提取六个关节角，使用本项目 URDF 计算每个关节
和 TCP，在 WRS 中绘制机械臂骨架、TCP 坐标系和整条 TCP 轨迹。它不依赖 WRS 中已有某个特定机器人
类，因此不会把其他机器人的模型或坐标约定带入 FAFU 数据。

先只校验数据和轨迹范围，不打开窗口：

```bash
fafu-arm-wrs-view \
  --root ./datasets/fafu_demo \
  --episode 0 \
  --source observation \
  --dry-run
```

打开本机 WRS：

```powershell
fafu-arm-wrs-view `
  --root ./datasets/fafu_demo `
  --episode 0 `
  --wrs-path D:\path\to\wrs `
  --speed 0.5
```

Linux：

```bash
fafu-arm-wrs-view \
  --root ./datasets/fafu_demo \
  --episode 0 \
  --wrs-path /path/to/wrs \
  --stride 2 \
  --speed 0.5
```

也可设置 `WRS_PATH` 后省略 `--wrs-path`。默认播放实测 observation；只有 joint/all action 数据才能
使用 `--source action`。`--stride 2` 每两帧显示一次但保持原时间比例，`--speed 0.5` 半速播放，
`--no-loop` 在最后一帧停止。若采集使用了定制 TCP，必须同时传同一份 `--urdf`。

WRS 查看器用于发现关节跳变、方向错误、TCP 轨迹异常和不可解释的重置运动。相机和数值曲线仍应使用
LeRobot 查看器；WRS 骨架没有 mesh、碰撞体或外部标定，不应作为碰撞安全证明。

## 6. 发布和隐私

所有首次采集命令必须显式使用：

```text
--dataset.push_to_hub=false
```

发布前逐条检查相机画面、任务文本和元数据，避免人脸、屏幕、工牌、地址、访问令牌或未公开实验装置
被上传。确需上传时先使用私有数据集：

```text
--dataset.push_to_hub=true --dataset.private=true
```

只有明确决定公开时才改为 `--dataset.private=false`，不要依赖组织账号默认可见性。上传已经完成本地
检查的数据集也可以使用 LeRobot/Hugging Face 官方工具，但发布前应保存不可变备份。

## 7. 回放前校验和低速回放

严格按顺序执行：

1. 支撑机械臂、清空工作空间并确认急停；
2. `fafu-arm-check` 检查软件、URDF 和 SDK；
3. `fafu-arm-check --connect --port ...` 只读真实关节；
4. `fafu-arm-dataset check ...` 校验 episode 和 action mode；
5. 用 LeRobot/WRS 查看器抽查轨迹；
6. 首次回放把每帧关节步长降到 `0.03 rad`。

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

回放 `ee_pose`、`ee_delta` 或 `all` 时，`robot.action_mode` 必须同名；`all` 还要设置经过验证的
`robot.all_control_source`。字段不匹配时保持严格校验开启并修复命令或数据集，不要绕过检查。

## 8. 故障恢复

### 字段、NaN 或 IK 错误

本帧会在硬件写入前被拒绝。停止采集，检查 robot/teleop action mode、数据集字段和上游输出；不要
关闭严格校验来掩盖问题。

### 当前 EE 位于工作空间外

控制器拒绝自动投影回边界，避免产生大幅意外运动。人工确认 URDF/TCP、零位和工作空间；支撑机械臂
后使用已验证的 joint 模式低速恢复，或重新测量边界。

### SDK 拒绝 servo、看门狗、串口断开或电机 fault

1. 立即停止采集/回放，异常运动时使用急停；
2. 支撑机械臂，不在故障状态循环自动重试；
3. 关闭程序和机械臂电源，检查串口、供电、急停和机械干涉；
4. 重新上电后先运行 SDK `01_smoke`，再运行 `fafu-arm-check --connect`；
5. 只做小步单关节方向测试，全部正常后恢复采集。

### 相机丢帧或断开

终止并重录当前 episode。检查 USB 带宽、相机 FPS/分辨率和编码负载，重新运行
`lerobot-find-cameras` 并确认设备编号和画面。

### 录制异常退出或数据无法读取

保留原目录，不要直接修改 Parquet/MP4。先复制备份，再运行 `fafu-arm-dataset info`；如果缺少
metadata、Parquet footer 损坏或视频未 finalize，不要续录或回放该目录。恢复到最后一份正常备份，
或新建数据集重新录制受影响 episode。

## 参考

- [LeRobot：录制和快捷键](https://huggingface.co/docs/lerobot/en/cheat-sheet)
- [LeRobot：数据集查看工具](https://huggingface.co/docs/lerobot/en/using_dataset_tools)
- [LeRobotDataset v3 格式](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3)
- [LeRobot：动作表示](https://huggingface.co/docs/lerobot/action_representations)
- [pytracik](https://pypi.org/project/pytracik/)
