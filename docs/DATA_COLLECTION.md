# Data Collection：录制、检查、查看与回放

本文是 FAFU Arm 数据集工作的唯一入口。安装见[部署指南](DEPLOYMENT.md)，运动学与 TCP 见[URDF 与 TCP](URDF.md)，训练见[Policy Training 指南](TRAINING.md)。

本文命令已用 LeRobot 0.6.x 验证；运行前进入仓库根目录并激活项目虚拟环境：

```bash
cd /path/to/fafu_arm_lerobot
# Linux: source .venv/bin/activate；Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip show lerobot lerobot_robot_fafu_arm
```

LeRobot 0.4.3–0.5.x 用户应按部署指南选择环境并用 `--help` 核对参数，不要混用不同版本命令。

## 1. 五分钟短录验收

正式采集前，先完成一个独立的 1 episode、10 秒 smoke 数据集：

1. 运行 `fafu-arm-check`、`lerobot-find-port`，按第 3 节确认 follower/leader 端口；
2. 运行 `lerobot-find-cameras opencv --output-dir outputs/camera_test --record-time-s 3`；
3. 替换第 4.1 节命令中的端口和相机路径，录制 1 条 episode，并用 `Esc` 正常结束；
4. 执行以下检查，再用 LeRobot 查看器核对视频、状态和动作：

```bash
fafu-arm-dataset info --root ./datasets/fafu_smoke
fafu-arm-dataset check --root ./datasets/fafu_smoke --action-mode joint --episode 0
fafu-arm-dataset preview --root ./datasets/fafu_smoke --episode 0 --rows 3
```

只有字段、相机、轨迹和正常退出均通过后才扩展录制；smoke 数据集不要与正式数据集续录或合并。

## 2. 数据字段与时序

### 2.1 一帧数据的时间关系

每个采样周期依次执行：

1. follower 读取关节、夹爪和相机 observation；
2. leader 产生 action；
3. follower 校验、限幅并下发 action；
4. LeRobot 把该 observation 和 action 写入同一帧。

因此同一行的 observation 是动作执行前状态，action 是接下来要执行的目标；下一行 observation
才反映该动作的执行结果。训练或分析时不要把同一行 action 当成已测量状态。

### 2.2 Action 模式

`robot.action_mode` 与 `teleop.action_mode` 必须相同。所有模式都含绝对 `gripper.pos`（rad）。

| 模式与 Action 字段 | 必需配置 | 用途 |
|---|---|---|
| `joint`：`joint1.pos` … `joint6.pos`（rad） | 两端设为 `joint`；首轮显式设 `robot.max_relative_target=0.03` | 推荐基线，最容易验证和回放 |
| `ee_pose`：`ee.x/y/z/wx/wy/wz` | 两端设为 `ee_pose`；设置实测 workspace 和 EE 单步限制 | 基座系绝对 TCP 位姿 |
| `ee_delta`：`ee_delta.x/y/z/wx/wy/wz` | 两端设为 `ee_delta`；设置实测 workspace 和 EE 单步限制 | 相邻控制帧的 TCP 增量 |
| `all`：同时保存以上三组 | 两端设为 `all`；必须设置 `robot.all_control_source=joint/ee_pose/ee_delta` 及对应限制 | 仅用于归档和离线分析 |

`all` 含冗余 action，当前 `fafu-arm-train` 训练入口不直接支持；训练应使用只含一种 action 表示的新数据集。

EE 位置/平移单位为 m，旋转使用 axis-angle 旋转向量，单位为 rad。EE 模式必须使用在实际
`base_link`/TCP 下测量的边界，不要复制其他工作台的数值：

```text
--robot.ee_workspace_min="[<x_min>, <y_min>, <z_min>]"
--robot.ee_workspace_max="[<x_max>, <y_max>, <z_max>]"
--robot.max_ee_translation_step_m=0.02
--robot.max_ee_rotation_step_rad=0.15
```

### 2.3 Observation、相机和公共字段

| `robot.observation_mode` | `observation.state` 中的内容 |
|---|---|
| `joint` | 关节/夹爪位置，可选速度和 effort |
| `ee_pose` | 绝对 EE 位姿和夹爪位置 |
| `all`（推荐） | 关节、速度、绝对 EE 位姿、相邻观测 EE delta、夹爪 |

相机字段独立于 `observation_mode`；配置 `robot.cameras` 后，各模式都会增加 `observation.images.<camera_name>`。
LeRobot 还写入 `timestamp`、`frame_index`、`episode_index`、`index` 和 `task_index`；实际 schema 以 `meta/info.json` 为准。

通用数据集推荐 `action_mode=joint`、`observation_mode=all`。`record_joint_velocity=true` 保存 SDK 的 rad/s；
开启 `record_motor_effort` 后保存的是 SDK `MotorState.torque` raw 值，不是标定后的六维力/力矩。

### 2.4 EE delta 定义

```text
delta_position = current_position - previous_position
delta_rotation = Log(previous_rotation.T @ current_rotation)
```

平移在 `base_link` 表达，旋转在上一帧 `tool_link` 局部坐标系表达。delta 是每个控制帧的位移，
不是速度；速度还需除以实测时间间隔。

首次计算或采样间隔超过 `delta_reset_timeout_s` 时 delta 为零。episode 重置期间仍会连续采样，
因此后续 episode 首帧可能相对重置阶段最后一个未保存采样计算，不保证为零。绝对 EE 位姿仍保存在
`observation_mode=all` 中，可用于恢复和复算。

### 2.5 请求动作与保存动作

follower 默认启用：

- `strict_action_features=true`：缺失/未知字段、NaN 或 Inf 会使整帧在硬件写入前失败；
- `write_sent_action_back=true`：将关节/EE 步长、workspace、URDF 和夹爪限位后的最终命令写回
  LeRobot 录制流水线。

例如请求 EE 平移 0.20 m、单步限制为 0.03 m，保存的 action 是 0.03 m。它仍是 SDK 接收的目标，
不是机械臂已到达的测量值；实际结果应看下一帧 observation。

## 3. 采集前检查

### 3.1 软件和串口

```bash
fafu-arm-check
lerobot-find-port
```

分别拔插 follower 和 leader，确认角色并记录两个不同端口。Ubuntu 优先使用重启后稳定的
`/dev/serial/by-id/...`；Windows 使用 `COM<n>`，拔插后应重新确认。只读连接检查不会使能电机或
发送运动命令：

```bash
# Windows 示例
fafu-arm-check --connect --port COM14
# Ubuntu 示例
fafu-arm-check --connect --port /dev/serial/by-id/<follower-device>
```

分别检查 follower 和 leader，均应成功读取 6 个关节状态。若角色、端口或读数不明确，不要开始遥操作。

### 3.2 相机

```bash
lerobot-find-cameras opencv --output-dir outputs/camera_test --record-time-s 3
```

将设备命名为稳定的逻辑键，例如 `front`、`wrist`；数据集字段对应
`observation.images.front`、`observation.images.wrist`。同一数据集续录时不能改名。相机只配置在
follower，leader 不需要相机参数。

Ubuntu 优先使用 `/dev/v4l/by-id/...`；Windows 使用数字编号时，USB 拔插后必须重新发现。确认相机
支持配置的分辨率和 FPS，并让相机 FPS 与 `dataset.fps` 一致。正式录制前用短录命令中的
`--display_data=true` 核对画面、角色和性能；长时间录制默认关闭显示，只有实测编码与显示负载稳定
时才开启。

### 3.3 安全清单

- 支撑未制动的 leader 重力轴，清空 follower 工作空间并确认急停可用；
- 检查零位、关节方向、软限位、URDF 和 TCP；
- 首次连接、短录和首次回放显式设 `robot.max_relative_target=0.03`；
- 完成小步响应和急停验证后，日常 joint 采集可选升至 `0.05`；代码默认 `0.15` 不是首轮安全值；
- EE 模式必须同时设置实测 `ee_workspace_min/max` 和保守的平移/旋转单步限制；
- 确认磁盘可容纳全部视频与临时编码文件，画面不含屏幕、人脸或其他敏感信息。

## 4. 录制、保存与续录

### 4.1 首次短录命令

以下命令使用推荐的 `joint` action、`all` observation，录制 1 条 10 秒数据并禁止上传。替换两个
串口和相机路径；单相机时删除 `wrist` 项。

```bash
lerobot-record \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/serial/by-id/<follower-device> \
  --robot.action_mode=joint \
  --robot.observation_mode=all \
  --robot.max_relative_target=0.03 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=fafu_leader \
  --teleop.id=fafu_leader \
  --teleop.port=/dev/serial/by-id/<leader-device> \
  --teleop.action_mode=joint \
  --dataset.repo_id=FAFU-Robotics/fafu_smoke \
  --dataset.root=./datasets/fafu_smoke \
  --dataset.single_task="pick and place" \
  --dataset.fps=30 \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false \
  --display_data=true
```

快捷键：`→` 提前保存当前 episode，`←` 丢弃并重录，`Esc` 正常结束并等待 Parquet、视频和 metadata
完成写入。不要直接断电、结束 Python 进程或读取仍在写入的 Parquet。

正常结束后，`meta/` 保存字段、统计和 episode 边界，`data/` 保存低维 Parquet 分片，`videos/`
保存各相机 MP4。多条 episode 可共用分片；读取时使用 metadata 的 `episode_index`，不要从文件名推断。

### 4.2 正式采集

短录验收通过后，新建正式数据集并只修改以下参数：

```text
--dataset.repo_id=FAFU-Robotics/fafu_demo
--dataset.root=./datasets/fafu_demo
--dataset.num_episodes=20
--dataset.episode_time_s=30
--dataset.reset_time_s=15
--display_data=false
```

保持相机逻辑名、分辨率、FPS、action/observation 模式和任务定义稳定。`display_data=true` 仅用于短录
或已验证性能充足的环境。完成小步响应验证后可选将 joint 的 `max_relative_target` 从 `0.03` 升至
`0.05`，并记录该变更；不要把代码默认的 `0.15` 直接用于首轮采集。

若采集 `ee_pose` 或 `ee_delta`，同时修改两端 action mode，并按第 2.2 节提供实测 workspace 和 EE
步长。`all` 还必须显式设置 `robot.all_control_source`，且只作为归档数据集。

正式采集前在数据目录外保存采集记录：主仓库与 SDK commit、Python/LeRobot 版本、URDF/TCP、SDK cfg、
端口与相机映射、曝光/FPS、完整命令和任务定义。不要手工修改 LeRobot 的 `meta/` 文件伪造该记录。

### 4.3 续录

续录必须使用相同 `repo_id`、`root`、字段、相机名、FPS 和 action mode，并增加 `--resume=true`。
`dataset.num_episodes` 是本次新增数量，不是续录后的总数：

```text
--dataset.repo_id=FAFU-Robotics/fafu_demo
--dataset.root=./datasets/fafu_demo
--dataset.num_episodes=5
--resume=true
```

续录前先完整备份数据目录，并执行第 5.1 节的 `info` 和 `check`。任何 schema 不一致都应新建
数据集，不要强行合并。

## 5. 验收、查看与读取

### 5.1 检查范围与验收

```bash
fafu-arm-dataset info --root ./datasets/fafu_demo
fafu-arm-dataset check --root ./datasets/fafu_demo --action-mode joint --episode 0
fafu-arm-dataset preview --root ./datasets/fafu_demo --episode 0 --rows 3
```

- `info` 汇总 metadata 中的格式、FPS、episode/frame 数和字段；
- `check` 只检查 `meta/info.json` 的 action 字段/顺序、模式、正 FPS、episode 范围、robot type 和
  相机声明；
- `preview` 读取指定 episode 的少量低维 Parquet 行，不扫描整集。

它们不检查全量 Parquet 的 NaN/Inf、视频完整解码、实际 FPS、相机—动作同步或任务质量。正式数据集
至少抽查第一、中间、最后 episode，并组合执行：

1. `preview` 检查 timestamp、frame/episode index 和数值跳变；
2. `lerobot-dataset-viz` 检查视频、相机角色、动作与观测对齐；
3. `fafu-arm-wrs-view --dry-run` 检查关节/TCP 范围；
4. 人工确认任务、重置边界、遮挡及敏感画面。

失败的 episode 应重录或隔离；不能仅凭 `check` 的 `[OK]` 就进入训练/回放。上述命令可加 `--json`
保存机器可读记录。

### 5.2 LeRobot 与 WRS 查看

```bash
lerobot-dataset-viz --repo-id FAFU-Robotics/fafu_demo --root ./datasets/fafu_demo --mode local --episode-index 0
```

缺少依赖时，在已锁定的 0.6.x 环境安装 `python -m pip install "lerobot[dataset-viz]>=0.6,<0.7"`；
其他版本应安装匹配的 extra。

WRS 无窗口检查只读数据并计算轨迹，不需要 WRS checkout：

```bash
fafu-arm-wrs-view --root ./datasets/fafu_demo --episode 0 --source observation --dry-run
```

打开三维窗口必须提供 WRS checkout，或预先设置 `WRS_PATH`：

```bash
fafu-arm-wrs-view --root ./datasets/fafu_demo --episode 0 --wrs-path "<path-to-wrs-checkout>" --speed 0.5
```

默认播放 observation；仅 `joint/all` action 可用 `--source action`。`--stride 2` 可降显示频率，
`--no-loop` 在末帧停止；定制 TCP 必须传入采集时的 `--urdf`。WRS 不解码相机且没有碰撞体，不能
证明相机同步或碰撞安全。

### 5.3 CSV 与 Python API

```bash
fafu-arm-dataset export --root ./datasets/fafu_demo --episode 0 --output ./exports/fafu_demo_ep000.csv
```

CSV 只含低维 state/action/timestamp，视频仍在 `videos/`；默认不覆盖，确认后加 `--force`。本项目 API
按 `episode_index` 筛选 v2.1/v3 Parquet，不连接机械臂、不解码视频：

```python
from lerobot_robot_fafu_arm.local_dataset import load_dataset_info, load_episode

info = load_dataset_info("./datasets/fafu_demo")
episode = load_episode("./datasets/fafu_demo", 0)
print(info.fps, info.total_episodes, len(episode))
measured = episode.joint_trajectory("observation")  # [frames, 6], rad
targets = episode.joint_trajectory("action")        # 仅 joint/all action
```

可用 `load_episode(..., columns=[...])` 限制列。`ee_pose/ee_delta` action 调用
`joint_trajectory("action")` 会报缺少字段，WRS 应读取 observation。相机解码和训练 DataLoader 使用
LeRobot 官方 API：

```python
from lerobot.datasets import LeRobotDataset
dataset = LeRobotDataset("FAFU-Robotics/fafu_demo", root="./datasets/fafu_demo", episodes=[0])
frame = dataset[0]
```

## 6. 发布和隐私

首次采集必须显式使用 `--dataset.push_to_hub=false`。发布前逐条检查相机、任务文本和 metadata，
排除人脸、屏幕、工牌、地址、令牌及未公开装置，并保存不可变本地备份。

首次上传必须显式设为私有：

```text
--dataset.push_to_hub=true --dataset.private=true
```

上传后再次检查远端可见性和内容；只有经批准公开时才改为 `--dataset.private=false`，不要依赖组织的
默认可见性。

## 7. 回放前校验和低速回放

按顺序执行：支撑机械臂并确认急停；运行 `fafu-arm-check` 和只读连接检查；执行 dataset
`check`；用 LeRobot 查看器和 WRS 抽查第一、中间、最后 episode；最后才接通运动并小步回放。

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

`ee_pose/ee_delta` 回放必须使用同名 `robot.action_mode`、实测 workspace 和保守 EE 步长。`all` 是归档
模式，不建议直接回放；确需回放时必须使用 `action_mode=all`，显式选择已验证的
`all_control_source` 并启用对应限制。字段不匹配时修复命令或数据，不要关闭严格校验。

## 8. 故障恢复

- **字段、NaN、Inf 或 IK 错误**：本帧会在硬件写入前被拒绝。停止采集，核对两端 action mode、
  数据字段、URDF/TCP 和 workspace；不要关闭严格校验。
- **当前 EE 位于 workspace 外**：控制器不会把当前位置自动投影回边界。支撑机械臂，核对零位和
  边界，再用已验证的 joint 模式小步恢复或重新测量边界。
- **相机丢帧/断开**：终止并重录当前 episode；检查 USB 带宽、FPS、分辨率和编码负载，重新运行
  `lerobot-find-cameras` 并核对画面。仍不稳定时一次只调整
  `dataset.num_image_writer_threads_per_camera`、`dataset.num_image_writer_processes`、
  `dataset.video_encoding_batch_size` 或 `dataset.streaming_encoding`，然后重新做双相机短录压力测试。
- **异常退出/数据不可读**：保留原目录并先复制备份，不要直接修改 Parquet/MP4。若 metadata 缺失、
  Parquet footer 损坏或视频未 finalize，不得续录或回放；恢复正常备份或新建数据集重录。

SDK 拒绝 servo、看门狗超时、串口断开或电机 fault 时：

1. 立即停止采集/回放，异常运动时使用急停；
2. 支撑机械臂，禁止在 fault 状态循环自动重试；
3. 断开程序和设备电源，检查串口、供电、急停和机械干涉；
4. 重新上电后先运行 SDK `01_smoke`，再运行 `fafu-arm-check --connect`；
5. 完成小步单关节方向测试后再恢复采集。

## 参考

- [LeRobot：录制和快捷键](https://huggingface.co/docs/lerobot/en/cheat-sheet)
- [LeRobot：数据集查看工具](https://huggingface.co/docs/lerobot/en/using_dataset_tools)
- [LeRobotDataset v3 格式](https://huggingface.co/docs/lerobot/v0.6.0/lerobot-dataset-v3)
- [LeRobot：动作表示](https://huggingface.co/docs/lerobot/action_representations)
