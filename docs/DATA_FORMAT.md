# 数据表示与采集模式

## 设计原则

FAFU Arm 将“观测状态”和“动作标签”分开配置：

- 观测默认同时保存关节位置、关节速度、绝对 EE 位姿和相邻帧 EE 增量，尽量保留可复用的原始信息。
- 每个训练数据集建议只选择一种主要动作表示，避免策略同时拟合三套冗余标签。
- 若需要一次归档全部动作表示，可使用 `all`；训练前应按实验目的筛选 action features。

LeRobot 会在设备字段之外自动保存帧时间戳、帧索引、episode 索引、任务文本/任务索引以及数据集元数据；
相机开启后还会保存对应的 RGB 或深度流。

## 动作模式

`robot.action_mode` 与 `teleop.action_mode` 必须设为相同值。

| 模式 | Arm action 字段 | 含义 | 适合场景 |
|---|---|---|---|
| `joint` | `joint1.pos` … `joint6.pos` | 目标关节角，rad | 默认；主从结构一致、回放稳定 |
| `ee_pose` | `ee.x/y/z/wx/wy/wz` | 基座坐标系中的绝对 TCP 位姿 | 工作空间固定、希望策略直接预测目标位姿 |
| `ee_delta` | `ee_delta.x/y/z/wx/wy/wz` | 相邻控制帧的 TCP 增量 | 跨初始姿态泛化、局部精细操作 |
| `all` | 上述三组全部保存 | 同时归档三套标签 | 数据分析或以后离线筛选，不建议直接作为默认训练输出 |

所有模式都包含绝对 `gripper.pos`，单位为 rad。位置及位置增量单位为 m；旋转使用旋转向量
（axis-angle / rotvec），单位为 rad。平移增量在 `base_link` 坐标系表达，旋转增量在上一帧
`tool_link` 局部坐标系表达。

`all` 模式下，`robot.all_control_source` 决定真机实际采用 `joint`、`ee_pose` 或 `ee_delta` 中的哪一组。
默认使用 `joint`，其余字段只作为同步标签保存。

## 观测模式

| `robot.observation_mode` | 保存字段 |
|---|---|
| `joint` | 关节/夹爪位置，以及可选速度和 effort |
| `ee_pose` | 绝对 EE 位姿与夹爪位置 |
| `all`（默认） | 上述全部字段，并增加相邻观测帧的 `ee_delta.*` |

默认 `record_joint_velocity=true`，直接读取 SDK 的 rad/s。`record_motor_effort=false` 默认关闭；开启后
`joint*.effort` 和 `gripper.effort` 保存 SDK `MotorState.torque` 的 raw 值，它不是经过标定的腕部力/力矩。

首帧以及两次采样间隔超过 `delta_reset_timeout_s` 时，EE delta 会置零。这可以避免 episode 重置期间
长时间停顿或人工重新摆放 leader 后产生一个虚假的大增量。

## 推荐配置

稳妥的通用数据集：动作保存关节角，状态同时保存关节与笛卡尔信息。

```bash
lerobot-record \
  --robot.type=fafu_follower \
  --robot.action_mode=joint \
  --robot.observation_mode=all \
  --teleop.type=fafu_leader \
  --teleop.action_mode=joint \
  ...
```

EE 增量策略数据集：

```bash
lerobot-record \
  --robot.type=fafu_follower \
  --robot.action_mode=ee_delta \
  --robot.observation_mode=all \
  --robot.max_ee_translation_step_m=0.02 \
  --robot.max_ee_rotation_step_rad=0.15 \
  --teleop.type=fafu_leader \
  --teleop.action_mode=ee_delta \
  ...
```

绝对 EE 位姿模式需要 leader 与 follower 使用一致的基座坐标定义和零位。首次上真机应同时设置
`ee_workspace_min` / `ee_workspace_max`，并从很小的 EE 步长开始验证。

## pytracik 适用性

pytracik 适合本项目的两项工作：

1. 用 FK 从编码器关节角计算 EE 位姿及相邻帧增量；
2. 用当前关节角作为 seed，把绝对 EE 或 EE 增量动作转换成关节目标。

但这些 EE 数据是“URDF 模型推算值”，不是外部测量值。精度受零位、连杆尺寸、TCP、装配误差、
减速器回差和结构形变影响。pytracik 也不负责碰撞检测、轨迹规划、动力学、奇异点风险评估或接触力估计。
因此项目在 IK 外仍保留工作空间、EE 单步、关节单步、URDF 限位和 SDK 软限位；接触任务需要真实
力/力矩传感器或经过标定的电机力矩信号。

## 参考资料

- [LeRobot：机器人与遥操作处理流水线](https://huggingface.co/docs/lerobot/main/en/processors_robots_teleop)
- [LeRobotDataset v3 数据格式](https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx)
- [pytracik 项目说明](https://pypi.org/project/pytracik/)
