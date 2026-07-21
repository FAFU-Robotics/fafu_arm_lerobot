# Policy Training：ACT 与后续算法

本文是 FAFU Arm 策略训练的唯一入口，覆盖动作表示选择、ACT 快速训练、评估、调参、断点续训和模型修改。
数据录制与检查见 [Data Collection 指南](DATA_COLLECTION.md)，真机部署见 [部署指南](DEPLOYMENT.md)。命令以
LeRobot 0.6 为主；项目仍可在 0.4.3–0.6.x 运行硬件接口，但旧版训练参数可能不同，训练机优先使用 0.6.x。

## 1. 当前实现与设计边界

本项目使用 LeRobot 官方 `ACTConfig`、`ACTPolicy`、processor 和 `lerobot-train`，不复制一份 ACT 源码。
`fafu-arm-train` 只负责三件事：

1. 检查本地数据的 action 字段、相机、样本可读性、统计量和 LeRobot 版本；
2. 根据 FAFU 动作表示生成可复现的官方训练命令；
3. 默认关闭模型上传和 W&B，只有显式授权才对外发送内容。

这能让当前实现持续获得 LeRobot 的修复，同时把后续算法扩展限制在清晰的训练层：

```text
src/lerobot_robot_fafu_arm/
├── follower.py / leader.py / kinematics.py   # 硬件与运动学，不依赖具体算法
└── training/
    ├── common.py                              # 所有算法共用的数据/运行时检查
    ├── act.py                                 # ACT 参数和官方命令适配
    └── cli.py                                 # fafu-arm-train；以后增加 diffusion/dp3 子命令
```

当确实要改变网络结构时，再按 LeRobot 官方规范增加独立的 `lerobot_policy_fafu_act` 策略插件；不要修改
虚拟环境中的 `site-packages/lerobot`。具体做法见第 8 节。

## 2. 先选动作表示

ACT 只要求 `action` 是固定维度的连续向量，因此 `joint`、`ee_delta` 和 `ee_pose` 都能训练。表示由数据集
`features.action.names` 决定，不是训练开始时临时转换。三种数据应分开录制、分开训练、使用相同任务划分比较。

| 模式 | 模型学习的 action | 优点 | 主要风险 | 建议 |
|---|---|---|---|---|
| `joint` | 6 个绝对关节目标 + 绝对夹爪位置 | 直接对应控制器，无 FK/IK 误差，最容易回放和定位问题 | 机器人构型相关，跨机械臂迁移较弱 | **首个基线，默认推荐** |
| `ee_delta` | 每控制帧的 TCP 顺序增量 + 绝对夹爪位置 | 局部动作近零、适合小范围精细操作 | 顺序增量会累计误差；每步都依赖 FK/IK、TCP、坐标系和采样率 | joint 基线稳定后做对照实验 |
| `ee_pose` | `base_link` 下绝对 TCP 位姿 + 绝对夹爪位置 | 目标含义直观，不累计增量 | 要求固定且准确的 base/TCP 标定；IK 可能在奇异点或边界失败 | 固定工位、标定可靠时使用 |

所以“delta EE 最多人用、普遍最推荐”并不正确。LeRobot 官方把绝对 joint action 作为默认，并明确指出顺序 delta
需要逐步累加、会产生误差累积。ACT 原论文和 LeRobot 入门流程也以关节状态/关节动作作为典型输入输出。对 FAFU Arm，
推荐顺序是：`joint` 建立可复现基线 → `ee_delta` 做同数据规模对照 → 标定充分时评估 `ee_pose`。

这里还要区分两个容易混淆的概念：

- 本项目 `ee_delta` 是相邻控制帧之间的**顺序增量**。平移在 `base_link` 表达，旋转是上一帧 tool 坐标系中的
  rotvec；follower 每帧从实测关节做 FK、应用增量、再用 pytracik 求 IK。
- LeRobot 文档中的 relative action 是整个 action chunk 都相对“本次推理开始状态”，不是顺序 delta。目前
  LeRobot 的自动 relative processor 主要用于 pi 系列，不能把它和本项目 `ee_delta` 当成同一表示。

`action_mode=all` 适合归档和离线研究，但会把三套冗余控制目标塞进同一个输出。本训练入口故意拒绝 `all`；训练前应
生成只含一种 action schema 的 LeRobot 数据集，而不是让 ACT 自己猜哪组字段用于控制。

### pytracik 在训练中的作用

pytracik 不参加神经网络反向传播。录制时 FK 生成 EE 标签，`ee_pose`/`ee_delta` 真机执行时 IK 把策略输出转成关节目标；
训练只读取已经保存的数值。因此 pytracik 很适合当前接口，但它不能消除 URDF/TCP 标定误差、不可达目标、奇异点或多解
跳变。使用 EE action 前必须先通过回放、WRS 轨迹和小步真机测试验证整条 FK/IK 链。

## 3. 新手五分钟启动

### 3.1 准备环境与数据

在项目虚拟环境中安装：

```bash
python -m pip install -e .
```

先录制一个单一任务数据集。建议从 50 条干净示范开始，固定相机、光照、TCP、FPS 和任务文字；第一轮使用
`action_mode=joint`、`observation_mode=all`。录制完成后先执行：

```bash
fafu-arm-dataset check --root ./datasets/fafu_demo --action-mode joint --episode 0
```

### 3.2 预检并生成命令

下面命令不会启动 GPU，只检查数据并打印最终 `lerobot-train` 命令，因此可以安全地先运行：

```bash
fafu-arm-train act --dataset-root ./datasets/fafu_demo --dataset-repo-id FAFU-Robotics/fafu_demo --action-mode joint --output-dir ./outputs/train/act_fafu_joint --device cuda
```

通过后在同一命令末尾增加 `--run`：

```bash
fafu-arm-train act --dataset-root ./datasets/fafu_demo --dataset-repo-id FAFU-Robotics/fafu_demo --action-mode joint --output-dir ./outputs/train/act_fafu_joint --device cuda --run
```

未指定 `--device` 时由 LeRobot 自动选择可用设备；CPU 可以用于流水线冒烟测试，但完整视觉 ACT 训练通常应使用 GPU。
默认值与官方基线接近：100,000 updates、batch 8、action chunk 100。FAFU 入口把 `n_action_steps` 设为 10，表示
部署时执行 10 帧就重新观察，而不是官方默认的 100 帧；30 Hz 下约每 0.33 秒闭环一次，更适合作为首轮真机设置。

训练结果保存在：

```text
outputs/train/act_fafu_joint/
├── checkpoints/<step>/
└── checkpoints/last/pretrained_model/
```

输出目录已存在时，新训练会被拒绝，防止覆盖旧实验。换一个目录，或按第 6 节断点续训。

### 3.3 隐私默认值

入口总是显式生成：

```text
--policy.push_to_hub=false --save_checkpoint_to_hub=false --wandb.enable=false
```

模型只在明确指定下上传，并默认创建私有仓库：

```text
--push-to-hub --policy-repo-id FAFU-Robotics/act_fafu_joint
```

只有确认可以公开时才再加 `--public`。W&B 也属于外部服务，只有确认配置、任务名和指标允许离开实验室时才加
`--wandb`。训练模型本身可能记忆画面或实验信息，不应因为它不是原始视频就默认公开。

## 4. 三种表示的训练命令

三条命令的主要区别只是数据集和 `--action-mode`；相机名、action 字段及部署时的 robot mode 必须与各自数据一致。

```bash
# 绝对关节目标：首选基线
fafu-arm-train act --dataset-root ./datasets/task_joint --dataset-repo-id FAFU-Robotics/task_joint --action-mode joint --output-dir ./outputs/train/act_joint --device cuda --run

# 顺序 EE 增量：确认 FK/IK、坐标系、步长和回放后使用
fafu-arm-train act --dataset-root ./datasets/task_ee_delta --dataset-repo-id FAFU-Robotics/task_ee_delta --action-mode ee_delta --output-dir ./outputs/train/act_ee_delta --device cuda --run

# 绝对 EE 位姿：固定 base/TCP 且 IK 稳定时使用
fafu-arm-train act --dataset-root ./datasets/task_ee_pose --dataset-repo-id FAFU-Robotics/task_ee_pose --action-mode ee_pose --output-dir ./outputs/train/act_ee_pose --device cuda --run
```

公平比较时应固定示范次数、任务初始状态、相机、训练 steps、seed 和评估起点；每个模型至少做 10 次独立真机评估，
报告成功率、人工干预次数、IK 失败和超时，而不只比较训练 loss。

## 5. 验证、观察训练与调参

### 5.1 推荐的第一组实验

先只改变一个变量：

1. `joint`，默认 ACT 参数，50 条高质量示范；
2. 同一配置换 3 个 seed，确认结果不是偶然；
3. 再比较 `ee_delta` 或 `ee_pose`，不要同时改网络大小、相机和学习率；
4. 数据质量稳定后才扩大模型或加入数据增强。

LeRobot 0.6 可以按 episode 留出离线验证集：

```text
--eval-split 0.1 --eval-steps 5000
```

例如追加到 `fafu-arm-train act`。少量数据时，留出集会明显减少训练示范；至少 50 条再考虑 10% 留出。验证 loss 只说明
对未参与训练的示范拟合程度，不等于真机成功率，最终仍要固定条件做 rollout。

### 5.2 调参顺序

| 参数 | 当前基线 | 何时调整 | 建议 |
|---|---:|---|---|
| `steps` | 100,000 | loss 仍下降且真机持续改善 | 先比较 50k/100k/200k checkpoint，防止只看最后一个 |
| `batch_size` | 8 | 显存富余或 OOM | 先试 8；OOM 降到 4/2，显存足可升 16/32；改变后重新比较学习率 |
| `chunk_size` | 100 帧 | 动作持续时间明显更短/长 | 30 Hz 下 30/60/100 分别约 1/2/3.3 秒预测范围 |
| `n_action_steps` | 10 帧 | 闭环频率与推理速度不平衡 | 越小越频繁看图、算力要求越高；必须 `<= chunk_size` |
| `policy.optimizer_lr` | `1e-5` | loss 不降或发散 | 依次小范围比较 `1e-5`、`3e-5`；不要同时改模型规模 |
| `policy.dropout` | 0.1 | 小数据明显过拟合 | 试 0.1/0.2；优先增加数据多样性 |
| `policy.kl_weight` | 10 | VAE 重构与多模态行为问题 | 初始不改；修改时同时记录 L1 与 KL 分量 |
| `policy.dim_model` | 512 | 数据和算力足、确定欠拟合 | 维度必须与 attention heads 合理整除；模型变大不替代好数据 |

高级参数通过 `--set KEY=VALUE` 原样交给 LeRobot，仍保留数据和隐私检查：

```bash
fafu-arm-train act --dataset-root ./datasets/fafu_demo --dataset-repo-id FAFU-Robotics/fafu_demo --action-mode joint --output-dir ./outputs/train/act_joint_lr3e5 --device cuda --set policy.optimizer_lr=3e-5 --set policy.dropout=0.2 --run
```

支持 AMP 的 GPU 可加 `--amp` 降低显存和提高吞吐；第一次先不用，确认基线无 NaN 后再比较。若希望使用 ACT temporal
ensembling，LeRobot 要求每帧重新推理：

```text
--n-action-steps 1 --temporal-ensemble-coeff 0.01
```

它通常更平滑，但显著增加推理频率；必须先测量训练机/部署机的端到端延迟，不能只看离线输出。

## 6. Checkpoint、续训与真机评估

LeRobot checkpoint 同时保存 policy、processor、优化器、step 和训练配置。续训应直接使用官方命令，不要把
`pretrained_model` 当作一个全新的实验：

```bash
lerobot-train --config_path=outputs/train/act_fafu_joint/checkpoints/last/pretrained_model/train_config.json --resume=true
```

需要改变总 steps 时可在续训命令增加相应官方覆盖参数。尽量保持 batch size 和进程数不变；LeRobot 0.6 虽能恢复
数据位置，但改变 world size/batch size 后不能保证每个 rank 的样本顺序完全一致。

真机评估先用本地 checkpoint、低速、小动作上限和随时可用的急停。`robot.action_mode` 必须与训练数据一致，相机名称、
数量、分辨率和顺序必须与训练时一致：

```bash
lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/act_fafu_joint/checkpoints/last/pretrained_model \
  --robot.type=fafu_follower \
  --robot.id=fafu_follower \
  --robot.port=/dev/serial/by-id/<follower-device> \
  --robot.action_mode=joint \
  --robot.max_relative_target=0.03 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task="pick and place" \
  --duration=30
```

评估顺序是：离线数据预检 → WRS/视频抽查 → 无负载或软物体低速测试 → 固定初始条件的正式统计。策略运行前确认
急停、工作空间、软限位和 watchdog；EE 模式还要记录 IK 失败、限幅次数和 TCP 边界命中率。

## 7. 常见坑

- **训练时换 action mode 不会转换数据。** ACT 自动适配 action 维度，但不会理解字段语义；预检失败应修数据，而不是
  绕过检查。
- **不要直接训练 `all`。** 三套冗余 action 会改变损失权重，也无法唯一决定真机控制源。
- **delta 不等于 relative。** 顺序 delta 的误差会累积；chunk 越长、重观察越慢，风险越明显。
- **EE 标签正确不代表 EE 控制可靠。** URDF、0.175 m TCP、关节零位、rotvec 约定、base 坐标和 pytracik seed
  任一不一致都会造成系统偏差；接近旋转角 π 时 rotvec 还可能发生表示跳变。
- **相机必须可复现。** 训练和部署的相机 key、数量、shape、安装位置、曝光和裁剪保持一致；模型不会自动识别两路相机
  被交换。
- **统计量必须对应当前数据。** 筛字段、拼数据或改变 action 表示后要重新生成 LeRobot stats；沿用旧 stats 会错误归一化。
- **ACT chunk 跨 episode 尾部会 padding。** 官方模型使用 `action_is_pad` 处理；后续自定义 loss 必须继续屏蔽 padding。
- **短 loss 不等于好策略。** 示范中的停顿、抖动、失败恢复和单一起点都可能被忠实学会；先修数据再扩模型。
- **本地训练也要同时传 repo id 和 root。** `repo_id` 是数据身份，`root` 指向具体本地树；遗漏 root 可能触发 Hub 查找。
- **不要覆盖输出目录。** 每个动作表示、seed 和关键参数使用独立目录，并保存完整命令、git commit 和 LeRobot 版本。
- **版本不要漂移。** 可复现实验锁定 LeRobot 0.6.x 小版本；升级后先做一次数据加载和 100-step 冒烟训练。

## 8. 如何修改 ACT，及以后增加 Diffusion/DP3

### 8.1 只改配置，不改模型代码

学习率、chunk、执行步数、dropout、层数、隐藏维度、VAE 和 KL 权重都属于配置实验，使用当前
`fafu-arm-train act` 参数或 `--set` 即可。先建立不可变基线，不需要 fork LeRobot。

### 8.2 真正修改网络或 loss

当要增加新输入、改变 encoder/decoder、加入辅助 loss 或改变 action head 时，按 LeRobot 官方的 out-of-tree policy
plugin 规范创建独立包，建议放在仓库未来的 `policies/` 工作区：

```text
policies/lerobot_policy_fafu_act/
├── pyproject.toml                         # distribution 名必须以 lerobot_policy_ 开头
└── src/lerobot_policy_fafu_act/
    ├── __init__.py
    ├── configuration_fafu_act.py          # @register_subclass("fafu_act")
    ├── modeling_fafu_act.py               # name = "fafu_act"
    └── processor_fafu_act.py              # make_fafu_act_pre_post_processors
```

三个名字必须一致，安装后即可使用 `--policy.type=fafu_act`。建议从官方 ACT 的接口而不是旧项目代码开始，并做到：

1. 固定并记录所基于的 LeRobot tag/commit，保留上游 Apache-2.0 版权声明；
2. config 显式校验 action/image/state feature；model 实现 `forward`、`select_action`、`predict_action_chunk`、`reset`；
3. processor 保持训练归一化与部署反归一化完全对称；
4. 自定义 loss 正确处理 `action_is_pad`，并分别记录主 loss、辅助 loss 和 KL；
5. 增加 shape、单步训练、保存/加载、processor round-trip 和真机输出限幅测试；
6. 用相同数据、seed、steps 和评估起点与官方 ACT 比较，确认修改带来的收益不是数据或配置差异。

稳定后在 `training/` 增加 `fafu_act.py` 适配及 CLI 子命令。这样硬件接口不需要改，官方 ACT 基线也始终可用。

### 8.3 后续算法

- **LeRobot Diffusion Policy**：优先复用官方 `policy.type=diffusion`，在 `training/diffusion.py` 增加依赖检查、默认参数和
  同一套 dataset preflight。它擅长多模态动作分布，但训练/推理成本通常高于 ACT。
- **DP3**：不是把 RGB ACT 的 policy type 改名即可。DP3 需要深度相机、相机内外参、深度尺度、点云裁剪/采样、3D
  encoder 和点云数据 schema；应作为独立 `lerobot_policy_fafu_dp3` 插件，并先扩展采集/processor，再接训练入口。
- **其他算法**：每个算法一个 `training/<algorithm>.py`，共用 `common.py`；算法私有的重依赖放 optional extra，不能让
  安装机械臂驱动时被迫安装全部训练栈。

## 参考资料

- [LeRobot 官方 ACT 指南](https://huggingface.co/docs/lerobot/act)
- [LeRobot 官方真实机器人模仿学习流程](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobot 官方 Action Representations](https://huggingface.co/docs/lerobot/action_representations)
- [LeRobot 官方 Adding a Policy](https://huggingface.co/docs/lerobot/bring_your_own_policies)
- [ACT 原论文：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy 原论文](https://arxiv.org/abs/2303.04137)
- [DP3 原论文](https://arxiv.org/abs/2403.03954)
