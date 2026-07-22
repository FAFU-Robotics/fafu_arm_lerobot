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
    ├── config_file.py                         # 严格、版本化的 YAML 读取
    ├── act.py                                 # ACT 参数和官方命令适配
    └── cli.py                                 # fafu-arm-train；以后增加 diffusion/dp3 子命令
configs/train/
├── act_baseline.yaml                          # 官方 ACT 的 FAFU 可复现实验基线
└── fafu_act_demo.yaml                         # 网络修改 demo 的实验配置
policies/lerobot_policy_fafu_act_demo/          # 独立的 ACT 网络修改示例
```

参数实验放 YAML，网络结构实验放独立 policy plugin；不要修改虚拟环境中的 `site-packages/lerobot`。仓库已包含一个
可以训练的残差 action head demo，具体做法见第 8 节。

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

正式实验推荐复制并修改 `configs/train/act_baseline.yaml`。至少设置 `dataset.root`、`dataset.repo_id`、
`dataset.action_mode` 和一个全新的 `run.output_dir`。然后先 dry-run；它不会启动 GPU，只检查数据并打印最终
`lerobot-train` 命令：

```bash
fafu-arm-train act --config configs/train/act_baseline.yaml
```

检查通过后启动训练：

```bash
fafu-arm-train act --config configs/train/act_baseline.yaml --run
```


命令行值会覆盖 YAML，适合临时做一个单变量实验：

```bash
fafu-arm-train act --config configs/train/act_baseline.yaml --seed 1001 --output-dir ./outputs/train/act_joint_seed1001 --run
```

不使用 YAML 也仍然支持完整命令行：

```bash
fafu-arm-train act --dataset-root ./datasets/fafu_demo --dataset-repo-id FAFU-Robotics/fafu_demo --action-mode joint --output-dir ./outputs/train/act_fafu_joint --device cuda
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

1. 保留 `act_baseline.yaml` 不变，复制为实验 YAML；
2. `joint`、默认 ACT 参数、至少 50 条高质量示范作为基线；
3. 同一配置换 3 个 seed，确认结果不是偶然；
4. 再比较 `ee_delta` 或 `ee_pose`，不要同时改网络大小、相机和学习率；
5. 数据质量稳定后才扩大模型或加入数据增强。

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

建议按下面四轮推进，每轮只保留表现更好的设置：

1. **数据/控制基线**：固定数据版本、相机、FPS、action mode、任务文本和评估起点；跑 seed 1000/1001/1002。
2. **优化器**：先比较 `optimizer_lr: 0.00001` 与 `0.00003`。如果 batch 改变，重新比较学习率。
3. **时间参数**：按任务持续时间比较 `chunk_size: 30/60/100`，再比较 `n_action_steps: 1/5/10/20`；后者不能大于前者。
4. **容量与正则**：确认欠拟合后再比较 `dim_model: 256/512`、`dropout: 0.1/0.2`；最后才动 `kl_weight: 1/10/20`。

每个实验至少记录：Git commit、YAML、数据集版本或哈希、seed、GPU、训练时长、最优/最后 checkpoint、离线 loss，及固定
初始条件下的真机成功次数。不要只根据训练 loss 选模型；关注成功率、完成时间、碰撞/越界次数和动作抖动。

### 5.3 为什么使用 YAML

YAML 比一长串命令更适合正式调参：可以 code review、提交版本、复现实验，也能清楚地区分数据、运行、策略、评估、外部
日志和上传设置。本项目的 YAML schema 会拒绝未知字段，拼错 `chunk_size` 不会静默使用默认值；`policy.extra` 则用于
自定义 policy 的标量参数。示例：

```yaml
run:
  output_dir: ./outputs/train/act_joint_lr3e5_seed1001
  seed: 1001
policy:
  optimizer_lr: 0.00003
  dropout: 0.2
```

仓库当前不提供 YAML 继承，避免多层配置合并后不知道最终值。每次实验复制基线文件、给输出目录和文件名加入变量与 seed，
训练前查看 dry-run 打印的最终命令。

少量临时覆盖仍可用 `--set KEY=VALUE`，并继续保留数据和隐私检查：

```bash
fafu-arm-train act --config configs/train/act_baseline.yaml --output-dir ./outputs/train/act_joint_lr3e5 --set policy.optimizer_lr=3e-5 --run
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

学习率、chunk、执行步数、dropout、层数、隐藏维度、VAE 和 KL 权重都属于配置实验，直接复制并修改
`configs/train/act_baseline.yaml`，不需要 fork LeRobot。只有“参数相同但计算图或 loss 要改变”时才写新 policy。

### 8.2 可运行 demo：给 ACT 增加残差 action head

仓库中的 `lerobot_policy_fafu_act_demo` 继承官方 `ACTPolicy`，保留其视觉编码器、Transformer、VAE、loss、padding 处理和
processor，只把官方线性 action head 包装为：

```text
ACT decoder feature h ── official Linear ───────────────┐
                      └─ LayerNorm → MLP → × scale ────┼─→ action
                                                        ┘
```

核心代码是：

```python
self.model.action_head = ResidualActionHead(
    base_head=self.model.action_head,
    feature_dim=config.dim_model,
    action_dim=config.action_feature.shape[0],
    ...,
)

# ResidualActionHead.forward
return self.base_head(features) + self.residual_scale * self.residual(features)
```

残差分支最后一层使用零初始化，所以刚创建时输出与官方 head 相同，随后再学习修正。这让“结构修改”可以和官方 ACT 做公平
对照，但它不代表未训练模型可以安全上真机。

安装 demo（LeRobot 0.6 环境）：

```bash
python -m pip install -e policies/lerobot_policy_fafu_act_demo
fafu-arm-train act --config configs/train/fafu_act_demo.yaml
fafu-arm-train act --config configs/train/fafu_act_demo.yaml --run
```

LeRobot 会发现名称以 `lerobot_policy_` 开头的已安装 distribution，并根据 `policy.type: fafu_act_demo` 加载它。训练、保存、
断点续训和 rollout 仍走官方命令。先用相同数据、seed、steps 对比 `act_baseline.yaml`，不要一开始同时改变网络和超参数。
该 demo 的自有 checkpoint 支持正常保存/加载；它不直接兼容 `policy.type=act` 的 action-head key。若要从官方 ACT
checkpoint 微调，应写一次显式迁移：复制所有 shape 相同的权重，并把官方 `action_head.weight/bias` 映射到
`action_head.base_head.weight/bias`，然后记录未加载的残差层；不要使用 `strict=False` 后忽略报告。

### 8.3 修改哪个文件

```text
policies/lerobot_policy_fafu_act_demo/
├── pyproject.toml
└── src/lerobot_policy_fafu_act_demo/
    ├── configuration_fafu_act_demo.py     # 新参数、默认值和范围校验
    ├── modeling_fafu_act_demo.py          # 网络层；当前 demo 修改 action head
    └── processor_fafu_act_demo.py         # 训练归一化和推理反归一化
```

- **改 action head**：修改 `ResidualActionHead`，同时保持输入最后一维为 `dim_model`、输出最后一维为 action dimension。
- **加 loss**：在 policy 中覆盖 `forward`；继续屏蔽 `batch["action_is_pad"]`，并分别返回主 loss、KL 和新增 loss 指标。
- **加状态/图像输入**：先在 config/processor 声明并归一化 feature，再修改模型 token/encoder；不能只在网络里读一个未声明字段。
- **大改 Transformer/VAE**：不要继续层层 monkey-patch。以固定的 LeRobot 0.6 tag 为基础实现自己的 model 类，保留许可证，
  但仍复用 `PreTrainedPolicy` 保存/加载契约和 processor 接口。

每个新结构应增加：输出 shape、零/随机输入前向、单步反向、padding loss、保存后加载输出一致、processor round-trip 测试。
配置类名、模型类名、processor 工厂名和 `policy.type` 必须遵循 LeRobot 的命名约定，否则动态加载会失败。改变参数名称或结构后，
旧 checkpoint 也可能无法严格加载；此时应升级自定义 policy 版本并明确迁移方式，不要悄悄复用同一类型名。

### 8.4 后续算法

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
