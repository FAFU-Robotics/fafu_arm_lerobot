# ACT Policy Training

本文是 FAFU Arm 的 ACT 训练、评估和模型修改指南。以下命令以 **LeRobot 0.6.x** 为准；硬件接口支持旧版 LeRobot，不代表训练参数兼容。正式实验应锁定并记录完整的小版本号。

数据录制、动作字段和 EE 坐标定义见 [Data Collection 指南](DATA_COLLECTION.md)，训练机与真机安装见 [部署指南](DEPLOYMENT.md)。

下文含行尾 `\` 的多行命令使用 Bash；PowerShell 请改为单行，或把行尾 `\` 换成反引号（反引号后不能有空格）。

## 1. 训练边界与动作选择

本项目复用 LeRobot 官方 `ACTConfig`、`ACTPolicy`、processor 和 `lerobot-train`。`fafu-arm-train` 负责数据预检、严格 YAML 读取、隐私默认值和官方命令生成，不复制 ACT 实现。参数实验写入 YAML；改变计算图或 loss 时使用独立 policy plugin，不要修改虚拟环境中的 `site-packages/lerobot`。

| 模式 | ACT 学习的 action | 优点 | 主要风险 | 使用顺序 |
|---|---|---|---|---|
| `joint` | 6 个绝对关节目标 + 绝对夹爪位置 | 直接对应控制器，无 FK/IK 误差，最容易定位问题 | 与机器人构型相关 | **首个基线，默认推荐** |
| `ee_delta` | TCP 顺序增量 + 绝对夹爪位置 | 局部动作接近零，适合小范围精细操作 | 增量误差累积；依赖 FK/IK、TCP、坐标系和 FPS | joint 基线稳定后对照 |
| `ee_pose` | `base_link` 下绝对 TCP 位姿 + 绝对夹爪位置 | 目标直观，不累计增量 | 依赖可靠标定；IK 边界/奇异点；rotvec 接近 π 时不连续 | 固定工位且标定可靠时评估 |

推荐顺序：`joint` 建立可复现基线 → `ee_delta` 做同规模对照 → 标定可靠后评估 `ee_pose`。三种表示必须分开录制、分开训练，并使用相同任务划分和评估条件。

精确的 `ee_delta` 公式、平移/旋转坐标系、保存时序和 `ee_pose` 字段见 [Data Collection：Action 模式](DATA_COLLECTION.md#22-action-模式)。本项目的 `ee_delta` 是相邻控制帧之间的顺序增量，不等同于 LeRobot 某些策略使用的 relative action。

`ee_pose` 的旋转使用 axis-angle rotvec。其模长接近 π 时存在分支不连续，同一姿态可能出现符号跳变；示范应避免跨越该分支，评估旋转误差应比较相对旋转，而不是逐分量相减。

`action_mode=all` 只用于归档和离线分析。训练入口会拒绝 `all`；ACT 每次训练只能使用一种明确的 action schema。

pytracik 不参与神经网络反向传播。录制时 FK 生成 EE 数据，真机执行 `ee_delta` 或 `ee_pose` 时 IK 把策略输出转换为关节目标。使用 EE action 前必须通过 URDF/TCP、回放、WRS 轨迹和小步真机测试验证整条 FK/IK 链。

## 2. 第一个可复现实验

### 2.1 环境与数据预检

在训练环境中安装项目并确认 LeRobot 版本：

```bash
python -m pip install -e ".[train]"
python -c "import importlib.metadata as m; print(m.version('lerobot'))"
```

第二条命令应输出已验证的 `0.6.x` 小版本。升级版本后必须重新执行数据加载和 100-step 冒烟训练。

基线 YAML 使用 `run.device: cuda`，视觉 backbone 首次训练会下载 ImageNet 权重。训练前确认目标 `cuda:N` 可用；无 CUDA 时可把设备改为 `cpu` 做慢速冒烟。离线机器应预先缓存权重；把 `policy.pretrained_backbone_weights` 设为 `null` 会改成随机初始化，属于不同实验，必须另存 YAML 并记录。推理可用 `--device` 明确设备，但任何设备都必须通过实时延迟预检。

第一轮建议使用至少 50 条干净的单任务示范，固定相机、光照、TCP、FPS 和任务文本，并选择 `action_mode=joint`、`observation_mode=all`。训练前检查数据：
当前 LeRobot 0.6 ACT 训练入口要求 `observation.state` 和至少一个三通道 RGB 相机特征；visual-only、灰度或单通道深度输入会在预检中被拒绝。

```bash
fafu-arm-dataset check --root ./datasets/fafu_demo --action-mode joint --episode 0
```

检查通过不代表视频和全部 episode 都正确；录制质量验收按 [Data Collection：检查范围与验收](DATA_COLLECTION.md#51-检查范围与验收)完成。

### 2.2 复制完整基线 YAML

`configs/train/act_baseline.yaml` 是完整、可直接运行的基线。保留原文件，复制一份实验配置：

```bash
# Linux
cp configs/train/act_baseline.yaml configs/train/act_joint_seed1000.yaml
# PowerShell
Copy-Item configs/train/act_baseline.yaml configs/train/act_joint_seed1000.yaml
```

在副本中至少核对以下字段：

| 字段 | 首轮建议 | 要求 |
|---|---|---|
| `dataset.root` | `./datasets/fafu_demo` | 指向实际本地数据目录 |
| `dataset.repo_id` | `FAFU-Robotics/fafu_demo` | 与数据元信息身份一致；本地训练也必须保留 |
| `dataset.action_mode` | `joint` | 必须与 `features.action.names` 一致 |
| `dataset.urdf_path` | `null` | `null` 使用随包 URDF；自定义时必须填写采集所用的同一文件 |
| `run.output_dir` | `./outputs/train/act_fafu_joint_seed1000` | 每个实验使用全新目录 |

不要删除基线 YAML 中其他组。该文件同时固定网络、优化器、评估、W&B 和 Hub 设置。

### 2.3 Dry-run、冒烟和正式训练

先做 dry-run；它只检查数据并打印最终 `lerobot-train` 命令，不启动训练：

```bash
fafu-arm-train act --config configs/train/act_joint_seed1000.yaml
```

再用独立输出目录完成 100-step 流水线冒烟：

```bash
fafu-arm-train act --config configs/train/act_joint_seed1000.yaml --steps 100 --output-dir ./outputs/train/smoke_act_fafu_joint_seed1000 --run
```

冒烟通过标准：预检无错误、数据和相机批次可读取、loss 为有限值、训练正常退出。它不能证明策略有效。

正式训练使用完整 YAML：

```bash
fafu-arm-train act --config configs/train/act_joint_seed1000.yaml --run
```

基线结果统一保存在：

```text
outputs/train/act_fafu_joint_seed1000/
├── fafu_inference_manifest.json
└── checkpoints/last/pretrained_model/
    └── fafu_inference_manifest.json
```

入口拒绝覆盖已有输出目录。新实验更换 YAML 文件名和 `run.output_dir`；需要续训时按第 5 节操作。

### 2.4 隐私默认值

入口始终显式生成：

```text
--policy.push_to_hub=false --save_checkpoint_to_hub=false --wandb.enable=false
```

模型只在明确授权后上传，新仓库默认私有：

```bash
fafu-arm-train act --config configs/train/act_joint_seed1000.yaml --push-to-hub --policy-repo-id FAFU-Robotics/act_fafu_joint --run
```

上传前，入口会查询当前 token 可访问的已有 Hub 模型仓库的真实 visibility；若它与本次请求的 private/public 状态不一致就拒绝继续，也不会借创建仓库改变已有仓库的可见性。新仓库默认私有；查询或上传需要具备目标仓库读写权限。只有确认模型和训练信息可以公开时才增加 `--public`。W&B 也是外部服务，仅在允许实验配置、任务名和指标离开实验室时启用 `--wandb`。

显式上传时，LeRobot 完成训练发布后，入口会选择 `checkpoints/last`（不可用时选择最大数字 step），验证 checkpoint manifest 与全部受绑定文件，再把模型、processor 及 manifest 作为一次 Hub commit 提交。若该提交失败，命令会报错但本地输出保留；远端 checkpoint 下载后仍须先通过无硬件预检，才能进入真机验收。

## 3. 切换动作表示

复制完整基线 YAML，只修改数据身份、动作模式和输出目录。以下是**字段片段，不是完整训练配置**：

```yaml
dataset:
  repo_id: FAFU-Robotics/task_ee_delta
  root: ./datasets/task_ee_delta
  action_mode: ee_delta
run:
  output_dir: ./outputs/train/act_fafu_ee_delta_seed1000
```

| 目标 | `dataset.action_mode` | 使用前验收 |
|---|---|---|
| 关节基线 | `joint` | 低速回放、关节限位和夹爪方向正确 |
| EE 增量 | `ee_delta` | FK/IK、坐标系、FPS、步长、workspace 和回放全部通过 |
| 绝对 EE | `ee_pose` | base/TCP 标定固定，工作空间内 IK 连续且稳定 |

不能只修改 YAML 中的模式来转换旧数据。预检失败时应重新生成符合目标 schema 的数据集；不要绕过字段检查。`all` 无对应训练配置。

## 4. 调参与实验管理

### 4.1 YAML 规则

每个正式实验都复制完整基线 YAML，只改变一个变量，并使用独立输出目录。YAML schema 会拒绝未知字段；仓库不提供 YAML 继承，避免多层合并掩盖最终值。少量临时覆盖可用 `--set KEY=VALUE`，但最终配置仍应回写到实验 YAML。

下面同样只是调参片段：

```yaml
run:
  output_dir: ./outputs/train/act_joint_lr3e5_seed1001
  seed: 1001
policy:
  optimizer_lr: 0.00003
```

### 4.2 参数起点

| 参数 | 基线 | 何时调整 | 建议范围或约束 |
|---|---:|---|---|
| `run.steps` | 100,000 | loss 仍下降且真机继续改善 | 比较 50k/100k/200k checkpoint |
| `run.batch_size` | 8 | OOM 或显存富余 | 2/4/8/16；改变后重新比较学习率 |
| `policy.chunk_size` | 100 | 任务动作持续时间不同 | 30 Hz 下 30/60/100 约对应 1/2/3.3 秒 |
| `policy.n_action_steps` | 10 | 闭环频率或推理延迟不合适 | 1/5/10/20，且必须 `<= chunk_size` |
| `policy.optimizer_lr` | `1e-5` | loss 不降或发散 | 先比较 `1e-5`、`3e-5` |
| `policy.dropout` | 0.1 | 小数据过拟合 | 0.1/0.2；优先提高数据多样性 |
| `policy.kl_weight` | 10 | VAE 行为需要分析 | 最后比较 1/10/20，并记录 L1 与 KL |
| `policy.dim_model` | 512 | 数据充足且确认欠拟合 | 256/512；必须能被 `n_heads` 整除 |

### 4.3 单一调参顺序

1. 完成 100-step 冒烟，再固定数据版本、相机 schema、FPS、action mode、任务文本和评估起点。
2. 用 joint 基线运行 seed 1000、1001、1002；不要只保留最好的一次。
3. 先比较学习率；batch size 仅因显存调整，改变后重新比较学习率。
4. 再比较 `chunk_size` 和 `n_action_steps`，测量部署端到端延迟。
5. 确认欠拟合后再改变网络容量、dropout 或 KL 权重。
6. 最后在同数据规模、steps、seed 和评估条件下比较 `ee_delta` 或 `ee_pose`。

基线的 `evaluation.eval_split=0.1`、`eval_steps=5000` 适合约 50 条以上示范的起始实验。数据很少时，留出集会明显减少训练样本。离线 loss 只反映对示范的拟合，不能替代真机成功率。

每个实验至少保存：Git commit、完整 YAML、数据集版本或哈希、LeRobot 小版本、seed、GPU、训练时长、最优/最后 checkpoint、离线指标，以及固定条件下的真机结果。真机比较至少使用 3 个 seed，每个模型至少 10 次独立 rollout，并报告成功率、完成时间、人工干预、碰撞/越界、超时和动作抖动。

### 4.4 Temporal ensembling

ACT temporal ensembling 要求每帧重新推理：

```yaml
policy:
  n_action_steps: 1
  temporal_ensemble_coeff: 0.01
```

它可能使动作更平滑，但会提高推理频率。启用前必须测量部署设备延迟，并与不启用的相同 seed 基线比较。

## 5. Checkpoint、续训与真机评估

LeRobot checkpoint 保存 policy、processor、优化器、step 和训练配置。基线续训命令为：

```bash
lerobot-train --config_path=outputs/train/act_fafu_joint_seed1000/checkpoints/last/pretrained_model/train_config.json --resume=true
```

续训尽量保持 batch size、进程数和数据不变；改变 world size 或 batch size 后，样本顺序不保证完全一致。不要把 `pretrained_model` 当成全新实验重新训练。

LeRobot 的 `checkpoints/last` 通常是符号链接；Windows 未启用开发者模式或没有创建链接权限时可能失败或缺失。此时从 `checkpoints/` 选择实际的数字 step 目录用于续训和推理，不要假定 `last` 存在。

官方 resume 绕过了本项目的训练收尾。续训成功后必须同步并重新校验所有 checkpoint manifest：

```bash
fafu-arm-train act --config configs/train/act_joint_seed1000.yaml --sync-manifest
```

`--sync-manifest` 不启动训练，不能与 `--run` 或 `--json` 同用；它重新核对数据、action mode 和 URDF，并为已有 `pretrained_model` 计算文件哈希。若还要发布续训结果，目标 Hub 模型仓库必须已存在且 visibility 匹配，并在该命令上再次显式增加 `--push-to-hub --policy-repo-id ...`（公开仓库再加 `--public`）；入口会原子提交受绑定的 checkpoint 文件和 manifest，不只是上传清单。

### 5.1 无硬件推理预检

训练成功后，`fafu-arm-train` 会在输出根写入未绑定的 schema 模板，并在每个 `pretrained_model` 写入 checkpoint 专属 manifest。除 state/action 字段顺序、相机 schema、FPS、action mode、robot type、URDF SHA-256 和 base/tool link 外，checkpoint manifest 还用 SHA-256 绑定 `config.json`、模型权重、pre/postprocessor JSON 及其引用的 safetensors。推理会在加载权重前核对文件集合和哈希，拒绝缺失、篡改或来自另一 checkpoint 的文件。

先复制并修改相机配置；声明的名称、数量、顺序、分辨率和 FPS 必须与训练数据一致：

```bash
cp configs/inference/opencv_camera.yaml configs/inference/lab_camera.yaml
```

无硬件预检不会打开相机，因此无法确认物理设备。真机运行前必须通过预览或短录制人工核对：每个 key 对应正确相机、画面确为三通道 RGB 且颜色/方向正确、实测 FPS 与 manifest 一致。

然后执行不连接硬件的完整预检：

```bash
fafu-arm-infer act \
  --checkpoint outputs/train/act_fafu_joint_seed1000/checkpoints/last/pretrained_model \
  --cameras configs/inference/lab_camera.yaml \
  --task "pick and place"
```

它会使用 `strict=True` 加载权重和 checkpoint 自带的 pre/postprocessor，逐项核对 policy、manifest 与机器人 schema，再用合成 state/image 完成两次完整前向并 reset。完整 chunk 的暖机延迟必须低于控制周期和 watchdog 较小者的 80%；预检不会加载 FAFU SDK、打开相机或串口。`examples/act_inference.py` 是同一生产入口的可读 Python 示例。

旧 checkpoint 没有 manifest 时，必须显式提供原训练数据、action mode 和采集时使用的 URDF；即使它等同于当前随包模型，也不能省略 `--urdf-path`，更不能按 7 维 shape 猜测：

```bash
fafu-arm-infer act \
  --checkpoint /path/to/legacy/pretrained_model \
  --dataset-root ./datasets/fafu_demo \
  --action-mode joint \
  --urdf-path /path/to/training_fafu_arm.urdf \
  --cameras configs/inference/lab_camera.yaml
```

### 5.2 有限时长真机推理

只有预检和人工相机核对都通过后才增加 `--run`。把 `FOLLOWER_DEVICE` 换成 `ls -l /dev/serial/by-id/` 显示的真实文件名，并把七个起始值换成现场确认的弧度值：

```bash
fafu-arm-infer act \
  --checkpoint outputs/train/act_fafu_joint_seed1000/checkpoints/last/pretrained_model \
  --cameras configs/inference/lab_camera.yaml \
  --port /dev/serial/by-id/FOLLOWER_DEVICE \
  --task "pick and place" \
  --start-joints J1_RAD J2_RAD J3_RAD J4_RAD J5_RAD J6_RAD GRIPPER_RAD \
  --max-relative-target 0.03 \
  --servo-max-velocity 0.3 \
  --duration 5 \
  --run
```

`--start-joints` 是首次观测的授权包络，不会自动把机械臂移动到该姿态；默认关节/夹爪容差分别为 0.15/0.25 rad，超出时会在第一条策略动作前停止。只有 EE pose、没有关节位置的 observation schema 改用 `--start-ee`。

action mode、observation mode、速度/力矩字段和控制 FPS 由 manifest 严格确定。每个实时 state/image 帧都会检查字段、shape、dtype、数值范围和 NaN/Inf；输出经过 checkpoint postprocessor 反归一化、有限值与字段顺序检查后，只能通过 `FafuFollower.send_action()` 下发，因此继续受关节/笛卡尔步长、URDF 限位、IK、workspace 和 servo watchdog 保护。连续超期或真实一秒窗口内超期达到阈值会停止；正常结束、任何异常和 `Ctrl+C` 都会以 `joint_release=brake` 断开，不会把关节切到自由状态。

所有 action mode 都会核对 URDF 哈希；使用自定义 URDF 的 checkpoint（包括 `joint`）必须传入训练/采集时的同一文件。`ee_delta` 与 `ee_pose` 还必须提供经过标定的 workspace，例如 `--urdf-path /path/to/training_fafu_arm.urdf --ee-workspace-min X Y Z --ee-workspace-max X Y Z`。先完成 joint 基线，再按 `ee_delta`、`ee_pose` 顺序低速验收。

LeRobot 0.6.0 的通用 `lerobot-rollout` 当前只把 `.pos` 标量送入 policy；它会遗漏 FAFU 的 EE action，以及 `all` observation 中的速度/EE 字段。因此本项目的三种 action 统一使用 `fafu-arm-infer`，不要以通用 rollout 命令替代 schema 预检。

评估顺序：离线数据预检 → 视频/WRS 抽查 → 无负载或软物体低速测试 → 固定初始条件的正式统计。启动前确认急停、workspace、软限位和 watchdog；EE 模式还要记录 IK 失败、限幅次数和 TCP 边界命中。回放与故障恢复步骤见 [Data Collection 指南](DATA_COLLECTION.md#7-回放前校验和低速回放)。

`fafu-arm-infer` 当前只输出控制时序摘要，不会保存 LeRobot rollout。正式评估必须另行记录每次 rollout 的视频、状态、成功/失败、人工干预和停止原因；需要复现控制细节时还应记录原始预测、安全处理后的下发动作及边界事件。

## 6. 常见问题

- **`root` 和 `repo_id` 缺一不可。** `repo_id` 是数据身份，`root` 指向本地数据树；遗漏 `root` 可能触发 Hub 查找。
- **action mode 不会转换数据。** 模式必须与 `features.action.names` 一致，`all` 会被训练入口拒绝。
- **相机 schema 必须固定。** 当前 ACT 路径只接受三通道 RGB；训练和部署的 key、数量、shape、顺序、安装位置、曝光和裁剪保持一致。
- **数据改变后重建 stats。** 筛字段、拼接数据或更换 action 表示后，旧归一化统计量不能沿用。
- **自定义 loss 必须屏蔽 padding。** ACT chunk 越过 episode 尾部时使用 `action_is_pad`；遗漏会让补齐帧参与损失。
- **输出目录不能复用。** 每个 action mode、seed 和关键参数使用独立目录，并保留完整 YAML。
- **锁定 LeRobot 0.6.x 小版本。** 依赖升级后先做数据预检和 100-step 冒烟，再恢复正式实验。
- **低 loss 不等于策略成功。** 示范中的停顿、抖动和失败动作也会被学习，最终结论来自固定条件的多次真机评估。
- **缺少 inference manifest 时不能猜 action。** 使用原训练数据的 `meta/info.json` 和明确 action mode 重建；字段同为 7 维不代表语义一致。

## 7. 修改 ACT 神经网络

### 7.1 何时写新 policy

学习率、chunk、执行步数、dropout、层数、隐藏维度、VAE 和 KL 权重都属于 YAML 实验。只有改变计算图、输入处理或 loss 时才创建新 policy plugin。

仓库中的 `lerobot_policy_fafu_act_demo` 面向 LeRobot `>=0.6,<0.7`，继承官方 `ACTPolicy`，保留视觉编码器、Transformer、VAE、padding 和 processor，只给线性 action head 增加零初始化残差分支：

```text
ACT feature ── official Linear ─────────────┐
            └─ LayerNorm → MLP → × scale ──┴─→ action
```

零初始化使新建模型的初始输出与官方 head 一致，之后再学习修正。它便于做结构对照，不代表未训练模型可以安全上真机。

### 7.2 运行残差 head demo

```bash
python -m pip install -e policies/lerobot_policy_fafu_act_demo
fafu-arm-train act --config configs/train/fafu_act_demo.yaml
fafu-arm-train act --config configs/train/fafu_act_demo.yaml --run
```

第一条安装 plugin；第二条完成数据预检和 dry-run；第三条训练。LeRobot 根据 `policy.type: fafu_act_demo` 动态加载该 plugin。训练机和推理机都必须安装完全相同的 plugin 版本或 Git commit；checkpoint 不会携带可执行的 plugin 代码，版本不一致会导致类型发现、参数名或处理流程不匹配。正式对照必须使用与 `act_baseline.yaml` 相同的数据、seed、steps 和评估条件。

### 7.3 文件职责与修改规则

| 文件 | 职责 |
|---|---|
| `configuration_fafu_act_demo.py` | 新参数、默认值、范围校验和 `policy.type` |
| `modeling_fafu_act_demo.py` | 网络层和 forward；当前实现残差 action head |
| `processor_fafu_act_demo.py` | 当前仅把调用转发给官方 ACT processor 工厂 |
| `configs/train/fafu_act_demo.yaml` | 可复现的 demo 实验参数 |

当前 processor 示例直接返回 `make_act_pre_post_processors(...)`，只是复用官方归一化/反归一化管线。只改注释、名称或复制这个包装文件不会替换标准处理器；要自定义处理，必须让注册工厂返回新的 pipeline，并保证训练、保存、严格加载和 round-trip 测试使用同一实现。

- 改 action head 时保持输入末维为 `dim_model`、输出末维为 action dimension。
- 增加 loss 时继续屏蔽 `batch["action_is_pad"]`，并分别返回主 loss、KL 和新增指标。
- 增加状态或图像时先在 config/processor 声明和归一化 feature，再修改 token/encoder。
- 大改 Transformer 或 VAE 时实现独立 model 类，并继续遵循 `PreTrainedPolicy` 和 processor 的保存/加载契约。

### 7.4 Checkpoint 与测试门槛

该 demo 自己保存的 checkpoint 可以正常续训和加载，但不直接兼容 `policy.type=act` 的 action-head key。从官方 ACT 迁移时必须显式映射：

```text
model.action_head.weight/bias
→ model.action_head.base_head.weight/bias
```

同时记录未加载的残差参数。不要使用 `strict=False` 后忽略 missing/unexpected keys。改变参数名或结构时应升级自定义 policy 版本，并提供迁移说明。

运行现有测试：

```bash
python -m pytest policies/lerobot_policy_fafu_act_demo/tests
```

结构修改的最低验收包括：输出 shape、零/随机输入前向、单步反向、padding loss、保存/加载输出一致、processor round-trip。训练前仍要完成第 2 节的 100-step 冒烟。

## 8. 后续算法扩展

- **Diffusion Policy**：优先复用 LeRobot 官方 policy，在 `training/diffusion.py` 增加依赖、YAML 适配和同一套数据预检。
- **DP3**：作为独立 plugin；先定义深度尺度、相机内外参、点云裁剪/采样和数据 schema，再实现 3D encoder 与训练入口。
- **其他算法**：每个算法一个 `training/<algorithm>.py`，共用 `training/common.py`；算法私有重依赖放入 optional extra，避免机械臂部署被迫安装完整训练栈。

## 参考资料

- [LeRobot 官方 ACT 指南](https://huggingface.co/docs/lerobot/act)
- [LeRobot 官方真实机器人模仿学习流程](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobot 0.6 官方 Policy Deployment](https://huggingface.co/docs/lerobot/v0.6.0/inference)
- [LeRobot 官方 Action Representations](https://huggingface.co/docs/lerobot/action_representations)
- [LeRobot 官方 Adding a Policy](https://huggingface.co/docs/lerobot/bring_your_own_policies)
- [ACT 原论文](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy 原论文](https://arxiv.org/abs/2303.04137)
- [DP3 原论文](https://arxiv.org/abs/2403.03954)
