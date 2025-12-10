<div align="center">
  <h1 align="center">PNDbotics RL GYM</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  🎮🚪 <strong>这是一个基于 PNDbotics 机器人实现强化学习的示例仓库，支持 PNDbotics adam_lite_12dof。</strong> 🚪🎮
</p>

<div align="center">

| <div align="center"> Isaac Gym </div> | <div align="center">  Mujoco </div> |  <div align="center"> Physical </div> |
|--- | --- | --- |
| [<img src="doc/pnd_gif/adam_lite_isaac.gif" width="240px">](https://oss-global-cdn.PNDbotics.com/static/5bbc5ab1d551407080ca9d58d7bec1c8.mp4) | [<img src="doc/pnd_gif/adam_lite_mujoco.gif" width="240px">](https://oss-global-cdn.PNDbotics.com/static/5aa48535ffd641e2932c0ba45c8e7854.mp4) | [<img src="https://oss-global-cdn.PNDbotics.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF" width="240px">](https://oss-global-cdn.PNDbotics.com/static/0818dcf7a6874b92997354d628adcacd.mp4) |

</div>
---

## 📦 安装配置

安装和配置步骤请参考 [setup.md](/doc/setup_zh.md)

## 🔁 流程说明

强化学习实现运动控制的基本流程为：

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: 通过 Gym 仿真环境，让机器人与环境互动，找到最满足奖励设计的策略。通常不推荐实时查看效果，以免降低训练效率。
- **Play**: 通过 Play 命令查看训练后的策略效果，确保策略符合预期。
- **Sim2Sim**: 将 Gym 训练完成的策略部署到其他仿真器，避免策略小众于 Gym 特性。
- **Sim2Real**: 将策略部署到实物机器人，实现运动控制。

## 🛠️ 使用指南

### 1. 训练

运行以下命令进行训练：

```bash
screen -S pndbotics
cd ~/Documents/pndbotics_rl_gym
conda activate pndbotics-rl
python legged_gym/scripts/train.py --task=adam_lite_12dof --headless
```

#### ⚙️  参数说明
- `--task`: 必选参数，值可选(adam_lite_12dof)
- `--headless`: 默认启动图形界面，设为 true 时不渲染图形界面（效率更高）
- `--resume`: 从日志中选择 checkpoint 继续训练
- `--experiment_name`: 运行/加载的 experiment 名称
- `--run_name`: 运行/加载的 run 名称
- `--load_run`: 加载运行的名称，默认加载最后一次运行
- `--checkpoint`: checkpoint 编号，默认加载最新一次文件
- `--num_envs`: 并行训练的环境个数
- `--seed`: 随机种子
- `--max_iterations`: 训练的最大迭代次数
- `--sim_device`: 仿真计算设备，指定 CPU 为 `--sim_device=cpu`
- `--rl_device`: 强化学习计算设备，指定 CPU 为 `--rl_device=cpu`

**默认保存训练结果**：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

如果想要在 Gym 中查看训练效果，可以运行以下命令：

```bash
python legged_gym/scripts/play.py --task=xxx 
```

**说明**：

- Play 启动参数与 Train 相同。
- 默认加载实验文件夹上次运行的最后一个模型。
- 可通过 `load_run` 和 `checkpoint` 指定其他模型。

#### ⚙️ 额外的 Play 参数

- `--test_default_pose`: 通过将所有动作设置为零来测试默认关节角度。这对于在没有策略控制的情况下验证机器人的默认站立姿势很有用。

#### 💾 导出网络

Play 会导出 Actor 网络，保存于 `logs/{experiment_name}/exported/policies` 中：
- 普通网络（MLP）导出为 `policy_1.pt`
- RNN 网络，导出为 `policy_lstm_1.pt`
  
### Play 效果

### Play Results

| Adam Lite 12DOF |
| --- |
| [![adam_lite_12dof](doc/pnd_gif/adam_lite_isaac.gif)]|


---

### 3. Sim2Sim (Mujoco)

支持在 Mujoco 仿真器中运行 Sim2Sim：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py {config_name}
```

#### 参数说明
- `config_name`: 配置文件，默认查询路径为 `deploy/deploy_mujoco/configs/`

#### 示例：运行 Adam Lite 12DOF

```bash
python deploy/deploy_mujoco/deploy_mujoco.py adam_lite_12dof.yaml
```

#### ➡️  替换网络模型

默认模型位于 `deploy/pre_train/{robot}/motion.pt`；自己训练模型保存于`logs/adam_lite_12dof/exported/policies/policy_lstm_1.pt`，只需替换 yaml 配置文件中 `policy_path`。

#### 运行效果

| Adam Lite 12DOF |
| --- |
| [![mujoco_adam_lite_12dof](doc/pnd_gif/adam_lite_mujoco.gif)]|

---

### 4. Sim2Real (实物部署)

实现实物部署前，确保机器人进入调试模式。详细步骤请参考 [实物部署指南](deploy/deploy_real/README.zh.md)：

```bash
python deploy/deploy_real/deploy_real.py {net_interface} {config_name}
```

#### 参数说明
- `net_interface`: 连接机器人网卡名称，如 `enp3s0`
- `config_name`: 配置文件，存在于 `deploy/deploy_real/configs/`，如 `adam_lite_12dof.yaml`

#### 运行效果

| Adam Lite 12DOF |
| --- |
| [![real_adam_lite_12dof](https://oss-global-cdn.pndbotics.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF)](https://oss-global-cdn.PNDbotics.com/static/0818dcf7a6874b92997354d628adcacd.mp4) |

---

## 📊 WandB 集成

本项目集成了 [Weights & Biases (WandB)](https://wandb.ai/) 用于高级实验追踪和可视化。WandB 提供实时指标记录、模型版本管理和协作实验管理功能。

### 功能特性

- **实时指标记录**: 追踪训练损失、奖励、回合长度和自定义指标
- **模型检查点**: 自动保存和版本化模型检查点
- **实验对比**: 并排比较多个训练运行
- **超参数追踪**: 记录所有配置参数以确保可重现性
- **梯度监控**: 在训练期间监视模型梯度和参数

### 安装

WandB 已包含在项目依赖中。如果尚未安装：

```bash
pip install wandb
```

### 设置

1. **创建 WandB 账户**，访问 [https://wandb.ai/](https://wandb.ai/)

2. **登录 WandB**：
   ```bash
   wandb login
   ```
   
   系统会提示您输入 API 密钥，可在 [https://wandb.ai/authorize](https://wandb.ai/authorize) 获取

3. **验证安装**（可选）：
   ```bash
   python scripts/test_wandb.py
   ```

### 使用方法

#### 在训练中启用 WandB

要在训练期间启用 WandB 记录，添加 `--wandb` 标志：

```bash
python legged_gym/scripts/train.py --task=adam_lite_12dof --wandb
```

或在任务配置文件中设置 WandB 配置：

```python
class AdamLite12DofCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        use_wandb = True  # 启用 WandB 记录
        wandb_project = "pndbotics_rl_gym"  # WandB 项目名称
        wandb_entity = None  # WandB 团队/用户名（可选）
        wandb_tags = ["adam_lite_12dof", "rough_terrain"]  # 用于组织的标签
```

#### 配置选项

您可以通过配置自定义 WandB 行为：

```python
class runner:
    # WandB 设置
    use_wandb = False          # 启用/禁用 WandB 记录
    wandb_project = "pndbotics_rl_gym"  # WandB 项目名称
    wandb_entity = None        # WandB 团队/用户名（可选）
    wandb_tags = []            # 运行的标签列表
```

#### 记录的指标

WandB 集成自动记录：

- **训练指标**：
  - `Loss/value_function`: 价值函数损失
  - `Loss/surrogate`: 策略代理损失
  - `Train/learning_rate`: 当前学习率
  - `Train/mean_reward`: 平均回合奖励
  - `Train/mean_episode_length`: 平均回合长度

- **时间信息**：
  - `Time/collection`: 数据收集时间
  - `Time/learn`: 学习更新时间

- **回合信息**：
  - 自定义奖励组件
  - 环境特定指标

- **模型工件**：
  - 带版本追踪的模型检查点
  - 带特殊 "final" 别名的最终模型

### 高级功能

#### 模型监视

集成会自动监视模型梯度和参数：

```python
# 这在 WandbOnPolicyRunner 中自动完成
wandb_logger.watch_model(model, log_freq=100)
```

#### 手动记录

您也可以手动记录自定义指标：

```python
from legged_gym.utils import WandbLogger

logger = WandbLogger(
    project="my_project",
    experiment_name="my_experiment",
    run_name="my_run"
)

# 记录自定义指标
logger.log({"custom_metric": value}, step=iteration)

# 记录视频
logger.log_video("path/to/video.mp4", name="policy_video", step=iteration)

# 保存模型
logger.save_model("path/to/model.pt", aliases=["best"])
```

### 离线模式

要在没有互联网连接或 WandB 记录的情况下运行训练：

```bash
WANDB_MODE=disabled python legged_gym/scripts/train.py --task=adam_lite_12dof
```

或在 Python 中设置：

```python
import os
os.environ['WANDB_MODE'] = 'disabled'
```

### 查看结果

启用 WandB 的训练运行后：

1. 控制台会打印一个 URL，例如：
   ```
   [WandB] ✓ View run at: https://wandb.ai/your-username/pndbotics_rl_gym/runs/xxxxx
   ```

2. 点击 URL 或访问 [https://wandb.ai/](https://wandb.ai/) 查看：
   - 实时训练指标和图表
   - 系统指标（GPU/CPU 使用率、内存）
   - 模型架构和梯度
   - 超参数和配置
   - 已保存的模型检查点

### 故障排除

#### 未登录

如果看到关于 WandB 未登录的错误：

```bash
wandb login <your_api_key>
```

#### 导入错误

如果找不到 WandB：

```bash
pip install wandb
```

#### 测试集成

运行测试脚本以验证一切设置正确：

```bash
python scripts/test_wandb.py
```

这将检查：
- ✅ WandB 安装
- ✅ 登录状态
- ✅ 与训练代码的集成
- ✅ 配置文件

### 示例工作流程

```bash
# 1. 安装并登录 WandB
pip install wandb
wandb login

# 2. 测试集成
python scripts/test_wandb.py

# 3. 使用 WandB 开始训练
python legged_gym/scripts/train.py --task=adam_lite_12dof --wandb

# 4. 在 https://wandb.ai/ 监控训练
# 5. 比较不同的运行和超参数
# 6. 从 WandB 下载模型检查点
```

---

## 🎉  致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): 构建训练与运行代码的基础。
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。
- [mujoco](https://github.com/google-deepmind/mujoco.git): 提供强大仿真功能。
- [pndbotics\_sdk2\_python](https://github.com/pndbotics/pndbotics_sdk2_python.git): 实物部署硬件通信接口。


---

## 🔖  许可证

本项目根据 [BSD 3-Clause License](./LICENSE) 授权：
1. 必须保留原始版权声明。
2. 禁止以项目名或组织名作举。
3. 声明所有修改内容。

详情请阅读完整 [LICENSE 文件](./LICENSE)。

