# 安装配置文档

## 系统要求

- **操作系统**：推荐使用 Ubuntu 20.04 或更高版本  
- **显卡**：Nvidia 显卡  
- **驱动版本**：建议使用 525 或更高版本  
- **Python 版本**：Python 3.8
- **支持的机器人**：Adam Lite

---

## 1. 创建虚拟环境

建议在虚拟环境中运行训练或部署程序，推荐使用 Conda 创建虚拟环境。如果您的系统中已经安装了 Conda，可以跳过步骤 1.1。

### 1.1 下载并安装 MiniConda

MiniConda 是 Conda 的轻量级发行版，适用于创建和管理虚拟环境。使用以下命令下载并安装：

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

安装完成后，初始化 Conda：

```bash
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

### 1.2 创建新环境

使用以下命令创建虚拟环境：

```bash
conda create -n pndbotics-rl python=3.8 -y
```

### 1.3 激活虚拟环境

```bash
conda activate pndbotics-rl
```

---

## 2. 安装依赖

### 2.1 安装 PyTorch

PyTorch 是一个神经网络计算框架，用于模型训练和推理。使用以下命令安装：

```bash
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 2.2 安装 Isaac Gym

Isaac Gym 是 Nvidia 提供的刚体仿真和训练框架。

#### 2.2.1 下载

从 Nvidia 官网下载 [Isaac Gym](https://developer.nvidia.com/isaac-gym)。

#### 2.2.2 安装

解压后进入 `isaacgym/python` 文件夹，执行以下命令安装：

```bash
cd isaacgym/python
pip install -e .
```

#### 2.2.3 验证安装

运行以下命令，若弹出窗口并显示 1080 个球下落，则安装成功：

```bash
cd examples
python 1080_balls_of_solitude.py
```

如有问题，可参考 `isaacgym/docs/index.html` 中的官方文档。

### 2.3 安装 rsl_rl

`rsl_rl` 是一个强化学习算法库。

#### 2.3.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
```

#### 2.3.2 切换分支

切换到 v1.0.2 分支：

```bash
cd rsl_rl
git checkout v1.0.2
```

#### 2.3.3 安装

```bash
pip install -e .
```

### 2.4 安装 pnd_rl_gym

#### 2.4.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/pndbotics/pnd_rl_gym.git
```

#### 2.4.2 安装

进入目录并安装：

```bash
cd pnd_rl_gym
pip install -e .
```

**说明**：此步骤会自动安装以下依赖包：
- `matplotlib`: 数据可视化
- `numpy==1.20`: 数值计算库
- `tensorboard`: 训练可视化工具
- `mujoco==3.2.3`: Mujoco 仿真器
- `pyyaml`: YAML 配置文件解析
- `wandb`: 实验追踪和可视化（可选）

### 2.5 配置 WandB（可选）

`WandB (Weights & Biases)` 是一个强大的实验追踪和可视化工具，用于监控训练过程、管理模型版本和对比不同的训练运行。

#### 2.5.1 创建账户

访问 [https://wandb.ai/](https://wandb.ai/) 创建免费账户。

#### 2.5.2 登录

```bash
wandb login
```

系统会提示您输入 API 密钥，可在 [https://wandb.ai/authorize](https://wandb.ai/authorize) 获取。

#### 2.5.3 验证安装（可选）

```bash
python scripts/test_wandb.py
```

#### 2.5.4 使用 WandB

在训练时添加 `--wandb` 标志即可启用：

```bash
python legged_gym/scripts/train.py --task=adam_lite --wandb
```

**说明**：如果不需要使用 WandB，可以跳过此步骤。训练时不添加 `--wandb` 标志即可。

### 2.6 安装 pndbotics_sdk_py（可选）

`pndbotics_sdk_py` 是用于与真实机器人通信的库。如果需要将训练的模型部署到物理机器人上运行，可以安装此库。

#### 2.6.1 下载

通过 Git 克隆仓库：

```bash
git clone https://github.com/pndbotics/pnd_sdk_python.git
```

#### 2.6.2 安装

进入目录并安装：

```bash
cd pnd_sdk_python
pip install -e .
```

---

## 3. 验证安装

### 3.1 验证 Isaac Gym

如前所述，运行 Isaac Gym 示例：

```bash
cd isaacgym/python/examples
python 1080_balls_of_solitude.py
```

### 3.2 验证 pnd_rl_gym

运行训练命令测试环境是否正常：

```bash
cd pnd_rl_gym
python legged_gym/scripts/train.py --task=adam_lite --headless --num_envs=64 --max_iterations=1
```

如果命令成功执行并生成日志文件，说明安装成功。

---

## 4. 常见问题

### 4.1 Isaac Gym 安装失败

- 确保已安装 Nvidia 驱动（版本 525+）
- 检查 Python 版本是否为 3.8
- 参考 `isaacgym/docs/index.html` 中的故障排除指南

### 4.2 CUDA 相关错误

- 确保 PyTorch 的 CUDA 版本与系统 CUDA 版本兼容
- 可使用 `nvidia-smi` 查看系统 CUDA 版本

### 4.3 numpy 版本冲突

- 本项目要求 `numpy==1.20`
- 如遇版本冲突，使用以下命令强制安装：
  ```bash
  pip install numpy==1.20 --force-reinstall
  ```

---

## 总结

按照上述步骤完成后，您已经准备好在虚拟环境中运行相关程序。完整的工作流程为：

1. **训练**：`python legged_gym/scripts/train.py --task=adam_lite`
2. **Play 验证**：`python legged_gym/scripts/play.py --task=adam_lite`
3. **Sim2Sim 部署**：`python deploy/deploy_mujoco/deploy_mujoco.py adam_lite.yaml`
4. **Sim2Real 部署**：`python deploy/deploy_real/deploy_real.py {网卡名称} adam_lite.yaml`

详细使用说明请参考主 [README_zh.md](../README_zh.md) 文档。

若遇到问题，请参考各组件的官方文档或检查依赖安装是否正确。

