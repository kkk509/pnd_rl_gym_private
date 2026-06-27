# PNDbotics RL Gym - Adam Pro 12DoF Workflow

This repository contains the current reinforcement-learning training and deployment workflow for the PNDbotics Adam Pro 12DoF leg controller.

The main task is:

```text
adam_pro_12dof
```

The full workflow is:

```text
Train in Isaac Gym -> Play and export policy -> Deploy in MuJoCo -> Deploy on the real robot
```

## Installation

Follow the environment setup guide first:

```text
doc/setup_en.md
```

Typical environment activation:

```bash
conda activate pndbotics-rl
```

## Key Files

Adam Pro training environment:

```text
legged_gym/envs/adam_pro_12dof/adam_pro_12dof_config.py
legged_gym/envs/adam_pro_12dof/adam_pro_12dof_env.py
```

Task registration:

```text
legged_gym/envs/__init__.py
```

Play, test, and export:

```text
legged_gym/scripts/play.py
legged_gym/scripts/export_lstm_onnx.py
```

MuJoCo deployment:

```text
deploy/deploy_mujoco/deploy_mujoco.py
deploy/deploy_mujoco/configs/adam_pro_12dof.yaml
resources/robots/adam_pro/scene_adam_pro_12dof_simplified_collision.xml
```

Real robot deployment:

```text
deploy/deploy_real/deploy_real.py
deploy/deploy_real/deploy_real_onnx.py
deploy/deploy_real/deploy_real_onnx_cpu.py
deploy/deploy_real/configs/adam_pro_12dof.yaml
```

## 1. Train Adam Pro 12DoF

Start training:

```bash
python legged_gym/scripts/train.py --task=adam_pro_12dof --headless
```

Useful options:

```bash
python legged_gym/scripts/train.py --task=adam_pro_12dof --headless --num_envs=4096
python legged_gym/scripts/train.py --task=adam_pro_12dof --headless --resume
python legged_gym/scripts/train.py --task=adam_pro_12dof --headless --load_run=<run_name> --checkpoint=<checkpoint_id>
```

Training outputs are saved under:

```text
logs/adam_pro_12dof/<date_time>_pro_12dof_leg_only/
```

The PPO config currently uses recurrent policy inference:

```python
policy_class_name = "ActorCriticRecurrent"
rnn_type = "lstm"
rnn_hidden_size = 128
```

## 2. Verify With Play

Run the trained policy in Isaac Gym:

```bash
python legged_gym/scripts/play.py --task=adam_pro_12dof
```

`play.py` is used for three things:

- Verify whether the policy walks correctly in the training simulator.
- Record base velocity, yaw, contact force, torque, and joint velocity plots.
- Export the policy for deployment when `EXPORT_POLICY = True`.

The exported JIT policy is saved to:

```text
logs/adam_pro_12dof/exported/policies/policy_lstm_1.pt
```

To test the default standing pose without policy actions:

```bash
python legged_gym/scripts/play.py --task=adam_pro_12dof --test_default_pose
```

## 3. Important Heading Command Detail

Adam Pro training uses heading correction:

```python
heading_command = True
```

The training logic is in:

```text
legged_gym/envs/base/legged_robot.py
```

It computes yaw command as:

```python
forward = quat_apply(self.base_quat, self.forward_vec)
heading = torch.atan2(forward[:, 1], forward[:, 0])
self.commands[:, 2] = torch.clip(
    0.5 * wrap_to_pi(self.commands[:, 3] - heading),
    -1.,
    1.,
)
```

Formula:

```text
yaw_cmd = clip(0.5 * wrap_to_pi(target_heading - current_heading), -1, 1)
```

This matters for deployment. If MuJoCo or real-robot deployment sends a fixed yaw command of `0` instead of this heading correction, the policy may drift and fail to walk in a straight line.

## 4. Deploy in MuJoCo

Run MuJoCo deployment:

```bash
python deploy/deploy_mujoco/deploy_mujoco.py adam_pro_12dof.yaml
```

MuJoCo config:

```text
deploy/deploy_mujoco/configs/adam_pro_12dof.yaml
```

Important fields:

```yaml
policy_path: "{LEGGED_GYM_ROOT_DIR}/logs/adam_pro_12dof/exported/policies/policy_lstm_1.pt"
xml_path: "{LEGGED_GYM_ROOT_DIR}/resources/robots/adam_pro/scene_adam_pro_12dof_simplified_collision.xml"
cmd_init: [0.8, 0, 0]
```

For debugging straight walking, first try a lower forward command:

```yaml
cmd_init: [0.6, 0, 0]
```

MuJoCo should reproduce the same heading correction used during training:

```python
target_heading = 0.0
current_heading = get_yaw_from_quat(quat)
cmd[2] = np.clip(
    0.5 * wrap_to_pi(target_heading - current_heading),
    -1.0,
    1.0,
)
```

If the robot turns harder in the wrong direction, check the yaw sign convention and test:

```python
cmd[2] = np.clip(
    0.5 * wrap_to_pi(current_heading - target_heading),
    -1.0,
    1.0,
)
```

## 5. Deploy on the Real Robot

Real robot config:

```text
deploy/deploy_real/configs/adam_pro_12dof.yaml
```

Run real deployment:

```bash
python deploy/deploy_real/deploy_real.py <net_interface> adam_pro_12dof.yaml
```

Example:

```bash
python deploy/deploy_real/deploy_real.py enp3s0 adam_pro_12dof.yaml
```

Before running policy control:

1. Put the robot in the expected debug/control mode.
2. Confirm the network interface with `ifconfig` or `ip addr`.
3. Confirm the robot can enter zero-torque state.
4. Press the start signal to continue.
5. Let the controller move the robot to default pose.
6. Press Button A to enter policy control.

The real deployment observation must match training:

```text
obs[0:3]   = base angular velocity * ang_vel_scale
obs[3:6]   = projected gravity
obs[6:9]   = command * command_scale
obs[9:21]  = joint position error
obs[21:33] = joint velocity
obs[33:45] = previous action
obs[45:47] = sin/cos phase
```

For heading correction on the real robot, compute yaw from IMU quaternion and write the corrected yaw command into `obs[8]`. The yaw correction value is already in rad/s, so do not multiply it by `max_cmd[2]` a second time.

Recommended logic:

```python
self.obs[6] = self.cmd[0] * self.config.max_cmd[0] * self.config.cmd_scale[0]
self.obs[7] = self.cmd[1] * self.config.max_cmd[1] * self.config.cmd_scale[1]
self.obs[8] = yaw_cmd * self.config.cmd_scale[2]
```

Where:

```python
yaw_cmd = np.clip(
    0.5 * wrap_to_pi(self.target_heading - current_heading),
    -self.config.max_cmd[2],
    self.config.max_cmd[2],
)
```

If the yaw correction direction is wrong on hardware, invert the sign convention and test carefully at low speed.

## 6. Optional ONNX Export

Export LSTM policy to ONNX:

```bash
python legged_gym/scripts/export_lstm_onnx.py --task=adam_pro_12dof
```

Output:

```text
logs/adam_pro_12dof/exported/onnx/policy_lstm.onnx
```

ONNX deployment scripts:

```text
deploy/deploy_real/deploy_real_onnx.py
deploy/deploy_real/deploy_real_onnx_cpu.py
```

Useful checks:

```bash
python scripts/test_onnx_inference.py
python scripts/test_onnx_LSTM_state.py
python scripts/test_onnx_vs_pytorch.py
```

## 7. WandB

The Adam Pro PPO config enables WandB logging by default:

```python
use_wandb = True
wandb_project = "pnd_adam_pro_12dof_locomotion"
wandb_tags = ["adam_pro_12dof", "rough_terrain", "lstm"]
```

Login before training if using WandB:

```bash
wandb login
```

Disable WandB for offline/local runs:

```bash
WANDB_MODE=disabled python legged_gym/scripts/train.py --task=adam_pro_12dof --headless
```

## Troubleshooting

### Play walks straight, but MuJoCo does not

Check these first:

- MuJoCo uses the same exported policy: `policy_lstm_1.pt`.
- `cmd_init` is not too aggressive; test `[0.3, 0, 0]` or `[0.6, 0, 0]`.
- Heading correction is applied before writing `obs[6:9]`.
- Yaw correction sign is correct.
- The robot starts from the default joint angles.

### Play itself does not walk straight

This is usually a training or command-mode issue, not MuJoCo:

- Confirm `heading_command = True` if the policy was trained with heading correction.
- Check average lateral velocity and yaw plots from `play.py`.
- Check asymmetric long-term actions on `hipRoll`, `hipYaw`, and `ankleRoll`.
- Review reward weights for lateral velocity, yaw tracking, and orientation.

### Real robot drifts in yaw

Check:

- IMU yaw sign and quaternion convention.
- Whether `imu_type` is `pelvis` or `torso`.
- Whether `transform_imu_data` is needed.
- Whether `yaw_cmd` is scaled once, not twice.
- Whether manual yaw input from the remote is overriding heading hold.

## Acknowledgments

This repository builds on:

- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git)
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [pndbotics_sdk2_python](https://github.com/pndbotics/pndbotics_sdk2_python.git)

## License

This project is licensed under the BSD 3-Clause License. See:

```text
LICENSE
```
