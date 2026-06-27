import torch

from legged_gym.envs.adam_pro_12dof.adam_pro_12dof_env import (
    AdamPro12dofRobot,
)


class AdamPro12dofHybridRobot(AdamPro12dofRobot):

    def _init_buffers(self):
        super()._init_buffers()

        # 命令目标值：命令切换时不直接送给策略，而是逐步逼近
        self.command_targets = torch.zeros_like(self.commands)

        # 每个环境独立维护步态相位
        self.phase = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self.phase_left = torch.zeros_like(self.phase)
        self.phase_right = torch.zeros_like(self.phase)
        self.leg_phase = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
            dtype=torch.float,
        )

        # 0 表示站立，1 表示步行；中间值表示过渡
        self.walk_weight = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        self.stand_weight = torch.ones_like(self.walk_weight)

    def _resample_commands(self, env_ids):
        """同时采样站立和步行命令。"""

        if len(env_ids) == 0:
            return

        # 保存当前已经平滑后的命令
        old_commands = self.commands[env_ids].clone()

        # 使用基类采样普通步行命令
        super()._resample_commands(env_ids)
        sampled_commands = self.commands[env_ids].clone()

        # 按指定概率将部分环境设置成纯站立
        stand_probability = self.cfg.commands.stand_probability

        stand_mask = (
            torch.rand(len(env_ids), device=self.device)
            < stand_probability
        )

        sampled_commands[stand_mask] = 0.0

        # sampled_commands 只作为目标，不直接跳变
        self.command_targets[env_ids] = sampled_commands
        self.commands[env_ids] = old_commands

    def _post_physics_step_callback(self):
        """更新命令、模式权重、脚部状态以及步态相位。"""

        # AdamPro 父类会更新脚部状态、采样命令、地形高度等
        previous_phase = self.phase.clone()
        super()._post_physics_step_callback()

        # 平滑命令，避免站立/步行瞬间切换
        transition_time = self.cfg.commands.transition_time
        alpha = min(self.dt / max(transition_time, 1e-6), 1.0)

        self.commands += alpha * (
            self.command_targets - self.commands
        )

        # 把 yaw 速度折算成等效运动强度
        linear_motion = torch.norm(
            self.commands[:, :2],
            dim=1,
        )
        yaw_motion = (
            self.cfg.commands.yaw_motion_scale
            * torch.abs(self.commands[:, 2])
        )
        motion = linear_motion + yaw_motion

        stand_threshold = self.cfg.commands.stand_threshold
        walk_threshold = self.cfg.commands.walk_threshold

        self.walk_weight[:] = torch.clamp(
            (motion - stand_threshold)
            / max(walk_threshold - stand_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        self.stand_weight[:] = 1.0 - self.walk_weight

        # 步态相位只在行走时推进
        period = self.cfg.gait.period
        self.phase[:] = torch.remainder(
            previous_phase
            + self.dt / period * self.walk_weight,
            1.0,
        )

        # 完全进入站立后，将相位固定为双脚支撑对应的 phase=0
        standing = self.walk_weight < 0.05
        self.phase[standing] = 0.0

        self.phase_left = self.phase
        self.phase_right = (
            self.phase + self.cfg.gait.phase_offset
        ) % 1.0

        self.leg_phase = torch.stack(
            (self.phase_left, self.phase_right),
            dim=1,
        )

    def reset_idx(self, env_ids):
        """重置时从零命令、站立模式开始。"""

        if len(env_ids) > 0 and hasattr(self, "command_targets"):
            self.commands[env_ids] = 0.0
            self.command_targets[env_ids] = 0.0
            self.phase[env_ids] = 0.0
            self.walk_weight[env_ids] = 0.0
            self.stand_weight[env_ids] = 1.0

        super().reset_idx(env_ids)

    # ------------------------------------------------------------------
    # 步行模式奖励
    # ------------------------------------------------------------------

    def _reward_contact(self):
        reward = super()._reward_contact()
        return reward * self.walk_weight

    def _reward_feet_swing_height(self):
        reward = super()._reward_feet_swing_height()
        return reward * self.walk_weight

    def _reward_feet_air_time(self):
        reward = super()._reward_feet_air_time()
        return reward * self.walk_weight

    # ------------------------------------------------------------------
    # 站立模式奖励
    # ------------------------------------------------------------------

    def _reward_stand_still(self):
        """站立时保持默认关节姿态。"""

        pose_error = torch.sum(
            torch.square(
                self.dof_pos - self.default_dof_pos
            ),
            dim=1,
        )
        return pose_error * self.stand_weight

    def _reward_stand_contact(self):
        """站立时鼓励双脚同时接触地面。"""

        contact = (
            self.contact_forces[
                :, self.feet_indices, 2
            ] > 1.0
        )

        both_feet_contact = torch.all(
            contact,
            dim=1,
        ).float()

        return both_feet_contact * self.stand_weight