import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi


class AdamPro12dofHybridRobot(LeggedRobot):
    """Standalone Adam Pro 12DOF stand / walk / turn environment.

    这个环境只继承 LeggedRobot，不再继承旧的 Adam Pro 环境类。
    Adam Pro 相关的 obs、脚部状态、奖励和 push curriculum 都在本类中维护。
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        if self.cfg.domain_rand.curriculum:
            self.push_vel_xy = self.cfg.domain_rand.initial_push_vel_xy
        else:
            self.push_vel_xy = self.cfg.domain_rand.max_push_vel_xy

    # ------------------------------------------------------------------
    # Buffers / observations
    # ------------------------------------------------------------------

    def _get_noise_scale_vec(self, cfg):
        """Noise scale matching the 47-dim hybrid observation layout."""

        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0  # commands
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.0  # previous actions
        noise_vec[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = 0.0  # sin/cos phase

        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        self.rigid_body_states_view = self.rigid_state
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

        # 命令目标值：命令切换时不直接送给策略，而是逐步逼近。
        self.command_targets = torch.zeros_like(self.commands)

        # 每个环境独立维护步态相位。
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.phase_left = torch.zeros_like(self.phase)
        self.phase_right = torch.zeros_like(self.phase)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)

        # 0 表示站立，1 表示行走/转弯，中间值表示过渡。
        self.walk_weight = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.stand_weight = torch.ones_like(self.walk_weight)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        """Update command smoothing, mode weights, feet state and gait phase."""

        self.update_feet_state()

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        transition_time = self.cfg.commands.transition_time
        alpha = min(self.dt / max(transition_time, 1e-6), 1.0)
        self.commands += alpha * (self.command_targets - self.commands)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # 把 yaw 速度折算成等效运动强度。
        linear_motion = torch.norm(self.commands[:, :2], dim=1)
        yaw_motion = self.cfg.commands.yaw_motion_scale * torch.abs(self.commands[:, 2])
        motion = linear_motion + yaw_motion

        stand_threshold = self.cfg.commands.stand_threshold
        walk_threshold = self.cfg.commands.walk_threshold

        self.walk_weight[:] = torch.clamp((motion - stand_threshold) / max(walk_threshold - stand_threshold, 1e-6), min=0.0, max=1.0)
        self.stand_weight[:] = 1.0 - self.walk_weight

        period = self.cfg.gait.period
        self.phase[:] = torch.remainder(self.phase + self.dt / period * self.walk_weight, 1.0)

        standing = self.walk_weight < 0.05
        self.phase[standing] = 0.0

        self.phase_left = self.phase
        self.phase_right = (self.phase + self.cfg.gait.phase_offset) % 1.0
        self.leg_phase = torch.stack((self.phase_left, self.phase_right), dim=1)


    def compute_observations(self):
        """47-dim obs: ang vel, gravity, command, dof pos/vel, action, phase."""

        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)

        self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions,
                                  sin_phase,
                                  cos_phase
                                  ), dim=-1)
        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                             self.base_ang_vel * self.obs_scales.ang_vel,
                                             self.projected_gravity,
                                             self.commands[:, :3] * self.commands_scale,
                                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                             self.dof_vel * self.obs_scales.dof_vel,
                                             self.actions,
                                             sin_phase,
                                             cos_phase
                                             ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # ------------------------------------------------------------------
    # Commands / phase
    # ------------------------------------------------------------------

    def _resample_commands(self, env_ids):
        """Sample stand, walk and in-place-turn command targets."""

        if len(env_ids) == 0:
            return

        old_commands = self.commands[env_ids].clone()
        sampled_commands = torch.zeros_like(old_commands)
        num_envs = len(env_ids)

        sampled_commands[:, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (num_envs, 1), device=self.device).squeeze(1)
        sampled_commands[:, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (num_envs, 1), device=self.device).squeeze(1)

        if self.cfg.commands.heading_command:
            sampled_commands[:, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (num_envs, 1), device=self.device).squeeze(1)
        else:
            sampled_commands[:, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (num_envs, 1), device=self.device).squeeze(1)

        # 避免极小 xy 命令制造“半走不走”的样本。
        # Unitree 风格初始命令范围很小，所以这里不能继续用 base env 的 0.2。
        small_command_threshold = self.cfg.commands.small_command_threshold
        sampled_commands[:, :2] *= (torch.norm(sampled_commands[:, :2], dim=1) > small_command_threshold).unsqueeze(1)

        stand_probability = self.cfg.commands.stand_probability
        turn_probability = self.cfg.commands.turn_probability
        mode_rand = torch.rand(num_envs, device=self.device)

        stand_mask = mode_rand < stand_probability
        turn_mask = (mode_rand >= stand_probability) & (mode_rand < stand_probability + turn_probability)

        sampled_commands[stand_mask] = 0.0

        if torch.any(turn_mask).item():
            num_turn_envs = int(torch.sum(turn_mask).item())
            sampled_commands[turn_mask, :2] = 0.0
            sampled_commands[turn_mask, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (num_turn_envs, 1), device=self.device).squeeze(1)
            if sampled_commands.shape[1] > 3:
                sampled_commands[turn_mask, 3] = 0.0

        # sampled_commands 只作为目标，不直接跳变。
        self.command_targets[env_ids] = sampled_commands
        self.commands[env_ids] = old_commands


    def reset_idx(self, env_ids):
        """Reset from zero command and standing phase."""

        if len(env_ids) > 0 and self.cfg.domain_rand.curriculum and hasattr(self, "push_vel_xy"):
            self.update_push_curriculum(env_ids)

        if len(env_ids) > 0 and hasattr(self, "command_targets"):
            self.commands[env_ids] = 0.0
            self.command_targets[env_ids] = 0.0
            self.phase[env_ids] = 0.0
            self.walk_weight[env_ids] = 0.0
            self.stand_weight[env_ids] = 1.0

        super().reset_idx(env_ids)

        if self.cfg.domain_rand.push_robots and hasattr(self, "push_vel_xy"):
            if "episode" not in self.extras:
                self.extras["episode"] = {}
            self.extras["episode"]["push_vel_xy"] = self.push_vel_xy

        if self.cfg.commands.curriculum:
            if "episode" not in self.extras:
                self.extras["episode"] = {}
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_y"] = self.command_ranges["lin_vel_y"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]

    # ------------------------------------------------------------------
    # Adam Pro specific callbacks
    # ------------------------------------------------------------------

    def _process_dof_props(self, props, env_id):
        """Store URDF limits after applying Adam Pro safety factors."""

        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item() * self.cfg.safety.pos_limit
                self.dof_pos_limits[i, 1] = props["upper"][i].item() * self.cfg.safety.pos_limit
                self.dof_vel_limits[i] = props["velocity"][i].item() * self.cfg.safety.vel_limit
                self.torque_limits[i] = props["effort"][i].item() * self.cfg.safety.torque_limit
        return props

    def update_command_curriculum(self, env_ids):
        """Unitree-style command range curriculum for stand / walk / turn.

        Push 不再自动升级；训练难度主要通过速度命令范围和 terrain level 慢慢放大。
        """
        if len(env_ids) == 0 or not hasattr(self.cfg.commands, "limit_ranges"):
            return

        tracking_lin = torch.mean(self.episode_sums.get("tracking_lin_vel", torch.zeros(self.num_envs, device=self.device))[env_ids]) / self.max_episode_length
        tracking_ang = torch.mean(self.episode_sums.get("tracking_ang_vel", torch.zeros(self.num_envs, device=self.device))[env_ids]) / self.max_episode_length

        lin_threshold = self.cfg.commands.command_curriculum_tracking_lin_threshold * self.reward_scales.get("tracking_lin_vel", 1.0)
        ang_threshold = self.cfg.commands.command_curriculum_tracking_ang_threshold * self.reward_scales.get("tracking_ang_vel", 1.0)

        if tracking_lin < lin_threshold or tracking_ang < ang_threshold:
            return

        step = self.cfg.commands.command_curriculum_step
        limit_ranges = self.cfg.commands.limit_ranges
        old_ranges = {name: self.command_ranges[name].copy() for name in ["lin_vel_x", "lin_vel_y", "ang_vel_yaw"]}

        for name in ["lin_vel_x", "lin_vel_y", "ang_vel_yaw"]:
            target_min, target_max = getattr(limit_ranges, name)
            self.command_ranges[name][0] = max(target_min, self.command_ranges[name][0] - step)
            self.command_ranges[name][1] = min(target_max, self.command_ranges[name][1] + step)

        if any(old_ranges[name] != self.command_ranges[name] for name in old_ranges):
            print(
                "[Hybrid Command Curriculum] "
                f"x={self.command_ranges['lin_vel_x']}, "
                f"y={self.command_ranges['lin_vel_y']}, "
                f"yaw={self.command_ranges['ang_vel_yaw']} "
                f"(lin={tracking_lin:.4f}, yaw={tracking_ang:.4f})"
            )

    def update_push_curriculum(self, env_ids):
        """ Implements curriculum learning for push disturbances.
        Gradually increases push velocity as the robot becomes more stable.
        Uses hysteresis mechanism to prevent oscillation.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self.cfg.domain_rand.curriculum:
            return

        # Initialize curriculum tracking variables
        if not hasattr(self, 'push_curriculum_counter'):
            self.push_curriculum_counter = 0
            self.push_performance_history = []

        # Only update curriculum every N resets to avoid oscillation
        # 每100次reset才评估一次，避免频繁调整
        self.push_curriculum_counter += 1
        if self.push_curriculum_counter % 100 != 0:
            return

        # Calculate stability metric: combination of tracking performance and survival
        if len(env_ids) > 0:
            # Use tracking reward and alive reward as indicators of stability
            tracking_reward = self.episode_sums.get("tracking_lin_vel", torch.zeros(self.num_envs, device=self.device))[env_ids]
            alive_reward = self.episode_sums.get("alive", torch.zeros(self.num_envs, device=self.device))[env_ids]

            # Normalize by episode length
            avg_tracking = torch.mean(tracking_reward) / self.max_episode_length
            avg_alive = torch.mean(alive_reward) / self.max_episode_length

            # Store in history for moving average (keep last 5 evaluations)
            performance_score = avg_tracking.item() + avg_alive.item()
            self.push_performance_history.append(performance_score)
            if len(self.push_performance_history) > 5:
                self.push_performance_history.pop(0)

            # Use moving average to smooth out noise
            smoothed_performance = np.mean(self.push_performance_history)

            # Hysteresis thresholds: 不同的增加和减少阈值，防止震荡
            # 增加难度需要更高的性能（保守）
            if "tracking_lin_vel" in self.reward_scales:
                increase_threshold = 0.85 * self.reward_scales["tracking_lin_vel"]  # 需要达到85%才增加
                decrease_threshold = 0.40 * self.reward_scales["tracking_lin_vel"]  # 低于40%才减少
            else:
                increase_threshold = 0.85
                decrease_threshold = 0.40

            if "alive" in self.reward_scales:
                alive_increase_threshold = 0.90 * self.reward_scales["alive"]
                alive_decrease_threshold = 0.50 * self.reward_scales["alive"]
            else:
                alive_increase_threshold = 0.135  # 0.9 * 0.15
                alive_decrease_threshold = 0.075  # 0.5 * 0.15

            old_push_vel = self.push_vel_xy

            # 增加难度：需要同时满足tracking和alive的高阈值
            if avg_tracking > increase_threshold and avg_alive > alive_increase_threshold:
                # 小步增加，避免突然变难
                self.push_vel_xy = np.clip(
                    self.push_vel_xy + 0.1,  # 从0.1改为0.05，更平滑
                    self.cfg.domain_rand.initial_push_vel_xy,
                    self.cfg.domain_rand.max_push_vel_xy_curriculum,
                )
                if self.push_vel_xy != old_push_vel:
                    print(f"✅ [Curriculum] Push velocity increased: {old_push_vel:.2f} → {self.push_vel_xy:.2f} m/s "
                          f"(tracking: {avg_tracking:.3f}, alive: {avg_alive:.3f})")

            # 减少难度：任一指标过低就减少（更快响应困难）
            elif avg_tracking < decrease_threshold or avg_alive < alive_decrease_threshold:
                # 减少时步长更大，快速降低难度帮助恢复
                self.push_vel_xy = np.clip(
                    self.push_vel_xy - 0.1,  # 减少时稍快一些
                    self.cfg.domain_rand.initial_push_vel_xy,
                    self.cfg.domain_rand.max_push_vel_xy_curriculum,
                )
                if self.push_vel_xy != old_push_vel:
                    print(f"⚠️  [Curriculum] Push velocity decreased: {old_push_vel:.2f} → {self.push_vel_xy:.2f} m/s "
                          f"(tracking: {avg_tracking:.3f}, alive: {avg_alive:.3f})")

            # 在稳定区间内（40%-85%）不做调整，避免震荡

    def _push_robots(self):
        """Random pushes using curriculum-adjusted velocity."""

        if not self.cfg.domain_rand.push_robots:
            return

        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return

        max_vel = self.push_vel_xy
        self.root_states[push_env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (len(push_env_ids), 2), device=self.device)

        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    # ------------------------------------------------------------------
    # Rewards shared by stand / walk / turn
    # ------------------------------------------------------------------

    def _reward_alive(self):
        return 1.0

    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.0

    def _reward_feet_distance(self):
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        min_dist = self.cfg.rewards.min_dist
        max_dist = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_dist, 0.0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        return torch.sum(torch.square(contact_feet_vel[:, :, :3]), dim=(1, 2))

    def _reward_ankle_pos(self):
        ankle_deviation = self.dof_pos[:, [4, 5, 10, 11]] - self.default_dof_pos[:, [4, 5, 10, 11]]
        return torch.sum(torch.square(ankle_deviation), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [1, 2, 7, 8]]), dim=1)

    def _reward_feet_lateral_deviation(self):
        base_quat = self.root_states[:, 3:7]
        feet_quat = self.feet_state[:, :, 3:7]

        forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, 3)
        body_forward_world = quat_apply(base_quat, forward_vec)
        body_forward_world = body_forward_world.unsqueeze(1).expand(-1, self.feet_num, -1)

        feet_quat_flat = feet_quat.reshape(-1, 4)
        foot_local_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        foot_local_forward_flat = foot_local_forward.unsqueeze(0).expand(self.num_envs * self.feet_num, 3)
        foot_forward_world_flat = quat_apply(feet_quat_flat, foot_local_forward_flat)
        foot_forward_world = foot_forward_world_flat.reshape(self.num_envs, self.feet_num, 3)

        body_forward_xy = body_forward_world[:, :, :2]
        foot_forward_xy = foot_forward_world[:, :, :2]
        body_forward_xy = body_forward_xy / (torch.norm(body_forward_xy, dim=2, keepdim=True) + 1e-8)
        foot_forward_xy = foot_forward_xy / (torch.norm(foot_forward_xy, dim=2, keepdim=True) + 1e-8)

        body_yaw = torch.atan2(body_forward_xy[:, :, 1], body_forward_xy[:, :, 0])
        foot_yaw = torch.atan2(foot_forward_xy[:, :, 1], foot_forward_xy[:, :, 0])

        yaw_diff = foot_yaw - body_yaw
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        return torch.sum(torch.square(yaw_diff), dim=1)

    def _reward_action_smoothness(self):
        if not hasattr(self, "last_last_actions"):
            self.last_last_actions = torch.zeros_like(self.actions)
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        action_diff_1 = self.actions - self.last_actions
        action_diff_2 = self.last_actions - self.last_last_actions
        action_jerk = action_diff_1 - action_diff_2
        self.last_last_actions = self.last_actions.clone()

        return torch.sum(torch.square(action_jerk), dim=1)

    # ------------------------------------------------------------------
    # Walk / turn rewards
    # ------------------------------------------------------------------

    def _reward_contact(self):
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            reward += ~(contact ^ is_stance)
        return reward * self.walk_weight

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.1) * ~contact
        return torch.sum(pos_error, dim=1) * self.walk_weight

    def _reward_feet_air_time(self):
        return super()._reward_feet_air_time() * self.walk_weight

    # ------------------------------------------------------------------
    # Stand rewards
    # ------------------------------------------------------------------

    def _reward_stand_still(self):
        pose_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return pose_error * self.stand_weight

    def _reward_stand_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        both_feet_contact = torch.all(contact, dim=1).float()
        return both_feet_contact * self.stand_weight

    def _reward_stand_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        return lin_vel_error * self.stand_weight

    def _reward_stand_ang_vel(self):
        yaw_vel_error = torch.square(self.base_ang_vel[:, 2])
        return yaw_vel_error * self.stand_weight
