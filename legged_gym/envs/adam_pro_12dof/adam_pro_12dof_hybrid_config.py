from legged_gym.envs.adam_pro_12dof.adam_pro_12dof_config import (
    AdamPro12dofRoughCfg,
    AdamPro12dofRoughCfgPPO,
)


class AdamPro12dofHybridCfg(AdamPro12dofRoughCfg):

    class env(AdamPro12dofRoughCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12

        # 需要让同一个 episode 中出现多次模式切换
        episode_length_s = 20

    class commands(AdamPro12dofRoughCfg.commands):
        curriculum = False
        num_commands = 4

        # 每 3 秒产生一次新的站立/步行目标
        resampling_time = 3.0

        # 使用直接 yaw 速度命令，避免 heading 控制偷偷覆盖站立命令
        heading_command = False

        # 40% 的命令片段为纯站立
        stand_probability = 0.4

        # 命令平滑时间常数
        transition_time = 0.25

        # 模式平滑区间
        stand_threshold = 0.05
        walk_threshold = 0.15

        # 将 yaw 角速度折算成运动强度
        yaw_motion_scale = 0.5

        class ranges(AdamPro12dofRoughCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.8]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.4, 0.4]
            heading = [0.0, 0.0]

    class gait:
        period = 0.8
        phase_offset = 0.5

    class rewards(AdamPro12dofRoughCfg.rewards):
        base_height_target = 0.88

        # 先保持与原步行任务一致，减少续训冲击
        only_positive_rewards = True

        class scales(AdamPro12dofRoughCfg.rewards.scales):
            # 通用速度与姿态
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = 1.0
            base_height = -10.0
            alive = 0.15

            # 站立模式
            stand_still = -1.0
            stand_contact = 1.0

            # 步行模式，由 walk_weight 控制
            feet_air_time = 0.05
            feet_swing_height = -30.0
            contact = 1.0

            # 两种模式共用
            contact_no_vel = -0.1
            feet_distance = 1.5
            feet_lateral_deviation = -5.0
            ankle_pos = -10.0

            dof_acc = -2.5e-7
            dof_vel = -1e-3
            dof_pos_limits = -5.0
            action_rate = -0.01
            action_smoothness = -0.01
            collision = -1.0

            # termination 在正奖励裁剪后加入
            termination = -5.0

    class terrain(AdamPro12dofRoughCfg.terrain):
        # 第一阶段先使用平地
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class domain_rand(AdamPro12dofRoughCfg.domain_rand):
        # 第一阶段关闭推扰动
        push_robots = False
        curriculum = False

        # 可以保留温和的摩擦和质量随机化
        randomize_friction = True
        friction_range = [0.5, 1.25]

        randomize_base_mass = True
        added_mass_range = [-0.05, 0.05]

    # 注意：noise 必须放在环境配置中，不能放在 PPO 配置中
    class noise(AdamPro12dofRoughCfg.noise):
        add_noise = True
        noise_level = 0.4


class AdamPro12dofHybridCfgPPO(AdamPro12dofRoughCfgPPO):

    class runner(AdamPro12dofRoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"

        max_iterations = 3000
        save_interval = 100

        # 暂时保持原 experiment_name，方便读取现有步行 checkpoint
        experiment_name = "adam_pro_12dof"
        run_name = "hybrid_stand_walk"

        resume = False
        load_run = -1
        checkpoint = -1

        use_wandb = True
        wandb_project = "pnd_adam_pro_12dof_locomotion"
        wandb_tags = [
            "adam_pro_12dof",
            "hybrid",
            "stand",
            "walk",
            "lstm",
        ]