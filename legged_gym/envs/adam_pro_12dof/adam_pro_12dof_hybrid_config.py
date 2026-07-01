from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class AdamPro12dofHybridCfg(LeggedRobotCfg):
    """Adam Pro 12DOF hybrid config.

    这个配置不继承原 Adam Pro 配置类，但参数按原 Adam Pro
    12DOF 训练配置展开，只做站立/行走/转弯训练所需的最小增量。
    """

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.92]
        default_joint_angles = {
            "hipPitch_Left": -0.32,
            "hipRoll_Left": 0.0,
            "hipYaw_Left": -0.18,
            "kneePitch_Left": 0.66,
            "anklePitch_Left": -0.29,
            "ankleRoll_Left": -0.0,

            "hipPitch_Right": -0.32,
            "hipRoll_Right": 0.0,
            "hipYaw_Right": 0.18,
            "kneePitch_Right": 0.66,
            "anklePitch_Right": -0.29,
            "ankleRoll_Right": 0.0,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12
        episode_length_s = 10

    class commands(LeggedRobotCfg.commands):
        # Unitree velocity_env_cfg 思路：
        # 先用很小的速度命令训练稳定性，再通过 curriculum 慢慢放大到 limit_ranges。
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 8.0

        # 原 adam_pro 使用 heading_command=True；hybrid 需要直接训练 yaw
        # velocity，才能覆盖“原地转弯”模式。这是命令侧的必要最小改动。
        heading_command = False

        # hybrid 增量：站立 / 原地转弯 / 普通行走采样。
        stand_probability = 0.4
        turn_probability = 0.2
        transition_time = 0.25
        stand_threshold = 0.05
        walk_threshold = 0.15
        yaw_motion_scale = 0.5
        small_command_threshold = 0.05
        command_curriculum_step = 0.05
        command_curriculum_tracking_lin_threshold = 0.75
        command_curriculum_tracking_ang_threshold = 0.65

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.1, 0.1]
            lin_vel_y = [-0.1, 0.1]
            ang_vel_yaw = [-0.1, 0.1]
            heading = [-0.0, 0.0]

        class limit_ranges:
            lin_vel_x = [-0.5, 0.8]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.4, 0.4]
            heading = [-0.0, 0.0]

    class gait:
        period = 0.8
        phase_offset = 0.5

    class safety:
        pos_limit = 0.9
        vel_limit = 0.9
        torque_limit = 0.85

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 2
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 5
        num_cols = 20
        terrain_proportions = [0.5, 0.5, 0.0, 0.0, 0.0]
        slope_treshold = 0.75

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.3, 1.0]
        randomize_base_mass = True
        added_mass_range = [-0.05, 0.05]

        # Unitree velocity_env_cfg 思路：push 是固定 interval event，
        # 不根据 reward 自动升级，避免 hybrid 站/转样本把 push 过早拉满。
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.2
        curriculum = True
        max_push_vel_xy_curriculum = 0.5
        initial_push_vel_xy = 0.0

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hipPitch": 305.0,
            "hipRoll": 700.0,
            "hipYaw": 405.0,
            "kneePitch": 305.0,
            "anklePitch": 30.0,
            "ankleRoll": 0.0,
        }
        damping = {
            "hipPitch": 6.1,
            "hipRoll": 30.0,
            "hipYaw": 6.1,
            "kneePitch": 6.1,
            "anklePitch": 3.0,
            "ankleRoll": 0.35,
        }
        action_scale = 0.5
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/adam_pro/adam_pro_12dof_simplified_collision.urdf"
        name = "adam_pro_12dof"
        foot_name = "toe"
        penalize_contacts_on = ["pelvis", "shin", "shoulder", "elbow", "thigh", "torso"]
        terminate_after_contacts_on = ["pelvis", "knee", "hip", "torso"]
        self_collisions = 0
        flip_visual_attachments = True

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.88
        min_dist = 0.2
        max_dist = 0.5

        class scales(LeggedRobotCfg.rewards.scales):
            # 原 adam_pro_12dof 奖励项。
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            base_height = -10.0

            dof_acc = -2.5e-7
            dof_vel = -1e-3
            dof_pos_limits = -5.0
            ankle_pos = -10.0
            action_rate = -0.01

            feet_air_time = 0.05
            feet_swing_height = -30.0
            contact = 1.0
            contact_no_vel = -0.1
            feet_distance = 1.5
            orientation = 1.0

            action_smoothness = -0.01
            feet_lateral_deviation = -5.0

            collision = -1.0
            alive = 0.15

            # hybrid 最小增量：站立段专用奖励。
            stand_still = -1.0
            stand_contact = 1.0

            # termination 在正奖励裁剪后加入。
            termination = -5.0

            # 显式关掉 base cfg 默认项，避免独立继承后意外启用。
            torques = 0.0
            feet_stumble = 0.0

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.4

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1


class AdamPro12dofHybridCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [128, 64]
        activation = "elu"
        rnn_type = "lstm"
        rnn_hidden_size = 128
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        schedule = "adaptive"
        num_learning_epochs = 5
        num_mini_batches = 4
        desired_kl = 0.01
        clip_param = 0.2

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 8000
        run_name = "hybrid_stand_walk_turn"
        experiment_name = "adam_pro_12dof_hybrid"
        num_steps_per_env = 32

        save_interval = 100

        resume = False
        load_run = -1
        checkpoint = -1

        use_wandb = True
        wandb_project = "pnd_adam_pro_12dof_locomotion"
        wandb_entity = None
        wandb_tags = ["adam_pro_12dof", "hybrid", "stand", "walk", "turn", "rough_terrain", "lstm"]
