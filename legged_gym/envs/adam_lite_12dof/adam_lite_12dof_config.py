from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AdamLite12dofRoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.85]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'hipPitch_Left': -0.1,
            'hipRoll_Left': 0,
            'hipYaw_Left': 0.,
            'kneePitch_Left': 0.3,
            'anklePitch_Left': -0.2,
            'ankleRoll_Left': 0,
            'hipPitch_Right': -0.1,
            'hipRoll_Right': 0,
            'hipYaw_Right': 0.,
            'kneePitch_Right': 0.3,
            'anklePitch_Right': -0.2,
            'ankleRoll_Right': 0,
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12
        episode_length_s = 10

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]
        push_robots = True  # Enable robot pushing for disturbance training
        push_interval_s = 5
        max_push_vel_xy = 1.5
        # Curriculum learning for push_robots
        curriculum = True  # Enable curriculum learning for push disturbances
        max_push_vel_xy_curriculum = 1.5  # Maximum push velocity to reach through curriculum
        initial_push_vel_xy = 0.0  # Initial push velocity (easier at start)

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        resampling_time = 8.0
        heading_command = False
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.6, 0.6]
            heading = [-3.14, 3.14]
      
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters - tuned for stability and tracking
        stiffness = {
            'hipPitch': 150,   # 230 Nm effort in URDF
            'hipRoll': 100,    # 160 Nm effort in URDF
            'hipYaw': 80,      # 105 Nm effort in URDF
            'kneePitch': 180,  # 230 Nm effort in URDF
            'anklePitch': 40,  # 40 Nm effort in URDF
            'ankleRoll': 20,   # 12 Nm effort in URDF
        }  # [N*m/rad]
        damping = {
            'hipPitch': 3.0,
            'hipRoll': 2.5,
            'hipYaw': 2.0,
            'kneePitch': 4.0,
            'anklePitch': 2.0,
            'ankleRoll': 1.5,
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.22
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/adam_lite/adam_lite_12dof.urdf'
        name = "adam_lite_12dof"
        foot_name = "toe"  # The foot links are toeLeft and toeRight
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis", "torso"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.80  # Target height for pelvis
        
        class scales(LeggedRobotCfg.rewards.scales):
            # Velocity Tracking
            tracking_lin_vel = 1.0      # Track commanded linear velocity
            tracking_ang_vel = 0.5      # Track commanded angular velocity
            lin_vel_z = -2.0            # Penalize vertical velocity
            ang_vel_xy = -0.05          # Penalize roll/pitch angular velocity
            
            # Pose Stability
            orientation = -0.5          # Keep body level (penalize roll/pitch)
            base_height = -3.0          # Maintain target height
            
            # Joint Control
            dof_acc = -2.5e-7           # Penalize joint accelerations
            dof_vel = -1e-3             # Penalize high joint velocities
            dof_pos_limits = -5.0       # Penalize approaching joint limits
            hip_pos = -0.5              # Regularize hip positions
            ankle_pos = -1.0            # Keep ankles stable (fix toe-up & roll issues)
            action_rate = -0.01         # Smooth actions (penalize rapid changes)
            
            # Foot Contact & Gait
            feet_air_time = 0.05        # Small value to lift feet without jumping
            feet_swing_height = -30.0   # Ensure proper swing clearance
            contact = 1.0               # Reward foot contact配合实现相位同步）
            contact_no_vel = -0.1       # Penalize foot contact while moving
            
            # Symmetry & Gait Quality
            feet_lateral_deviation = -5.0  # Keep feet pointing forward
            
            # Safety & Survival
            collision = -1.0            # Penalize collisions with forbidden body parts
            alive = 0.15                # Encourage staying alive
            
            # Optional (currently disabled)
            # torques = -0.00001        # Penalize high torques (encourage efficiency)
            # foot_slip = -0.1          # Penalize lateral foot slip during stance

class AdamLite12dofRoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [128, 64]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 128
        rnn_num_layers = 1
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.02
        learning_rate = 8.e-4
        schedule = 'adaptive'
        num_learning_epochs = 3
        num_mini_batches = 8
        desired_kl = 0.006
        clip_param = 0.15
    
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 5000
        run_name = ''
        experiment_name = 'adam_lite_12dof'
        num_steps_per_env = 32
        
        # logging
        save_interval = 100 # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1 #"Oct17_14-14-05_push*" #"Oct16_22-05-20_" #"Oct16_12-36-25_" #"Oct14_21-46-15_*" #"Oct14_21-46-15_*" #"Oct14_21-46-15_" #"Oct15_01-10-49_" #"Oct14_02-52-45_" #"Oct13_02-15-48_" # -1 = last run
        checkpoint = -1# -1 = last saved model

        
        # WandB Configuration
        use_wandb = True  # Enable WandB logging
        wandb_project = 'pnd_adam_lite_12dof_locomotion'  # WandB project name
        wandb_entity = None  # WandB entity (optional)
        wandb_tags = ['adam_lite_12dof', 'rough_terrain', 'lstm']  # Tags

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.4


