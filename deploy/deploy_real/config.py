from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            self.arm_waist_kps = config["arm_waist_kps"]
            self.arm_waist_kds = config["arm_waist_kds"]
            self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]

            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            # Hybrid stand/walk deployment parameters. Defaults keep older
            # configs usable while allowing the hybrid deploy script to match
            # the MuJoCo command smoothing and phase logic.
            self.command_transition_time = config.get("command_transition_time", 0.25)
            self.stand_threshold = config.get("stand_threshold", 0.05)
            self.walk_threshold = config.get("walk_threshold", 0.15)
            self.yaw_motion_scale = config.get("yaw_motion_scale", 0.5)
            self.gait_period = config.get("gait_period", 0.8)
            self.gait_phase_offset = config.get("gait_phase_offset", 0.5)

            # Real-robot safety filters. Use permissive defaults unless a
            # deployment config explicitly tightens them.
            self.action_clip = config.get("action_clip", None)
            self.action_filter_alpha = config.get("action_filter_alpha", 1.0)
            self.lock_ankles = config.get("lock_ankles", False)
            self.use_heading_hold = config.get("use_heading_hold", False)
