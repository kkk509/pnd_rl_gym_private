from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import time

from pndbotics_sdk_py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from pndbotics_sdk_py.idl.default import (
    pnd_adam_msg_dds__HandCmd_,
    pnd_adam_msg_dds__LowCmd_,
    pnd_adam_msg_dds__LowState_,
)
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import HandCmd_, LowCmd_, LowState_

from common.command_helper import create_damping_cmd, init_cmd_adam, MotorMode
from common.remote_controller import KeyMap, RemoteController
from common.rotation_helper import get_gravity_orientation, transform_imu_data, ypr_to_quaternion
from config import Config


class OnnxLstmPolicy:
    def __init__(self, policy_path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required on Jetson. Install the Jetson-compatible "
                "onnxruntime-gpu package, or install onnxruntime for CPU testing."
            ) from exc

        providers = []
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(policy_path, providers=providers)
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

        required_inputs = {"obs", "h_in", "c_in"}
        required_outputs = {"action", "h_out", "c_out"}
        if not required_inputs.issubset(set(self.input_names)):
            raise ValueError(f"ONNX inputs must include {required_inputs}, got {self.input_names}")
        if not required_outputs.issubset(set(self.output_names)):
            raise ValueError(f"ONNX outputs must include {required_outputs}, got {self.output_names}")

        h_shape = self.session.get_inputs()[self.input_names.index("h_in")].shape
        c_shape = self.session.get_inputs()[self.input_names.index("c_in")].shape
        self.h = np.zeros(self._resolve_state_shape(h_shape), dtype=np.float32)
        self.c = np.zeros(self._resolve_state_shape(c_shape), dtype=np.float32)

        print("ONNX Runtime providers:", self.session.get_providers())
        print("ONNX inputs:", self.input_names)
        print("ONNX outputs:", self.output_names)
        print("LSTM h shape:", self.h.shape)
        print("LSTM c shape:", self.c.shape)

    @staticmethod
    def _resolve_state_shape(shape):
        # Exported model should be [num_layers, batch, hidden_size].
        # Dynamic axes may appear as strings or None; real deployment uses batch=1.
        resolved = []
        for dim in shape:
            if isinstance(dim, int):
                resolved.append(dim)
            else:
                resolved.append(1)
        return tuple(resolved)

    def reset(self):
        self.h.fill(0.0)
        self.c.fill(0.0)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None, :]
        obs = obs.astype(np.float32, copy=False)

        action, h_out, c_out = self.session.run(
            ["action", "h_out", "c_out"],
            {
                "obs": obs,
                "h_in": self.h,
                "c_in": self.c,
            },
        )
        self.h = h_out.astype(np.float32, copy=False)
        self.c = c_out.astype(np.float32, copy=False)
        return action.squeeze(0).astype(np.float32, copy=False)


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        self.policy = OnnxLstmPolicy(config.policy_path)

        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.counter = 0

        if config.msg_type == "adam_lite":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(23)
            self.low_state = pnd_adam_msg_dds__LowState_(23)
            self.hand_cmd = None
            self.hand_pub = None
        elif config.msg_type == "adam_pro":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(31)
            self.low_state = pnd_adam_msg_dds__LowState_(31)
            self.hand_cmd = pnd_adam_msg_dds__HandCmd_()
            self.close_hand = np.array([500] * 12, dtype=int)
            self.hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
            self.hand_pub.Init()
        else:
            raise ValueError(f"Invalid msg_type: {config.msg_type}")

        self.mode_pr_ = MotorMode.PR

        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.wait_for_low_state()
        init_cmd_adam(self.low_cmd, self.mode_pr_)

    def low_state_handler(self, msg: LowState_):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmd_):
        self.lowcmd_publisher_.Write(cmd)

    def publish_hand_cmd(self):
        if self.hand_pub is None:
            return
        for i in range(12):
            self.hand_cmd.position[i] = self.close_hand[i]
        self.hand_pub.Write(self.hand_cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            print("wait for low state")
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        self.policy.reset()

        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)

        init_dof_pos = np.zeros(len(dof_idx), dtype=np.float32)
        for i, motor_idx in enumerate(dof_idx):
            init_dof_pos[i] = self.low_state.motor_state[motor_idx].q

        for step in range(num_step):
            alpha = step / num_step
            for j, motor_idx in enumerate(dof_idx):
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1.0 - alpha) + default_pos[j] * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0

            self.publish_hand_cmd()
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        self.policy.reset()

        while self.remote_controller.button[KeyMap.A] != 1:
            for i, motor_idx in enumerate(self.config.leg_joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0

            for i, motor_idx in enumerate(self.config.arm_waist_joint2motor_idx):
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0

            self.publish_hand_cmd()
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

        self.policy.reset()
        self.counter = 0
        self.action.fill(0.0)

    def run(self):
        self.publish_hand_cmd()
        self.counter += 1

        for i, motor_idx in enumerate(self.config.leg_joint2motor_idx):
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq

        quat = ypr_to_quaternion(
            self.low_state.imu_state.ypr[0],
            self.low_state.imu_state.ypr[1],
            self.low_state.imu_state.ypr[2],
        )
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel,
            )

        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = (self.qj - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = self.dqj * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2.0 * np.pi * phase)
        cos_phase = np.cos(2.0 * np.pi * phase)

        self.cmd[0] = self.remote_controller.get_walk_x_direction_speed()
        self.cmd[1] = self.remote_controller.get_walk_y_direction_speed()
        self.cmd[2] = self.remote_controller.get_walk_yaw_direction_speed()

        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + 2 * num_actions] = dqj_obs
        self.obs[9 + 2 * num_actions : 9 + 3 * num_actions] = self.action
        self.obs[9 + 3 * num_actions] = sin_phase
        self.obs[9 + 3 * num_actions + 1] = cos_phase

        self.action = self.policy(self.obs)
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        for i, motor_idx in enumerate(self.config.leg_joint2motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        for i, motor_idx in enumerate(self.config.arm_waist_joint2motor_idx):
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0.0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0.0

        # Keep the same ankle special handling as deploy_real.py.
        self.low_cmd.motor_cmd[4].q *= 0.5
        self.low_cmd.motor_cmd[10].q *= 0.5
        self.low_cmd.motor_cmd[5].q = self.low_state.motor_state[5].q
        self.low_cmd.motor_cmd[11].q = self.low_state.motor_state[11].q

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)

    def damping_state(self):
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface connected to the robot")
    parser.add_argument("config", type=str, help="config file name in deploy/deploy_real/configs")
    args = parser.parse_args()

    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(1, args.net)
    controller = Controller(config)

    try:
        controller.zero_torque_state()
        controller.move_to_default_pos()
        controller.default_pos_state()

        while True:
            controller.run()
            if controller.remote_controller.button[KeyMap.B] == 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        controller.damping_state()
        print("Exit")
