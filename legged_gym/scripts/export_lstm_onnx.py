import os
import copy

import isaacgym
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


class LSTMPolicyONNX(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).cpu()
        self.rnn = copy.deepcopy(actor_critic.memory_a.rnn).cpu()

    def forward(self, obs, h_in, c_in):
        out, (h_out, c_out) = self.rnn(obs.unsqueeze(0), (h_in, c_in))
        action = self.actor(out.squeeze(0))
        return action, h_out, c_out


def export(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )

    model = LSTMPolicyONNX(runner.alg.actor_critic)
    model.eval()

    obs_dim = env_cfg.env.num_observations
    action_dim = env_cfg.env.num_actions
    hidden_size = train_cfg.policy.rnn_hidden_size
    num_layers = train_cfg.policy.rnn_num_layers

    obs = torch.zeros(1, obs_dim)
    h = torch.zeros(num_layers, 1, hidden_size)
    c = torch.zeros(num_layers, 1, hidden_size)

    out_dir = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
        "onnx",
    )
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "policy_lstm.onnx")

    torch.onnx.export(
        model,
        (obs, h, c),
        out_path,
        input_names=["obs", "h_in", "c_in"],
        output_names=["action", "h_out", "c_out"],
        opset_version=13,
        dynamic_axes={
            "obs": {0: "batch"},
            "h_in": {1: "batch"},
            "c_in": {1: "batch"},
            "action": {0: "batch"},
            "h_out": {1: "batch"},
            "c_out": {1: "batch"},
        },
    )

    print("Exported ONNX to:", out_path)
    print("obs_dim:", obs_dim)
    print("action_dim:", action_dim)
    print("hidden_size:", hidden_size)
    print("num_layers:", num_layers)


if __name__ == "__main__":
    args = get_args()
    export(args)
