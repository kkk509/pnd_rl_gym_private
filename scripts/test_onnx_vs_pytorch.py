import argparse
import copy
import sys

import isaacgym  # noqa: F401
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F403,F401
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


def parse_custom_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--onnx_path",
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/adam_pro_12dof/exported/onnx/policy_lstm.onnx",
        help="Path to the exported LSTM ONNX model.",
    )
    parser.add_argument("--num_tests", type=int, default=20, help="Number of sequential inference steps.")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Maximum allowed absolute error.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for test observations.")
    custom_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return custom_args


def load_pytorch_policy(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.test = True

    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
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
    return model, env_cfg, train_cfg


def main():
    custom_args = parse_custom_args()
    args = get_args()
    if not args.task:
        args.task = "adam_pro_12dof"
    args.headless = True

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: pip install onnx onnxruntime"
        ) from exc

    onnx_model = onnx.load(custom_args.onnx_path)
    onnx.checker.check_model(onnx_model)

    torch_model, env_cfg, train_cfg = load_pytorch_policy(args)

    obs_dim = env_cfg.env.num_observations
    action_dim = env_cfg.env.num_actions
    hidden_size = train_cfg.policy.rnn_hidden_size
    num_layers = train_cfg.policy.rnn_num_layers

    sess = ort.InferenceSession(custom_args.onnx_path, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(custom_args.seed)
    h_np = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    c_np = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)

    max_action_diff = 0.0
    max_h_diff = 0.0
    max_c_diff = 0.0

    print("Comparing PyTorch wrapper and ONNX:")
    print(f"  task:        {args.task}")
    print(f"  onnx_path:   {custom_args.onnx_path}")
    print(f"  obs_dim:     {obs_dim}")
    print(f"  action_dim:  {action_dim}")
    print(f"  lstm state:  ({num_layers}, 1, {hidden_size})")
    print(f"  num_tests:   {custom_args.num_tests}")
    print(f"  tolerance:   {custom_args.tolerance:g}")

    for i in range(custom_args.num_tests):
        if i == 0:
            obs_np = np.zeros((1, obs_dim), dtype=np.float32)
        else:
            obs_np = rng.normal(0.0, 0.2, size=(1, obs_dim)).astype(np.float32)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np)
            h_t = torch.from_numpy(h_np)
            c_t = torch.from_numpy(c_np)
            action_t, h_t_next, c_t_next = torch_model(obs_t, h_t, c_t)

        action_onnx, h_onnx, c_onnx = sess.run(
            ["action", "h_out", "c_out"],
            {
                "obs": obs_np,
                "h_in": h_np,
                "c_in": c_np,
            },
        )

        action_pt = action_t.numpy()
        h_pt = h_t_next.numpy()
        c_pt = c_t_next.numpy()

        action_diff = float(np.max(np.abs(action_pt - action_onnx)))
        h_diff = float(np.max(np.abs(h_pt - h_onnx)))
        c_diff = float(np.max(np.abs(c_pt - c_onnx)))

        max_action_diff = max(max_action_diff, action_diff)
        max_h_diff = max(max_h_diff, h_diff)
        max_c_diff = max(max_c_diff, c_diff)

        print(
            f"step {i:02d}: "
            f"action_diff={action_diff:.3e}, "
            f"h_diff={h_diff:.3e}, "
            f"c_diff={c_diff:.3e}"
        )

        h_np = h_onnx.astype(np.float32)
        c_np = c_onnx.astype(np.float32)

    max_diff = max(max_action_diff, max_h_diff, max_c_diff)
    print("\nSummary:")
    print(f"  max_action_diff: {max_action_diff:.6e}")
    print(f"  max_h_diff:      {max_h_diff:.6e}")
    print(f"  max_c_diff:      {max_c_diff:.6e}")
    print(f"  max_diff:        {max_diff:.6e}")

    if max_diff > custom_args.tolerance:
        raise SystemExit(f"FAILED: max_diff {max_diff:.6e} > tolerance {custom_args.tolerance:.6e}")

    print("PASSED: ONNX output matches PyTorch wrapper.")


if __name__ == "__main__":
    main()
