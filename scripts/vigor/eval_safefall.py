"""Evaluate / watch the K1 safe-fall policy (Isaac).

    # numbers over 256 falls (head hit rate, peak impacts, arm load share), split by fall direction
    python scripts/vigor/eval_safefall.py --run flat --headless
    # watch forward falls in the GUI at real speed, camera on env 0
    python scripts/vigor/eval_safefall.py --run flat --direction forward --follow --real_time --num_envs 16
    # passive baseline (zero actions = hold the walking stance)
    python scripts/vigor/eval_safefall.py --passive --headless

Writes logs/vigor/eval/<timestamp>_safefall/summary.json (+ videos/ with --video).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate the K1 safe-fall policy.")
parser.add_argument("--task", type=str, default="Booster-K1-SafeFall-Flat-Play-v0")
parser.add_argument("--run", type=str, default=None, help="run folder regex (latest if omitted)")
parser.add_argument("--checkpoint", type=str, default=None, help="file name pattern or absolute path")
parser.add_argument("--passive", action="store_true", default=False, help="zero actions (hold the stance)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--direction", choices=["forward", "backward", "left", "right"], default=None)
parser.add_argument("--follow", action="store_true", default=False, help="camera follows env 0")
parser.add_argument("--real_time", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=750)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import time
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import JY_train.tasks  # noqa: F401
from JY_train.tasks.manager_based.vigor import safefall_math as sf

from common import resolve_checkpoint


def main():
    device = args_cli.device or "cuda:0"
    if args_cli.direction:
        sf.DIRECTIONS = tuple((a, float(name == args_cli.direction)) for (a, _), name in zip(sf.DIRECTIONS, sf.DIRECTION_NAMES))
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    if args_cli.follow:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (1.6, -2.2, 0.9)
        env_cfg.viewer.lookat = (0.0, 0.0, -0.1)
    out_dir = os.path.abspath(os.path.join("logs", "vigor", "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_safefall"))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, os.path.join(out_dir, "videos"), step_trigger=lambda s: s == 0,
                                       video_length=args_cli.video_length, disable_logger=True)
    agent = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    base = env.unwrapped
    path = "passive"
    if args_cli.passive:
        policy = lambda obs: torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)  # noqa: E731
    else:
        path = resolve_checkpoint(os.path.abspath(os.path.join("logs", "rsl_rl", agent.experiment_name)), args_cli.run, args_cli.checkpoint)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=device)
        runner.load(path, load_optimizer=False)
        policy = runner.get_inference_policy(device=device)
    print(f"[SafeFall] evaluating {path}")

    monitor = base.command_manager.get_term("monitor")
    monitor.collect = True
    obs, _ = env.get_observations()
    with torch.inference_mode():
        while simulation_app.is_running() and len(monitor.finished) < args_cli.episodes:
            t0 = time.time()
            obs, _, _, _ = env.step(policy(obs))
            if args_cli.real_time:
                time.sleep(max(0.0, base.step_dt - (time.time() - t0)))
    summary = sf.summarize(monitor.finished[: args_cli.episodes]) | {"checkpoint": path, "task": args_cli.task}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary | {"episodes_detail": monitor.finished[: args_cli.episodes]}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[SafeFall] wrote {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
