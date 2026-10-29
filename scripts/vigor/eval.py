"""Evaluate a VIGOR teacher or student with the paper's recovery metrics (and optionally record a video).

    # teacher (rsl_rl checkpoint)
    python scripts/vigor/eval.py --policy teacher --task Booster-K1-Vigor-Teacher-Flat-Play-v0 --run <regex> --headless
    # student (distill.py checkpoint)
    python scripts/vigor/eval.py --policy student --task Booster-K1-Vigor-Student-Flat-Play-v0 --run <regex> --headless --video

Writes logs/vigor/eval/<timestamp>_<policy>/summary.json (+ videos/).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate VIGOR policies.")
parser.add_argument("--policy", choices=["teacher", "student"], required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--run", type=str, default=None, help="run folder regex (latest if omitted)")
parser.add_argument("--checkpoint", type=str, default=None, help="file name pattern or absolute path")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=500)
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
import torch
from datetime import datetime

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import JY_train.tasks  # noqa: F401
from JY_train.tasks.manager_based.vigor import spec
from JY_train.tasks.manager_based.vigor.mdp.rewards import upright_mask
from JY_train.tasks.manager_based.vigor.modules import VigorStudent
from JY_train.tasks.manager_based.vigor.torch_math import tilt_angle

from common import RecoveryTracker, Teacher, resolve_checkpoint


def main():
    device = args_cli.device or "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    out_dir = os.path.abspath(os.path.join("logs", "vigor", "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.policy}"))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, os.path.join(out_dir, "videos"), step_trigger=lambda s: s == 0,
                                       video_length=args_cli.video_length, disable_logger=True)
    base = env.unwrapped
    om = base.observation_manager
    n_act = base.action_manager.total_action_dim

    if args_cli.policy == "teacher":
        agent = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        root = os.path.abspath(os.path.join("logs", "rsl_rl", agent.experiment_name))
        path = resolve_checkpoint(root, args_cli.run, args_cli.checkpoint)
        teacher = Teacher(path, om.group_obs_dim["policy"][0], om.group_obs_dim["critic"][0], n_act, agent.policy.to_dict(), device)
        act = lambda obs: teacher(obs["policy"])[0]  # noqa: E731
    else:
        cfg = load_cfg_from_registry(args_cli.task, "vigor_distill_cfg_entry_point")
        root = os.path.abspath(os.path.join("logs", "rsl_rl", cfg.experiment_name))
        path = resolve_checkpoint(root, args_cli.run, args_cli.checkpoint)
        ck = torch.load(path, map_location=device, weights_only=False)
        d = ck["distill_cfg"]
        student = VigorStudent(ck["deploy"]["term_dims"], ck["deploy"]["history"], n_act, d["latent_dim"],
                               d["actor_hidden_dims"], d["fusion_hidden_dims"], d["activation"]).to(device).eval()
        student.load_state_dict(ck["student_state_dict"])
        mean, std = ck["obs_mean"].to(device), ck["obs_std"].to(device)
        act = lambda obs: student((obs["student"] - mean) / std)[0]  # noqa: E731
    print(f"[VIGOR] evaluating {args_cli.policy}: {path}")

    robot = base.scene["robot"]
    contact = base.scene.sensors["contact_forces"]
    head_ids = contact.find_bodies(spec.HEAD_BODY_NAMES)[0]
    tracker = RecoveryTracker(base.num_envs, base.step_dt, device)
    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running() and len(tracker.episodes) < args_cli.episodes:
            obs, _, term, trunc, _ = env.step(act(obs))
            forces = contact.data.net_forces_w.norm(dim=-1)
            tracker.step(
                upright=upright_mask(base),
                tilt=tilt_angle(robot.data.projected_gravity_b),
                head_contact=(forces[:, head_ids] > 1.0).any(dim=-1),
                max_contact_force=forces.max(dim=-1).values,
                done=term | trunc,
            )
    summary = tracker.summary() | {"policy": args_cli.policy, "checkpoint": path, "task": args_cli.task}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary | {"episodes_detail": tracker.episodes}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[VIGOR] wrote {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
