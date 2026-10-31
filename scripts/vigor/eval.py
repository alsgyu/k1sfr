"""Evaluate a VIGOR teacher or student with the paper's recovery metrics (and optionally record a video).

    # teacher (rsl_rl checkpoint)
    python scripts/vigor/eval.py --policy teacher --task Booster-K1-Vigor-Teacher-Flat-Play-v0 --run <regex> --headless
    # student (distill.py checkpoint)
    python scripts/vigor/eval.py --policy student --task Booster-K1-Vigor-Student-Flat-Play-v0 --run <regex> --headless --video
    # watch only push starts in the GUI at real speed, camera on env 0
    python scripts/vigor/eval.py --policy teacher --task Booster-K1-Vigor-Teacher-Flat-Play-v0 --regime push --follow --real_time

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
parser.add_argument("--regime", choices=["fall", "standup", "handover", "push"], default=None,
                    help="start every episode in this regime (default: the training mix)")
parser.add_argument("--follow", action="store_true", default=False, help="camera follows env 0 instead of the grid")
parser.add_argument("--real_time", action="store_true", default=False, help="slow the GUI down to real time")
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

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import JY_train.tasks  # noqa: F401
from JY_train.tasks.manager_based.vigor import spec
from JY_train.tasks.manager_based.vigor.mdp.commands import REGIME_NAMES
from JY_train.tasks.manager_based.vigor.mdp.rewards import trunk_height, upright_mask
from JY_train.tasks.manager_based.vigor.modules import VigorStudent

from common import RecoveryTracker, Teacher, resolve_checkpoint


def main():
    device = args_cli.device or "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    if args_cli.regime:
        env_cfg.commands.motion.regime_probs = tuple(float(n == args_cli.regime) for n in REGIME_NAMES)
    if args_cli.follow:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (1.6, -2.2, 0.9)
        env_cfg.viewer.lookat = (0.0, 0.0, -0.1)
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
    command = base.command_manager.get_term("motion")
    tracker = RecoveryTracker(base.num_envs, base.step_dt, device, spec.FULL_FALL_TRUNK_HEIGHT)
    obs, _ = env.reset()
    ep_regime = command.regime.clone()  # regime of the running episode (the command resamples on reset)
    with torch.inference_mode():
        while simulation_app.is_running() and len(tracker.episodes) < args_cli.episodes:
            t0 = time.time()
            obs, _, term, trunc, _ = env.step(act(obs))
            forces = contact.data.net_forces_w.norm(dim=-1)
            done = term | trunc
            tracker.step(
                upright=upright_mask(base),
                trunk_height=trunk_height(base, robot),
                head_contact=(forces[:, head_ids] > 1.0).any(dim=-1),
                max_contact_force=forces.max(dim=-1).values,
                done=done,
                regime_names=[REGIME_NAMES[r] for r in ep_regime.tolist()],
            )
            ep_regime = torch.where(done, command.regime, ep_regime)
            if args_cli.real_time:
                time.sleep(max(0.0, base.step_dt - (time.time() - t0)))
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
