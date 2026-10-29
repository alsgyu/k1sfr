"""Dump the walking policy's interface (joint order, defaults, obs layout) from the real walking env.

play_walk_fall.py drives the walking policy on the 22-DoF model, so it must use
exactly the joint order / defaults / observation layout the walking task had.
Reading them from the instantiated env (instead of re-typing them) is what keeps
the two policies in sync.

    python scripts/vigor/export_walk_spec.py --task jy_curriculum_v1-Play --headless --out logs/vigor/walk_spec.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="jy_curriculum_v1-Play")
parser.add_argument("--out", type=str, default="logs/vigor/walk_spec.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch

from isaaclab_tasks.utils import parse_env_cfg

import JY_train.tasks  # noqa: F401

EXPECTED_TERMS = ["command", "gait_phase", "projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"]


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=1)
    # the walking tasks perturb default_joint_pos on reset; record the nominal values
    if getattr(env_cfg.events, "add_joint_default_pos", None) is not None:
        env_cfg.events.add_joint_default_pos = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    joint_ids = term._joint_ids if not isinstance(term._joint_ids, slice) else list(range(robot.num_joints))
    scale = term._scale[0].tolist() if torch.is_tensor(term._scale) else [float(term._scale)] * term.action_dim
    om = env.observation_manager
    names = om.active_terms["policy"]
    dims = [d[0] for d in om.group_obs_term_dim["policy"]]
    if names != EXPECTED_TERMS:
        print(f"[WARN] walking policy terms {names} differ from {EXPECTED_TERMS}; update play_walk_fall.py")
    cmd_cfg = env_cfg.commands.motion
    spec = {
        "task": args_cli.task,
        "robot_joint_names": list(robot.joint_names),
        "action_joint_names": list(term._joint_names),
        "action_scale": scale,
        "use_default_offset": bool(term.cfg.use_default_offset),
        "default_joint_pos": robot.data.default_joint_pos[0, joint_ids].tolist(),
        "stiffness": robot.data.default_joint_stiffness[0, joint_ids].tolist(),
        "damping": robot.data.default_joint_damping[0, joint_ids].tolist(),
        "obs_terms": names,
        "obs_dims": dims,
        "obs_dim": om.group_obs_dim["policy"][0],
        "default_internal_parameters": list(cmd_cfg.default_internal_parameters),
        "gait_frequency_index": 3,
        "policy_dt": env.step_dt,
        "urdf": env_cfg.scene.robot.spawn.asset_path,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(spec, f, indent=2)
    print(json.dumps(spec, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
