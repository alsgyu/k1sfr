"""Walking policy + VIGOR fall-safety student on one 22-DoF K1, switching like the real controller would.

WALK  --(tilt >= 20 deg)-->  FALL (student)  --(upright & still for 1 s)-->  RESUME (walk, zero command,
arms blended back to the walking pose)  --(2 s)-->  WALK.  Random velocity kicks knock the robot over.

Both policies' inputs are built here from the articulation state, using the specs exported
next to each policy, so nothing is re-typed by hand:
    walk:    logs/vigor/walk_spec.json             (scripts/vigor/export_walk_spec.py)
    student: <exported>/vigor_student_<it>_spec.json (scripts/vigor/distill.py / export_policy.py)

    python scripts/vigor/play_walk_fall.py --headless --video \
        --walk_policy logs/rsl_rl/JY_walk_001_symmetry/<run>/exported/<name>.pt \
        --student_policy logs/rsl_rl/k1_vigor_student/<run>/exported/vigor_student_<it>.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Booster-K1-Vigor-WalkFall-Play-v0")
parser.add_argument("--walk_policy", type=str, required=True, help="TorchScript exported by scripts/rsl_rl/play.py")
parser.add_argument("--walk_spec", type=str, default="logs/vigor/walk_spec.json")
parser.add_argument("--student_policy", type=str, required=True, help="TorchScript exported by the VIGOR distillation")
parser.add_argument("--student_spec", type=str, default=None, help="defaults to <student_policy stem>_spec.json")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--duration", type=float, default=60.0, help="seconds of simulated time")
parser.add_argument("--push_interval", type=float, nargs=2, default=(4.0, 7.0))
parser.add_argument("--push_speed", type=float, nargs=2, default=(0.8, 1.8), help="m/s velocity kick")
parser.add_argument("--command_interval", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1500)
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
import math
import os
import torch
from datetime import datetime

from isaaclab_tasks.utils import parse_env_cfg

import JY_train.tasks  # noqa: F401
from JY_train.tasks.manager_based.vigor import spec
from JY_train.tasks.manager_based.vigor.mdp.rewards import head_height
from JY_train.tasks.manager_based.vigor.torch_math import tilt_angle

WALK, FALL, RESUME = 0, 1, 2
WALK_TERMS = ["command", "gait_phase", "projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions"]


def _uniform(lo, hi, shape, device):
    return lo + (hi - lo) * torch.rand(shape, device=device)


def main():
    device = args_cli.device or "cuda:0"
    torch.manual_seed(args_cli.seed)
    with open(args_cli.walk_spec) as f:
        wspec = json.load(f)
    student_spec = args_cli.student_spec or os.path.splitext(args_cli.student_policy)[0] + "_spec.json"
    with open(student_spec) as f:
        sspec = json.load(f)
    if wspec["obs_terms"] != WALK_TERMS:
        raise RuntimeError(f"walking obs terms {wspec['obs_terms']} != {WALK_TERMS}")
    if sspec["spec_version"] != spec.SPEC_VERSION:
        raise RuntimeError(f"student built for {sspec['spec_version']}, code is {spec.SPEC_VERSION}")

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.duration + 10.0
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.8, -2.4, 1.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    out_dir = os.path.abspath(os.path.join("logs", "vigor", "walk_fall", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, os.path.join(out_dir, "videos"), step_trigger=lambda s: s == 0,
                                       video_length=args_cli.video_length, disable_logger=True)
    base = env.unwrapped
    robot = base.scene["robot"]
    N, dt = base.num_envs, base.step_dt
    names = list(robot.joint_names)
    if sspec["joint_names"] != names:
        raise RuntimeError("student was trained on a different articulation joint order")
    if abs(wspec["policy_dt"] - dt) > 1e-6 or abs(sspec["policy_dt"] - dt) > 1e-6:
        raise RuntimeError(f"policy dt mismatch: walk {wspec['policy_dt']}, student {sspec['policy_dt']}, env {dt}")
    ids = lambda ns: torch.tensor([names.index(n) for n in ns], device=device)  # noqa: E731
    env_order = ids(spec.ALL_JOINT_NAMES)  # action term order of the play env
    walk_ids = ids(wspec["action_joint_names"])
    vig_ids = ids(sspec["action_joint_names"])
    arm_ids = ids(spec.ARM_JOINT_NAMES)
    walk_default = torch.tensor(wspec["default_joint_pos"], device=device)
    walk_scale = torch.tensor(wspec["action_scale"], device=device)
    vig_default = torch.tensor(sspec["default_joint_pos"], device=device)
    vig_scale = torch.tensor(sspec["action_scale"], device=device)
    if not torch.allclose(robot.data.default_joint_pos[0], vig_default, atol=1e-4):
        raise RuntimeError("play env default pose differs from the student's training default pose")
    if not torch.allclose(vig_default[walk_ids], walk_default, atol=1e-4):
        print("[WARN] walking and VIGOR leg default poses differ; check K1_FULL_DEFAULT_JOINT_POS")
    internal = torch.tensor(wspec["default_internal_parameters"], device=device)
    term_dims, T = sspec["term_dims"], sspec["history"]

    walk_policy = torch.jit.load(args_cli.walk_policy, map_location=device).eval()
    student = torch.jit.load(args_cli.student_policy, map_location=device).eval()

    mode = torch.full((N,), WALK, dtype=torch.long, device=device)
    cmd = torch.zeros(N, 3, device=device)
    phase = torch.zeros(N, device=device)
    walk_last = torch.zeros(N, len(walk_ids), device=device)
    vig_last = torch.zeros(N, len(vig_ids), device=device)
    hist = torch.zeros(N, T, sum(term_dims), device=device)
    fresh = torch.ones(N, dtype=torch.bool, device=device)
    stable_t = torch.zeros(N, device=device)
    resume_t = torch.zeros(N, device=device)
    arm_from = robot.data.joint_pos[:, arm_ids].clone()
    push_t = _uniform(*args_cli.push_interval, (N,), device)
    cmd_t = torch.zeros(N, device=device)
    fall_start = torch.zeros(N, device=device)
    head_hit = torch.zeros(N, dtype=torch.bool, device=device)
    contact = base.scene.sensors["contact_forces"]
    head_body_ids = contact.find_bodies(spec.HEAD_BODY_NAMES)[0]
    events, t_sim = [], 0.0
    counts = {"pushes": 0}

    obs, _ = env.reset()
    steps = int(args_cli.duration / dt)
    with torch.inference_mode():
        for step in range(steps):
            d = robot.data
            q, dq = d.joint_pos, d.joint_vel
            g, w = d.projected_gravity_b, d.root_ang_vel_b
            tilt = tilt_angle(g)

            # ---- commands (resampled while walking)
            cmd_t -= dt
            new_cmd = cmd_t <= 0
            if new_cmd.any():
                k = int(new_cmd.sum())
                c = torch.stack((_uniform(-0.3, 0.8, k, device), _uniform(-0.3, 0.3, k, device), _uniform(-0.6, 0.6, k, device)), -1)
                c[torch.rand(k, device=device) < 0.2] = 0.0
                cmd[new_cmd] = c
                cmd_t[new_cmd] = args_cli.command_interval
            active_cmd = torch.where((mode == WALK).unsqueeze(-1), cmd, torch.zeros_like(cmd))
            moving = active_cmd.norm(dim=-1) > 0.05
            freq = torch.where(moving, internal[0].expand(N), torch.zeros(N, device=device))
            phase = torch.where(moving, torch.fmod(phase + dt * freq, 1.0), torch.zeros_like(phase))

            # ---- walking policy input (same layout as the walking task's policy group)
            command = torch.cat((active_cmd, freq.unsqueeze(-1), internal[1:].expand(N, -1)), -1)
            gait = torch.stack((torch.cos(2 * math.pi * phase), torch.sin(2 * math.pi * phase)), -1)
            walk_obs = torch.cat((command, gait, g, w, q[:, walk_ids] - walk_default, dq[:, walk_ids], walk_last), -1)
            if step == 0 and walk_obs.shape[-1] != wspec["obs_dim"]:
                raise RuntimeError(f"walking obs {walk_obs.shape[-1]} != spec {wspec['obs_dim']} ({wspec['obs_dims']})")
            walk_act = walk_policy(walk_obs)

            # ---- student input: proprio history kept for every env, also while walking
            frame = torch.cat((w, g, q - vig_default, dq, vig_last), -1)
            hist = torch.roll(hist, shifts=-1, dims=1)
            hist[:, -1] = frame
            hist[fresh] = frame[fresh].unsqueeze(1).expand(-1, T, -1)
            fresh[:] = False
            parts, start = [], 0
            for dd in term_dims:
                parts.append(hist[:, :, start : start + dd].reshape(N, -1))
                start += dd
            vig_act = student(torch.cat(parts, -1))

            # ---- joint targets per mode
            target = vig_default.expand(N, -1).clone()
            walking = mode != FALL
            target[:, walk_ids] = torch.where(walking.unsqueeze(-1), walk_default + walk_scale * walk_act, target[:, walk_ids])
            blend = (resume_t / 0.5).clamp(0, 1).unsqueeze(-1)
            arms = torch.where((mode == RESUME).unsqueeze(-1), arm_from + (vig_default[arm_ids] - arm_from) * blend, vig_default[arm_ids].expand(N, -1))
            target[:, arm_ids] = torch.where(walking.unsqueeze(-1), arms, target[:, arm_ids])
            vig_target = vig_default[vig_ids] + vig_scale * vig_act
            target[:, vig_ids] = torch.where((mode == FALL).unsqueeze(-1), vig_target, target[:, vig_ids])

            walk_last = torch.where(walking.unsqueeze(-1), walk_act, walk_last)
            # the student sees whatever was commanded, expressed in its own action space
            vig_last = torch.where((mode == FALL).unsqueeze(-1), vig_act, (target[:, vig_ids] - vig_default[vig_ids]) / vig_scale)

            obs, _, term, trunc, _ = env.step(target[:, env_order])
            t_sim += dt

            # ---- mode switching
            d = robot.data
            tilt = tilt_angle(d.projected_gravity_b)
            forces = contact.data.net_forces_w.norm(dim=-1)
            head_hit |= (mode == FALL) & (forces[:, head_body_ids] > 1.0).any(-1)
            start_fall = walking & (tilt >= math.radians(spec.ACTIVATION_TILT_DEG))
            for i in start_fall.nonzero(as_tuple=True)[0].tolist():
                fall_start[i], head_hit[i] = t_sim, False
            mode[start_fall] = FALL
            stable_t[start_fall] = 0.0

            upright = (tilt < math.radians(15.0)) & (head_height(base, robot) > 0.85 * spec.HEAD_HEIGHT_NOMINAL) \
                & (d.root_lin_vel_b.norm(dim=-1) < 0.3) & (d.root_ang_vel_b.norm(dim=-1) < 0.6)
            stable_t = torch.where((mode == FALL) & upright, stable_t + dt, torch.zeros_like(stable_t))
            recovered = (mode == FALL) & (stable_t >= 1.0)
            for i in recovered.nonzero(as_tuple=True)[0].tolist():
                events.append({"env": i, "t_fall": round(fall_start[i].item(), 2), "recovered": True,
                               "recovery_time": round(t_sim - 1.0 - fall_start[i].item(), 2), "head_contact": bool(head_hit[i])})
            if recovered.any():
                mode[recovered] = RESUME
                resume_t[recovered] = 0.0
                arm_from[recovered] = d.joint_pos[recovered][:, arm_ids]
                walk_last[recovered] = d.joint_pos[recovered][:, walk_ids] - walk_default
                phase[recovered] = 0.0
            resume_t += dt
            mode[(mode == RESUME) & (resume_t >= 2.0)] = WALK

            # ---- pushes while walking
            push_t -= dt
            kick = (push_t <= 0) & (mode == WALK)
            if kick.any():
                env_ids = kick.nonzero(as_tuple=True)[0]
                ang = _uniform(-math.pi, math.pi, len(env_ids), device)
                speed = _uniform(*args_cli.push_speed, len(env_ids), device)
                vel = torch.cat((d.root_lin_vel_w[env_ids], d.root_ang_vel_w[env_ids]), -1)
                vel[:, 0] += speed * torch.cos(ang)
                vel[:, 1] += speed * torch.sin(ang)
                robot.write_root_velocity_to_sim(vel, env_ids=env_ids)
                push_t[env_ids] = _uniform(*args_cli.push_interval, (len(env_ids),), device)
                counts["pushes"] += len(env_ids)

            done = term | trunc
            if done.any():
                fresh |= done
                mode[done] = WALK
                walk_last[done] = 0.0
                vig_last[done] = 0.0
            if step % 250 == 0:
                print(f"t={t_sim:6.1f}s  walking {(mode == WALK).sum().item():3d}  falling {(mode == FALL).sum().item():3d}  "
                      f"resuming {(mode == RESUME).sum().item():3d}  recoveries {len(events)}")

    still_down = (mode == FALL).nonzero(as_tuple=True)[0].tolist()
    for i in still_down:
        events.append({"env": i, "t_fall": round(fall_start[i].item(), 2), "recovered": False,
                       "time_on_ground": round(t_sim - fall_start[i].item(), 2), "head_contact": bool(head_hit[i])})
    rec = [e for e in events if e["recovered"]]
    summary = {
        "sim_time_s": round(t_sim, 1), "envs": N, "pushes": counts["pushes"], "falls": len(events),
        "recovered": len(rec), "not_recovered_at_end": len(still_down),
        "mean_recovery_time_s": round(sum(e["recovery_time"] for e in rec) / len(rec), 2) if rec else None,
        "head_contact_falls": sum(e["head_contact"] for e in events),
        "walk_policy": os.path.abspath(args_cli.walk_policy), "student_policy": os.path.abspath(args_cli.student_policy),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary | {"events": events}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[VIGOR] wrote {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
