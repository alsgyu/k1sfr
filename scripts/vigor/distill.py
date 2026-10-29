"""Distill the privileged VIGOR teacher into the deployable student (paper Sec. III-D).

    python scripts/vigor/distill.py --task Booster-K1-Vigor-Student-Rough-v0 --teacher_run <teacher run regex> --headless

Rollouts mix teacher- and student-driven envs (rho = a*rho_teacher + (1-a)*rho_student,
a annealed 1 -> 0) and every visited state is labelled by the teacher with its
goal-in-context latent z_g and action a_t:
    L = lambda_z ||z~_g - z_g||^2 + lambda_a ||a~_t - a_t||^2
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VIGOR student distillation.")
parser.add_argument("--task", type=str, default="Booster-K1-Vigor-Student-Flat-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--teacher_run", type=str, default=None, help="regex of the teacher run folder (latest if omitted)")
parser.add_argument("--teacher_checkpoint", type=str, default=None, help="file name pattern or absolute path")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--resume", type=str, default=None, help="student checkpoint to continue from")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from datetime import datetime

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import JY_train.tasks  # noqa: F401
from JY_train.tasks.manager_based.vigor import spec
from JY_train.tasks.manager_based.vigor.modules import VigorStudent

from common import Teacher, export_student, resolve_checkpoint, student_layout_stats


def main():
    cfg = load_cfg_from_registry(args_cli.task, "vigor_distill_cfg_entry_point")
    teacher_agent = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.max_iterations is not None:
        cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed
    device = args_cli.device or cfg.device
    torch.manual_seed(cfg.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = cfg.seed
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    om = env.observation_manager
    n_obs, n_critic = om.group_obs_dim["policy"][0], om.group_obs_dim["critic"][0]
    n_student, n_act = om.group_obs_dim["student"][0], env.action_manager.total_action_dim
    term_dims = [d for _, d in spec.PROPRIO_TERMS]
    assert n_obs == spec.TEACHER_OBS_DIM, f"teacher obs {n_obs} != spec {spec.TEACHER_OBS_DIM}"
    assert n_student == cfg.history * spec.PROPRIO_DIM, f"student obs {n_student} != {cfg.history}x{spec.PROPRIO_DIM}"
    assert [d[0] for d in om.group_obs_term_dim["student"]] == [cfg.history * d for d in term_dims]

    teacher_root = os.path.abspath(os.path.join("logs", "rsl_rl", cfg.teacher_experiment_name))
    teacher_path = resolve_checkpoint(teacher_root, args_cli.teacher_run, args_cli.teacher_checkpoint)
    teacher = Teacher(teacher_path, n_obs, n_critic, n_act, teacher_agent.policy.to_dict(), device)
    print(f"[VIGOR] teacher: {teacher_path}")

    student = VigorStudent(term_dims, cfg.history, n_act, cfg.latent_dim, cfg.actor_hidden_dims,
                           cfg.fusion_hidden_dims, cfg.activation).to(device)
    # Student inputs use the teacher's proprio statistics (fixed), so its current-frame
    # input matches what the copied policy head saw during PPO.
    mean, std = student_layout_stats(*teacher.proprio_stats(spec.PROPRIO_DIM), term_dims, cfg.history)
    obs_mean, obs_std = mean.to(device), std.to(device)
    start_it = 0
    if args_cli.resume:
        ck = torch.load(args_cli.resume, map_location=device, weights_only=False)
        student.load_state_dict(ck["student_state_dict"])
        obs_mean, obs_std, start_it = ck["obs_mean"].to(device), ck["obs_std"].to(device), ck["iteration"]
    elif cfg.init_head_from_teacher:
        student.head.load_state_dict(teacher.policy.actor.head.state_dict())
        print("[VIGOR] student policy head initialised from the teacher head")
    optim = torch.optim.Adam(student.parameters(), lr=cfg.learning_rate)

    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", cfg.experiment_name,
                                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args_cli.run_name}" if args_cli.run_name else "")))
    os.makedirs(log_dir, exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "distill.yaml"), cfg)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir)
    robot = env.scene["robot"]
    act_term = env.action_manager.get_term("joint_pos")
    scale = act_term._scale if torch.is_tensor(act_term._scale) else torch.full((env.num_envs, n_act), float(act_term._scale))
    deploy = {
        "joint_names": list(robot.joint_names),
        "action_joint_names": list(spec.ACTION_JOINT_NAMES),
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "action_scale": scale[0].tolist(),
        "stiffness": robot.data.default_joint_stiffness[0].tolist(),
        "damping": robot.data.default_joint_damping[0].tolist(),
        "policy_dt": env.step_dt,
        "history": cfg.history,
        "term_names": [n for n, _ in spec.PROPRIO_TERMS],
        "term_dims": term_dims,
        "activation_tilt_deg": spec.ACTIVATION_TILT_DEG,
    }
    print_dict(deploy, nesting=2)

    obs, _ = env.reset()
    ep_ret = torch.zeros(env.num_envs, device=device)
    ep_len = torch.zeros(env.num_envs, device=device)
    finished_ret, finished_len = [], []
    S, N = cfg.num_steps_per_env, env.num_envs
    buf_obs = torch.zeros(S, N, n_student, device=device)
    buf_z = torch.zeros(S, N, cfg.latent_dim, device=device)
    buf_a = torch.zeros(S, N, n_act, device=device)
    anneal = max(1, int(cfg.alpha_anneal_fraction * cfg.max_iterations))

    for it in range(start_it, cfg.max_iterations):
        t0 = time.time()
        alpha = max(0.0, 1.0 - it / anneal)
        teacher_driven = (torch.rand(N, device=device) < alpha).unsqueeze(-1)
        student.eval()
        for s in range(S):
            with torch.no_grad():
                a_t, z_t = teacher(obs["policy"])
                x = (obs["student"] - obs_mean) / obs_std
                a_s, _ = student(x)
            buf_obs[s], buf_z[s], buf_a[s] = x, z_t, a_t
            obs, rew, term, trunc, _ = env.step(torch.where(teacher_driven, a_t, a_s))
            ep_ret += rew
            ep_len += 1
            done = (term | trunc).nonzero(as_tuple=True)[0]
            if len(done):
                finished_ret += ep_ret[done].tolist()
                finished_len += ep_len[done].tolist()
                ep_ret[done] = 0
                ep_len[done] = 0
        collect = time.time() - t0

        student.train()
        flat_obs, flat_z, flat_a = buf_obs.view(-1, n_student), buf_z.view(-1, cfg.latent_dim), buf_a.view(-1, n_act)
        mb = flat_obs.shape[0] // cfg.num_mini_batches
        stats = {"latent": 0.0, "action": 0.0}
        for _ in range(cfg.num_learning_epochs):
            perm = torch.randperm(flat_obs.shape[0], device=device)
            for i in range(cfg.num_mini_batches):
                idx = perm[i * mb : (i + 1) * mb]
                a_pred, z_pred = student(flat_obs[idx])
                l_z = (z_pred - flat_z[idx]).square().sum(-1).mean()
                l_a = (a_pred - flat_a[idx]).square().sum(-1).mean()
                loss = cfg.lambda_latent * l_z + cfg.lambda_action * l_a
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                optim.step()
                stats["latent"] += l_z.item()
                stats["action"] += l_a.item()
        k = cfg.num_learning_epochs * cfg.num_mini_batches
        writer.add_scalar("Loss/latent", stats["latent"] / k, it)
        writer.add_scalar("Loss/action", stats["action"] / k, it)
        writer.add_scalar("Distill/alpha", alpha, it)
        if finished_ret:
            writer.add_scalar("Train/mean_reward", sum(finished_ret[-200:]) / len(finished_ret[-200:]), it)
            writer.add_scalar("Train/mean_episode_length", sum(finished_len[-200:]) / len(finished_len[-200:]), it)
        if it % 10 == 0:
            r = sum(finished_ret[-200:]) / max(1, len(finished_ret[-200:]))
            print(f"[{it:5d}/{cfg.max_iterations}] alpha {alpha:.2f}  L_z {stats['latent'] / k:.4f}  "
                  f"L_a {stats['action'] / k:.4f}  ep_rew {r:7.2f}  collect {collect:.1f}s  total {time.time() - t0:.1f}s")
        if (it + 1) % cfg.save_interval == 0 or it + 1 == cfg.max_iterations:
            path = os.path.join(log_dir, f"model_{it + 1}.pt")
            torch.save({
                "student_state_dict": student.state_dict(),
                "obs_mean": obs_mean.cpu(),
                "obs_std": obs_std.cpu(),
                "iteration": it + 1,
                "distill_cfg": cfg.to_dict(),
                "teacher_checkpoint": teacher_path,
                "spec_version": spec.SPEC_VERSION,
                "deploy": deploy,
            }, path)
    writer.close()
    export_student(path)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
