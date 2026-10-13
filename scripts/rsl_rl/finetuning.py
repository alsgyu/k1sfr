# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--finetune_checkpoint",
    type=str,
    default=None,
    help="Initialize finetuning from policy/normalizer weights without restoring the old PPO optimizer.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import omni
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import JY_train.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _attach_symmetry_normalizers(runner):
    """Expose runner-owned normalizers to raw-space symmetry callbacks.

    This keeps empirical normalization enabled without modifying RSL-RL.  The
    callback module opts in by defining ``NORMALIZER_ATTR``.
    """
    symmetry_cfg = getattr(runner.alg, "symmetry", None)
    if not symmetry_cfg:
        return

    callback = symmetry_cfg.get("data_augmentation_func")
    callback_globals = getattr(callback, "__globals__", {})
    normalizer_attr = callback_globals.get("NORMALIZER_ATTR")
    if not normalizer_attr:
        return

    policy_normalizer = runner.obs_normalizer
    critic_normalizer = runner.privileged_obs_normalizer
    if not callable(getattr(policy_normalizer, "inverse", None)):
        # empirical_normalization=False uses torch.nn.Identity; observations
        # are already raw and the symmetry callback needs no bridge.
        print("[INFO] Symmetry callback is using raw observations (normalization disabled).")
        return
    if not callable(getattr(critic_normalizer, "inverse", None)):
        raise TypeError(
            "Critic empirical normalizer does not provide inverse(); "
            "raw-space symmetry cannot be constructed safely."
        )

    symmetry_env = symmetry_cfg.get("_env", runner.env)
    target_env = symmetry_env.unwrapped
    setattr(
        target_env,
        normalizer_attr,
        {
            "policy": policy_normalizer,
            "critic": critic_normalizer,
        },
    )
    print(
        "[INFO] Attached policy/critic empirical normalizers to the "
        "raw-space symmetry callback."
    )


# Expands a checkpoint tensor only along its final observation dimension.
def _copy_prefix_tensor(source: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    """Copy an old observation prefix while retaining the new tensor defaults."""
    if source.ndim == 0 or source.ndim != target.ndim:
        raise ValueError(f"Cannot expand {name}: checkpoint {tuple(source.shape)} -> model {tuple(target.shape)}.")
    if source.shape[:-1] != target.shape[:-1] or source.shape[-1] >= target.shape[-1]:
        raise ValueError(f"Cannot expand {name}: checkpoint {tuple(source.shape)} -> model {tuple(target.shape)}.")

    expanded = target.clone()
    expanded[..., : source.shape[-1]].copy_(source)
    return expanded


# [stop finetuning] 기존 locomotion network weight를 동일한 finetuning network에 복원한다.
def _load_transfer_policy(runner, checkpoint: dict) -> None:
    """Restore matching policy parameters while supporting a future input-prefix expansion."""
    source_state = checkpoint["model_state_dict"]
    target_state = runner.alg.policy.state_dict()
    expandable_layers = {"actor.0.weight", "critic.0.weight"}

    for name, target in target_state.items():
        if name not in source_state:
            raise KeyError(f"Transfer checkpoint is missing policy parameter {name!r}.")
        source = source_state[name].to(device=target.device, dtype=target.dtype)
        if source.shape == target.shape:
            target_state[name] = source
        elif name in expandable_layers:
            expanded = target.clone()
            expanded.zero_()
            expanded[..., : source.shape[-1]].copy_(source)
            target_state[name] = expanded
        else:
            raise ValueError(
                f"Unexpected policy shape mismatch for {name}: "
                f"checkpoint {tuple(source.shape)} -> model {tuple(target.shape)}."
            )

    runner.alg.policy.load_state_dict(target_state, strict=True)


# [stop finetuning] 기존 empirical-normalization 통계를 새 runner에 복원한다.
def _load_transfer_normalizer(normalizer, source_state: dict, label: str) -> None:
    """Expand old normalization statistics by copying their observation prefix."""
    target_state = normalizer.state_dict()
    for name, target in target_state.items():
        source = source_state.get(name)
        if source is None:
            raise KeyError(f"Transfer checkpoint is missing {label} normalizer parameter {name!r}.")
        source = source.to(device=target.device, dtype=target.dtype)
        if source.shape == target.shape:
            target_state[name] = source
        else:
            target_state[name] = _copy_prefix_tensor(source, target, f"{label}.{name}")
    normalizer.load_state_dict(target_state, strict=True)


# [stop finetuning] 기존 optimizer를 제외하고 policy와 normalization만 가져온다.
def _load_transfer_checkpoint(runner, checkpoint_path: str) -> None:
    """Load old locomotion weights/statistics and start a fresh finetuning optimizer."""
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Transfer checkpoint does not exist: {checkpoint_path}")

    print(f"[INFO]: Initializing finetuning from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=runner.device, weights_only=False)
    _load_transfer_policy(runner, checkpoint)

    if "obs_norm_state_dict" in checkpoint:
        _load_transfer_normalizer(runner.obs_normalizer, checkpoint["obs_norm_state_dict"], "policy")
    if "privileged_obs_norm_state_dict" in checkpoint:
        _load_transfer_normalizer(
            runner.privileged_obs_normalizer,
            checkpoint["privileged_obs_norm_state_dict"],
            "critic",
        )

    print(
        "[INFO]: Transferred policy and available normalization statistics. "
        "PPO optimizer and iteration start fresh."
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.finetune_checkpoint and agent_cfg.resume:
        raise ValueError("Use either --finetune_checkpoint or --resume, not both.")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    _attach_symmetry_normalizers(runner)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if args_cli.finetune_checkpoint:
        _load_transfer_checkpoint(runner, args_cli.finetune_checkpoint)
    elif agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
