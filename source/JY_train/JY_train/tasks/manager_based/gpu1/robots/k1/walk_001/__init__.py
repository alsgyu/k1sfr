"""Gym registrations for K1 locomotion symmetry experiments."""

import gymnasium as gym


def _register(task_id: str, agent_cfg: str, env_cfg: str = "RoughWoStateEstimationEnvCfg"):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:{agent_cfg}",
        },
    )


_register("jy_gpu1_v0", "SymmetryPPORunnerCfg")
_register("jy_gpu1_loss_v0", "MirrorLossPPORunnerCfg")
_register("jy_gpu1_aug_v0", "MirrorAugPPORunnerCfg")
_register(
    "jy_gpu1_v0-Play",
    "SymmetryPPORunnerCfg",
    "PlayFlatWoStateEstimationEnvCfg",
)
