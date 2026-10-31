"""Gym registrations for the K1 safe-fall task (see vigor/README.md)."""

import gymnasium as gym

_AGENTS = "JY_train.tasks.manager_based.vigor.agents.rsl_rl_ppo_cfg"

for _id, _cfg in (("Booster-K1-SafeFall-Flat-v0", "SafeFallEnvCfg"), ("Booster-K1-SafeFall-Flat-Play-v0", "SafeFallPlayEnvCfg")):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{_AGENTS}:SafeFallPPORunnerCfg",
        },
    )
