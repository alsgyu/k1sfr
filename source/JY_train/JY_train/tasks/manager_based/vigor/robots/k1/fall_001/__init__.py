"""Gym registrations for K1 VIGOR fall safety (see vigor/README.md)."""

import gymnasium as gym

_AGENTS = "JY_train.tasks.manager_based.vigor.agents.rsl_rl_ppo_cfg"


def _register(task_id: str, env_cfg: str):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{_AGENTS}:VigorTeacherPPORunnerCfg",
            "vigor_distill_cfg_entry_point": f"{_AGENTS}:VigorDistillCfg",
        },
    )


_register("Booster-K1-Vigor-Teacher-Flat-v0", "TeacherFlatEnvCfg")
_register("Booster-K1-Vigor-Teacher-Rough-v0", "TeacherRoughEnvCfg")
_register("Booster-K1-Vigor-Teacher-Flat-Play-v0", "TeacherFlatPlayEnvCfg")
_register("Booster-K1-Vigor-Teacher-Rough-Play-v0", "TeacherRoughPlayEnvCfg")
_register("Booster-K1-Vigor-Student-Flat-v0", "StudentFlatEnvCfg")
_register("Booster-K1-Vigor-Student-Rough-v0", "StudentRoughEnvCfg")
_register("Booster-K1-Vigor-Student-Flat-Play-v0", "StudentFlatPlayEnvCfg")
_register("Booster-K1-Vigor-Student-Rough-Play-v0", "StudentRoughPlayEnvCfg")
_register("Booster-K1-Vigor-WalkFall-Play-v0", "WalkFallPlayEnvCfg")
_register("Booster-K1-Vigor-WalkFall-Rough-Play-v0", "WalkFallRoughPlayEnvCfg")
