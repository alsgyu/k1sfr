"""Privileged goal-in-context teacher for rsl_rl 2.3.x (the version scripts/rsl_rl/train.py targets).

``OnPolicyRunner`` resolves ``policy.class_name`` with ``eval`` inside its own
module, so the class is registered there on import.  Task registration imports
this package, which means the stock ``scripts/rsl_rl/train.py`` trains it as is.
"""

from __future__ import annotations

import warnings
from importlib.metadata import version

import rsl_rl.runners.on_policy_runner as _on_policy_runner
from rsl_rl.modules import ActorCritic

from .networks import GoalInContextActor

_RSL_RL_VERSION = version("rsl-rl-lib")
if not _RSL_RL_VERSION.startswith("2.3."):
    warnings.warn(f"VIGOR teacher was written against rsl-rl-lib 2.3.x, found {_RSL_RL_VERSION}.")


class VigorTeacherActorCritic(ActorCritic):
    """ActorCritic whose actor is F(k_t, h_t) -> z_g followed by pi_theta(z_g, o_t)."""

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, proprio_dim: int, latent_dim: int = 32,
                 encoder_hidden_dims=(256, 128), actor_hidden_dims=(512, 256, 128),
                 critic_hidden_dims=(512, 256, 128), activation="elu", init_noise_std=1.0,
                 noise_std_type: str = "scalar", **kwargs):
        super().__init__(num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims=list(actor_hidden_dims),
                         critic_hidden_dims=list(critic_hidden_dims), activation=activation,
                         init_noise_std=init_noise_std, noise_std_type=noise_std_type, **kwargs)
        self.actor = GoalInContextActor(
            proprio_dim, num_actor_obs - proprio_dim, num_actions, latent_dim,
            list(encoder_hidden_dims), list(actor_hidden_dims), activation,
        )
        print(f"[VIGOR] teacher actor: {self.actor}")


_on_policy_runner.VigorTeacherActorCritic = VigorTeacherActorCritic
