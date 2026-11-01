"""Left-right symmetry augmentation for the K1 safe-fall task (rsl_rl 2.3 ``data_augmentation_func``).

PPO also trains on the mirror image (sagittal plane) of every rollout sample, so mirrored falls get
mirrored motions and a straight forward fall a symmetric one; without it the policy broke symmetry at
random and always planted the right fist further up.  The observation layout is read from the env's
observation manager (term names and sizes; within a term the history is oldest first); joint terms
follow the articulation's joint order, the last action the action order.  Mirroring has to commute
with the observation processing, so the task uses fixed observation scales instead of empirical
normalisation (rsl_rl stores normalised observations before calling this).
"""

from __future__ import annotations

import torch

try:
    from . import safefall_math as sf
    from . import spec
except ImportError:  # loaded as a top-level module by the laptop-side scripts
    import safefall_math as sf
    import spec

# observation term -> how it mirrors (safefall_math.mirror kinds; "actions" = joints in action order)
TERM_KINDS = {
    "base_ang_vel": "axial",
    "projected_gravity": "polar",
    "base_lin_vel": "polar",
    "joint_pos": "joints",
    "joint_vel": "joints",
    "last_action": "actions",
    "heights": "none",
    "contact_forces": "contacts",
    "facing": "none",
    "triggered": "none",
}


def _layout(env, group: str) -> list[tuple[slice, str, int, list[str] | None]]:
    """(slice, mirror kind, values per frame, joint names) per term of an observation group (cached)."""
    cache = env.__dict__.setdefault("_safefall_mirror_layout", {})
    if group not in cache:
        om = env.observation_manager
        joint_names = list(env.scene["robot"].data.joint_names)
        parts, start = [], 0
        for name, dim in zip(om.active_terms[group], om.group_obs_term_dim[group]):
            kind = TERM_KINDS[name]  # KeyError: a new observation term needs a mirror rule here
            names = joint_names if kind == "joints" else list(spec.ACTION_JOINT_NAMES) if kind == "actions" else None
            width = len(names) if names else {"axial": 3, "polar": 3, "contacts": len(sf.CONTACT_GROUPS)}.get(kind, 1)
            parts.append((slice(start, start + dim[0]), "joints" if names else kind, width, names))
            start += dim[0]
        cache[group] = parts
    return cache[group]


def mirror_obs(obs: torch.Tensor, parts) -> torch.Tensor:
    out = obs.clone()
    n = obs.shape[0]
    for sl, kind, width, names in parts:
        out[:, sl] = sf.mirror(obs[:, sl].reshape(n, -1, width), kind, names).reshape(n, -1)
    return out


def data_augmentation_func(obs=None, actions=None, env=None, obs_type: str = "policy"):
    """Originals first, then their mirror images (the rsl_rl 2.3 convention)."""
    base = env.unwrapped
    out_obs = out_actions = None
    if obs is not None:
        out_obs = torch.cat((obs, mirror_obs(obs, _layout(base, obs_type))))
    if actions is not None:
        out_actions = torch.cat((actions, sf.mirror(actions, "joints", spec.ACTION_JOINT_NAMES)))
    return out_obs, out_actions
