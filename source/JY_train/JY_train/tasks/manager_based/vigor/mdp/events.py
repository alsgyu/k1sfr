"""Torque suppression events: free-fall at episode start and partial actuator dropouts (paper Sec. III-A/E).

Gains are zeroed on the explicit PD actuators of the selected joints and
restored later.  Declare ``torque_dropout_clear`` before and
``torque_dropout_on_reset`` after ``randomize_actuator_gains`` so freshly
randomised gains are the ones that get saved and restored.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _DropoutState:
    def __init__(self, env: ManagerBasedRLEnv, asset: Articulation):
        n, j, dev = env.num_envs, asset.num_joints, env.device
        self.until = torch.zeros(n, device=dev)
        self.mask = torch.zeros(n, j, dtype=torch.bool, device=dev)
        self.kp = torch.zeros(n, j, device=dev)
        self.kd = torch.zeros(n, j, device=dev)
        self.groups = []
        for act in asset.actuators.values():
            ids = act.joint_indices
            if isinstance(ids, slice):
                ids = torch.arange(j, device=dev)[ids]
            self.groups.append((act, torch.as_tensor(ids, device=dev)))


def _state(env: ManagerBasedRLEnv, asset: Articulation) -> _DropoutState:
    st = getattr(env, "_vigor_dropout", None)
    if st is None:
        st = _DropoutState(env, asset)
        env._vigor_dropout = st
    return st


def _episode_time(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    return env.episode_length_buf[env_ids].float() * env.step_dt


def _restore(st: _DropoutState, env_ids: torch.Tensor) -> None:
    for act, ids in st.groups:
        m = st.mask[env_ids][:, ids]
        if not m.any():
            continue
        rows = env_ids.unsqueeze(1).expand(-1, len(ids))
        kp, kd = act.stiffness[env_ids], act.damping[env_ids]
        act.stiffness[env_ids] = torch.where(m, st.kp[rows, ids], kp)
        act.damping[env_ids] = torch.where(m, st.kd[rows, ids], kd)
    st.mask[env_ids] = False


def _start(env: ManagerBasedRLEnv, st: _DropoutState, env_ids: torch.Tensor, joint_ids: torch.Tensor,
           joint_fraction: tuple[float, float], duration_s: tuple[float, float]) -> None:
    if len(env_ids) == 0:
        return
    n = len(env_ids)
    frac = joint_fraction[0] + (joint_fraction[1] - joint_fraction[0]) * torch.rand(n, 1, device=env.device)
    pick = torch.zeros(n, st.mask.shape[1], dtype=torch.bool, device=env.device)
    pick[:, joint_ids] = torch.rand(n, len(joint_ids), device=env.device) < frac
    for act, ids in st.groups:
        m = pick[:, ids]
        if not m.any():
            continue
        rows = env_ids.unsqueeze(1).expand(-1, len(ids))
        st.kp[rows, ids] = act.stiffness[env_ids]
        st.kd[rows, ids] = act.damping[env_ids]
        act.stiffness[env_ids] = act.stiffness[env_ids].masked_fill(m, 0.0)
        act.damping[env_ids] = act.damping[env_ids].masked_fill(m, 0.0)
    st.mask[env_ids] = pick
    dur = duration_s[0] + (duration_s[1] - duration_s[0]) * torch.rand(n, device=env.device)
    st.until[env_ids] = _episode_time(env, env_ids) + dur


def torque_dropout_clear(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset mode: undo any dropout still active in the finished episode."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    _restore(_state(env, asset), env_ids)


def torque_dropout_on_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor, probability: float,
                            joint_fraction: tuple[float, float], duration_s: tuple[float, float],
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset mode: short free-fall at episode start before control takes over."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    st = _state(env, asset)
    chosen = env_ids[torch.rand(len(env_ids), device=env.device) < probability]
    joint_ids = torch.arange(asset.num_joints, device=env.device)[asset_cfg.joint_ids]
    # episode_length_buf is zeroed after the events run; time starts at 0
    _start(env, st, chosen, joint_ids, joint_fraction, duration_s)
    st.until[chosen] -= _episode_time(env, chosen)


def torque_dropout_step(env: ManagerBasedRLEnv, env_ids: torch.Tensor, probability_per_s: float,
                        joint_fraction: tuple[float, float], duration_s: tuple[float, float],
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Interval mode (every step): expire dropouts and occasionally start partial actuator failures."""
    asset: Articulation = env.scene[asset_cfg.name]
    st = _state(env, asset)
    env_ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    active = st.mask[env_ids].any(dim=1)
    expired = env_ids[active & (_episode_time(env, env_ids) >= st.until[env_ids])]
    if len(expired):
        _restore(st, expired)
    idle = env_ids[~st.mask[env_ids].any(dim=1)]
    chosen = idle[torch.rand(len(idle), device=env.device) < probability_per_s * env.step_dt]
    joint_ids = torch.arange(asset.num_joints, device=env.device)[asset_cfg.joint_ids]
    _start(env, st, chosen, joint_ids, joint_fraction, duration_s)
