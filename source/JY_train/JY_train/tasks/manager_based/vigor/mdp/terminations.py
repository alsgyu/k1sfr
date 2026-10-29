from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_state(env: ManagerBasedRLEnv, max_lin_vel: float, max_ang_vel: float,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Simulation blow-up only; falling itself is part of the task and never terminates."""
    asset: Articulation = env.scene[asset_cfg.name]
    d = asset.data
    finite = torch.isfinite(d.root_state_w).all(dim=-1) & torch.isfinite(d.joint_pos).all(dim=-1)
    lin = d.root_lin_vel_w.norm(dim=-1) > max_lin_vel
    ang = d.root_ang_vel_w.norm(dim=-1) > max_ang_vel
    return ~finite | lin | ang
