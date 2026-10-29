from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HoldJointPositionAction(ActionTerm):
    """Zero-dimensional action term that keeps the given joints at their default positions."""

    cfg: HoldJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: HoldJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        target = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class HoldJointPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = HoldJointPositionAction
    joint_names: list[str] = MISSING
