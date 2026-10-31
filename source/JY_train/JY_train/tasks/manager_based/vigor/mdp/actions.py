from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .. import safefall_math as sf

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


class ReflexJointPositionAction(JointPositionAction):
    """Joint position targets = ``safefall_math.reflex_targets(projected gravity)`` + scale * action.

    Zero action falls into the hand-off plank on the side the chest faces and holds it; the policy
    learns offsets from there.  The joints must be ``spec.ACTION_JOINT_NAMES`` in that order."""

    cfg: ReflexJointPositionActionCfg

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)  # raw * scale (no default offset), clipped if configured
        self._processed_actions = self._processed_actions + sf.reflex_targets(self._asset.data.projected_gravity_b)


@configclass
class ReflexJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = ReflexJointPositionAction
    use_default_offset: bool = False
