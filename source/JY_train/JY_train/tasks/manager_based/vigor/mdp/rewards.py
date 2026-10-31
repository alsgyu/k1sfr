"""Reward terms of paper Table I (motion imitation, regularization & safety, post-recovery).

Imitation compares root-relative keypoints p_i - p_0 in the world frame against the
reference rotated by the episode's yaw offset, so the coarse terrain projection of
flat-ground demonstrations never shows up as a tracking error (paper Sec. III-C,
AbsTrack ablation) and nothing depends on a per-frame heading.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .. import spec
from ..torch_math import quat_error_magnitude
from .commands import VigorKeyframeCommand
from .observations import terrain_center_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _command(env: ManagerBasedRLEnv, name: str) -> VigorKeyframeCommand:
    return env.command_manager.get_term(name)


def _height_gate(c: VigorKeyframeCommand, sigma: float | None, margin: float = 0.05) -> torch.Tensor | float:
    """exp(-(shortfall / sigma)^2) for a root lower than the reference's (B-Human get-up reward):
    matching the limbs while lying when the reference has already risen earns nothing."""
    if not sigma:
        return 1.0
    shortfall = (c.ref_root_height() - c.robot_root_height() - margin).clamp_min(0.0)
    return torch.exp(-(shortfall / sigma).square())


# ----------------------------------------------------------------- imitation
def track_body_pos_exp(env: ManagerBasedRLEnv, command_name: str, std: float, height_sigma: float | None = None) -> torch.Tensor:
    c = _command(env, command_name)
    err = (c.ref_rel_body_pos() - c.robot_rel_body_pos()).square().sum(-1).mean(-1)
    return torch.exp(-err / std**2) * _height_gate(c, height_sigma)


def track_body_rot_exp(env: ManagerBasedRLEnv, command_name: str, std: float, height_sigma: float | None = None) -> torch.Tensor:
    c = _command(env, command_name)
    robot_q = c.robot.data.body_quat_w[:, c.body_ids]
    err = quat_error_magnitude(c.ref_body_quat(), robot_q).square().mean(-1)
    return torch.exp(-err / std**2) * _height_gate(c, height_sigma)


def track_body_lin_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    c = _command(env, command_name)
    err = (c.ref_body_lin_vel() - c.robot.data.body_lin_vel_w[:, c.body_ids]).square().sum(-1).mean(-1)
    return torch.exp(-err / std**2)


def track_body_ang_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    c = _command(env, command_name)
    err = (c.ref_body_ang_vel() - c.robot.data.body_ang_vel_w[:, c.body_ids]).square().sum(-1).mean(-1)
    return torch.exp(-err / std**2)


def track_joint_pos_exp(env: ManagerBasedRLEnv, command_name: str, std: float, height_sigma: float | None = None) -> torch.Tensor:
    c = _command(env, command_name)
    ref = c.data.joint_pos[c.frame][:, c.joint_map][:, c.action_joint_ids]
    err = (ref - c.robot.data.joint_pos[:, c.action_joint_ids]).square().mean(-1)
    return torch.exp(-err / std**2) * _height_gate(c, height_sigma)


def track_joint_vel_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    c = _command(env, command_name)
    ref = c.data.joint_vel[c.frame][:, c.joint_map][:, c.action_joint_ids] * c.speed.unsqueeze(-1)
    err = (ref - c.robot.data.joint_vel[:, c.action_joint_ids]).square().mean(-1)
    return torch.exp(-err / std**2)


# ------------------------------------------------------------ contacts
def _touching(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > threshold


def ref_contact_match(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, contact_ids: list[int],
                      threshold: float = 1.0) -> torch.Tensor:
    """Fraction of ``sensor_cfg`` bodies (one per ``contact_ids`` entry, same order) on the ground while the
    reference has them there -- plants the hands when the reference braces."""
    ref = _command(env, command_name).ref_contact()[:, contact_ids]
    return (ref & _touching(env, sensor_cfg, threshold)).float().mean(-1)


def unexpected_contact(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, contact_id: int,
                       threshold: float = 1.0) -> torch.Tensor:
    """Any ``sensor_cfg`` body on the ground although the reference keeps that part off it."""
    ref = _command(env, command_name).ref_contact()[:, contact_id]
    return (_touching(env, sensor_cfg, threshold).any(-1) & ~ref).float()


# ------------------------------------------------------- regularization/safety
def torque_limit_excess(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """sum max(0, |tau| - tau_max): the part of the PD torque the motors cannot deliver."""
    asset: Articulation = env.scene[asset_cfg.name]
    d = asset.data
    return (d.computed_torque - d.applied_torque).abs()[:, asset_cfg.joint_ids].sum(-1)


class momentum_change(ManagerTermBase):
    """||p_t - p_{t-1}|| of the whole-body linear momentum."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        self.mass = self.asset.data.default_mass.to(env.device).unsqueeze(-1)
        self.prev = torch.zeros(env.num_envs, 3, device=env.device)
        # the new episode state is written by the command after this reset, so skip one step
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        p = (self.mass * self.asset.data.body_lin_vel_w).sum(dim=1)
        out = (p - self.prev).norm(dim=-1).masked_fill(self.fresh, 0.0)
        self.prev.copy_(p)
        self.fresh.zero_()
        return out


def body_yank(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """sum_b ||F_b,t - F_b,t-1||^2 over the contact-force history."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hist = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    return (hist[:, 0] - hist[:, 1]).square().sum(-1).sum(-1)


# ------------------------------------------------------------ post-recovery
_HEAD_IDS: dict[int, int] = {}


def trunk_height(env: ManagerBasedRLEnv, asset: Articulation, sensor_name: str = "height_scanner") -> torch.Tensor:
    return asset.data.root_pos_w[:, 2] - terrain_center_height(env, sensor_name)


def head_height(env: ManagerBasedRLEnv, asset: Articulation, sensor_name: str = "height_scanner") -> torch.Tensor:
    if id(asset) not in _HEAD_IDS:
        _HEAD_IDS[id(asset)] = asset.find_bodies("Head_2")[0][0]
    return asset.data.body_pos_w[:, _HEAD_IDS[id(asset)], 2] - terrain_center_height(env, sensor_name)


def upright_mask(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 sensor_name: str = "height_scanner") -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tilt_ok = asset.data.projected_gravity_b[:, 2] < -math.cos(math.radians(25.0))
    return tilt_ok & (head_height(env, asset, sensor_name) > 0.75 * spec.HEAD_HEIGHT_NOMINAL)


def head_height_exp(env: ManagerBasedRLEnv, std: float, gated: bool = True,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_name: str = "height_scanner") -> torch.Tensor:
    """exp(-(h - h_nominal)^2 / std^2); ``gated=False`` turns it into a dense keep-your-head-up signal."""
    asset: Articulation = env.scene[asset_cfg.name]
    err = (head_height(env, asset, sensor_name) - spec.HEAD_HEIGHT_NOMINAL).square()
    r = torch.exp(-err / std**2)
    return r * upright_mask(env, asset_cfg, sensor_name) if gated else r


def base_lin_vel_l2_upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b.square().sum(-1) * upright_mask(env, asset_cfg)


def base_ang_vel_l2_upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b.square().sum(-1) * upright_mask(env, asset_cfg)
