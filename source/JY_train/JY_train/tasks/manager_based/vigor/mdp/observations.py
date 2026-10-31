from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from ..torch_math import quat_apply_inverse
from .commands import VigorKeyframeCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# central 3x3 block of the 15x15 height-scan grid (index symmetric in x/y ordering)
_CENTER_IDS = [r * 15 + c for r in (6, 7, 8) for c in (6, 7, 8)]


def terrain_center_height(env: ManagerBasedEnv, sensor_name: str = "height_scanner") -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_name]
    z = torch.nan_to_num(sensor.data.ray_hits_w[:, _CENTER_IDS, 2], nan=0.0, posinf=0.0, neginf=0.0)
    return z.mean(dim=1)


def keyframe_goal(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """k_t: features of the next sparse keyframes (paper Sec. III-B)."""
    command: VigorKeyframeCommand = env.command_manager.get_term(command_name)
    return command.command


def terrain_context(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """h_t: terrain heights relative to the ground under the trunk, plus trunk height above it."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    z = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    center = z[:, _CENTER_IDS].mean(dim=1, keepdim=True)
    shape = (z - center).clamp(-1.0, 1.0)
    trunk = (asset.data.root_pos_w[:, 2:3] - center).clamp(-0.5, 1.5)
    return torch.cat((shape, trunk), dim=-1)


def robot_keypoints(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Privileged: current keypoints relative to the root, in the root frame (critic only)."""
    command: VigorKeyframeCommand = env.command_manager.get_term(command_name)
    rel = command.robot_rel_body_pos()[:, 1:]
    q = command.robot.data.root_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
    return quat_apply_inverse(q, rel).reshape(env.num_envs, -1)


def contact_flags(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1)
    return (forces > threshold).float()
