from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length

def root_height_below_minimum(env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        root_height = root_height - terrain_height

    return root_height < minimum_height

def root_velocity_above_threshold(env: ManagerBasedRLEnv, max_velocity_square: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_square = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)
    vel_square += torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    return vel_square > max_velocity_square

def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1),dim=1)[0] > threshold, dim=1)