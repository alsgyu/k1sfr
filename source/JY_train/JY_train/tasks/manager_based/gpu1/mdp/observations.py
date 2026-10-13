from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def gait_phase(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    phase = command.gait_process # 현재 보행 진행도 (0 ~ 1)
    # phase를 원 위의 좌표로 변환 -> 주기가 끝나는 부분에서 발생하는 문제 해결
    return torch.stack(
        (torch.cos(2.0 * math.pi * phase), torch.sin(2.0 * math.pi * phase)),
        dim=-1,
    )

# 몸통의 질량이 얼마나 변형됐는지에 대한 값을 가져옴
def base_mass_scaled(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    value = getattr(env, "base_mass_scaled", None)
    if value is None:
        return torch.zeros(env.num_envs, 4, device=asset.device)
    return value

def base_height(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2:3]

    if sensor_cfg is not None:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)
        base_height = base_height - terrain_height

    return base_height

# body_id 1개 꺼내는 함수
def _single_body_id(asset_cfg: SceneEntityCfg) -> int:
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        return 0
    if isinstance(body_ids, int):
        return body_ids
    if len(body_ids) != 1:
        raise ValueError(f"Expected exactly one body for push obs, got: {body_ids}")
    return body_ids[0]


def push_force(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    value = getattr(env, "pushing_forces", None)
    if value is None:
        return torch.zeros(env.num_envs, 3, device=asset.device)

    body_id = _single_body_id(asset_cfg)
    return value[:, body_id, :] * 0.1


def push_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    value = getattr(env, "pushing_torques", None)
    if value is None:
        return torch.zeros(env.num_envs, 3, device=asset.device)

    body_id = _single_body_id(asset_cfg)
    return value[:, body_id, :] * 0.5