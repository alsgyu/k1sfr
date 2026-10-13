from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if not isinstance(env_ids, slice) and not isinstance(joint_ids, slice):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos

def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)

def randomize_base_com_mass(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, base_com_randomization: dict, base_mass_randomization: dict, recompute_inertia: bool = True):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
        env_ids_sim = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids_cpu = env_ids.cpu()
        env_ids_sim = env_ids.to(asset.device)

    if isinstance(asset_cfg.body_ids, slice):
        raise ValueError("randomize_base_com_mass needs one base body, e.g. body_names='Trunk'.")

    body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")
    if body_ids.numel() != 1:
        raise ValueError(f"Expected one base body, got: {asset_cfg.body_ids}")

    body_id = int(body_ids.item()) 

    # 버퍼 초기화 -> com x,y,z축 랜덤 노이즈 / 질량 랜덤 노이즈
    if not hasattr(env, "base_mass_scaled"):
        env.base_mass_scaled = torch.zeros(env.num_envs, 4, device=asset.device)

    coms = asset.root_physx_view.get_coms().clone()
    com_x, noise_x = _apply_randomization(coms[env_ids_cpu, body_id, 0], base_com_randomization, True)
    com_y, noise_y = _apply_randomization(coms[env_ids_cpu, body_id, 1], base_com_randomization, True)
    com_z, noise_z = _apply_randomization(coms[env_ids_cpu, body_id, 2], base_com_randomization, True)

    coms[env_ids_cpu, body_id, 0] = com_x
    coms[env_ids_cpu, body_id, 1] = com_y
    coms[env_ids_cpu, body_id, 2] = com_z
    asset.root_physx_view.set_coms(coms, env_ids_cpu)

    masses = asset.root_physx_view.get_masses().clone()
    default_mass = asset.data.default_mass[env_ids_cpu, body_id].clone().to(masses.device)
    masses[env_ids_cpu, body_id] = default_mass

    new_mass, noise_mass = _apply_randomization(masses[env_ids_cpu, body_id], base_mass_randomization, True)
    masses[env_ids_cpu, body_id] = new_mass
    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    # 관성 재계산 여부
    if recompute_inertia:
        ratios = new_mass / default_mass # 질량 변화 비율 계산
        inertias = asset.root_physx_view.get_inertias().clone()
        default_inertia = asset.data.default_inertia[env_ids_cpu, body_id].clone().to(inertias.device)
        # 질량 비율만큼 관성 스케일링
        inertias[env_ids_cpu, body_id] = default_inertia * ratios[:, None].to(inertias.device)
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

    env.base_mass_scaled[env_ids_sim, 0] = noise_x.to(asset.device)
    env.base_mass_scaled[env_ids_sim, 1] = noise_y.to(asset.device)
    env.base_mass_scaled[env_ids_sim, 2] = noise_z.to(asset.device)
    env.base_mass_scaled[env_ids_sim, 3] = noise_mass.to(asset.device)

def _apply_randomization(tensor: torch.Tensor, params: dict | None, return_noise: bool = False):
    if params is None:
        return (tensor, torch.zeros_like(tensor)) if return_noise else tensor

    if params["distribution"] == "gaussian":
        mean, std = params["range"]
        noise = torch.randn_like(tensor)
        noise_value = mean + std * noise
    elif params["distribution"] == "uniform":
        low, high = params["range"]
        noise = torch.rand_like(tensor)
        noise_value = low + (high - low) * noise
    else:
        raise ValueError(f"Invalid distribution: {params['distribution']}")

    if params["operation"] == "additive":
        result = tensor + noise_value
    elif params["operation"] == "scaling":
        result = tensor * noise_value
    else:
        raise ValueError(f"Invalid operation: {params['operation']}")

    # Privileged observation must contain the physical perturbation, not the
    # raw U(0, 1) / N(0, 1) random draw.  This also gives COM-y a well-defined
    # sign under left-right reflection.
    return (result, noise_value) if return_noise else result

def kick_by_setting_velocity(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                             lin_vel_randomization: dict | None = None,
                             ang_vel_randomization: dict | None = None):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)

    root_vel = asset.data.root_vel_w[env_ids].clone()
    root_vel[:, 0:3] = _apply_randomization(root_vel[:, 0:3], lin_vel_randomization)
    root_vel[:, 3:6] = _apply_randomization(root_vel[:, 3:6], ang_vel_randomization)

    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

def external_push(env, env_ids, asset_cfg: SceneEntityCfg, push_interval_s: float, push_duration_s: float, force_randomization: dict, torque_randomization: dict):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device) # 없으면 전체 환경

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, device=asset.device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device=asset.device)

    num_bodies = len(body_ids)

    if not hasattr(env, "pushing_forces"): # 외력을 저장할 버퍼 최초 생성
        env.pushing_forces = torch.zeros(env.num_envs, asset.num_bodies, 3, device=asset.device)
        env.pushing_torques = torch.zeros_like(env.pushing_forces)

    interval_steps = max(1, int(math.ceil(push_interval_s / env.step_dt))) # 초 단위를 step 수로 변경
    duration_steps = max(1, int(math.ceil(push_duration_s / env.step_dt)))
    phase = int(env.common_step_counter) % interval_steps # 얼마나 진행했는지

    if phase == 0:
        force = _apply_randomization(torch.zeros(len(env_ids), num_bodies, 3, device=asset.device),
                                     force_randomization)
        torque = _apply_randomization(torch.zeros(len(env_ids), num_bodies, 3, device=asset.device),
                                      torque_randomization)
        # 외력 버퍼에 저장
        env.pushing_forces[env_ids[:, None], body_ids, :] = force
        env.pushing_torques[env_ids[:, None], body_ids, :] = torque

    elif phase == duration_steps:
        env.pushing_forces[env_ids[:, None], body_ids, :].zero_()
        env.pushing_torques[env_ids[:, None], body_ids, :].zero_()

    forces = env.pushing_forces[env_ids[:, None], body_ids, :]
    torques = env.pushing_torques[env_ids[:, None], body_ids, :]

    asset.set_external_force_and_torque(
        forces,
        torques,
        env_ids=env_ids,
        body_ids=asset_cfg.body_ids,
    )
