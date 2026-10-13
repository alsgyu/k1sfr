from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import (
    randomize_actuator_gains as _isaac_randomize_actuator_gains,
    reset_joints_by_offset as _isaac_reset_joints_by_offset,
)
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

from .curriculums import (
    get_walk_20k_curriculum_stage,
    scale_additive_range,
    scale_gaussian_std_range,
    scale_multiplicative_range,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None, device: str | torch.device) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=device)
    return env_ids.to(device)


def _select_curriculum_active_env_ids(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    active_ratio: float,
    device: str | torch.device,
) -> torch.Tensor:
    env_ids = _resolve_env_ids(env, env_ids, device)
    if active_ratio <= 0.0:
        return env_ids[:0]
    if active_ratio >= 1.0:
        return env_ids
    return env_ids[torch.rand(len(env_ids), device=device) < active_ratio]


def _scale_randomization_params(params: dict | None, level: float) -> dict | None:
    if params is None:
        return None

    scaled = dict(params)
    value_range = params["range"]
    operation = params["operation"]
    distribution = params["distribution"]

    if operation in ("add", "additive", "abs"):
        if distribution == "gaussian":
            scaled["range"] = scale_gaussian_std_range(value_range, level)
        else:
            scaled["range"] = scale_additive_range(value_range, level)
    elif operation in ("scale", "scaling"):
        scaled["range"] = scale_multiplicative_range(value_range, level)
    else:
        raise ValueError(f"Unsupported randomization operation: {operation}")

    return scaled


def _reset_joint_default_pos_to_nominal(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_curriculum_default_joint_pos_nominal"):
        env._curriculum_default_joint_pos_nominal = asset.data.default_joint_pos.clone()

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
        asset.data.default_joint_pos[env_ids, :] = env._curriculum_default_joint_pos_nominal[env_ids, :]
        env.action_manager.get_term("joint_pos")._offset[env_ids, :] = asset.data.default_joint_pos[env_ids, :]
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=asset.device)
        asset.data.default_joint_pos[env_ids[:, None], joint_ids] = env._curriculum_default_joint_pos_nominal[
            env_ids[:, None], joint_ids
        ]
        env.action_manager.get_term("joint_pos")._offset[env_ids[:, None], joint_ids] = asset.data.default_joint_pos[
            env_ids[:, None], joint_ids
        ]


def _reset_base_com_mass_to_nominal(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids_cpu = env_ids.cpu()

    if isinstance(asset_cfg.body_ids, slice):
        raise ValueError("curriculum base_com reset needs one base body, e.g. body_names='Trunk'.")
    body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")
    if body_ids.numel() != 1:
        raise ValueError(f"Expected one base body, got: {asset_cfg.body_ids}")
    body_id = int(body_ids.item())

    if not hasattr(env, "_curriculum_nominal_coms"):
        env._curriculum_nominal_coms = asset.root_physx_view.get_coms().clone()

    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids_cpu, body_id, :3] = env._curriculum_nominal_coms[env_ids_cpu, body_id, :3]
    asset.root_physx_view.set_coms(coms, env_ids_cpu)

    masses = asset.root_physx_view.get_masses().clone()
    masses[env_ids_cpu, body_id] = asset.data.default_mass[env_ids_cpu, body_id].to(masses.device)
    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    inertias = asset.root_physx_view.get_inertias().clone()
    inertias[env_ids_cpu, body_id] = asset.data.default_inertia[env_ids_cpu, body_id].to(inertias.device)
    asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

    if hasattr(env, "base_mass_scaled"):
        env.base_mass_scaled[env_ids.to(asset.device), :] = 0.0


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


def curriculum_randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    num_steps_per_iteration: int = 24,
):
    stage = get_walk_20k_curriculum_stage(env, num_steps_per_iteration)
    asset: Articulation = env.scene[asset_cfg.name]
    all_env_ids = _resolve_env_ids(env, env_ids, asset.device)

    # [20%][35%][55%][75%][90%][100%] reset to nominal before applying stage randomization.
    _reset_joint_default_pos_to_nominal(env, all_env_ids, asset_cfg)

    if stage.randomization_level <= 0.0 or pos_distribution_params is None:
        return

    scaled_pos_distribution_params = scale_additive_range(
        pos_distribution_params,
        stage.randomization_level,
    )
    randomize_joint_default_pos(
        env,
        all_env_ids,
        asset_cfg=asset_cfg,
        pos_distribution_params=scaled_pos_distribution_params,
        operation=operation,
        distribution=distribution,
    )


def curriculum_randomize_base_com_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    base_com_randomization: dict,
    base_mass_randomization: dict,
    recompute_inertia: bool = True,
    num_steps_per_iteration: int = 24,
):
    stage = get_walk_20k_curriculum_stage(env, num_steps_per_iteration)
    asset: Articulation = env.scene[asset_cfg.name]
    all_env_ids = _resolve_env_ids(env, env_ids, asset.device)

    # [20%][35%][55%][75%][90%][100%] reset nominal envs before stage COM/mass randomization.
    _reset_base_com_mass_to_nominal(env, all_env_ids, asset_cfg)

    if stage.randomization_level <= 0.0:
        return

    randomize_base_com_mass(
        env,
        all_env_ids,
        asset_cfg=asset_cfg,
        base_com_randomization=_scale_randomization_params(base_com_randomization, stage.randomization_level),
        base_mass_randomization=_scale_randomization_params(base_mass_randomization, stage.randomization_level),
        recompute_inertia=recompute_inertia,
    )


def curriculum_kick_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_randomization: dict | None = None,
    ang_vel_randomization: dict | None = None,
    num_steps_per_iteration: int = 24,
):
    stage = get_walk_20k_curriculum_stage(env, num_steps_per_iteration)
    asset: Articulation = env.scene[asset_cfg.name]

    # [0%][20%][45%][70%][90%][100%] apply kick only to non-nominal envs.
    active_ratio = 1.0 - stage.nominal_ratio
    active_env_ids = _select_curriculum_active_env_ids(env, env_ids, active_ratio, asset.device)
    if len(active_env_ids) == 0 or stage.disturbance_level <= 0.0:
        return

    kick_by_setting_velocity(
        env,
        active_env_ids,
        asset_cfg=asset_cfg,
        lin_vel_randomization=_scale_randomization_params(lin_vel_randomization, stage.disturbance_level),
        ang_vel_randomization=_scale_randomization_params(ang_vel_randomization, stage.disturbance_level),
    )


def curriculum_external_push(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    push_interval_s: float,
    push_duration_s: float,
    force_randomization: dict,
    torque_randomization: dict,
    num_steps_per_iteration: int = 24,
):
    stage = get_walk_20k_curriculum_stage(env, num_steps_per_iteration)
    asset: Articulation = env.scene[asset_cfg.name]
    all_env_ids = _resolve_env_ids(env, env_ids, asset.device)

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, device=asset.device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device=asset.device)

    num_bodies = len(body_ids)
    if not hasattr(env, "pushing_forces"):
        env.pushing_forces = torch.zeros(env.num_envs, asset.num_bodies, 3, device=asset.device)
        env.pushing_torques = torch.zeros_like(env.pushing_forces)

    if stage.disturbance_level <= 0.0:
        env.pushing_forces[all_env_ids[:, None], body_ids, :].zero_()
        env.pushing_torques[all_env_ids[:, None], body_ids, :].zero_()
        asset.set_external_force_and_torque(
            env.pushing_forces[all_env_ids[:, None], body_ids, :],
            env.pushing_torques[all_env_ids[:, None], body_ids, :],
            env_ids=all_env_ids,
            body_ids=asset_cfg.body_ids,
        )
        return

    interval_steps = max(1, int(math.ceil(stage.push_interval_s / env.step_dt)))
    duration_steps = max(1, int(math.ceil(stage.push_duration_s / env.step_dt)))
    phase = int(env.common_step_counter) % interval_steps

    if phase == 0:
        env.pushing_forces[all_env_ids[:, None], body_ids, :].zero_()
        env.pushing_torques[all_env_ids[:, None], body_ids, :].zero_()

        # [0%][20%][45%][70%][90%][100%] push grows to scratch_symmetry final value.
        active_ratio = 1.0 - stage.nominal_ratio
        active_env_ids = _select_curriculum_active_env_ids(env, all_env_ids, active_ratio, asset.device)
        if len(active_env_ids) > 0:
            force = _apply_randomization(
                torch.zeros(len(active_env_ids), num_bodies, 3, device=asset.device),
                _scale_randomization_params(force_randomization, stage.disturbance_level),
            )
            torque = _apply_randomization(
                torch.zeros(len(active_env_ids), num_bodies, 3, device=asset.device),
                _scale_randomization_params(torque_randomization, stage.disturbance_level),
            )
            env.pushing_forces[active_env_ids[:, None], body_ids, :] = force
            env.pushing_torques[active_env_ids[:, None], body_ids, :] = torque
    elif phase == duration_steps:
        env.pushing_forces[all_env_ids[:, None], body_ids, :].zero_()
        env.pushing_torques[all_env_ids[:, None], body_ids, :].zero_()

    asset.set_external_force_and_torque(
        env.pushing_forces[all_env_ids[:, None], body_ids, :],
        env.pushing_torques[all_env_ids[:, None], body_ids, :],
        env_ids=all_env_ids,
        body_ids=asset_cfg.body_ids,
    )


def curriculum_reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_range: tuple[float, float] = (-0.05, 0.05),
    velocity_range: tuple[float, float] = (-0.1, 0.1),
    num_steps_per_iteration: int = 24,
):
    stage = get_walk_20k_curriculum_stage(env, num_steps_per_iteration)
    asset: Articulation = env.scene[asset_cfg.name]
    all_env_ids = _resolve_env_ids(env, env_ids, asset.device)

    # [20%][35%][55%][75%][90%][100%] joint reset noise grows to scratch_symmetry final value.
    _isaac_reset_joints_by_offset(
        env,
        all_env_ids,
        asset_cfg=asset_cfg,
        position_range=scale_additive_range(position_range, stage.randomization_level),
        velocity_range=scale_additive_range(velocity_range, stage.randomization_level),
    )


def curriculum_randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float],
    damping_distribution_params: tuple[float, float],
    operation: Literal["scale"] = "scale",
    distribution: Literal["uniform"] = "uniform",
    num_steps_per_iteration: int = 24,
):
    stage = get_walk_20k_curriculum_stage(env, num_steps_per_iteration)
    asset: Articulation = env.scene[asset_cfg.name]

    # [20%][35%][55%][75%][90%][100%] gain randomization grows to 0.9~1.1.
    all_env_ids = _resolve_env_ids(env, env_ids, asset.device)
    if stage.randomization_level <= 0.0:
        return

    _isaac_randomize_actuator_gains(
        env,
        all_env_ids,
        asset_cfg=asset_cfg,
        stiffness_distribution_params=scale_multiplicative_range(
            stiffness_distribution_params,
            stage.randomization_level,
        ),
        damping_distribution_params=scale_multiplicative_range(
            damping_distribution_params,
            stage.randomization_level,
        ),
        operation=operation,
        distribution=distribution,
    )
