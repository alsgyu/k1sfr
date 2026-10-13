"""Left-right reflection for the K1 scratch velocity task.

The callback follows the RSL-RL 2.3.1 convention: the returned batch contains
the original samples first and their mirrored copies second.
"""

from __future__ import annotations

import torch


POLICY_OBS_DIM = 54
CRITIC_OBS_DIM = 68
ACTION_DIM = 12

# Observation slices from ObservationsCfg.
COMMAND = slice(0, 10)
GAIT_PHASE = slice(10, 12)
PROJECTED_GRAVITY = slice(12, 15)
BASE_ANG_VEL = slice(15, 18)
JOINT_POS = slice(18, 30)
JOINT_VEL = slice(30, 42)
LAST_ACTION = slice(42, 54)

BASE_MASS_COM = slice(54, 58)
BASE_LIN_VEL = slice(58, 61)
BASE_HEIGHT = slice(61, 62)
PUSH_FORCE = slice(62, 65)
PUSH_TORQUE = slice(65, 68)

NORMALIZER_ATTR = "_rsl_rl_symmetry_normalizers"


def _unwrap_env(env):
    """Resolve the Isaac Lab environment behind Gym/RSL-RL wrappers."""
    seen = set()
    while id(env) not in seen:
        seen.add(id(env))
        if hasattr(env, "scene"):
            return env
        if hasattr(env, "unwrapped"):
            candidate = env.unwrapped
        elif hasattr(env, "env"):
            candidate = env.env
        else:
            break
        if candidate is env:
            break
        env = candidate
    raise AttributeError("Unable to resolve an environment with a 'scene' attribute.")


def _get_normalizer(env, obs_type: str):
    """Get the RSL-RL normalizer attached by the project training script."""
    env = _unwrap_env(env)
    normalizers = getattr(env, NORMALIZER_ATTR, None)
    if normalizers is None:
        return None
    return normalizers.get(obs_type)


def _normalize_without_update(normalizer, values: torch.Tensor) -> torch.Tensor:
    """Apply an empirical normalizer without changing its running statistics."""
    was_training = normalizer.training
    try:
        normalizer.eval()
        return normalizer(values)
    finally:
        normalizer.train(was_training)


def _mirror_normalized_observations(
    obs: torch.Tensor,
    env,
    obs_type: str,
    raw_mirror_func,
) -> torch.Tensor:
    """Mirror in raw space even when PPO stores empirically-normalized obs.

    RSL-RL 2.3.1 stores normalized observations in rollout storage before its
    symmetry callback runs.  A signed permutation generally does not commute
    with per-channel empirical normalization, so the correct operation is:

        normalized -> raw -> mirror(raw) -> normalized.
    """
    normalizer = _get_normalizer(env, obs_type)
    if normalizer is None:
        # CPU algebra tests and normalization-disabled runners use raw obs.
        return raw_mirror_func(obs, env)
    if not callable(getattr(normalizer, "inverse", None)):
        raise TypeError(
            f"The {obs_type!r} symmetry normalizer must provide inverse()."
        )

    raw_obs = normalizer.inverse(obs)
    mirrored_raw_obs = raw_mirror_func(raw_obs, env)
    return _normalize_without_update(normalizer, mirrored_raw_obs)


def _joint_permutation(env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the action-local left/right permutation from runtime joint names."""
    env = _unwrap_env(env)
    names = env.scene["robot"].data.joint_names
    selected_names = [
        name
        for name in names
        if any(token in name for token in ("Hip_", "Knee_", "Ankle_"))
    ]
    if len(selected_names) != ACTION_DIM:
        raise ValueError(
            f"Expected {ACTION_DIM} K1 leg joints, found {len(selected_names)}: "
            f"{selected_names}"
        )

    local_index = {name: index for index, name in enumerate(selected_names)}
    permutation = []
    signs = []
    for name in selected_names:
        counterpart = (
            name.replace("Left_", "@_")
            .replace("Right_", "Left_")
            .replace("@_", "Right_")
        )
        if counterpart not in local_index:
            raise ValueError(
                f"Mirror counterpart for {name!r} is missing: {counterpart!r}."
            )
        permutation.append(local_index[counterpart])
        signs.append(-1.0 if ("_Roll" in name or "_Yaw" in name) else 1.0)

    return (
        torch.tensor(permutation, dtype=torch.long, device=device),
        torch.tensor(signs, dtype=torch.float32, device=device),
    )


def _mirror_joints(
    values: torch.Tensor, permutation: torch.Tensor, signs: torch.Tensor
) -> torch.Tensor:
    if values.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"Expected a {ACTION_DIM}-D joint/action tensor, got {values.shape}."
        )
    return values[..., permutation] * signs


def _mirror_command(command: torch.Tensor) -> torch.Tensor:
    """Mirror [vx, vy, wz, frequency, foot-yaws, body targets, foot offsets]."""
    if command.shape[-1] != 10:
        raise ValueError(f"Expected a 10-D command, got {command.shape}.")
    out = command.clone()
    out[..., 0] = command[..., 0]  # vx
    out[..., 1] = -command[..., 1]  # vy
    out[..., 2] = -command[..., 2]  # yaw rate
    out[..., 3] = command[..., 3]  # gait frequency
    out[..., 4] = -command[..., 5]  # left foot yaw <- mirrored right
    out[..., 5] = -command[..., 4]  # right foot yaw <- mirrored left
    out[..., 6] = command[..., 6]  # body pitch
    out[..., 7] = -command[..., 7]  # body roll
    out[..., 8] = -command[..., 8]  # left-right forward offset
    out[..., 9] = command[..., 9]  # lateral foot-distance offset
    return out


def mirror_policy_observations(obs: torch.Tensor, env) -> torch.Tensor:
    """Apply sagittal reflection and a half-cycle gait-phase shift."""
    if obs.shape[-1] != POLICY_OBS_DIM:
        raise ValueError(
            f"Expected {POLICY_OBS_DIM}-D policy observations, got {obs.shape}."
        )
    permutation, signs = _joint_permutation(env, obs.device)
    out = obs.clone()

    out[..., COMMAND] = _mirror_command(obs[..., COMMAND])
    # phi -> (phi + 0.5) mod 1 negates both sin/cos channels.
    out[..., GAIT_PHASE] = -obs[..., GAIT_PHASE]
    out[..., PROJECTED_GRAVITY] = obs[..., PROJECTED_GRAVITY] * obs.new_tensor(
        [1.0, -1.0, 1.0]
    )
    # Angular velocity is an axial vector under sagittal reflection.
    out[..., BASE_ANG_VEL] = obs[..., BASE_ANG_VEL] * obs.new_tensor(
        [-1.0, 1.0, -1.0]
    )
    out[..., JOINT_POS] = _mirror_joints(
        obs[..., JOINT_POS], permutation, signs
    )
    out[..., JOINT_VEL] = _mirror_joints(
        obs[..., JOINT_VEL], permutation, signs
    )
    out[..., LAST_ACTION] = _mirror_joints(
        obs[..., LAST_ACTION], permutation, signs
    )
    return out


def mirror_critic_observations(obs: torch.Tensor, env) -> torch.Tensor:
    """Mirror the 54-D common state and 14-D privileged simulator state."""
    if obs.shape[-1] != CRITIC_OBS_DIM:
        raise ValueError(
            f"Expected {CRITIC_OBS_DIM}-D critic observations, got {obs.shape}."
        )
    out = obs.clone()
    out[..., :POLICY_OBS_DIM] = mirror_policy_observations(
        obs[..., :POLICY_OBS_DIM], env
    )

    # [com_dx, com_dy, com_dz, mass_scale]
    out[..., BASE_MASS_COM] = obs[..., BASE_MASS_COM] * obs.new_tensor(
        [1.0, -1.0, 1.0, 1.0]
    )
    out[..., BASE_LIN_VEL] = obs[..., BASE_LIN_VEL] * obs.new_tensor(
        [1.0, -1.0, 1.0]
    )
    out[..., BASE_HEIGHT] = obs[..., BASE_HEIGHT]
    out[..., PUSH_FORCE] = obs[..., PUSH_FORCE] * obs.new_tensor(
        [1.0, -1.0, 1.0]
    )
    # Torque is an axial vector.
    out[..., PUSH_TORQUE] = obs[..., PUSH_TORQUE] * obs.new_tensor(
        [-1.0, 1.0, -1.0]
    )
    return out


def mirror_actions(actions: torch.Tensor, env) -> torch.Tensor:
    permutation, signs = _joint_permutation(env, actions.device)
    return _mirror_joints(actions, permutation, signs)


def data_augmentation_func(obs, actions, env, obs_type):
    """Return original+mirrored batches for RSL-RL symmetry training."""
    if obs is None:
        augmented_obs = None
    elif obs_type == "policy":
        augmented_obs = torch.cat(
            (
                obs,
                _mirror_normalized_observations(
                    obs, env, "policy", mirror_policy_observations
                ),
            ),
            dim=0,
        )
    elif obs_type == "critic":
        augmented_obs = torch.cat(
            (
                obs,
                _mirror_normalized_observations(
                    obs, env, "critic", mirror_critic_observations
                ),
            ),
            dim=0,
        )
    else:
        raise ValueError(f"Unsupported observation type: {obs_type!r}.")

    augmented_actions = (
        None
        if actions is None
        else torch.cat((actions, mirror_actions(actions, env)), dim=0)
    )
    return augmented_obs, augmented_actions

