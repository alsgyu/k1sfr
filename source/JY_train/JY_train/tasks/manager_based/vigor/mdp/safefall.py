"""Isaac terms of the K1 safe-fall task.

The reward math, start distribution and outcome definitions live in ``safefall_math``
(shared with the laptop-side MuJoCo check); this module only gathers the tensors.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from .. import safefall_math as sf
from .. import spec

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class _Ids:
    def __init__(self, robot: Articulation, sensor: ContactSensor):
        self.groups = {g: sensor.find_bodies(names, preserve_order=True)[0] for g, names in sf.CONTACT_GROUPS.items()}
        self.head = robot.find_bodies(sf.HEAD_BODY)[0][0]
        self.hands = robot.find_bodies(sf.HAND_BODIES, preserve_order=True)[0]
        self.joints = robot.find_joints(spec.ALL_JOINT_NAMES, preserve_order=True)[0]


def fall_state(env: ManagerBasedEnv, robot_name: str = "robot", sensor_name: str = "contact_forces") -> sf.FallState:
    """FallState of every env, computed once per env step (all reward terms and the monitor share it).

    Contact forces are the peak over the sensor history, which spans the policy step's physics substeps."""
    cache = getattr(env, "_safefall_cache", None)
    step = getattr(env, "common_step_counter", 0)
    if cache is not None and cache[0] == step:
        return cache[1]
    robot: Articulation = env.scene[robot_name]
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    if not hasattr(env, "_safefall_ids"):
        env._safefall_ids = _Ids(robot, sensor)
    ids = env._safefall_ids
    peak = sensor.data.net_forces_w_history.norm(dim=-1).amax(dim=1)  # (N, bodies)
    ground = env.scene.env_origins[:, 2]  # flat ground
    pos, vel = robot.data.body_pos_w, robot.data.body_lin_vel_w
    state = sf.FallState(
        force={g: peak[:, i].sum(dim=-1) for g, i in ids.groups.items()},
        head_h=pos[:, ids.head, 2] - ground,
        trunk_h=robot.data.root_pos_w[:, 2] - ground,
        hand_h=pos[:, ids.hands, 2].min(dim=-1).values - ground,
        head_vz=vel[:, ids.head, 2],
        trunk_vz=robot.data.root_lin_vel_w[:, 2],
        gravity_b=robot.data.projected_gravity_b,
        q=robot.data.joint_pos[:, ids.joints],
        facing=getattr(env, "safefall_facing", torch.zeros_like(ground)),
    )
    env._safefall_cache = (step, state, sf.terms(state))
    return state


# -------------------------------------------------------------------- rewards
def safefall_term(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """One unweighted term of ``safefall_math.terms`` (weight it with ``safefall_math.WEIGHTS[name]``)."""
    fall_state(env)
    return env._safefall_cache[2][name]


# ---------------------------------------------------------------- observations
def contact_force_groups(env: ManagerBasedEnv) -> torch.Tensor:
    """Privileged: peak ground contact force per body group [body weights]."""
    s = fall_state(env)
    return torch.stack([s.force[g] for g in sf.CONTACT_GROUPS], dim=-1) / sf.WEIGHT


def body_heights(env: ManagerBasedEnv) -> torch.Tensor:
    """Privileged: trunk, head and lower-hand heights above the ground."""
    s = fall_state(env)
    return torch.stack((s.trunk_h, s.head_h, s.hand_h), dim=-1)


def fall_facing(env: ManagerBasedEnv) -> torch.Tensor:
    """Privileged: cos of the episode's fall azimuth from the heading (+1 forward, -1 backward)."""
    return fall_state(env).facing.unsqueeze(-1)


# ---------------------------------------------------------------------- events
def reset_to_fall_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_noise: float = 0.1,
    stride_spread: float = 0.25,
    spawn_margin: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Walking stance tipping or pushed in a random direction, or held in a hand-off plank
    (``safefall_math.sample_starts``)."""
    robot: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    st = sf.sample_starts(n, env.device)
    q = robot.data.default_joint_pos[env_ids].clone()
    head = robot.find_joints(["AAHead_yaw", "Head_pitch"])[0]
    noise = (2.0 * torch.rand_like(q) - 1.0) * joint_noise
    noise[:, head] = 0.0
    q += noise
    spread = (2.0 * torch.rand(n, device=env.device) - 1.0) * stride_spread  # mid-stride legs
    q[:, robot.find_joints("Left_Hip_Pitch")[0][0]] += spread
    q[:, robot.find_joints("Right_Hip_Pitch")[0][0]] -= spread
    planks = torch.tensor([sf.PLANK_POSES["front"], sf.PLANK_POSES["back"]], device=env.device)[st["plank"].clamp_min(0)]
    on = st["plank"] >= 0
    body = len(spec.HEAD_JOINT_NAMES)  # the head stays held at its default
    joints = robot.find_joints(spec.ALL_JOINT_NAMES[body:], preserve_order=True)[0]
    q[:, joints] = torch.where(on.unsqueeze(-1), planks[:, body:] + 0.5 * noise[:, joints], q[:, joints])
    limits = robot.data.soft_joint_pos_limits[env_ids]
    q = torch.clamp(q, limits[..., 0], limits[..., 1])

    pos = env.scene.env_origins[env_ids] + st["root_rel"]
    pos[:, 2] += 0.09 * torch.sin(st["tilt"]) + spawn_margin  # toe/heel edge dips below the pivot
    robot.write_root_pose_to_sim(torch.cat((pos, st["root_quat"]), dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.cat((st["lin_vel"], st["ang_vel"]), dim=-1), env_ids=env_ids)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    if not hasattr(env, "safefall_direction"):
        env.safefall_direction = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.safefall_facing = torch.zeros(env.num_envs, device=env.device)
    env.safefall_direction[env_ids] = st["direction"]
    env.safefall_facing[env_ids] = st["facing"]


# --------------------------------------------------------------------- monitor
class SafeFallMonitor(CommandTerm):
    """Issues no command; logs the per-episode outcomes of ``safefall_math.OutcomeTracker`` as
    Metrics/<term>/<outcome> (means over the episodes that ended in the step).  Eval scripts set
    ``collect`` to receive every finished episode in ``finished``."""

    cfg: SafeFallMonitorCfg

    def __init__(self, cfg: SafeFallMonitorCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.tracker = sf.OutcomeTracker(self.num_envs, self.device)
        self.metrics.update(self.tracker.v)  # same tensors: CommandTerm.reset logs and zeroes them
        self.direction = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.collect, self.finished = False, []

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        )
        ran = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ran[ids] = self._env.episode_length_buf[ids] > 0  # not the initial reset
        # the step's cached state predates the resets: it is the last state of these episodes
        self.tracker.update(fall_state(self._env), live=ran)
        if self.collect:
            done = ids[ran[ids]]
            self.finished += self.tracker.episodes(done, self.direction)
        return super().reset(env_ids)

    def _update_metrics(self):
        self.tracker.update(fall_state(self._env), live=self._env.episode_length_buf > 0)
        # last consumer of the pre-reset state: observations of reset envs must see the new episode
        self._env._safefall_cache = None

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.tracker.clear(ids, values=False)  # values were logged and zeroed by CommandTerm.reset
        self.direction[ids] = getattr(self._env, "safefall_direction", self.direction)[ids]

    def _update_command(self):
        pass


@configclass
class SafeFallMonitorCfg(CommandTermCfg):
    class_type: type = SafeFallMonitor
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
