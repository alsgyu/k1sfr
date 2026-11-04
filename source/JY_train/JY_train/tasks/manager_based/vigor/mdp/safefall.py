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
        self.bodies = sensor.find_bodies(sf.CONTACT_BODIES, preserve_order=True)[0]
        self.head = robot.find_bodies(sf.HEAD_BODY)[0][0]
        self.hands = robot.find_bodies(sf.HAND_BODIES, preserve_order=True)[0]
        self.feet = robot.find_bodies(sf.FOOT_BODIES, preserve_order=True)[0]
        self.joints = robot.find_joints(spec.ALL_JOINT_NAMES, preserve_order=True)[0]


# the flat terrain's collider (TerrainImporterCfg prim_path "/World/ground", terrain_type "plane"; the path
# Isaac Lab's own contact sensor test filters on)
GROUND_COLLIDER = "/World/ground/terrain/GroundPlane/CollisionPlane"


def _ground_forces(env: ManagerBasedEnv, sensor: ContactSensor) -> torch.Tensor:
    """(N, substeps, bodies) normal force from the ground on each sensor body [N].  Self-contacts left out:
    the K1 collides with itself, and the net forces counted a fist pressed against the trunk as a chest /
    back hit (MuJoCo: all of the trunk contacts of side falls) and a fist on the thigh as a fist down."""
    matrix = sensor.data.force_matrix_w_history
    if matrix is None:
        raise RuntimeError("SafeFall needs the contact sensor filtered on the ground (filter_prim_paths_expr=[GROUND_COLLIDER])")
    ground = matrix.sum(dim=3).norm(dim=-1)
    if not getattr(env, "_safefall_ground_ok", False):  # a filter matching no collider would read zero, silently
        total = sensor.data.net_forces_w_history.norm(dim=-1).sum()
        if total > 10.0 * sf.WEIGHT:
            if ground.sum() < 0.05 * total:
                raise RuntimeError(f"SafeFall: no ground forces through {GROUND_COLLIDER} while the net contact forces "
                                   f"sum to {float(total):.0f} N: the ground collider prim path is wrong")
            env._safefall_ground_ok = True
    return ground


def fall_state(env: ManagerBasedEnv, robot_name: str = "robot", sensor_name: str = "contact_forces") -> sf.FallState:
    """FallState of every env, computed once per env step (all reward terms and the monitor share it).

    Contact forces are ground contacts only, the peak (contact detection) and the mean (impulse / step, the
    impact) over the sensor history, which spans the policy step's physics substeps."""
    cache = getattr(env, "_safefall_cache", None)
    step = getattr(env, "common_step_counter", 0)
    if cache is not None and cache[0] == step:
        return cache[1]
    robot: Articulation = env.scene[robot_name]
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    if not hasattr(env, "_safefall_ids"):
        env._safefall_ids = _Ids(robot, sensor)
    ids = env._safefall_ids
    history = _ground_forces(env, sensor)  # (N, substeps, bodies)
    peak = history.amax(dim=1)
    ground = env.scene.env_origins[:, 2]  # flat ground
    pos, vel = robot.data.body_pos_w, robot.data.body_lin_vel_w
    fist_pos, fist_vel = sf.fist_tips(pos[:, ids.hands], robot.data.body_quat_w[:, ids.hands], vel[:, ids.hands],
                                      robot.data.body_ang_vel_w[:, ids.hands])
    _start_buffers(env)
    force = {g: peak[:, i].sum(dim=-1) for g, i in ids.groups.items()}
    fist_touched = env.safefall_fist_down.clone()
    fist_on = torch.stack([force["fist_l"], force["fist_r"]], dim=-1) > sf.CONTACT_THRESHOLD
    env.safefall_fist_down |= env.safefall_triggered.unsqueeze(-1) & fist_on
    state = sf.FallState(
        force=force,
        body_force=history[:, :, ids.bodies].mean(dim=1),
        head_h=pos[:, ids.head, 2] - ground,
        trunk_h=robot.data.root_pos_w[:, 2] - ground,
        hand_h=pos[:, ids.hands, 2].min(dim=-1).values - ground,
        fist_h=fist_pos[..., 2] - ground.unsqueeze(-1),
        fist_vz=fist_vel[..., 2],
        foot_h=pos[:, ids.feet, 2] - ground.unsqueeze(-1),
        arm_width=sf.arm_width(robot.data.root_pos_w, robot.data.root_quat_w, pos[:, ids.hands], fist_pos),
        fist_touched=fist_touched,
        head_vz=vel[:, ids.head, 2],
        trunk_vz=robot.data.root_lin_vel_w[:, 2],
        gravity_b=robot.data.projected_gravity_b,
        q=robot.data.joint_pos[:, ids.joints],
        facing=getattr(env, "safefall_facing", torch.zeros_like(ground)),
        side=getattr(env, "safefall_side", torch.ones_like(ground)),
    )
    env._safefall_cache = (step, state, sf.terms(state))
    return state


# -------------------------------------------------------------------- rewards
def _guarded(env: ManagerBasedRLEnv, value: torch.Tensor) -> torch.Tensor:
    """Zero for envs that blew up this step (terminated by bad_state; rewards are computed before the
    reset) and for any non-finite value: one NaN reward turns the whole PPO update into NaN.  The bound
    is far above any physical value of the unweighted terms (joint_acc_l2 reaches ~1e7 at impacts)."""
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0e9, 1.0e9)
    if hasattr(env, "safefall_triggered"):  # nothing before the fall detector hands over
        value = value * env.safefall_triggered
    if "bad_state" in env.termination_manager.active_terms:
        value = torch.where(env.termination_manager.get_term("bad_state"), torch.zeros_like(value), value)
    return value


def safefall_term(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """One unweighted term of ``safefall_math.terms`` (weight it with ``safefall_math.WEIGHTS[name]``)."""
    fall_state(env)
    return _guarded(env, env._safefall_cache[2][name])


def safefall_residual(env: ManagerBasedRLEnv) -> torch.Tensor:
    """safefall_math.residual of the policy's actions (legs weighted RESIDUAL_LEGS)."""
    return _guarded(env, sf.residual(env.action_manager.action))


def guarded_reward(env: ManagerBasedRLEnv, term: str) -> torch.Tensor:
    """The ``isaaclab.envs.mdp`` reward ``term`` (by name: cfg params go through hydra as strings) with
    blown-up envs and non-finite values zeroed."""
    import isaaclab.envs.mdp as isaac_mdp

    return _guarded(env, getattr(isaac_mdp, term)(env))


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


def fall_time(env: ManagerBasedEnv) -> torch.Tensor:
    """Deployable: time since the fall detector handed over [s], up to FALL_TIME_MAX (the legs' reflex follows it)."""
    _start_buffers(env)
    return env.safefall_fall_time.clamp_max(sf.FALL_TIME_MAX).unsqueeze(-1)


def fall_triggered(env: ManagerBasedEnv) -> torch.Tensor:
    """Privileged: 1 once the fall detector has handed over to the policy."""
    _start_buffers(env)
    return env.safefall_triggered.float().unsqueeze(-1)


# --------------------------------------------------------------------- trigger
def _start_buffers(env: ManagerBasedEnv):
    if not hasattr(env, "safefall_direction"):
        n = env.num_envs
        env.safefall_direction = torch.zeros(n, dtype=torch.long, device=env.device)
        env.safefall_facing = torch.zeros(n, device=env.device)
        env.safefall_triggered = torch.ones(n, dtype=torch.bool, device=env.device)  # set per episode at reset
        env.safefall_hold = torch.zeros(n, len(spec.ACTION_JOINT_NAMES), device=env.device)
        env.safefall_side = torch.ones(n, device=env.device)  # latched reflex pose: +1 front, -1 back
        env.safefall_joint_bias = torch.zeros(n, len(spec.ALL_JOINT_NAMES), device=env.device)  # encoder offsets
        env.safefall_fall_time = torch.zeros(n, device=env.device)  # since the trigger [s]
        env.safefall_fist_down = torch.zeros(n, 2, dtype=torch.bool, device=env.device)  # left / right fist touched since the trigger


def update_trigger(env: ManagerBasedEnv, gravity_b: torch.Tensor) -> torch.Tensor:
    """The fall detector: latch per env once the tilt reaches ``TRIGGER_TILT_DEG``; the facing of the
    fall (which side should end up down) and the reflex side are taken from the tilt direction at that
    moment, and the reflex side follows the chest only once it clearly faces the other way."""
    _start_buffers(env)
    new = ~env.safefall_triggered & (-gravity_b[:, 2] < sf.TRIGGER_COS)
    facing = sf.facing_from_gravity(gravity_b)
    env.safefall_facing[new] = facing[new]
    env.safefall_side[new] = sf.initial_side(facing[new])
    env.safefall_fall_time[:] = torch.where(env.safefall_triggered, env.safefall_fall_time + env.step_dt, 0.0)
    env.safefall_triggered |= new
    env.safefall_side[:] = sf.update_side(env.safefall_side, gravity_b)
    return env.safefall_triggered


def joint_pos_measured(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint_pos_rel as the encoders report it: with the per-env calibration offset (randomize_joint_bias)."""
    robot: Articulation = env.scene[asset_cfg.name]
    _start_buffers(env)
    if not hasattr(env, "_safefall_canonical_ids"):
        env._safefall_canonical_ids = robot.find_joints(spec.ALL_JOINT_NAMES, preserve_order=True)[0]
    bias = torch.zeros_like(robot.data.joint_pos)
    bias[:, env._safefall_canonical_ids] = env.safefall_joint_bias
    return robot.data.joint_pos - robot.data.default_joint_pos + bias


def randomize_joint_bias(env: ManagerBasedEnv, env_ids: torch.Tensor | None, joint_names: str = ".*_(Hip|Knee|Ankle)_.*",
                         offset_range: tuple[float, float] = (-0.03, 0.03)):
    """Encoder calibration offsets per env (as the team's walking task: legs +-0.03 rad): the joints
    settle at target - offset and read position + offset."""
    import re

    _start_buffers(env)
    ids = env_ids if env_ids is not None else torch.arange(env.num_envs, device=env.device)
    cols = [i for i, n in enumerate(spec.ALL_JOINT_NAMES) if re.fullmatch(joint_names, n)]
    lo, hi = offset_range
    env.safefall_joint_bias[ids[:, None], torch.tensor(cols, device=env.device)] = lo + (hi - lo) * torch.rand(len(ids), len(cols), device=env.device)


# ---------------------------------------------------------------------- events
def reset_to_fall_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    joint_noise: float = 0.05,
    joint_vel_noise: float = 0.1,
    stride_spread: float = 0.25,
    spawn_margin: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Standing or walking on the ground and pushed in a random direction, or resting in a hand-off
    pose (``safefall_math.sample_starts``).  The start pose is held until the fall detector triggers."""
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
    q[:, robot.find_joints(spec.ALL_JOINT_NAMES, preserve_order=True)[0]] += st["joint_offset"]  # lifted leg
    poses = torch.tensor([sf.HANDOFF_POSES["front"], sf.HANDOFF_POSES["back"]], device=env.device)[st["handoff"].clamp_min(0)]
    on = st["handoff"] >= 0
    body = len(spec.HEAD_JOINT_NAMES)  # the head stays held at its default
    joints = robot.find_joints(spec.ALL_JOINT_NAMES[body:], preserve_order=True)[0]
    q[:, joints] = torch.where(on.unsqueeze(-1), poses[:, body:] + 0.5 * noise[:, joints], q[:, joints])
    limits = robot.data.soft_joint_pos_limits[env_ids]
    q = torch.clamp(q, limits[..., 0], limits[..., 1])

    pos = env.scene.env_origins[env_ids] + st["root_rel"]
    pos[:, 2] += spawn_margin  # standing: soles 0-2 cm above the ground (joint noise tilts the feet)
    robot.write_root_pose_to_sim(torch.cat((pos, st["root_quat"]), dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.cat((st["lin_vel"], st["ang_vel"]), dim=-1), env_ids=env_ids)
    dq = (2.0 * torch.rand_like(q) - 1.0) * joint_vel_noise * (~on).unsqueeze(-1)
    robot.write_joint_state_to_sim(q, dq, env_ids=env_ids)
    _start_buffers(env)
    env.safefall_direction[env_ids] = st["direction"]
    env.safefall_facing[env_ids] = st["facing"]
    env.safefall_triggered[env_ids] = on  # hand-off starts are the policy's from the first step
    env.safefall_fall_time[env_ids] = 0.0
    env.safefall_fist_down[env_ids] = on.unsqueeze(-1)  # resting on the fists already
    env.safefall_side[env_ids] = 1.0 - 2.0 * st["handoff"].clamp_min(0).float()  # falls: set at the trigger
    env.safefall_hold[env_ids] = q[:, robot.find_joints(spec.ACTION_JOINT_NAMES, preserve_order=True)[0]]


# --------------------------------------------------------------------- monitor
class SafeFallMonitor(CommandTerm):
    """Issues no command; logs the per-episode outcomes of ``safefall_math.OutcomeTracker`` as
    Metrics/<term>/<outcome> (means over the episodes that ended in the step).  Eval scripts set
    ``collect`` to receive every finished episode in ``finished``."""

    cfg: SafeFallMonitorCfg

    def __init__(self, cfg: SafeFallMonitorCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.tracker = sf.OutcomeTracker(self.num_envs, self.device, dt=env.step_dt)
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
