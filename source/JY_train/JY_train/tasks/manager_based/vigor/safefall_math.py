"""Safe-fall task terms on plain tensors (no Isaac imports).

Shared by the Isaac task (``mdp/safefall.py``) and the laptop-side MuJoCo check
(``scripts/vigor/safefall_mujoco.py``), so the reward trained on the GPU is the one that
was checked locally.  Impact mitigation only (another team handles getting up): land on
the fists, nothing else -- head, trunk, knees, thighs and forearms stay off the ground --
and keep holding the fist plank (fists + feet) with stiff arms until the end of the
episode, in one of two fixed poses the get-up starts from (``PLANK_POSES``).  Falls keep the
side they started on: forward falls end in the front plank, backward falls in the back one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    from . import spec
    from .torch_math import quat_apply, quat_from_axis_angle, quat_from_yaw, quat_mul
except ImportError:  # loaded as a top-level module by the laptop-side scripts
    import spec
    from torch_math import quat_apply, quat_from_axis_angle, quat_from_yaw, quat_mul

MASS = 19.67  # K1_22dof_jy.urdf total mass [kg]
WEIGHT = MASS * 9.81
EPISODE_S = 4.0  # fall (~0.6 s) + holding the plank

# links with collision shapes, grouped; only fists and feet may touch the ground
CONTACT_GROUPS = {
    "head": ["Head_1", "Head_2"],
    "trunk": ["Trunk"],
    "fist_l": ["left_hand_link"],
    "fist_r": ["right_hand_link"],
    "arms": ["Left_Arm_2", "Left_Arm_3", "Right_Arm_2", "Right_Arm_3"],  # upper arms, forearms
    "legs": ["Left_Hip_Roll", "Left_Hip_Yaw", "Left_Shank", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Shank"],  # hips, thighs, knees
    "feet": ["left_foot_link", "right_foot_link"],
}
BAD_GROUPS = ("head", "trunk", "arms", "legs")
HEAD_BODY = "Head_2"
HAND_BODIES = ["left_hand_link", "right_hand_link"]

CONTACT_THRESHOLD = 5.0  # N
UPRIGHT_COS = math.cos(math.radians(20.0))
FALLING_COS = math.cos(math.radians(30.0))
PLANK_MIN_TRUNK = 0.15  # m: trunk origin height while held up on the fists
HEAD_CLEAR = 0.15  # m: Head_2 origin below this is about to touch (head_low shapes before contact)
POSE_KERNEL = 0.5  # rad^2: mean squared joint error at which plank_pose has dropped to 1/e

# Hand-off poses for the get-up.  Symmetric, held on the PD gains alone (arms straight under the
# shoulders carry the load through the bones, ~1 Nm at the shoulders), survive 0.6 m/s pushes of
# the trunk in any direction, head 0.4 m up.  Joint angles as settled under load, in
# spec.ALL_JOINT_NAMES order (= INHA-Recovery-Motion K1_JOINT_ORDER); found, checked and exported
# with root height / orientation by scripts/vigor/plank_pose.py.
PLANK_POSES = {
    # chest down, push-up plank on the fists: root 0.315 m, trunk pitched 66 deg forward, head 0.41 m
    "front": (0.000, 0.055, -1.174, -1.251, 0.297, -0.140, -1.174, 1.251, 0.297, 0.140, -0.146, 0.205, 0.001, 0.174, -0.044, -0.002, -0.146, -0.205, -0.001, 0.174, -0.044, 0.002),
    # chest up, crab on fists and flat feet, knees bent: root 0.216 m, leaning back 23 deg, head 0.45 m
    "back": (0.000, -0.015, 0.614, -1.247, 0.295, -0.149, 0.614, 1.247, 0.295, 0.149, -1.645, 0.199, 0.004, 1.166, -0.167, 0.003, -1.645, -0.199, -0.004, 1.166, -0.167, -0.003),
}
# root height [m] and trunk pitch [rad] of those poses resting on fists and feet (plank_pose.py check;
# heights from geom bounding boxes, up to ~1 cm high)
PLANK_ROOTS = {"front": (0.327, 1.154), "back": (0.227, -0.385)}
# chest_z over which the reflex target blends from the front plank to the back one (the back plank
# itself has chest_z 0.38, so it gets < 1 % of the front pose)
REFLEX_BLEND = 0.07


def reflex_targets(gravity_b: torch.Tensor) -> torch.Tensor:
    """Joint targets the policy's actions are offsets from, (N, 20) in spec.ACTION_JOINT_NAMES order.

    The hand-off plank on the side the chest faces, blended through chest-sideways: with zero action
    the robot already reaches for the ground and ends in a plank it holds, and the policy only learns
    the landing and balance on top.  Measured from the walking stance instead, holding a plank took
    actions of -3.4 on shoulder or hip pitch and the policy never kept it.  On the robot this is the
    same function of the IMU's projected gravity (``gravity_b``, (N, 3))."""
    body = len(spec.HEAD_JOINT_NAMES)
    poses = torch.tensor([PLANK_POSES["front"][body:], PLANK_POSES["back"][body:]], device=gravity_b.device)
    front = torch.sigmoid(gravity_b[:, :1] / REFLEX_BLEND)  # chest down: chest_z = -gravity_x < 0
    return front * poses[0] + (1.0 - front) * poses[1]

# per-second weights (the env multiplies by its step dt, as Isaac's RewardManager does).
# Impacts last only a few 20 ms steps while the plank pays for seconds, so the contact weights
# are large: one head hit (~3 steps) costs more than holding the plank for the whole episode.
WEIGHTS = {
    "upright": 3.0,  # not falling stays the best outcome
    "plank": 1.5,  # held up on the fists (+ feet) only
    "fist_share": 2.0,  # share of the ground load (feet aside) on the fists
    "hands_lead": 0.5,  # fists go down before head and trunk
    "plank_pose": 4.0,  # the plank held in the front / back hand-off pose
    "head_contact": -50.0,
    "head_impact": -20.0,  # per body weight
    "head_low": -5.0,  # head origin below HEAD_CLEAR: about to touch
    "trunk_contact": -5.0,  # per body weight
    "leg_contact": -15.0,  # per body weight (hips, thighs, knees)
    "arm_contact": -2.0,  # per body weight (upper arms, forearms)
    "fist_impact": -1.0,  # per body weight above 2: let the elbows give instead of landing rigid
    "head_descent": -1.0,
    "trunk_descent": -0.5,
    # turned round mid-fall (forward fall ending chest up, backward chest down).  The plank does not
    # care which side faces the ground and a supine fists-and-feet table is easier to hold, so turning
    # also voids plank and fist_share and costs more than they pay
    "facing": -5.0,
}


@dataclass
class FallState:
    force: dict[str, torch.Tensor]  # (N,) per group: sum over its bodies of the peak contact force in the step [N]
    head_h: torch.Tensor  # Head_2 origin above the ground [m]
    trunk_h: torch.Tensor  # root (Trunk) origin above the ground [m]
    hand_h: torch.Tensor  # lower hand origin above the ground [m]
    head_vz: torch.Tensor  # vertical velocities [m/s]
    trunk_vz: torch.Tensor
    gravity_b: torch.Tensor  # (N, 3) unit gravity in the trunk frame (projected gravity)
    q: torch.Tensor  # (N, 22) joint positions, spec.ALL_JOINT_NAMES order
    facing: torch.Tensor  # cos(fall azimuth from the heading): +1 forward fall, -1 backward, ~0 sideways

    @property
    def up(self) -> torch.Tensor:
        """cos(trunk tilt)."""
        return -self.gravity_b[:, 2]

    @property
    def chest_z(self) -> torch.Tensor:
        """World z of the trunk's forward axis: < 0 chest down, > 0 chest up."""
        return -self.gravity_b[:, 0]


def chest_z(quat: torch.Tensor) -> torch.Tensor:
    """World z of the body x (forward) axis of (w, x, y, z) quaternions (= -projected_gravity_x)."""
    w, x, y, z = quat.unbind(-1)
    return 2.0 * (x * z - w * y)


def _on(x: torch.Tensor) -> torch.Tensor:
    return x > CONTACT_THRESHOLD


def upright(s: FallState) -> torch.Tensor:
    return (s.up > UPRIGHT_COS) & (s.trunk_h > 0.4)


def plank(s: FallState) -> torch.Tensor:
    """0.5 per fist on the ground while nothing but fists and feet touches and the trunk is held up."""
    f = s.force
    clean = ~_on(sum(f[g] for g in BAD_GROUPS)) & (s.trunk_h > PLANK_MIN_TRUNK) & ~upright(s)
    return clean * (0.5 * _on(f["fist_l"]).float() + 0.5 * _on(f["fist_r"]).float())


def plank_target(s: FallState) -> torch.Tensor:
    """(N, 22) hand-off pose for each env: the front plank with the chest down, the back one up."""
    poses = torch.tensor([PLANK_POSES["front"], PLANK_POSES["back"]], device=s.q.device)
    return poses[(s.chest_z >= 0.0).long()]


def pose_error(s: FallState) -> torch.Tensor:
    """Mean squared error to the hand-off pose over the actuated (non-head) joints [rad^2]."""
    head = len(spec.HEAD_JOINT_NAMES)
    return (s.q[:, head:] - plank_target(s)[:, head:]).square().mean(dim=-1)


def terms(s: FallState) -> dict[str, torch.Tensor]:
    """Unweighted per-step terms; multiply by WEIGHTS[name] * dt."""
    f = s.force
    fists = f["fist_l"] + f["fist_r"]
    load = fists + sum(f[g] for g in BAD_GROUPS)
    share = torch.where(load > 0.3 * WEIGHT, fists / load.clamp_min(1e-3), torch.zeros_like(load))
    # before head/trunk touch down: the fists go below head and trunk (reach for the ground)
    falling = (s.up < FALLING_COS) & ~_on(f["head"] + f["trunk"]) & (s.trunk_h > PLANK_MIN_TRUNK)
    lead = torch.sigmoid((torch.minimum(s.head_h, s.trunk_h) - s.hand_h) / 0.05) * falling
    bw = lambda x: (x / WEIGHT).clamp(0.0, 4.0)  # noqa: E731
    turned = (s.facing * s.chest_z).clamp(0.0, 1.0)
    return {
        "upright": upright(s).float(),
        "plank": plank(s) * (1.0 - turned),
        "fist_share": share * (1.0 - turned),
        "hands_lead": lead,
        "plank_pose": plank(s) * (1.0 - turned) * torch.exp(-pose_error(s) / POSE_KERNEL),
        "head_contact": _on(f["head"]).float(),
        "head_impact": bw(f["head"]),
        "head_low": ((HEAD_CLEAR - s.head_h) / HEAD_CLEAR).clamp(0.0, 1.0),
        "trunk_contact": bw(f["trunk"]),
        "leg_contact": bw(f["legs"]),
        "arm_contact": bw(f["arms"]),
        "fist_impact": (fists / WEIGHT - 2.0).clamp(0.0, 4.0),
        # descending fast close to the ground means the arms did not brake the fall
        "head_descent": (-s.head_vz).clamp_min(0.0).square() * torch.exp(-s.head_h / 0.25),
        "trunk_descent": (-s.trunk_vz).clamp_min(0.0).square() * torch.exp(-(s.trunk_h - 0.1).clamp_min(0.0) / 0.2),
        "facing": turned,
    }


def reward(s: FallState, dt: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t = terms(s)
    weighted = {k: WEIGHTS[k] * dt * v for k, v in t.items()}
    return sum(weighted.values()), weighted


# --------------------------------------------------------------------- starts
# The walking stance tipping (hand-off from the walking policy) or pushed, in a direction
# relative to the heading: (azimuth, probability), each spread +-45 deg.
DIRECTIONS = ((0.0, 0.45), (math.pi, 0.2), (math.pi / 2, 0.175), (-math.pi / 2, 0.175))
DIRECTION_NAMES = ("forward", "backward", "left", "right", "plank")
# Episodes that start already held in a hand-off plank (front or back, joints jittered): the policy
# experiences the target poses and learns to hold them, which a fall alone rarely reaches.  Their
# direction is "plank"; eval scripts set this to 0 so they measure falls only.
PLANK_START_PROB = 0.2
TIP_PROB = 0.7
TIP_TILT_DEG, TIP_OMEGA, TIP_SPEED = (10.0, 30.0), (0.5, 2.5), (0.0, 0.6)
PUSH_TILT_DEG, PUSH_OMEGA, PUSH_SPEED = (0.0, 5.0), (0.0, 1.0), (0.8, 2.0)


def _u(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


def sample_starts(n: int, device="cpu") -> dict[str, torch.Tensor]:
    """Root state of the stance rotated about a ground pivot under the feet (world frame, pivot at the origin)."""
    yaw = _u(-math.pi, math.pi, n, device)
    direction = torch.multinomial(torch.tensor([p for _, p in DIRECTIONS], device=device), n, replacement=True)
    azim = torch.tensor([a for a, _ in DIRECTIONS], device=device)[direction] + _u(-math.pi / 4, math.pi / 4, n, device)
    tip = torch.rand(n, device=device) < TIP_PROB
    tilt = torch.deg2rad(torch.where(tip, _u(*TIP_TILT_DEG, n, device), _u(*PUSH_TILT_DEG, n, device)))
    omega = torch.where(tip, _u(*TIP_OMEGA, n, device), _u(*PUSH_OMEGA, n, device))
    speed = torch.where(tip, _u(*TIP_SPEED, n, device), _u(*PUSH_SPEED, n, device))
    zeros = torch.zeros(n, device=device)
    dir_w = torch.stack((torch.cos(yaw + azim), torch.sin(yaw + azim), zeros), -1)
    axis = torch.stack((-dir_w[:, 1], dir_w[:, 0], zeros), -1)  # tipping towards dir_w
    q_tilt = quat_from_axis_angle(axis, tilt)
    root_quat = quat_mul(q_tilt, quat_from_yaw(yaw))
    up = torch.tensor([0.0, 0.0, spec.TRUNK_HEIGHT_NOMINAL], device=device).expand(n, 3)
    root_rel = quat_apply(q_tilt, up)
    ang_vel = axis * omega.unsqueeze(-1)
    lin_vel = torch.cross(ang_vel, root_rel, dim=-1) + dir_w * speed.unsqueeze(-1)
    facing = torch.cos(azim)
    # plank starts: -1 none, 0 front, 1 back (the env sets the joints to PLANK_POSES)
    plank = torch.where(torch.rand(n, device=device) < PLANK_START_PROB, torch.randint(0, 2, (n,), device=device), -1)
    on = plank >= 0
    height, pitch = (torch.tensor([PLANK_ROOTS[k][i] for k in ("front", "back")], device=device)[plank.clamp_min(0)]
                     for i in (0, 1))
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
    root_quat = torch.where(on.unsqueeze(-1), quat_mul(quat_from_yaw(yaw), quat_from_axis_angle(y_axis, pitch)), root_quat)
    root_rel = torch.where(on.unsqueeze(-1), torch.stack((zeros, zeros, height), -1), root_rel)
    lin_vel, ang_vel = (torch.where(on.unsqueeze(-1), torch.zeros_like(v), v) for v in (lin_vel, ang_vel))
    return {"root_quat": root_quat, "root_rel": root_rel, "lin_vel": lin_vel, "ang_vel": ang_vel,
            "direction": torch.where(on, len(DIRECTIONS), direction), "tilt": torch.where(on, zeros, tilt),
            "facing": torch.where(on, 1.0 - 2.0 * plank.float(), facing), "plank": plank}


# ------------------------------------------------------------------- outcomes
class OutcomeTracker:
    """Per-episode outcomes, vectorised over envs.

    head_hit / trunk_hit / knee_hit : head / trunk / hips-thighs-knees touched the ground
    head/trunk/leg_peak : largest contact force on them [body weights]
    fist_share  : share of the largest ground load (feet aside) carried by the fists
    fists_first : the first touch-down (feet aside) was taken mainly by the fists
    plank       : fraction of the time after the first touch-down spent held up on the fists only
    upright_end : upright at the last step
    flipped     : turned round by the last step (forward fall chest up / backward fall chest down)
    plank_end   : on both fists, nothing else touching, at the last step
    pose_rms    : RMS joint error to the hand-off pose at the last step [rad]
    Episode dicts also carry ``end`` (joint positions, trunk height, projected gravity at the last step).
    """

    KEYS = ("head_hit", "trunk_hit", "knee_hit", "head_peak", "trunk_peak", "leg_peak", "fist_share", "fists_first",
            "plank", "upright_end", "flipped", "plank_end", "pose_rms")

    def __init__(self, num_envs: int, device="cpu"):
        z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        self.v = {k: z() for k in self.KEYS}
        self.peak_load, self.touched = z(), torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.after, self.planked = z(), z()  # steps since the first touch-down / of them on the fists
        self.end = torch.zeros(num_envs, len(spec.ALL_JOINT_NAMES) + 4, device=device)  # q, trunk_h, gravity_b

    def update(self, s: FallState, live: torch.Tensor | None = None):
        f, v = s.force, self.v
        live = torch.ones_like(self.touched) if live is None else live
        fists = f["fist_l"] + f["fist_r"]
        load = fists + sum(f[g] for g in BAD_GROUPS)
        first = live & ~self.touched & _on(load)
        biggest = torch.stack([f[g] for g in BAD_GROUPS], dim=-1).amax(dim=-1)
        touched = self.touched | first
        after = self.after + touched.float()
        planked = self.planked + touched.float() * (plank(s) > 0).float()
        new = {
            "head_hit": torch.maximum(v["head_hit"], _on(f["head"]).float()),
            "trunk_hit": torch.maximum(v["trunk_hit"], _on(f["trunk"]).float()),
            "knee_hit": torch.maximum(v["knee_hit"], _on(f["legs"]).float()),
            "head_peak": torch.maximum(v["head_peak"], f["head"] / WEIGHT),
            "trunk_peak": torch.maximum(v["trunk_peak"], f["trunk"] / WEIGHT),
            "leg_peak": torch.maximum(v["leg_peak"], f["legs"] / WEIGHT),
            "fist_share": torch.where(load > self.peak_load, fists / load.clamp_min(1e-3), v["fist_share"]),
            "fists_first": torch.where(first, (fists >= biggest).float(), v["fists_first"]),
            "plank": planked / after.clamp_min(1.0),
            "upright_end": upright(s).float(),
            "flipped": (s.facing * s.chest_z > 0.25).float(),
            "plank_end": (plank(s) == 1.0).float(),
            "pose_rms": pose_error(s).sqrt(),
        }
        for k, x in new.items():  # in place: SafeFallMonitor logs these very tensors
            v[k].copy_(torch.where(live, x, v[k]))
        self.peak_load.copy_(torch.where(live, torch.maximum(self.peak_load, load), self.peak_load))
        self.after.copy_(torch.where(live, after, self.after))
        self.planked.copy_(torch.where(live, planked, self.planked))
        end = torch.cat((s.q, s.trunk_h.unsqueeze(-1), s.gravity_b), dim=-1)
        self.end.copy_(torch.where(live.unsqueeze(-1), end, self.end))
        self.touched |= live & first

    def episodes(self, ids: torch.Tensor, direction: torch.Tensor | None = None) -> list[dict]:
        out = []
        for i in ids.tolist():
            e = {k: float(self.v[k][i]) for k in self.KEYS}
            nq = len(spec.ALL_JOINT_NAMES)
            end = [round(x, 4) for x in self.end[i].tolist()]
            e["end"] = {"q": end[:nq], "trunk_h": end[nq], "gravity_b": end[nq + 1:]}
            if direction is not None:
                e["direction"] = DIRECTION_NAMES[int(direction[i])]
            out.append(e)
        return out

    def clear(self, ids: torch.Tensor, values: bool = True):
        if values:
            for k in self.KEYS:
                self.v[k][ids] = 0.0
        self.peak_load[ids] = 0.0
        self.after[ids] = 0.0
        self.planked[ids] = 0.0
        self.touched[ids] = False

    def close(self, ids: torch.Tensor, direction: torch.Tensor | None = None) -> list[dict]:
        """Outcome dicts of the episodes of ``ids``, then clear them for the next episode."""
        out = self.episodes(ids, direction)
        self.clear(ids)
        return out


def summarize(episodes: list[dict]) -> dict:
    def stats(e):
        n = len(e)
        if not n:
            return {"episodes": 0}
        return {"episodes": n} | {k: sum(x[k] for x in e) / n for k in OutcomeTracker.KEYS}

    out = stats(episodes)
    if episodes and "direction" in episodes[0]:
        out["by_direction"] = {d: stats([x for x in episodes if x["direction"] == d]) for d in DIRECTION_NAMES}
    return out


def end_poses(episodes: list[dict]) -> dict:
    """Last-step poses of the episodes that ended in a clean two-fist plank, per side (chest down =
    front), next to the target hand-off pose: what the get-up actually receives."""
    out = {}
    for side, sign in (("front", 1.0), ("back", -1.0)):
        ends = [e["end"] for e in episodes if e["plank_end"] == 1.0 and sign * e["end"]["gravity_b"][0] > 0.0]
        out[side] = {"episodes": len(ends), "share_of_all": len(ends) / max(len(episodes), 1),
                     "target_joint_pos": list(PLANK_POSES[side])}
        if ends:
            q = torch.tensor([x["q"] for x in ends])
            r = lambda t: [round(v, 3) for v in t.tolist()]  # noqa: E731
            out[side] |= {
                "joint_pos_median": r(q.median(dim=0).values),
                "joint_pos_std": r(q.std(dim=0)) if len(ends) > 1 else None,
                "trunk_h_median": round(float(torch.tensor([x["trunk_h"] for x in ends]).median()), 3),
                "projected_gravity_median": r(torch.tensor([x["gravity_b"] for x in ends]).median(dim=0).values),
            }
    return out
