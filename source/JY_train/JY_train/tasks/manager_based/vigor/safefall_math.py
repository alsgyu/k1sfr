"""Safe-fall task terms on plain tensors (no Isaac imports).

Shared by the Isaac task (``mdp/safefall.py``) and the laptop-side MuJoCo check
(``scripts/vigor/safefall_mujoco.py``), so the reward trained on the GPU is the one that
was checked locally.  Impact mitigation only (another team handles getting up): land on
the fists, nothing else -- head, trunk, knees, thighs and forearms stay off the ground --
and keep holding the fist plank (fists + feet) with stiff arms until the end of the
episode; lowering down afterwards is a damping release on the robot.
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

# per-second weights (the env multiplies by its step dt, as Isaac's RewardManager does).
# Impacts last only a few 20 ms steps while the plank pays for seconds, so the contact weights
# are large: one head hit (~3 steps) costs more than holding the plank for the whole episode.
WEIGHTS = {
    "upright": 3.0,  # not falling stays the best outcome
    "plank": 1.5,  # held up on the fists (+ feet) only
    "fist_share": 2.0,  # share of the ground load (feet aside) on the fists
    "hands_lead": 0.5,  # fists go down before head and trunk
    "head_contact": -30.0,
    "head_impact": -10.0,  # per body weight
    "trunk_contact": -5.0,  # per body weight
    "leg_contact": -15.0,  # per body weight (hips, thighs, knees)
    "arm_contact": -2.0,  # per body weight (upper arms, forearms)
    "fist_impact": -1.0,  # per body weight above 2: let the elbows give instead of landing rigid
    "head_descent": -1.0,
    "trunk_descent": -0.5,
}


@dataclass
class FallState:
    force: dict[str, torch.Tensor]  # (N,) per group: sum over its bodies of the peak contact force in the step [N]
    head_h: torch.Tensor  # Head_2 origin above the ground [m]
    trunk_h: torch.Tensor  # root (Trunk) origin above the ground [m]
    hand_h: torch.Tensor  # lower hand origin above the ground [m]
    head_vz: torch.Tensor  # vertical velocities [m/s]
    trunk_vz: torch.Tensor
    up: torch.Tensor  # cos(trunk tilt) = -projected_gravity_z


def _on(x: torch.Tensor) -> torch.Tensor:
    return x > CONTACT_THRESHOLD


def upright(s: FallState) -> torch.Tensor:
    return (s.up > UPRIGHT_COS) & (s.trunk_h > 0.4)


def plank(s: FallState) -> torch.Tensor:
    """0.5 per fist on the ground while nothing but fists and feet touches and the trunk is held up."""
    f = s.force
    clean = ~_on(sum(f[g] for g in BAD_GROUPS)) & (s.trunk_h > PLANK_MIN_TRUNK) & ~upright(s)
    return clean * (0.5 * _on(f["fist_l"]).float() + 0.5 * _on(f["fist_r"]).float())


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
    return {
        "upright": upright(s).float(),
        "plank": plank(s),
        "fist_share": share,
        "hands_lead": lead,
        "head_contact": _on(f["head"]).float(),
        "head_impact": bw(f["head"]),
        "trunk_contact": bw(f["trunk"]),
        "leg_contact": bw(f["legs"]),
        "arm_contact": bw(f["arms"]),
        "fist_impact": (fists / WEIGHT - 2.0).clamp(0.0, 4.0),
        # descending fast close to the ground means the arms did not brake the fall
        "head_descent": (-s.head_vz).clamp_min(0.0).square() * torch.exp(-s.head_h / 0.25),
        "trunk_descent": (-s.trunk_vz).clamp_min(0.0).square() * torch.exp(-(s.trunk_h - 0.1).clamp_min(0.0) / 0.2),
    }


def reward(s: FallState, dt: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t = terms(s)
    weighted = {k: WEIGHTS[k] * dt * v for k, v in t.items()}
    return sum(weighted.values()), weighted


# --------------------------------------------------------------------- starts
# The walking stance tipping (hand-off from the walking policy) or pushed, in a direction
# relative to the heading: (azimuth, probability), each spread +-45 deg.
DIRECTIONS = ((0.0, 0.45), (math.pi, 0.2), (math.pi / 2, 0.175), (-math.pi / 2, 0.175))
DIRECTION_NAMES = ("forward", "backward", "left", "right")
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
    return {"root_quat": root_quat, "root_rel": root_rel, "lin_vel": lin_vel, "ang_vel": ang_vel,
            "direction": direction, "tilt": tilt}


# ------------------------------------------------------------------- outcomes
class OutcomeTracker:
    """Per-episode outcomes, vectorised over envs.

    head_hit / trunk_hit / knee_hit : head / trunk / hips-thighs-knees touched the ground
    head/trunk/leg_peak : largest contact force on them [body weights]
    fist_share  : share of the largest ground load (feet aside) carried by the fists
    fists_first : the first touch-down (feet aside) was taken mainly by the fists
    plank       : fraction of the time after the first touch-down spent held up on the fists only
    upright_end : upright at the last step
    """

    KEYS = ("head_hit", "trunk_hit", "knee_hit", "head_peak", "trunk_peak", "leg_peak", "fist_share", "fists_first",
            "plank", "upright_end")

    def __init__(self, num_envs: int, device="cpu"):
        z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        self.v = {k: z() for k in self.KEYS}
        self.peak_load, self.touched = z(), torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.after, self.planked = z(), z()  # steps since the first touch-down / of them on the fists

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
        }
        for k, x in new.items():  # in place: SafeFallMonitor logs these very tensors
            v[k].copy_(torch.where(live, x, v[k]))
        self.peak_load.copy_(torch.where(live, torch.maximum(self.peak_load, load), self.peak_load))
        self.after.copy_(torch.where(live, after, self.after))
        self.planked.copy_(torch.where(live, planked, self.planked))
        self.touched |= live & first

    def episodes(self, ids: torch.Tensor, direction: torch.Tensor | None = None) -> list[dict]:
        out = []
        for i in ids.tolist():
            e = {k: float(self.v[k][i]) for k in self.KEYS}
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
