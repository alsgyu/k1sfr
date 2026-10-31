"""Safe-fall task terms on plain tensors (no Isaac imports).

Shared by the Isaac task (``mdp/safefall.py``) and the laptop-side MuJoCo check
(``scripts/vigor/safefall_mujoco.py``), so the reward trained on the GPU is the one that
was checked locally.  The goal is impact mitigation only (another team handles getting
up): catch the fall on the arms so they take the load, keep the head (stereo camera) off
the ground and slow the head and trunk down before they reach it.
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

# links with collision shapes, grouped
CONTACT_GROUPS = {
    "head": ["Head_1", "Head_2"],
    "trunk": ["Trunk"],
    "arms": ["Left_Arm_2", "Left_Arm_3", "left_hand_link", "Right_Arm_2", "Right_Arm_3", "right_hand_link"],
    "legs": ["Left_Hip_Roll", "Left_Hip_Yaw", "Left_Shank", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Shank"],
    "feet": ["left_foot_link", "right_foot_link"],
}
HEAD_BODY = "Head_2"
HAND_BODIES = ["left_hand_link", "right_hand_link"]

CONTACT_THRESHOLD = 5.0  # N
UPRIGHT_COS = math.cos(math.radians(20.0))
FALLING_COS = math.cos(math.radians(30.0))
# static share of the body weight the trunk may carry while lying (only more counts as impact)
TRUNK_REST_SHARE = 0.6

# per-second weights (the env multiplies by its step dt, as Isaac's RewardManager does)
WEIGHTS = {
    "upright": 1.0,  # standing is still the safest outcome
    "head_contact": -2.0,
    "head_impact": -2.0,
    "trunk_impact": -1.0,
    "arm_share": 1.0,
    "hands_lead": 0.5,
    "head_descent": -0.5,
    "trunk_descent": -0.3,
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


def terms(s: FallState) -> dict[str, torch.Tensor]:
    """Unweighted per-step terms; multiply by WEIGHTS[name] * dt."""
    f = s.force
    head_on = f["head"] > CONTACT_THRESHOLD
    body_on = (f["head"] + f["trunk"]) > CONTACT_THRESHOLD
    upright = (s.up > UPRIGHT_COS) & (s.trunk_h > 0.4)
    # share of the non-foot ground load carried by the arms while the body is on the ground
    load = f["arms"] + f["trunk"] + f["head"] + f["legs"]
    share = torch.where(load > 0.3 * WEIGHT, f["arms"] / load.clamp_min(1e-3), torch.zeros_like(load))
    # before head/trunk touch down: the hands go below head and trunk (reach for the ground)
    falling = (s.up < FALLING_COS) & ~body_on & (s.trunk_h > 0.15)
    lead = torch.sigmoid((torch.minimum(s.head_h, s.trunk_h) - s.hand_h) / 0.05) * falling
    return {
        "upright": upright.float(),
        "head_contact": head_on.float(),
        "head_impact": (f["head"] / WEIGHT).clamp(0.0, 4.0),
        "trunk_impact": (f["trunk"] / WEIGHT - TRUNK_REST_SHARE).clamp(0.0, 4.0),
        "arm_share": share,
        "hands_lead": lead,
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

    head_hit         : the head touched the ground
    head/trunk_peak  : largest contact force on head / trunk [body weights]
    arm_share        : share of the largest non-foot ground load carried by the arms
    arms_first       : the first non-foot ground contact was taken mainly by the arms
    fell             : the trunk went below spec.FULL_FALL_TRUNK_HEIGHT
    upright_end      : upright at the last step
    """

    KEYS = ("head_hit", "head_peak", "trunk_peak", "arm_share", "arms_first", "fell", "upright_end")

    def __init__(self, num_envs: int, device="cpu"):
        z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        self.v = {k: z() for k in self.KEYS}
        self.peak_load, self.touched = z(), torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, s: FallState, live: torch.Tensor | None = None):
        f, v = s.force, self.v
        live = torch.ones_like(self.touched) if live is None else live
        load = f["arms"] + f["trunk"] + f["head"] + f["legs"]
        new = {
            "head_hit": torch.maximum(v["head_hit"], (f["head"] > CONTACT_THRESHOLD).float()),
            "head_peak": torch.maximum(v["head_peak"], f["head"] / WEIGHT),
            "trunk_peak": torch.maximum(v["trunk_peak"], f["trunk"] / WEIGHT),
            "arm_share": torch.where(load > self.peak_load, f["arms"] / load.clamp_min(1e-3), v["arm_share"]),
            "fell": torch.maximum(v["fell"], (s.trunk_h < spec.FULL_FALL_TRUNK_HEIGHT).float()),
            "upright_end": ((s.up > UPRIGHT_COS) & (s.trunk_h > 0.4)).float(),
        }
        first = live & ~self.touched & (load > CONTACT_THRESHOLD)
        arms_lead = (f["arms"] >= torch.maximum(torch.maximum(f["trunk"], f["head"]), f["legs"])).float()
        new["arms_first"] = torch.where(first, arms_lead, v["arms_first"])
        for k, x in new.items():  # in place: SafeFallMonitor logs these very tensors
            v[k].copy_(torch.where(live, x, v[k]))
        self.peak_load.copy_(torch.where(live, torch.maximum(self.peak_load, load), self.peak_load))
        self.touched |= live & (load > CONTACT_THRESHOLD)

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
