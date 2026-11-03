"""Safe-fall task terms on plain tensors (no Isaac imports).

Shared by the Isaac task (``mdp/safefall.py``) and the laptop-side MuJoCo check
(``scripts/vigor/safefall_mujoco.py``), so the reward trained on the GPU is the one that
was checked locally.  Impact mitigation only (another team handles getting up): spread the
landing over several sturdy contacts and over time (``impact``) -- fists with the elbows
giving, the knees when falling forward, the hips when falling backward -- keep head, chest,
back and upper arms off the ground, and end in one of two fixed multi-contact poses the get-up
starts from (``HANDOFF_POSES``): kneeling on fists, knees and toes, or sitting on the hips with
the fists behind.  Falls keep the side they started on: forward falls end kneeling, backward
falls sitting.
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
EPISODE_S = 5.0  # pushed until the trigger (~0.2-1 s) + landing and holding the hand-off pose
# The fall detector on the robot: the safe-fall policy takes over from the walking policy once the
# IMU tilt reaches this; until then its actions are ignored and nothing is rewarded.
TRIGGER_TILT_DEG = 20.0
TRIGGER_COS = math.cos(math.radians(TRIGGER_TILT_DEG))

# links with collision shapes, grouped (hand_link is forearm + fist)
CONTACT_GROUPS = {
    "head": ["Head_1", "Head_2"],
    "trunk": ["Trunk"],
    "fist_l": ["left_hand_link"],
    "fist_r": ["right_hand_link"],
    "arms": ["Left_Arm_2", "Left_Arm_3", "Right_Arm_2", "Right_Arm_3"],  # shoulders, upper arms
    "legs": ["Left_Hip_Roll", "Left_Hip_Yaw", "Left_Shank", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Shank"],  # hips, thighs, knees
    "feet": ["left_foot_link", "right_foot_link"],
}
CONTACT_BODIES = [b for bodies in CONTACT_GROUPS.values() for b in bodies]  # FallState.body_force order
ARM_BODIES = {"left": ["left_hand_link", "Left_Arm_3", "Left_Arm_2"], "right": ["right_hand_link", "Right_Arm_3", "Right_Arm_2"]}
HEAD_BODY = "Head_2"
HAND_BODIES = ["left_hand_link", "right_hand_link"]
FOOT_BODIES = ["left_foot_link", "right_foot_link"]
# foot origins (ankles) above this are lifted: standing 0.03 m, kneeling on the toes 0.12, sitting 0.08
FOOT_FREE = 0.15
# hip roll / yaw further than this from the hand-off pose spread or twist the legs
SPREAD_JOINTS = ("Left_Hip_Roll", "Left_Hip_Yaw", "Right_Hip_Roll", "Right_Hip_Yaw")
SPREAD_FREE = 0.1  # rad
# fist tip: centre of the rounded far end of hand_link in its frame (the arm points along +-y); it is
# FIST_TOUCH_H up when the fist touches the ground
FIST_TIPS = ((0.0, 0.2, 0.0), (0.0, -0.2, 0.0))
FIST_TOUCH_H = 0.03
# the sturdy contacts the hand-off pose of each side rests on besides the fists (the pelvis also counts
# for the back); ``support`` pays a quarter per fist and per one of these on the ground
SUPPORT_BODIES = {"front": ["Left_Shank", "Right_Shank"], "back": ["Left_Hip_Roll", "Right_Hip_Roll"]}
# The trunk's collision shapes are a chest box and, below the hip joints, a pelvis cylinder.  Tilted less
# than this from upright only the pelvis reaches the ground (the box needs ~80 deg): a trunk contact then
# is sitting on the hips, otherwise it is the chest or the back.
PELVIS_TILT_DEG = 70.0
PELVIS_COS = math.cos(math.radians(PELVIS_TILT_DEG))

CONTACT_THRESHOLD = 5.0  # N
UPRIGHT_COS = math.cos(math.radians(20.0))
FALLING_COS = math.cos(math.radians(30.0))
SUPPORT_MIN_TRUNK = 0.08  # m: trunk origin height while held off the ground (sitting: 0.12)
HEAD_CLEAR = 0.15  # m: Head_2 origin below this is about to touch (head_low shapes before contact)
POSE_KERNEL = 0.5  # rad^2: mean squared joint error at which handoff_pose has dropped to 1/e

# Impact, the quantity the landing minimises.  Per body: the ground's push averaged over the policy step
# (impulse / 20 ms) in body weights, beyond IMPACT_FREE, squared, times IMPACT_WEIGHTS.  A landing's total
# impulse is set by the fall (momentum), so a linear sum over bodies only moves it to the cheapest one --
# with the fists the only contact allowed, that was the arms, 2-6 body weights each through 14 Nm motors.
# The square makes sharing pay (n bodies taking equal parts cost 1/n) and so does stretching it in time
# (joints that give while they take load): constant force over the longest stroke is the optimum.
IMPACT_FREE = 0.25  # body weights a body carries for free (each contact of the hand-off poses holds 0.1-0.3)
IMPACT_CLIP = 4.0  # body weights beyond IMPACT_FREE (bounds one step's cost)
IMPACT_WEIGHTS = {  # by how weak the joints behind the body are
    "head": 0.0,  # its own terms (head_contact, head_impact)
    "trunk": 3.0,  # chest or back; the pelvis counts as "pelvis"
    "pelvis": 1.0,
    "fist_l": 1.5,  # forearm + fist: 14 Nm shoulder and elbow motors
    "fist_r": 1.5,
    "arms": 3.0,
    "legs": 1.0,  # hips, thighs, knees: 68-112 Nm motors
    "feet": 0.5,
}

# Hand-off poses for the get-up: symmetric PD targets the K1 rests in on several contacts, each carrying
# 0.1-0.3 body weights (the old fist planks put 0.62 on the arms), held on the PD gains alone (arms
# < 2 Nm), surviving 0.3 / 0.6 m/s pushes of the trunk in any direction, head 0.31-0.33 m up, inside the
# soft joint limits.  Under load
# the joints settle up to ~0.08 rad from these targets (handoff_poses.json has the settled angles and the
# root state).  spec.ALL_JOINT_NAMES order (= INHA-Recovery-Motion K1_JOINT_ORDER); found, checked and
# exported by scripts/vigor/handoff_pose.py.
HANDOFF_POSES = {
    # chest down, kneeling on fists, knees and toes, elbows a little bent: fists 0.21 each, knees 0.18, toes 0.14
    "front": (0.000, 0.000, -1.200, -1.300, 0.300, -0.400, -1.200, 1.300, 0.300, 0.400, -1.050, 0.150, 0.000, 1.650, 0.050, 0.000, -1.050, -0.150, 0.000, 1.650, 0.050, 0.000),
    # chest up, sitting on the hips, fists behind, heels down: hips 0.30 each, fists 0.09, heels 0.11
    "back": (0.000, 0.000, 0.950, -1.150, 0.300, -0.200, 0.950, 1.150, 0.300, 0.200, -1.800, 0.150, 0.000, 0.850, -0.350, 0.000, -1.800, -0.150, 0.000, 0.850, -0.350, 0.000),
}
# root height [m] and trunk pitch [rad] resting in those poses (handoff_pose.py check)
HANDOFF_ROOTS = {"front": (0.274, 1.089), "back": (0.119, -0.292)}
# The reflex picks the front or back pose by a latched side (+1 front, -1 back): set from the tilt
# direction at the trigger, switched only once the chest clearly faces the other way.  Picking by the
# sign of chest_z each step flipped the legs between the two poses whenever a side fall rolled through
# chest-sideways, and the policy flailed its legs.
SIDE_SWITCH = 0.25  # |chest_z| beyond which the latched side follows the chest (sitting has 0.29, kneeling -0.89)
# The legs go to the hand-off pose gradually, from the joint targets held at the trigger (smoothstep over
# LEG_RAMP_S): jumping there at once folded them in the air (feet 0.25-0.4 m up) and put knees or hips down
# before the fists in 44-100 % of the reflex-only falls.  Front: from about when the fists touch (0.15-0.2 s
# after the trigger), so the knees come down as the arms give.  Back: at once; reflex-only backward falls
# hit the head in 22 % with 0.5 s, 29 % with 0.25 s, > 60 % with 0.75 s, and 95 % squatting first (rolls
# back like a ball).  The arms go to the pose at once (reach for the ground).
LEG_DELAY_S = {"front": 0.15, "back": 0.0}
LEG_RAMP_S = 0.5
FALL_TIME_MAX = 1.0  # s: the actor observes the time since the trigger up to this
LEG_ACTIONS = [i for i, n in enumerate(spec.ACTION_JOINT_NAMES) if "_Hip_" in n or "_Knee_" in n or "_Ankle_" in n]


def update_side(side: torch.Tensor, gravity_b: torch.Tensor) -> torch.Tensor:
    """Latched reflex side after this step's gravity (chest_z = -gravity_x)."""
    chest = -gravity_b[:, 0]
    return torch.where(chest < -SIDE_SWITCH, 1.0, torch.where(chest > SIDE_SWITCH, -1.0, side))


def leg_blend(side: torch.Tensor, fall_time: torch.Tensor) -> torch.Tensor:
    """How far the legs' reflex has moved from the trigger pose to the hand-off pose (0-1, smoothstep)."""
    delay = torch.where(side > 0, LEG_DELAY_S["front"], LEG_DELAY_S["back"])
    x = ((fall_time - delay) / LEG_RAMP_S).clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def reflex_targets(side: torch.Tensor, hold: torch.Tensor | None = None, fall_time: torch.Tensor | None = None) -> torch.Tensor:
    """Joint targets the policy's actions are offsets from, (N, 20) in spec.ACTION_JOINT_NAMES order.

    The hand-off pose of the latched side (``update_side``): with zero action the robot reaches for the
    ground and ends in a pose it holds, and the policy only learns the landing and balance on top.
    Measured from the walking stance instead, holding a plank took actions of -3.4 on shoulder or hip
    pitch and the policy never kept it.  Given the joint targets held at the trigger (``hold``, same
    order) and the time since it, the legs blend from those to the pose (``leg_blend``).  On the robot
    this is the same latch on the IMU's projected gravity, the targets at the switch and the time since."""
    body = len(spec.HEAD_JOINT_NAMES)
    poses = torch.tensor([HANDOFF_POSES["front"][body:], HANDOFF_POSES["back"][body:]], device=side.device)
    target = poses[(side < 0).long()]
    if hold is None:
        return target
    w = leg_blend(side, fall_time).unsqueeze(-1)
    legs = torch.zeros(target.shape[-1], dtype=torch.bool, device=side.device)
    legs[LEG_ACTIONS] = True
    return torch.where(legs, hold + w * (target - hold), target)

# per-second weights (the env multiplies by its step dt, as Isaac's RewardManager does).
# Impacts last only a few 20 ms steps while the pose pays for seconds, so the contact weights are
# large: one head hit (~3 steps) costs more than holding the pose for the whole episode.
WEIGHTS = {
    "upright": 3.0,  # not falling stays the best outcome
    "support": 1.5,  # resting on the fists and the knees / hips, head, chest, back and upper arms clear
    "hands_lead": 0.5,  # fists go down before head and trunk
    "handoff_pose": 4.0,  # supported in the front / back hand-off pose
    # per (body weight)^2 beyond IMPACT_FREE, fragility-weighted, summed over bodies: ~-4 for a fist plank
    # landing (0.8 (body weight)^2 s), well under a head hit
    "impact": -5.0,
    "head_contact": -50.0,
    "head_impact": -20.0,  # per body weight
    "head_low": -5.0,  # head origin below HEAD_CLEAR: about to touch
    "trunk_contact": -5.0,  # per body weight on the chest or back (not the pelvis)
    "head_descent": -1.0,
    "trunk_descent": -0.5,
    # per (m/s)^2 of a fist tip moving down near the ground: reach out early and let the body come down on
    # the arms.  Swung down by the reflex the fists hit at 2.3-2.8 m/s (the trunk falls at 1.3-1.5), one
    # 20 ms step of 3 body weights through the arm that the impact term alone did not remove
    "fist_descent": -2.0,
    # head, trunk, upper arms or legs on the ground before any fist has touched: the fists land first
    "early_contact": -30.0,
    # per m of foot origin above FOOT_FREE (each foot): the feet stay down
    "foot_lift": -10.0,
    # per rad^2 of hip roll / yaw beyond SPREAD_FREE from the hand-off pose: no spread or twisted legs
    "leg_spread": -10.0,
    # turned round mid-fall (forward fall ending chest up, backward chest down).  Support does not care
    # which side faces the ground, so turning also voids support and handoff_pose and costs more than they pay
    "facing": -5.0,
}
# regularisation of the residual on the reflex (per second, per sum of squared actions, legs x RESIDUAL_LEGS):
# leave the reflex only where the landing needs it, no flailing.  The legs only need to follow the reflex
# down; with the arms' weight the policy spread and twisted them (hip yaw 0.37 rad)
RESIDUAL_WEIGHT = -0.01
RESIDUAL_LEGS = 5.0


def residual(actions: torch.Tensor) -> torch.Tensor:
    """Sum of squared actions with the legs weighted RESIDUAL_LEGS (times RESIDUAL_WEIGHT is the penalty)."""
    scale = torch.ones(actions.shape[-1], device=actions.device)
    scale[LEG_ACTIONS] = RESIDUAL_LEGS
    return (scale * actions.square()).sum(dim=-1)


@dataclass
class FallState:
    force: dict[str, torch.Tensor]  # (N,) per group: sum over its bodies of the peak contact force in the step [N]
    body_force: torch.Tensor  # (N, len(CONTACT_BODIES)) contact force per body averaged over the step [N]
    head_h: torch.Tensor  # Head_2 origin above the ground [m]
    trunk_h: torch.Tensor  # root (Trunk) origin above the ground [m]
    hand_h: torch.Tensor  # lower hand origin above the ground [m]
    fist_h: torch.Tensor  # (N, 2) fist tips (FIST_TIPS) above the ground, left / right [m]
    fist_vz: torch.Tensor  # (N, 2) their vertical velocities [m/s]
    foot_h: torch.Tensor  # (N, 2) foot origins (FOOT_BODIES) above the ground [m]
    fists_before: torch.Tensor  # (N,) bool: a fist touched the ground in an earlier step of the fall
    head_vz: torch.Tensor  # vertical velocities [m/s]
    trunk_vz: torch.Tensor
    gravity_b: torch.Tensor  # (N, 3) unit gravity in the trunk frame (projected gravity)
    q: torch.Tensor  # (N, 22) joint positions, spec.ALL_JOINT_NAMES order
    facing: torch.Tensor  # cos(fall azimuth from the heading): +1 forward fall, -1 backward, ~0 sideways
    side: torch.Tensor  # latched reflex side: +1 front (kneeling), -1 back (sitting)

    @property
    def up(self) -> torch.Tensor:
        """cos(trunk tilt)."""
        return -self.gravity_b[:, 2]

    @property
    def chest_z(self) -> torch.Tensor:
        """World z of the trunk's forward axis: < 0 chest down, > 0 chest up."""
        return -self.gravity_b[:, 0]


def fist_tips(hand_pos: torch.Tensor, hand_quat: torch.Tensor, hand_vel: torch.Tensor | None = None,
              hand_ang_vel: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """World positions (N, 2, 3) of the fist tips from the HAND_BODIES poses, and their velocities when the
    bodies' linear / angular velocities (world frame) are given."""
    r = quat_apply(hand_quat, torch.tensor(FIST_TIPS, device=hand_pos.device).expand_as(hand_pos))
    vel = None if hand_vel is None else hand_vel + torch.cross(hand_ang_vel, r, dim=-1)
    return hand_pos + r, vel


def chest_z(quat: torch.Tensor) -> torch.Tensor:
    """World z of the body x (forward) axis of (w, x, y, z) quaternions (= -projected_gravity_x)."""
    w, x, y, z = quat.unbind(-1)
    return 2.0 * (x * z - w * y)


def _on(x: torch.Tensor) -> torch.Tensor:
    return x > CONTACT_THRESHOLD


def upright(s: FallState) -> torch.Tensor:
    return (s.up > UPRIGHT_COS) & (s.trunk_h > 0.4)


def pelvis_down(s: FallState) -> torch.Tensor:
    """A trunk contact would be the pelvis (trunk tilted less than PELVIS_TILT_DEG)."""
    return s.up >= PELVIS_COS


def chest_back(s: FallState) -> torch.Tensor:
    """Peak trunk contact force when it is the chest or the back [N]."""
    return torch.where(pelvis_down(s), torch.zeros_like(s.force["trunk"]), s.force["trunk"])


def support(s: FallState) -> torch.Tensor:
    """A quarter for each of the four contacts of the latched side's hand-off pose on the ground (two
    fists; two knees in front, two hips behind, where the pelvis stands for both) while head, upper arms,
    chest and back stay off it and the trunk is held up: 1 = resting on all of them.  With the fists
    alone a forward fall kept the knees off the ground and put the whole landing on the arms."""
    f = s.force
    clean = ~_on(f["head"] + f["arms"] + chest_back(s)) & (s.trunk_h > SUPPORT_MIN_TRUNK) & ~upright(s)
    body = lambda name: _on(s.body_force[:, CONTACT_BODIES.index(name)])  # noqa: E731
    front = [body(b) for b in SUPPORT_BODIES["front"]]
    pelvis = _on(s.force["trunk"]) & pelvis_down(s)
    back = [body(b) | pelvis for b in SUPPORT_BODIES["back"]]
    sturdy = [torch.where(s.side > 0, fb, bb).float() for fb, bb in zip(front, back)]
    return clean * 0.25 * (_on(f["fist_l"]).float() + _on(f["fist_r"]).float() + sturdy[0] + sturdy[1])


_BODY_GROUP = [g for g, bodies in CONTACT_GROUPS.items() for _ in bodies]


def impact(s: FallState) -> torch.Tensor:
    """Sum over bodies of IMPACT_WEIGHTS x (step-mean ground force / weight - IMPACT_FREE)_+^2."""
    w = torch.tensor([IMPACT_WEIGHTS[g] for g in _BODY_GROUP], device=s.q.device).repeat(len(s.q), 1)
    w[:, CONTACT_BODIES.index("Trunk")] = torch.where(pelvis_down(s), IMPACT_WEIGHTS["pelvis"], IMPACT_WEIGHTS["trunk"])
    excess = (s.body_force / WEIGHT - IMPACT_FREE).clamp(0.0, IMPACT_CLIP)
    return (w * excess.square()).sum(dim=-1)


# ------------------------------------------------------------------ mirroring
# Left-right reflection through the sagittal plane (symmetry augmentation, asymmetry outcome).  On this
# URDF pitch joints keep their sign and roll / yaw joints flip (checked by forward kinematics); body-frame
# angular velocity flips x and z, vectors (gravity, linear velocity) flip y.
MIRROR_KINDS = ("axial", "polar", "joints", "contacts", "none")


def _joint_mirror(names: tuple[str, ...] | list[str]) -> tuple[list[int], list[float]]:
    swap = lambda n: n.replace("Left", "@").replace("Right", "Left").replace("@", "Right")  # noqa: E731
    return [list(names).index(swap(n)) for n in names], [1.0 if "pitch" in n.lower() else -1.0 for n in names]


def mirror(x: torch.Tensor, kind: str, names=None) -> torch.Tensor:
    """Mirror the last dimension of ``x``: an angular velocity ("axial"), a vector ("polar"), joint values
    in the order of ``names`` ("joints"), CONTACT_GROUPS forces ("contacts") or a symmetric quantity."""
    if kind == "axial":
        return x * torch.tensor([-1.0, 1.0, -1.0], device=x.device)
    if kind == "polar":
        return x * torch.tensor([1.0, -1.0, 1.0], device=x.device)
    if kind == "joints":
        perm, sign = _joint_mirror(names)
        return x[..., perm] * torch.tensor(sign, device=x.device)
    if kind == "contacts":
        groups = list(CONTACT_GROUPS)
        perm = [groups.index({"fist_l": "fist_r", "fist_r": "fist_l"}.get(g, g)) for g in groups]
        return x[..., perm]
    if kind == "none":
        return x
    raise ValueError(f"unknown mirror kind {kind!r}")


def asymmetry(q: torch.Tensor) -> torch.Tensor:
    """RMS left-right asymmetry of (N, 22) joint positions over the body joints [rad] (0 = symmetric pose)."""
    head = len(spec.HEAD_JOINT_NAMES)
    return (0.5 * (q - mirror(q, "joints", spec.ALL_JOINT_NAMES)))[:, head:].square().mean(dim=-1).sqrt()


def handoff_target(s: FallState) -> torch.Tensor:
    """(N, 22) hand-off pose of each env's latched side."""
    poses = torch.tensor([HANDOFF_POSES["front"], HANDOFF_POSES["back"]], device=s.q.device)
    return poses[(s.side < 0).long()]


def leg_spread(s: FallState) -> torch.Tensor:
    """Sum over hip roll / yaw of (|deviation from the hand-off pose| - SPREAD_FREE)_+^2 [rad^2]."""
    ids = [spec.ALL_JOINT_NAMES.index(n) for n in SPREAD_JOINTS]
    return ((s.q[:, ids] - handoff_target(s)[:, ids]).abs() - SPREAD_FREE).clamp_min(0.0).square().sum(dim=-1)


def early_contact(s: FallState) -> torch.Tensor:
    """Head, trunk, upper arms or legs touch while no fist has touched yet (a fist in the same step is too late)."""
    f = s.force
    return ~s.fists_before & _on(f["head"] + f["trunk"] + f["arms"] + f["legs"])


def pose_error(s: FallState) -> torch.Tensor:
    """Mean squared error to the hand-off pose over the actuated (non-head) joints [rad^2]."""
    head = len(spec.HEAD_JOINT_NAMES)
    return (s.q[:, head:] - handoff_target(s)[:, head:]).square().mean(dim=-1)


def terms(s: FallState) -> dict[str, torch.Tensor]:
    """Unweighted per-step terms; multiply by WEIGHTS[name] * dt."""
    f = s.force
    # before head/trunk touch down: the fists go below head and trunk (reach for the ground)
    falling = (s.up < FALLING_COS) & ~_on(f["head"] + f["trunk"]) & (s.trunk_h > SUPPORT_MIN_TRUNK)
    lead = torch.sigmoid((torch.minimum(s.head_h, s.trunk_h) - s.hand_h) / 0.05) * falling
    bw = lambda x: (x / WEIGHT).clamp(0.0, 4.0)  # noqa: E731
    turned = (s.facing * s.chest_z).clamp(0.0, 1.0)
    held = support(s) * (1.0 - turned)
    return {
        "upright": upright(s).float(),
        "support": held,
        "hands_lead": lead,
        "handoff_pose": held * torch.exp(-pose_error(s) / POSE_KERNEL),
        "impact": impact(s),
        "head_contact": _on(f["head"]).float(),
        "head_impact": bw(f["head"]),
        "head_low": ((HEAD_CLEAR - s.head_h) / HEAD_CLEAR).clamp(0.0, 1.0),
        "trunk_contact": bw(chest_back(s)),
        # descending fast close to the ground means nothing braked the fall
        "head_descent": (-s.head_vz).clamp_min(0.0).square() * torch.exp(-s.head_h / 0.25),
        "trunk_descent": (-s.trunk_vz).clamp_min(0.0).square() * torch.exp(-(s.trunk_h - 0.1).clamp_min(0.0) / 0.2),
        "fist_descent": ((-s.fist_vz).clamp_min(0.0).square() * torch.exp(-(s.fist_h - FIST_TOUCH_H).clamp_min(0.0) / 0.05)).sum(-1),
        "early_contact": early_contact(s).float(),
        "foot_lift": (s.foot_h - FOOT_FREE).clamp_min(0.0).sum(dim=-1),
        "leg_spread": leg_spread(s),
        "facing": turned,
    }


def reward(s: FallState, dt: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t = terms(s)
    weighted = {k: WEIGHTS[k] * dt * v for k, v in t.items()}
    return sum(weighted.values()), weighted


# --------------------------------------------------------------------- starts
# Standing on the ground, or walking (forward speed, one foot lifted mid-swing), then pushed in a
# direction relative to the heading: (azimuth, probability), each spread +-45 deg.  Before the
# trigger the env holds the start pose (the walking policy's job on the robot), so the policy starts
# from whatever state a push leaves the robot in at 20 deg.
DIRECTIONS = ((0.0, 0.45), (math.pi, 0.2), (math.pi / 2, 0.175), (-math.pi / 2, 0.175))
DIRECTION_NAMES = ("forward", "backward", "left", "right", "handoff")
# Episodes that start already resting in a hand-off pose (front or back, joints jittered): the policy
# experiences the target poses and learns to hold them, which a fall alone rarely reaches.  Their
# direction is "handoff"; eval scripts set this to 0 so they measure falls only.
HANDOFF_START_PROB = 0.2
WALK_PROB = 0.5
WALK_SPEED = (0.2, 0.6)  # m/s along the heading
SWING_HIP, SWING_KNEE = (-0.4, -0.1), (0.3, 0.7)  # extra flexion of the lifted leg [rad]; ankle keeps the sole flat
PUSH_SPEED = (0.6, 2.5)  # trunk velocity change of the push [m/s] (eval also 2.5-3.0)


def _u(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


def facing_from_gravity(gravity_b: torch.Tensor) -> torch.Tensor:
    """cos of the tilt direction from the heading (+1 tipping forward, -1 backward, 0 sideways)."""
    return gravity_b[:, 0] / gravity_b[:, :2].norm(dim=-1).clamp_min(1e-6)


def sample_starts(n: int, device="cpu") -> dict[str, torch.Tensor]:
    """Root state (world frame, root above the origin; the env grounds the feet), joint offsets from the
    stance (canonical order: the lifted leg of walking starts), and the push; or a hand-off pose."""
    yaw = _u(-math.pi, math.pi, n, device)
    direction = torch.multinomial(torch.tensor([p for _, p in DIRECTIONS], device=device), n, replacement=True)
    azim = torch.tensor([a for a, _ in DIRECTIONS], device=device)[direction] + _u(-math.pi / 4, math.pi / 4, n, device)
    zeros = torch.zeros(n, device=device)
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw), zeros), -1)
    push = torch.stack((torch.cos(yaw + azim), torch.sin(yaw + azim), zeros), -1) * _u(*PUSH_SPEED, n, device).unsqueeze(-1)
    walk = torch.rand(n, device=device) < WALK_PROB
    lin_vel = push + heading * (walk * _u(*WALK_SPEED, n, device)).unsqueeze(-1)
    root_quat = quat_from_yaw(yaw)
    root_rel = torch.stack((zeros, zeros, zeros + spec.TRUNK_HEIGHT_NOMINAL), -1)
    names = spec.ALL_JOINT_NAMES
    joint_offset = torch.zeros(n, len(names), device=device)
    left = torch.rand(n, device=device) < 0.5
    hip, knee = _u(*SWING_HIP, n, device) * walk, _u(*SWING_KNEE, n, device) * walk
    for side, swing in (("Left", left), ("Right", ~left)):
        joint_offset[:, names.index(f"{side}_Hip_Pitch")] = hip * swing
        joint_offset[:, names.index(f"{side}_Knee_Pitch")] = knee * swing
        joint_offset[:, names.index(f"{side}_Ankle_Pitch")] = -(hip + knee) * swing
    facing = torch.cos(azim)  # until the trigger, which sets it from the actual tilt direction
    # hand-off starts: -1 none, 0 front, 1 back (the env sets the joints to HANDOFF_POSES)
    handoff = torch.where(torch.rand(n, device=device) < HANDOFF_START_PROB, torch.randint(0, 2, (n,), device=device), -1)
    on = handoff >= 0
    height, pitch = (torch.tensor([HANDOFF_ROOTS[k][i] for k in ("front", "back")], device=device)[handoff.clamp_min(0)]
                     for i in (0, 1))
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
    root_quat = torch.where(on.unsqueeze(-1), quat_mul(quat_from_yaw(yaw), quat_from_axis_angle(y_axis, pitch)), root_quat)
    root_rel = torch.where(on.unsqueeze(-1), torch.stack((zeros, zeros, height), -1), root_rel)
    lin_vel = torch.where(on.unsqueeze(-1), torch.zeros_like(lin_vel), lin_vel)
    return {"root_quat": root_quat, "root_rel": root_rel, "lin_vel": lin_vel, "ang_vel": torch.zeros_like(lin_vel),
            "joint_offset": torch.where(on.unsqueeze(-1), torch.zeros_like(joint_offset), joint_offset),
            "direction": torch.where(on, len(DIRECTIONS), direction), "walk": walk & ~on,
            "facing": torch.where(on, 1.0 - 2.0 * handoff.float(), facing), "handoff": handoff}


# ------------------------------------------------------------------- outcomes
LANDING_S = 0.5  # arm_share window after the first touch-down


class OutcomeTracker:
    """Per-episode outcomes, vectorised over envs.

    head_hit / trunk_hit : head / chest or back touched the ground
    head_peak / trunk_peak : largest contact force on them [body weights]
    impact      : time integral of the ``impact`` term over the episode [(body weight)^2 s]
    arm_peak    : largest step-mean ground force through one arm (forearm + fist, upper arm, shoulder) [body weights]
    max_peak    : largest step-mean ground force on any single body but the feet [body weights]
    arm_share   : share of the arms in the ground impulse (feet aside) over LANDING_S after the first touch-down
    fist_speed  : fastest downward speed of a fist tip within 5 cm of touching the ground [m/s]
    fists_first : no head, trunk, upper-arm or leg contact before the first fist contact (``early_contact``)
    foot_lift   : highest foot origin above the ground [m] (standing 0.03, kneeling 0.12, sitting 0.08)
    leg_spread  : largest hip roll / yaw deviation from the hand-off pose [rad]
    support     : fraction of the time after the first touch-down spent supported (``support``)
    upright_end : upright at the last step
    flipped     : turned round by the last step (forward fall chest up / backward fall chest down)
    handoff_end : resting on both fists and both knees / hips, head, chest, back and upper arms clear, at the last step
    pose_rms    : RMS joint error to the hand-off pose at the last step [rad]
    asym_rms    : RMS left-right asymmetry of the pose at the last step [rad]
    Episode dicts also carry ``end`` (joint positions, trunk height, projected gravity, side at the last step).
    """

    KEYS = ("head_hit", "trunk_hit", "head_peak", "trunk_peak", "impact", "arm_peak", "max_peak", "arm_share",
            "fist_speed", "fists_first", "foot_lift", "leg_spread", "support", "upright_end", "flipped", "handoff_end",
            "pose_rms", "asym_rms")

    def __init__(self, num_envs: int, device="cpu", dt: float = spec.POLICY_DT):
        z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        self.dt = dt
        self.v = {k: z() for k in self.KEYS}
        self.touched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.early = torch.zeros(num_envs, dtype=torch.bool, device=device)  # an early_contact happened
        self.after, self.supported = z(), z()  # steps since the first touch-down / of them supported
        self.arm_imp, self.all_imp = z(), z()  # landing impulse through the arms / all but the feet [N s]
        self.end = torch.zeros(num_envs, len(spec.ALL_JOINT_NAMES) + 5, device=device)  # q, trunk_h, gravity_b, side
        index = {b: i for i, b in enumerate(CONTACT_BODIES)}
        self.arm_ids = [[index[b] for b in ARM_BODIES[k]] for k in ("left", "right")]
        self.other_ids = [i for b, i in index.items() if b not in CONTACT_GROUPS["feet"]]

    def update(self, s: FallState, live: torch.Tensor | None = None):
        f, v = s.force, self.v
        live = torch.ones_like(self.touched) if live is None else live
        fists = f["fist_l"] + f["fist_r"]
        first = live & ~self.touched & _on(fists + f["head"] + f["trunk"] + f["arms"] + f["legs"])
        early = self.early | early_contact(s)
        spread = [spec.ALL_JOINT_NAMES.index(n) for n in SPREAD_JOINTS]
        touched = self.touched | first
        after = self.after + touched.float()
        supported = self.supported + touched.float() * (support(s) > 0).float()
        arm = torch.stack([s.body_force[:, ids].sum(dim=-1) for ids in self.arm_ids], dim=-1)  # (N, 2)
        landing = (touched & (self.after * self.dt < LANDING_S)).float() * self.dt
        arm_imp = self.arm_imp + landing * arm.sum(dim=-1)
        all_imp = self.all_imp + landing * s.body_force[:, self.other_ids].sum(dim=-1)
        new = {
            "head_hit": torch.maximum(v["head_hit"], _on(f["head"]).float()),
            "trunk_hit": torch.maximum(v["trunk_hit"], _on(chest_back(s)).float()),
            "head_peak": torch.maximum(v["head_peak"], f["head"] / WEIGHT),
            "trunk_peak": torch.maximum(v["trunk_peak"], chest_back(s) / WEIGHT),
            "impact": v["impact"] + impact(s) * self.dt,
            "arm_peak": torch.maximum(v["arm_peak"], arm.amax(dim=-1) / WEIGHT),
            "max_peak": torch.maximum(v["max_peak"], s.body_force[:, self.other_ids].amax(dim=-1) / WEIGHT),
            "arm_share": arm_imp / all_imp.clamp_min(1e-6),
            "fist_speed": torch.maximum(v["fist_speed"], ((-s.fist_vz) * (s.fist_h < FIST_TOUCH_H + 0.05)).amax(dim=-1)),
            "fists_first": 1.0 - early.float(),
            "foot_lift": torch.maximum(v["foot_lift"], s.foot_h.amax(dim=-1)),
            "leg_spread": torch.maximum(v["leg_spread"], (s.q[:, spread] - handoff_target(s)[:, spread]).abs().amax(dim=-1)),
            "support": supported / after.clamp_min(1.0),
            "upright_end": upright(s).float(),
            "flipped": (s.facing * s.chest_z > 0.25).float(),
            "handoff_end": (support(s) == 1.0).float(),
            "pose_rms": pose_error(s).sqrt(),
            "asym_rms": asymmetry(s.q),
        }
        for k, x in new.items():  # in place: SafeFallMonitor logs these very tensors
            v[k].copy_(torch.where(live, x, v[k]))
        for buf, x in ((self.after, after), (self.supported, supported), (self.arm_imp, arm_imp), (self.all_imp, all_imp)):
            buf.copy_(torch.where(live, x, buf))
        end = torch.cat((s.q, s.trunk_h.unsqueeze(-1), s.gravity_b, s.side.unsqueeze(-1)), dim=-1)
        self.end.copy_(torch.where(live.unsqueeze(-1), end, self.end))
        self.touched |= live & first
        self.early |= live & early

    def episodes(self, ids: torch.Tensor, direction: torch.Tensor | None = None) -> list[dict]:
        out = []
        for i in ids.tolist():
            e = {k: float(self.v[k][i]) for k in self.KEYS}
            nq = len(spec.ALL_JOINT_NAMES)
            end = [round(x, 4) for x in self.end[i].tolist()]
            e["end"] = {"q": end[:nq], "trunk_h": end[nq], "gravity_b": end[nq + 1:nq + 4], "side": end[nq + 4]}
            if direction is not None:
                e["direction"] = DIRECTION_NAMES[int(direction[i])]
            out.append(e)
        return out

    def clear(self, ids: torch.Tensor, values: bool = True):
        if values:
            for k in self.KEYS:
                self.v[k][ids] = 0.0
        for buf in (self.after, self.supported, self.arm_imp, self.all_imp):
            buf[ids] = 0.0
        self.touched[ids] = False
        self.early[ids] = False

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


SCORE_KEYS = ("head_hit", "handoff_end", "fists_first", "trunk_hit", "impact", "arm_peak", "arm_share", "fist_speed",
              "foot_lift", "leg_spread", "pose_rms", "asym_rms")
IMPACT_SCORE = 0.5  # score per unit of the impact outcome


def score(summary: dict) -> float:
    """One number to rank checkpoints (higher is better): the head first, then ending clean in the
    hand-off pose, the fists landing first, chest / back contact, the impact and the distance to the
    hand-off pose."""
    s = summary
    return (s["handoff_end"] + s["fists_first"] - 3.0 * s["head_hit"] - s["trunk_hit"] - IMPACT_SCORE * s["impact"]
            - s["pose_rms"])


def end_poses(episodes: list[dict]) -> dict:
    """Last-step poses of the episodes that ended resting on all four hand-off contacts, per latched side, next to the
    target hand-off pose: what the get-up actually receives.  ``samples`` keeps every such end state
    (joints, trunk height, projected gravity, side) for the get-up to start from."""
    out = {}
    for side, sign in (("front", 1.0), ("back", -1.0)):
        ends = [e["end"] for e in episodes if e["handoff_end"] == 1.0 and e["end"]["side"] == sign]
        out[side] = {"episodes": len(ends), "share_of_all": len(ends) / max(len(episodes), 1),
                     "target_joint_pos": list(HANDOFF_POSES[side])}
        if ends:
            q = torch.tensor([x["q"] for x in ends])
            r = lambda t: [round(v, 3) for v in t.tolist()]  # noqa: E731
            out[side] |= {
                "joint_pos_median": r(q.median(dim=0).values),
                "joint_pos_std": r(q.std(dim=0)) if len(ends) > 1 else None,
                "trunk_h_median": round(float(torch.tensor([x["trunk_h"] for x in ends]).median()), 3),
                "projected_gravity_median": r(torch.tensor([x["gravity_b"] for x in ends]).median(dim=0).values),
                "samples": ends,
            }
    return out
