"""Isaac-free checks for the VIGOR pieces (run on the laptop and on the GPU machine).

    python scripts/vigor/test_offline.py            # needs torch, numpy, rsl-rl-lib 2.3.x
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import sys
import tempfile

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
VIGOR = REPO / "source/JY_train/JY_train/tasks/manager_based/vigor"
sys.path.insert(0, str(VIGOR))  # spec / torch_math / keyframes as top-level modules
sys.path.insert(0, str(REPO / "scripts/vigor/data"))

import spec  # noqa: E402
import torch_math as tm  # noqa: E402

try:
    import onnxruntime as _ort
except ImportError:  # optional: ONNX parity is checked where onnxruntime exists
    _ort = None


def _load(name: str, path: pathlib.Path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _load_package(name: str, path: pathlib.Path):
    s = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


modules = _load_package("vigor_modules", VIGOR / "modules")  # networks + rsl_rl teacher
networks = sys.modules["vigor_modules.networks"]


def test_spec():
    assert spec.PROPRIO_DIM == 70 and spec.GOAL_DIM == 70 and spec.TERRAIN_DIM == 226
    assert spec.TEACHER_OBS_DIM == spec.PROPRIO_DIM + spec.GOAL_DIM + spec.TERRAIN_DIM
    assert len(set(spec.ALL_JOINT_NAMES)) == spec.NUM_JOINTS == 22
    assert spec.KEY_BODY_NAMES[0] == spec.ROOT_BODY_NAME
    n = len(np.arange(-spec.HEIGHT_SCAN_SIZE[0] / 2, spec.HEIGHT_SCAN_SIZE[0] / 2 + 1e-9, spec.HEIGHT_SCAN_RESOLUTION))
    assert n * n == spec.HEIGHT_SCAN_POINTS, n


def test_quaternions():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(64, 4), dim=-1)
    v = torch.randn(64, 3)
    assert torch.allclose(tm.quat_apply_inverse(q, tm.quat_apply(q, v)), v, atol=1e-5)
    yaw = torch.rand(64) * 6 - 3
    assert torch.allclose(tm.yaw_from_quat(tm.quat_from_yaw(yaw)), yaw, atol=1e-5)
    # goal features live in the robot root frame: invariant to any global rotation of robot + reference
    rel = torch.randn(8, 9, 3)
    robot_q = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    ref_q = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    g = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    feat = lambda rq, fq, r: torch.cat((tm.quat_apply_inverse(rq.unsqueeze(1).expand(-1, 9, -1), r).flatten(1),  # noqa: E731
                                        tm.quat_to_6d(tm.quat_mul(tm.quat_conjugate(rq), fq))), -1)
    a = feat(robot_q, ref_q, rel)
    b = feat(tm.quat_mul(g, robot_q), tm.quat_mul(g, ref_q), tm.quat_apply(g.unsqueeze(1).expand(-1, 9, -1), rel))
    assert torch.allclose(a, b, atol=1e-5)
    six = tm.quat_to_6d(robot_q)
    assert torch.allclose(six[:, :3].norm(dim=-1), torch.ones(8), atol=1e-5) and (six[:, :3] * six[:, 3:]).sum(-1).abs().max() < 1e-5
    # torch.where with python scalars is used when sampling start regimes
    assert torch.where(torch.tensor([True, False]), 1.0, 2.0).tolist() == [1.0, 2.0]
    g = tm.projected_gravity(tm.quat_from_axis_angle(torch.tensor([[1.0, 0, 0]]), torch.tensor([math.radians(20)])))
    assert abs(math.degrees(tm.tilt_angle(g).item()) - 20) < 1e-3


def test_dataset():
    keyframes = _load("keyframes", VIGOR / "keyframes.py")
    d = keyframes.KeyframeDataset("k1_fall_keyframes_v3.npz", strict_hash=True)
    print("   ", d.summary())
    assert d.rel_body_pos.shape == (d.num_frames, len(spec.KEY_BODY_NAMES), 3)
    assert d.rel_body_pos[:, 0].abs().max() < 1e-5  # root at origin
    assert len(d.brace_clips) > 0, "no brace-and-rise clips"
    # brace-and-rise clips: both hands planted at the brace, head and trunk never on the ground,
    # and the clip ends standing in the walking stance
    c = spec.REF_CONTACT_NAMES
    for i in d.brace_clips.tolist():
        s, n = int(d.clip_start[i]), int(d.clip_length[i])
        ref = d.ref_contact[s : s + n]
        assert ref[int(d.lowest[i]), c.index("left_hand")] and ref[int(d.lowest[i]), c.index("right_hand")], d.clip_names[i]
        assert not ref[:, [c.index("head"), c.index("trunk")]].any(), d.clip_names[i]
        assert abs(float(d.root_height[s + n - 1]) - (spec.TRUNK_HEIGHT_NOMINAL - spec.SOLE_OFFSET)) < 0.01
    # every other clip ends standing too (K1 get-up spliced after the human fall)
    ends = d.clip_start + d.clip_length - 1
    assert (d.root_height[ends, 0] > 0.45).all()
    assert (d.fall_onset < d.lowest).all() and (d.lowest < d.getup_end).all() and (d.getup_end < d.clip_length).all()
    clip = torch.arange(d.num_clips)
    local, g = d.future_keyframes(clip, torch.zeros(d.num_clips) + 3.2, spec.NUM_FUTURE_KEYFRAMES)
    assert (local[:, 0] == d.stride).all() and (local[:, 1] == 2 * d.stride).all()
    assert (g[:, 0] == d.clip_start + d.stride).all()
    # clips are grounded: the lowest collision point sits on the floor
    assert d.root_height.min() > 0.0
    # FK of the stored joints reproduces the stored keypoints (same URDF as Isaac)
    from urdf_fk import URDFModel

    m = URDFModel(str(REPO / "source/JY_train/JY_train/assets/robots/urdf/K1_22dof_jy.urdf"))
    idx = np.array([0, d.num_frames // 2, d.num_frames - 1])
    fk = m.fk(d.root_pos[idx].numpy(), d.root_quat[idx].numpy(), d.joint_pos[idx].numpy(), spec.ALL_JOINT_NAMES)
    for k, b in enumerate(spec.KEY_BODY_NAMES):
        assert np.allclose(fk[b][0], d.body_pos[idx, k].numpy(), atol=2e-4), b
    # reference stays within the soft joint limits Isaac enforces
    lo, hi = m.limits(spec.ALL_JOINT_NAMES)
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.9
    assert (d.joint_pos.numpy() >= mid - half - 1e-4).all() and (d.joint_pos.numpy() <= mid + half + 1e-4).all()


def test_k1_getup_stance():
    """k1_getup.STANCE (Isaac-free copy) must equal the robot's K1_FULL_DEFAULT_JOINT_POS."""
    import ast
    import re

    import k1_getup

    tree = ast.parse((REPO / "source/JY_train/JY_train/assets/robots/booster_k1_full.py").read_text())
    node = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "K1_FULL_DEFAULT_JOINT_POS" for t in n.targets))
    patterns = ast.literal_eval(node)
    for name in spec.ALL_JOINT_NAMES:
        value = next(v for k, v in patterns.items() if re.fullmatch(k, name))
        assert abs(k1_getup.STANCE[name] - value) < 1e-9, name


def test_safefall_math():
    """Starts stand upright on the ground and are pushed the way they claim; the reward prefers a landing
    shared over several bodies and ending in the hand-off pose."""
    import safefall_math as sf

    st = sf.sample_starts(4096)
    falls = st["handoff"] < 0
    cos_tilt = 1 - 2 * (st["root_quat"][:, 1] ** 2 + st["root_quat"][:, 2] ** 2)  # trunk z-axis vs world z
    assert torch.allclose(cos_tilt[falls], torch.ones(int(falls.sum())), atol=1e-6)  # standing upright
    assert torch.allclose(st["root_rel"][falls, 2], torch.full((int(falls.sum()),), spec.TRUNK_HEIGHT_NOMINAL))
    heading = torch.stack((1 - 2 * (st["root_quat"][:, 2] ** 2 + st["root_quat"][:, 3] ** 2),
                           2 * (st["root_quat"][:, 1] * st["root_quat"][:, 2] + st["root_quat"][:, 0] * st["root_quat"][:, 3])), -1)
    walk_speed = (st["lin_vel"][:, :2] * heading).sum(-1)  # forward speed (push included)
    assert (st["lin_vel"][:, 2] == 0).all() and (st["lin_vel"][falls, :2].norm(dim=-1) > 0).all()  # a push against the walk can nearly cancel it
    assert ((walk_speed[st["direction"] == 0]) > 0.6 * math.cos(math.pi / 4) - 1e-6).all()  # pushed forward
    # walking starts lift one leg, the ankle keeps the sole flat; standing starts lift none
    names = spec.ALL_JOINT_NAMES
    knees = st["joint_offset"][:, [names.index("Left_Knee_Pitch"), names.index("Right_Knee_Pitch")]]
    assert ((knees > 0).sum(-1) == st["walk"].long()).all() and (knees.max(-1).values[st["walk"]] >= 0.3).all()
    sole = sum(st["joint_offset"][:, [names.index(f"{s}_{j}") for s in ("Left", "Right")]] for j in ("Hip_Pitch", "Knee_Pitch", "Ankle_Pitch"))
    assert torch.allclose(sole, torch.zeros_like(sole), atol=1e-6)
    assert (st["facing"][st["direction"] == 0] > 0.7).all() and (st["facing"][st["direction"] == 1] < -0.7).all()
    # the facing set at the trigger follows the tilt direction
    g = torch.tensor([[math.sin(0.4), 0.0, -math.cos(0.4)], [-math.sin(0.4), 0.0, -math.cos(0.4)], [0.0, math.sin(0.4), -math.cos(0.4)]])
    assert torch.allclose(sf.facing_from_gravity(g), torch.tensor([1.0, -1.0, 0.0]), atol=1e-6)
    assert math.isclose(sf.TRIGGER_COS, math.cos(math.radians(20.0)))
    held = st["handoff"] >= 0  # hand-off starts: chest down in front, up in back, at rest
    assert 0.1 < held.float().mean() < 0.3 and (st["direction"][held] == len(sf.DIRECTIONS)).all()
    assert (sf.chest_z(st["root_quat"])[st["handoff"] == 0] < -0.8).all() and (sf.chest_z(st["root_quat"])[st["handoff"] == 1] > -sf.SIDE_SWITCH["front"]).all()
    assert (st["lin_vel"][held] == 0).all() and torch.equal(st["facing"][held], 1.0 - 2.0 * st["handoff"][held].float())
    n = 4
    z = torch.zeros(n)
    body = {b: i for i, b in enumerate(sf.CONTACT_BODIES)}

    def state(trunk_h, chest_z=-0.9, facing=1.0, head_h=0.3, q=None, side=None, bodies=None, fist_h=sf.FIST_TOUCH_H,
              fist_vz=0.0, foot_h=0.05, fists_before=True, widths=(0.32, 0.28), **force):
        """Group forces in body weights (split evenly over the group's bodies for body_force, unless
        ``bodies`` gives per-body step-mean forces in body weights)."""
        f = {g: z + force.get(g, 0.0) * sf.WEIGHT for g in sf.CONTACT_GROUPS}
        per_body = torch.zeros(n, len(sf.CONTACT_BODIES))
        for g, names in sf.CONTACT_GROUPS.items():
            for b in names:
                per_body[:, body[b]] = force.get(g, 0.0) * sf.WEIGHT / len(names)
        for b, v in (bodies or {}).items():
            per_body[:, body[b]] = v * sf.WEIGHT
        side = (1.0 if chest_z < 0 else -1.0) if side is None else side
        q = torch.tensor(sf.HANDOFF_POSES["front" if side > 0 else "back"]) if q is None else torch.as_tensor(q)
        grav = torch.tensor([-chest_z, 0.0, -math.sqrt(1 - chest_z**2)]).expand(n, 3)
        return sf.FallState(force=f, body_force=per_body, head_h=z + head_h, trunk_h=z + trunk_h, hand_h=z + 0.02,
                            fist_h=torch.full((n, 2), fist_h), fist_vz=torch.full((n, 2), fist_vz),
                            foot_h=torch.full((n, 2), foot_h), fists_before=torch.full((n,), fists_before),
                            arm_width=torch.tensor(widths).expand(n, 2), head_vz=z,
                            trunk_vz=z, gravity_b=grav, q=q.float().expand(n, -1), facing=z + facing, side=z + side)

    kneel = dict(trunk_h=0.27, fist_l=0.2, fist_r=0.2, legs=0.35, feet=0.25)
    kneeling, _ = sf.reward(state(**kneel), 0.02)
    lying, _ = sf.reward(state(0.06, chest_z=-0.99, fist_l=0.1, fist_r=0.1, trunk=0.5, legs=0.3), 0.02)
    faceplant, _ = sf.reward(state(0.06, chest_z=-0.99, head=2.0, trunk=3.0), 0.02)
    assert (sf.support(state(**kneel)) == 1).all() and (sf.impact(state(**kneel)) == 0).all()  # every contact under IMPACT_FREE
    assert (kneeling > lying).all() and (lying > faceplant).all()
    # support counts the knees in front (hips behind): held up on the fists alone is half of it
    on_fists = state(0.3, fist_l=0.3, fist_r=0.3, feet=0.4)
    knees = dict(trunk_h=0.27, fist_l=0.2, fist_r=0.2, bodies={"Left_Shank": 0.18, "Right_Shank": 0.18})
    assert (sf.support(on_fists) == 0.5).all() and (sf.support(state(**knees)) == 1).all()
    assert (sf.support(state(**knees, side=-1.0)) == 0.5).all()  # knees are not the back pose's contacts
    # the impact: the same load costs less shared over more bodies, and less on legs than on arms
    fists_only = sf.impact(state(0.27, bodies={"left_hand_link": 2.0, "right_hand_link": 2.0}))
    shared = sf.impact(state(0.27, bodies={"left_hand_link": 1.0, "right_hand_link": 1.0, "Left_Shank": 1.0, "Right_Shank": 1.0}))
    on_knees = sf.impact(state(0.27, bodies={"Left_Shank": 2.0, "Right_Shank": 2.0}))
    assert (shared < 0.5 * fists_only).all() and (on_knees < fists_only).all()
    # sitting on the pelvis is the hips (no chest / back contact, pelvis weight); lying on the back is not
    sitting = state(0.12, chest_z=0.29, facing=-1.0, fist_l=0.1, fist_r=0.1, trunk=0.6, feet=0.2)
    lying_back = state(0.06, chest_z=0.99, facing=-1.0, fist_l=0.1, fist_r=0.1, trunk=0.6, feet=0.2)
    assert (sf.terms(sitting)["trunk_contact"] == 0).all() and (sf.support(sitting) == 1).all()
    assert (sf.terms(lying_back)["trunk_contact"] > 0).all() and (sf.support(lying_back) == 0).all()
    assert (sf.impact(lying_back) > sf.impact(sitting)).all()
    # a forward fall that turned round onto its back loses the pose's worth; a backward one does not
    sit = dict(trunk_h=0.12, fist_l=0.1, fist_r=0.1, legs=0.6, feet=0.2, chest_z=0.29)
    turned, _ = sf.reward(state(**sit, side=-1.0), 0.02)
    backward, _ = sf.reward(state(**sit | {"facing": -1.0}), 0.02)
    assert (turned < backward).all() and (backward > 0).all()
    # support pays more in the hand-off pose of the latched side
    off_pose, _ = sf.reward(state(**kneel, q=torch.tensor(sf.HANDOFF_POSES["front"]) + 0.4), 0.02)
    wrong_side, _ = sf.reward(state(**kneel, side=-1.0, q=torch.tensor(sf.HANDOFF_POSES["front"])), 0.02)
    assert (kneeling > off_pose).all() and (kneeling > wrong_side).all()
    assert torch.allclose(sf.terms(state(**sit | {"facing": -1.0}))["handoff_pose"], torch.ones(n))
    assert (sf.terms(state(**kneel))["handoff_pose"] == 1).all() and (sf.pose_error(state(**kneel)) == 0).all()
    # fists coming down fast just above the ground cost (high up, or slowly, they do not)
    slam = sf.terms(state(**kneel | {"fist_h": 0.05, "fist_vz": -2.5}))["fist_descent"]
    high = sf.terms(state(**kneel | {"fist_h": 0.4, "fist_vz": -2.5}))["fist_descent"]
    soft = sf.terms(state(**kneel | {"fist_h": 0.05, "fist_vz": -0.5}))["fist_descent"]
    assert (slam > 5.0).all() and (high < 0.01 * slam).all() and (soft < 0.05 * slam).all()
    # fist tips sit 0.2 m out along the hand link's +-y, and move with its rotation
    hand_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 2, [[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]] * 2])
    tip, tip_vel = sf.fist_tips(torch.zeros(2, 2, 3), hand_quat, torch.zeros(2, 2, 3), torch.tensor([1.0, 0.0, 0.0]).expand(2, 2, 3))
    assert torch.allclose(tip[0], torch.tensor([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]), atol=1e-6)
    assert torch.allclose(tip[1], torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, -0.2]]), atol=1e-6)  # +90 deg about x
    assert torch.allclose(tip_vel[0, :, 2], torch.tensor([0.2, -0.2]), atol=1e-6)  # w x r
    # knees down before any fist: early contact; feet lifted and legs spread cost
    early = dict(trunk_h=0.4, legs=0.5, fists_before=False, fist_h=0.2)
    assert (sf.terms(state(**early))["early_contact"] == 1).all()
    assert (sf.terms(state(**early | {"fist_l": 0.3}))["early_contact"] == 0).all()  # landing together is fine
    assert (sf.terms(state(**early | {"fist_h": sf.FIST_TOUCH_H + 0.02}))["early_contact"] == 0).all()  # fists about to touch
    assert (sf.terms(state(**early | {"legs": 0.0, "fist_l": 0.3}))["early_contact"] == 0).all()
    # arms flung out: elbows / fists wider across the trunk than the hand-off pose + 5 cm
    assert (sf.terms(state(**kneel))["arm_width"] == 0).all()
    assert torch.allclose(sf.terms(state(**kneel | {"widths": (0.32, 0.48)}))["arm_width"], torch.full((n,), 0.15**2))
    assert (sf.terms(state(**sit | {"facing": -1.0, "widths": sf.ARM_WIDTH_MAX["back"]}))["arm_width"] == 0).all()
    back_wide = sf.terms(state(**sit | {"facing": -1.0, "widths": sf.ARM_WIDTH_MAX["front"]}))["arm_width"]  # per side
    assert torch.allclose(back_wide, torch.full((n,), 0.02**2 + 0.07**2))
    assert (sf.terms(state(**kneel))["early_contact"] == 0).all() and (sf.terms(state(**kneel))["foot_lift"] == 0).all()
    assert torch.allclose(sf.terms(state(**kneel | {"foot_h": 0.35}))["foot_lift"], torch.full((n,), 0.4))
    assert (sf.terms(state(**kneel | {"foot_h": 0.35, "fists_before": False}))["foot_lift"] == 0).all()  # in the air: free
    spread_q = torch.tensor(sf.HANDOFF_POSES["front"])
    spread_q[[spec.ALL_JOINT_NAMES.index(j) for j in ("Left_Hip_Yaw", "Right_Hip_Yaw")]] = torch.tensor([0.4, -0.4])
    assert (sf.terms(state(**kneel))["leg_spread"] == 0).all()
    assert torch.allclose(sf.terms(state(**kneel, q=spread_q))["leg_spread"], torch.full((n,), 2 * 0.3**2))
    act = torch.zeros(2, len(spec.ACTION_JOINT_NAMES))
    act[0, sf.LEG_ACTIONS[0]], act[1, spec.ACTION_JOINT_NAMES.index("ALeft_Shoulder_Pitch")] = 1.0, 1.0
    assert torch.allclose(sf.residual(act), torch.tensor([sf.RESIDUAL_LEGS, 1.0]))
    # head about to touch costs before contact
    low_head, _ = sf.reward(state(**kneel | {"head_h": 0.08}), 0.02)
    assert (low_head < kneeling).all()
    # zero action targets the hand-off pose of the latched side; the side follows the chest only past -0.5 / 0.25
    assert spec.ACTION_JOINT_NAMES == spec.ALL_JOINT_NAMES[len(spec.HEAD_JOINT_NAMES):]
    reflex = sf.reflex_targets(torch.tensor([1.0, -1.0]))
    assert torch.allclose(reflex[0], torch.tensor(sf.HANDOFF_POSES["front"][2:])) and torch.allclose(reflex[1], torch.tensor(sf.HANDOFF_POSES["back"][2:]))
    # given the held pose and the time since the trigger the legs blend from it (arms at once); front waits
    hold = torch.zeros(2, len(spec.ACTION_JOINT_NAMES))
    legs, arms = sf.LEG_ACTIONS, [i for i in range(hold.shape[1]) if i not in sf.LEG_ACTIONS]
    side = torch.tensor([1.0, -1.0])
    at0 = sf.reflex_targets(side, hold, torch.zeros(2))
    assert torch.equal(at0[:, legs], hold[:, legs]) and torch.allclose(at0[:, arms], reflex[:, arms])
    mid = sf.reflex_targets(side, hold, torch.full((2,), sf.LEG_DELAY_S["front"] + 0.5 * sf.LEG_RAMP_S["front"]))
    assert torch.allclose(mid[0, legs], 0.5 * reflex[0, legs])  # smoothstep(0.5) = 0.5
    assert (mid[1, legs].abs() > 0.5 * reflex[1, legs].abs() - 1e-6).all()  # back started at once
    done = sf.reflex_targets(side, hold, torch.full((2,), 2.0))
    assert torch.allclose(done, reflex)
    g = torch.tensor([[0.6, 0.0, -0.8], [-0.5, 0.0, -0.87], [0.3, 0.0, -0.95], [-0.1, 0.0, -0.99]])
    assert torch.equal(sf.update_side(torch.tensor([-1.0, 1.0, -1.0, 1.0]), g), torch.tensor([1.0, -1.0, -1.0, 1.0]))  # sitting leaning forward stays back
    assert torch.equal(sf.update_side(torch.tensor([1.0]), torch.tensor([[-0.288, 0.0, -0.958]])), torch.tensor([-1.0]))  # sitting is back
    assert torch.equal(sf.initial_side(torch.tensor([1.0, 0.1, -0.3, -0.7, -1.0])), torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0]))  # sideways: kneel
    # the tracker integrates the impact, measures the arms' share and hands the last-step pose to end_poses
    tr = sf.OutcomeTracker(n)
    landing = state(0.27, fist_l=1.0, fist_r=1.0, legs=1.0, bodies={"left_hand_link": 1.0, "right_hand_link": 1.0,
                                                                    "Left_Shank": 1.0, "Right_Shank": 1.0})
    tr.update(landing)
    tr.update(state(**kneel))
    e = tr.close(torch.arange(n))
    assert math.isclose(e[0]["impact"], float(sf.impact(landing)[0]) * spec.POLICY_DT, rel_tol=1e-5)
    assert math.isclose(e[0]["arm_peak"], 1.0, rel_tol=1e-4) and 0.4 < e[0]["arm_share"] < 0.6 and e[0]["handoff_end"] == 1.0
    assert e[0]["fists_first"] == 1.0 and math.isclose(e[0]["foot_lift"], 0.05, rel_tol=1e-5) and e[0]["leg_spread"] < 1e-6
    tr.update(state(**early))
    assert tr.close(torch.arange(n))[0]["fists_first"] == 0.0
    ends = sf.end_poses(e)
    assert ends["front"]["episodes"] == n and ends["back"]["episodes"] == 0 and len(ends["front"]["samples"]) == n
    assert torch.allclose(torch.tensor(ends["front"]["joint_pos_median"]), torch.tensor(sf.HANDOFF_POSES["front"]), atol=1e-3)


def test_safefall_symmetry():
    """The Isaac augmentation reads the layout from the observation manager and mirrors every term."""
    import types

    import safefall_math as sf
    import safefall_symmetry as sym

    joints = list(spec.ALL_JOINT_NAMES)
    robot_order = joints[1::2] + joints[0::2]  # the articulation's order differs from the canonical one
    terms = [("base_ang_vel", 3), ("projected_gravity", 3), ("joint_pos", 22), ("joint_vel", 22), ("last_action", 20)]
    h = 3
    om = types.SimpleNamespace(active_terms={"policy": [t for t, _ in terms]}, group_obs_term_dim={"policy": [(h * d,) for _, d in terms]})
    base = types.SimpleNamespace(observation_manager=om, scene={"robot": types.SimpleNamespace(data=types.SimpleNamespace(joint_names=robot_order))})
    env = types.SimpleNamespace(unwrapped=base)
    obs, act = torch.randn(5, h * 70), torch.randn(5, 20)
    aug_obs, aug_act = sym.data_augmentation_func(obs=obs, actions=act, env=env, obs_type="policy")
    assert aug_obs.shape == (10, h * 70) and torch.equal(aug_obs[:5], obs) and torch.equal(aug_act[:5], act)
    twice, _ = sym.data_augmentation_func(obs=aug_obs[5:], env=env, obs_type="policy")
    assert torch.allclose(twice[5:], obs)  # mirroring twice is the identity
    # a left hip roll in the newest joint_pos frame becomes the negated right hip roll in the same frame
    probe = torch.zeros(1, h * 70)
    jp = 2 * 3 * h  # joint_pos block: after ang vel and gravity (h frames of 3 each)
    probe[0, jp + 2 * 22 + robot_order.index("Left_Hip_Roll")] = 0.3
    m, _ = sym.data_augmentation_func(obs=probe, env=env, obs_type="policy")
    assert torch.isclose(m[1, jp + 2 * 22 + robot_order.index("Right_Hip_Roll")], torch.tensor(-0.3))
    assert torch.isclose(m[1, jp + 2 * 22 + robot_order.index("Left_Hip_Roll")], torch.tensor(0.0))
    # the hand-off poses and the reflex are symmetric, so mirrored actions mean mirrored joint targets
    for pose in sf.HANDOFF_POSES.values():
        assert torch.allclose(sf.mirror(torch.tensor(pose), "joints", spec.ALL_JOINT_NAMES), torch.tensor(pose))
    assert torch.equal(sf.mirror(torch.arange(7.0), "contacts"), torch.tensor([0.0, 1, 3, 2, 4, 5, 6]))


def test_teacher_ppo():
    import rsl_rl.runners.on_policy_runner as opr
    from rsl_rl.algorithms import PPO

    assert opr.VigorTeacherActorCritic is modules.VigorTeacherActorCritic  # OnPolicyRunner can eval() it
    n_env, n_steps = 16, 8
    n_critic = spec.TEACHER_OBS_DIM + 40
    # same kwargs OnPolicyRunner passes from VigorTeacherActorCriticCfg.to_dict()
    policy = modules.VigorTeacherActorCritic(
        spec.TEACHER_OBS_DIM, n_critic, spec.NUM_ACTIONS, proprio_dim=spec.PROPRIO_DIM, latent_dim=spec.LATENT_DIM,
        encoder_hidden_dims=[256, 128], actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu", init_noise_std=1.0, noise_std_type="scalar",
    )
    z = policy.actor.encode(torch.randn(5, spec.TEACHER_OBS_DIM))
    assert torch.allclose(z.norm(dim=-1), torch.full((5,), math.sqrt(spec.LATENT_DIM)), atol=1e-4)
    assert policy.actor[0].in_features == spec.TEACHER_OBS_DIM
    alg = PPO(policy, num_learning_epochs=2, num_mini_batches=2, device="cpu")
    alg.init_storage("rl", n_env, n_steps, [spec.TEACHER_OBS_DIM], [n_critic], [spec.NUM_ACTIONS])
    for _ in range(n_steps):
        obs, cobs = torch.randn(n_env, spec.TEACHER_OBS_DIM), torch.randn(n_env, n_critic)
        alg.act(obs, cobs)
        alg.process_env_step(torch.randn(n_env), torch.zeros(n_env), {})
    alg.compute_returns(torch.randn(n_env, n_critic))
    losses = alg.update()
    print("    ppo losses:", {k: round(float(v), 4) for k, v in (losses.items() if isinstance(losses, dict) else enumerate(losses))})


def _isaac_history(frames: list[torch.Tensor], term_dims: list[int]) -> torch.Tensor:
    """Mimic Isaac Lab: per term, (T, d) oldest-first flattened, terms concatenated."""
    parts, start = [], 0
    for d in term_dims:
        parts.append(torch.stack([f[:, start : start + d] for f in frames], dim=1).reshape(frames[0].shape[0], -1))
        start += d
    return torch.cat(parts, dim=-1)


def test_student_layout_and_export():
    dims = [d for _, d in spec.PROPRIO_TERMS]
    T = spec.STUDENT_HISTORY
    frames = [torch.randn(4, spec.PROPRIO_DIM) for _ in range(T)]
    obs = _isaac_history(frames, dims)
    enc = networks.ProprioHistoryEncoder(dims, T)
    rebuilt = enc.frames(obs)
    assert torch.equal(rebuilt[:, -1], frames[-1]) and torch.equal(rebuilt[:, 0], frames[0])
    student = networks.VigorStudent(dims, T, spec.NUM_ACTIONS)
    act, z = student(obs)
    assert act.shape == (4, spec.NUM_ACTIONS) and z.shape == (4, spec.LATENT_DIM)
    wrapped = networks.NormalizedActor(student, torch.zeros(obs.shape[1]), torch.ones(obs.shape[1])).eval()
    traced = torch.jit.trace(wrapped, obs[:1])
    assert torch.allclose(traced(obs), act, atol=1e-5)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "student.onnx")
        torch.onnx.export(wrapped, obs[:1], path, input_names=["obs"], output_names=["actions"],
                          dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17, dynamo=False)
        if _ort is not None:
            out = _ort.InferenceSession(path).run(None, {"obs": obs.numpy()})[0]
            assert np.allclose(out, act.detach().numpy(), atol=1e-4)




def test_export_student():
    sys.path.insert(0, str(REPO / "scripts/vigor"))
    import json as _json

    from common import export_student, student_layout_stats

    dims = [d for _, d in spec.PROPRIO_TERMS]
    T = spec.STUDENT_HISTORY
    student = networks.VigorStudent(dims, T, spec.NUM_ACTIONS)
    mean, std = student_layout_stats(torch.randn(spec.PROPRIO_DIM), torch.rand(spec.PROPRIO_DIM) + 0.5, dims, T)
    assert mean.numel() == T * spec.PROPRIO_DIM
    with tempfile.TemporaryDirectory() as tmp:
        ck = os.path.join(tmp, "model_7.pt")
        torch.save({  # same keys distill.py writes
            "student_state_dict": student.state_dict(), "obs_mean": mean, "obs_std": std, "iteration": 7,
            "distill_cfg": {"latent_dim": spec.LATENT_DIM, "actor_hidden_dims": [512, 256, 128],
                            "fusion_hidden_dims": [256], "activation": "elu"},
            "teacher_checkpoint": "teacher.pt", "spec_version": spec.SPEC_VERSION,
            "deploy": {"joint_names": spec.ALL_JOINT_NAMES, "action_joint_names": spec.ACTION_JOINT_NAMES,
                       "default_joint_pos": [0.0] * 22, "action_scale": [0.2] * 20, "stiffness": [80.0] * 22,
                       "damping": [2.0] * 22, "policy_dt": 0.02, "history": T,
                       "term_names": [n for n, _ in spec.PROPRIO_TERMS], "term_dims": dims, "activation_tilt_deg": 20.0},
        }, ck)
        out = export_student(ck)
        x = torch.randn(3, T * spec.PROPRIO_DIM)
        ref = student((x - mean) / std)[0]
        assert torch.allclose(torch.jit.load(out["pt"])(x), ref, atol=1e-5)
        if _ort is not None:
            onnx_out = _ort.InferenceSession(out["onnx"]).run(None, {"obs": x.numpy()})[0]
            assert np.allclose(onnx_out, ref.detach().numpy(), atol=1e-4)
        s = _json.load(open(out["spec"]))
        assert s["obs_dim"] == T * spec.PROPRIO_DIM and s["spec_version"] == spec.SPEC_VERSION


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
