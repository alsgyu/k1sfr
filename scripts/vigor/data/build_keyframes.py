"""Build the VIGOR keyframe dataset from GMR-retargeted LAFAN1 clips (laptop side, no Isaac).

Steps per source clip:
  1. resample 30 -> 50 fps, map joints to spec.ALL_JOINT_NAMES, zero the (held) head,
     clamp to the Isaac soft joint limits (factor 0.9 as in BOOSTER_K1_CFG);
  2. ground every frame: shift the root so the lowest collision point touches z = 0;
  3. cut stand -> fall -> lowest-pose segments and continue each with the K1 front or
     back get-up (k1_getup.py; human get-ups do not fit the K1), then the walking stance;
  4. add synthetic forward brace-and-rise clips: fold into the hands-and-feet pike of the
     K1 front get-up and stand up from it without lying down;
  5. keep keyframes every ``stride`` frames and rebuild a *coarse* trajectory by
     interpolating between them (paper: sparse ~5 Hz structural priors);
  6. forward kinematics on the coarse trajectory -> keypoint poses, velocities and the
     reference ground contacts of hands / head / trunk.

Writes ``<repo>/data/vigor/<name>.npz`` and records its sha256 in the git-tracked
``vigor/data_manifest.json`` so the GPU machine can verify it received the same file.

    python scripts/vigor/data/build_keyframes.py --retarget_dir ~/vigor_work/retarget \
        --mesh_dir ~/vigor_work/GMR/assets/booster_k1/meshes
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import importlib.util
import json
import os
import math
import pathlib
import struct
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

import k1_getup
from urdf_fk import URDFModel, matrix_to_quat, quat_to_matrix

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
VIGOR_DIR = REPO / "source/JY_train/JY_train/tasks/manager_based/vigor"
URDF = REPO / "source/JY_train/JY_train/assets/robots/urdf/K1_22dof_jy.urdf"
SOFT_LIMIT_FACTOR = 0.9  # BOOSTER_K1_CFG.soft_joint_pos_limit_factor


def load_spec():
    s = importlib.util.spec_from_file_location("vigor_spec", VIGOR_DIR / "spec.py")
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


spec = load_spec()


# ----------------------------------------------------------------- geometry
def _stl_points(path: str) -> np.ndarray:
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    if 84 + n * 50 == len(data):
        tri = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
        return tri["v"].reshape(-1, 3).astype(np.float64)
    pts = [l.split()[1:] for l in data.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")]
    return np.asarray(pts, dtype=np.float64)


def _corners(lo, hi) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def collision_points(urdf_path: str, mesh_dir: str) -> dict[str, np.ndarray]:
    """Bounding-box corners of every collision shape, in its link frame."""
    from urdf_fk import rpy_to_matrix

    out: dict[str, list] = {}
    for link in ET.parse(urdf_path).getroot().findall("link"):
        pts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rot = rpy_to_matrix([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
            g = col.find("geometry")[0]
            if g.tag == "box":
                h = np.array([float(v) for v in g.get("size").split()]) / 2
                local = _corners(-h, h)
            elif g.tag == "cylinder":
                r, l = float(g.get("radius")), float(g.get("length"))
                local = _corners([-r, -r, -l / 2], [r, r, l / 2])
            elif g.tag == "sphere":
                r = float(g.get("radius"))
                local = _corners([-r] * 3, [r] * 3)
            elif g.tag == "mesh":
                p = _stl_points(os.path.join(mesh_dir, os.path.basename(g.get("filename"))))
                local = _corners(p.min(0), p.max(0))
            else:
                continue
            pts.append(local @ rot.T + xyz)
        if pts:
            out[link.get("name")] = np.concatenate(pts)
    return out


# --------------------------------------------------------------- utilities
def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(max=1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    small = s < 1e-6
    w0 = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, s))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1, s))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample(t_src: np.ndarray, t_dst: np.ndarray, pos=None, quat=None, lin=None):
    idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(t_src) - 2)
    a = ((t_dst - t_src[idx]) / (t_src[idx + 1] - t_src[idx])).clip(0, 1)[:, None]
    out = []
    for arr in (pos, lin):
        if arr is not None:
            out.append(arr[idx] * (1 - a) + arr[idx + 1] * a)
    if quat is not None:
        out.append(slerp(quat[idx], quat[idx + 1], a))
    return out


def quat_conj(q):
    return q * np.array([1, -1, -1, -1])


def quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def angular_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a (T, ..., 4) quaternion sequence (central differences)."""
    qn = np.concatenate([q[1:], q[-1:]])
    qp = np.concatenate([q[:1], q[:-1]])
    dq = quat_mul(qn, quat_conj(qp))
    dq = dq * np.where(dq[..., :1] < 0, -1, 1)
    angle = 2 * np.arctan2(np.linalg.norm(dq[..., 1:], axis=-1), dq[..., 0])
    axis = dq[..., 1:] / np.maximum(np.linalg.norm(dq[..., 1:], axis=-1, keepdims=True), 1e-9)
    span = np.full(q.shape[0], 2.0)
    span[0] = span[-1] = 1.0
    return axis * (angle / (span.reshape((-1,) + (1,) * (angle.ndim - 1)) * dt))[..., None]


def finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(x, dt, axis=0)


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def trunk_up(q: np.ndarray) -> np.ndarray:
    return 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)


# ------------------------------------------------------------ segmentation
def segment(root_z, up, fps, stand_z, args):
    zs = smooth(root_z, 5)
    standing = (zs > args.stand_ratio * stand_z) & (up > 0.85)
    low = zs < args.fall_z
    segs, i, n = [], 0, len(zs)
    while i < n:
        if not low[i]:
            i += 1
            continue
        lo_start = i
        while i < n and low[i]:
            i += 1
        lo_end = i
        onset = lo_start
        while onset > 0 and not standing[onset]:
            onset -= 1
        if not standing[onset]:
            continue
        end = lo_end
        stable = int(0.3 * fps)
        while end < n - stable and not standing[end : end + stable].all():
            end += 1
        if end >= n - stable:
            break
        lowest = lo_start + int(np.argmin(zs[lo_start:lo_end]))
        dur = (end - onset) / fps
        if args.min_duration <= dur <= args.max_duration and (not segs or onset > segs[-1][2]):
            segs.append((onset, lowest, end, min(n - 1, end + int(args.post_roll * fps))))
        i = end
    return segs


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retarget_dir", required=True)
    p.add_argument("--mesh_dir", required=True, help="K1 meshes (GMR assets/booster_k1/meshes or booster_assets)")
    p.add_argument("--name", default="k1_fall_keyframes_v3.npz")
    p.add_argument("--out_dir", default=str(REPO / "data" / "vigor"))
    p.add_argument("--fall_z", type=float, default=0.40, help="trunk height that counts as fallen")
    p.add_argument("--stand_ratio", type=float, default=0.88)
    p.add_argument("--pre_roll", type=float, default=0.6)
    p.add_argument("--post_roll", type=float, default=1.2)
    p.add_argument("--min_duration", type=float, default=1.5)
    p.add_argument("--max_duration", type=float, default=7.0)
    p.add_argument("--max_joint_vel", type=float, default=25.0, help="reject segments with retarget glitches")
    p.add_argument("--stand_hold", type=float, default=1.5, help="walking stance held after each get-up (s)")
    p.add_argument("--brace_hips", default="-1.8,-2.0,-2.2,-2.4", help="hip pitch of the synthetic brace pikes")
    p.add_argument("--lafan1_zip_sha256", default=None, help="provenance only: sha256 of the LAFAN1 archive used")
    args = p.parse_args()

    model = URDFModel(str(URDF))
    contact = collision_points(str(URDF), os.path.expanduser(args.mesh_dir))
    fps, stride = spec.KEYFRAME_FPS, spec.KEYFRAME_STRIDE
    dt = 1.0 / fps
    names = spec.ALL_JOINT_NAMES
    lo, hi = model.limits(names)
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_LIMIT_FACTOR
    soft_lo, soft_hi = mid - half, mid + half
    head_ids = [names.index(n) for n in spec.HEAD_JOINT_NAMES]
    arm_ids = [names.index(n) for n in spec.ARM_JOINT_NAMES]
    body_names = spec.KEY_BODY_NAMES
    feet, hands, head = ["left_foot_link", "right_foot_link"], ["left_hand_link", "right_hand_link"], spec.HEAD_BODY_NAMES
    stance_q = np.clip(k1_getup.stance(names), soft_lo, soft_hi)  # elbow yaw 0 sits on the hard limit
    hold = stride * round(args.stand_hold * fps / stride)

    def fk_bodies(root_pos, root_quat, q):
        return model.fk(root_pos, root_quat, q, names)

    def link_low(fk, links: list[str]) -> np.ndarray:
        """(T,) lowest collision point over ``links``."""
        zmin = None
        for link in links:
            pos, rot = fk[link]
            z = (np.einsum("tij,pj->tpi", rot, contact[link])[..., 2] + pos[:, None, 2]).min(axis=1)
            zmin = z if zmin is None else np.minimum(zmin, z)
        return zmin

    def lowest_point(fk):
        return link_low(fk, list(contact))

    def touches(fk, links: list[str], tol: float = 0.025) -> np.ndarray:
        return link_low(fk, links) - lowest_point(fk) < tol

    def clearance(q, quat) -> dict[str, float]:
        fk = fk_bodies(np.zeros((1, 3)), quat[None], q[None])
        g = lowest_point(fk)[0]
        return {link: link_low(fk, [link])[0] - g for link in contact}

    def solve_pitch(q_of, roll, pitch0, groups, span=0.8):
        """Trunk pitch that puts every link group on the ground at once (min-max clearance)."""
        best = None
        for pitch in pitch0 + np.linspace(-span, span, 161):
            q = q_of(pitch)
            c = clearance(q, k1_getup.rpy_quat(roll, pitch))
            cost = max(min(c[l] for l in g) for g in groups)
            if best is None or cost < best[0]:
                best = (cost, pitch, q)
        return best

    def ground(q, quat, anchor_xy):
        """Root trajectory with the lowest point on z = 0 and the feet midpoint fixed at ``anchor_xy``."""
        rp = np.zeros((len(q), 3))
        fk = fk_bodies(rp, quat, q)
        rp[:, 2] = -lowest_point(fk)
        rp[:, :2] = anchor_xy - (fk[feet[0]][0][:, :2] + fk[feet[1]][0][:, :2]) / 2
        return rp

    def densify(keys):
        """[(frame, q, quat), ...] -> per-frame q, quat from the first to the last key frame."""
        f = np.array([k[0] for k in keys], dtype=np.float64)
        q, quat = resample(f, np.arange(f[0], f[-1] + 1), pos=np.stack([k[1] for k in keys]), quat=np.stack([k[2] for k in keys]))
        return q, quat

    # --- K1 get-ups (frames after keyframe 0, q, roll, pitch); yaw is applied per clip
    getups, getup_end = {}, {}
    for kind, m in k1_getup.GETUPS.items():
        poses, f = [], 0
        for k, (q_src, (roll, pitch), dur) in enumerate(zip(m.joint_pos, m.torso_roll_pitch, m.duration)):
            q = np.clip(k1_getup.canonical(q_src, names), soft_lo, soft_hi)
            if k in k1_getup.HEAD_SUPPORT[kind]:
                pitch = solve_pitch(lambda _: q, roll, pitch, [head, feet[:1], feet[1:]])[1]
            f += k1_getup.quantize(dur, fps, stride) if k else 0
            poses.append((f, q, roll, pitch))
            if k == 0:  # settle before pushing up
                f += stride
                poses.append((f, q, roll, pitch))
        f += 2 * stride
        poses.append((f, stance_q, 0.0, 0.0))
        getups[kind], getup_end[kind] = poses, poses[-2][0]  # keyframe 6: standing
        print(f"K1 {kind} get-up: {f / fps:.1f} s to the walking stance, pitches "
              f"{[round(float(x[3]), 2) for x in poses]}")

    out = {k: [] for k in ("joint_pos", "joint_vel", "root_pos", "root_quat", "body_pos", "body_quat", "body_lin_vel",
                           "body_ang_vel", "ref_contact")}
    meta = {k: [] for k in ("clip_names", "clip_start", "clip_length", "fall_onset", "lowest", "getup_end", "fall_dir",
                            "clip_hand_support")}
    rejected, cursor = [], 0

    def emit(name, qs, rp, rq, onset, lowest, getup, hand_support) -> bool:
        nonlocal cursor
        dq = np.abs(finite_diff(qs, dt)).max()
        if dq > args.max_joint_vel:
            rejected.append((name, f"joint vel {dq:.0f} rad/s"))
            return False
        n = len(qs)
        # sparse keyframes -> coarse reference (keeps the final frame as a keyframe too)
        key_idx = np.unique(np.append(np.arange(0, n, stride), n - 1))
        tk, tf = key_idx * dt, np.arange(n) * dt
        cq, crp, crq = resample(tk, tf, pos=qs[key_idx], lin=rp[key_idx], quat=rq[key_idx])
        # flat ground: every coarse pose touches z = 0 (interpolation can dip below or float above it)
        crp[:, 2] -= lowest_point(fk_bodies(crp, crq, cq))
        fk = fk_bodies(crp, crq, cq)
        bpos = np.stack([fk[b][0] for b in body_names], axis=1)
        bquat = matrix_to_quat(np.stack([fk[b][1] for b in body_names], axis=1))
        for i in range(1, n):  # sign continuity for finite differences
            flip = np.sum(bquat[i] * bquat[i - 1], axis=-1) < 0
            bquat[i, flip] *= -1
        # fall direction: horizontal trunk-axis lean when it first passes 45 deg, in the onset heading frame
        ups = trunk_up(crq)
        tipped = ups[onset : lowest + 1] < np.cos(np.radians(45))
        j = onset + int(np.argmax(tipped)) if tipped.any() else lowest
        zaxis = quat_to_matrix(crq[j])[:, 2]
        w, x, y, z = crq[onset]
        yaw0 = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        lean = np.arctan2(zaxis[1], zaxis[0]) - yaw0
        out["joint_pos"].append(cq)
        out["joint_vel"].append(finite_diff(cq, dt))
        out["root_pos"].append(crp)
        out["root_quat"].append(crq)
        out["body_pos"].append(bpos)
        out["body_quat"].append(bquat)
        out["body_lin_vel"].append(finite_diff(bpos, dt))
        out["body_ang_vel"].append(angular_velocity(bquat, dt))
        out["ref_contact"].append(np.stack([touches(fk, links, tol) for links, tol in zip(spec.REF_CONTACT_BODIES, spec.REF_CONTACT_TOL)], axis=-1))
        meta["clip_names"].append(name)
        meta["clip_start"].append(cursor)
        meta["clip_length"].append(n)
        meta["fall_onset"].append(onset)
        meta["lowest"].append(lowest)
        meta["getup_end"].append(getup)
        meta["fall_dir"].append(float(np.arctan2(np.sin(lean), np.cos(lean))))
        meta["clip_hand_support"].append(hand_support)
        cursor += n
        return True

    # --- LAFAN1 falls, cut at the lowest pose and continued with the closer K1 get-up
    yaws = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    kinds = {}
    sources = sorted(glob.glob(os.path.join(os.path.expanduser(args.retarget_dir), "*.npz")))
    for path in sources:
        src = np.load(path)
        src_names = [str(n) for n in src["joint_names"]]
        q_src = src["joint_pos"][:, [src_names.index(n) for n in names]]
        t_src = np.arange(len(q_src)) / float(src["fps"])
        t = np.arange(0.0, t_src[-1], dt)
        q, root_pos, root_quat = resample(t_src, t, pos=q_src, lin=src["root_pos"], quat=src["root_quat"])
        q[:, head_ids] = 0.0
        q = np.clip(q, soft_lo, soft_hi)
        root_pos = root_pos.copy()
        root_pos[:, 2] -= smooth(lowest_point(fk_bodies(root_pos, root_quat, q)), 5)

        clip = pathlib.Path(path).stem
        for k, (onset, lowest, _getup, _end) in enumerate(segment(root_pos[:, 2], trunk_up(root_quat), fps, spec.TRUNK_HEIGHT_NOMINAL, args)):
            start = max(0, onset - int(args.pre_roll * fps))
            j = lowest - start
            qs, rp, rq = q[start : lowest + 1], root_pos[start : lowest + 1], root_quat[start : lowest + 1]
            # closest keyframe-0 orientation (prone or supine) over all headings
            best = None
            for kind, poses in getups.items():
                cand = np.stack([k1_getup.rpy_quat(poses[0][2], poses[0][3], yw) for yw in yaws])
                ang = 2 * np.arccos(np.abs(cand @ rq[-1]).clip(max=1.0))
                i = int(np.argmin(ang))
                if best is None or ang[i] < best[2]:
                    best = (kind, yaws[i], ang[i])
            kind, yaw, ang = best
            e = stride * int(np.ceil((j + fps * (0.3 + 0.5 * ang / np.pi)) / stride))
            keys = [(j, qs[-1], rq[-1])] + [(e + f, qk, k1_getup.rpy_quat(r, pt, yaw)) for f, qk, r, pt in getups[kind]]
            keys.append((keys[-1][0] + hold, stance_q, keys[-1][2]))
            tq, tquat = densify(keys)
            fk_j = fk_bodies(rp[-1:], rq[-1:], qs[-1:])
            anchor = (fk_j[feet[0]][0][0, :2] + fk_j[feet[1]][0][0, :2]) / 2
            trp = ground(tq, tquat, anchor)
            if emit(f"{clip}_s{k}_{kind}", np.concatenate([qs[:-1], tq]), np.concatenate([rp[:-1], trp]),
                    np.concatenate([rq[:-1], tquat]), onset - start, e, e + getup_end[kind], hand_support=False):
                kinds[kind] = kinds.get(kind, 0) + 1

    # --- forward push: fold into the hands-and-feet pike (front keyframe 3) and rise without lying down
    q3 = np.clip(k1_getup.canonical(k1_getup.GETUPS["front"].joint_pos[k1_getup.BRACE_KEYFRAME], names), soft_lo, soft_hi)
    sym = k1_getup.mirrored(0.5 * (q3 + k1_getup.mirrored(q3, names, "Right")), names, "Left")  # averaged legs
    sym[arm_ids] = k1_getup.mirrored(q3, names, "Right")[arm_ids]  # the source's right hand is the one on the ground
    knee = [names.index(f"{s}_Knee_Pitch") for s in ("Left", "Right")]
    hip = [names.index(f"{s}_Hip_Pitch") for s in ("Left", "Right")]
    ankle = [names.index(f"{s}_Ankle_Pitch") for s in ("Left", "Right")]
    rise = getups["front"][k1_getup.BRACE_KEYFRAME + 2 :]  # front keyframes 4..6 and the stance (+1: settle key)
    onset_pitch = math.radians(spec.ACTIVATION_TILT_DEG) * 0.6
    n_brace = 0
    for hip_pitch in [float(v) for v in args.brace_hips.split(",")]:
        def brace_q(pitch, hp=hip_pitch):
            qb = sym.copy()
            qb[hip] = hp
            qb[ankle] = -(pitch + hp + qb[knee])  # soles flat on the ground
            return np.clip(qb, soft_lo, soft_hi)

        cost, pitch_b, qb = solve_pitch(brace_q, 0.0, 2.0, [[h] for h in hands] + [[f] for f in feet])
        c = clearance(qb, k1_getup.rpy_quat(0.0, pitch_b))
        print(f"brace hip {hip_pitch:+.1f}: trunk pitch {pitch_b:.2f}, hands/feet off the ground by <= {cost * 100:.1f} cm, "
              f"head {min(c[h] for h in head) * 100:.0f} cm, trunk {c['Trunk'] * 100:.0f} cm")
        for reach in (2, 3):  # strides from the onset to the brace: 0.4 s / 0.6 s
            for brace_hold in (stride, 2 * stride):  # hands planted for 0.2 / 0.4 s
                keys = [(0, stance_q, k1_getup.rpy_quat(0.0, 0.0)), (stride, stance_q, k1_getup.rpy_quat(0.0, onset_pitch))]
                for i in range(1, reach):
                    a = i / reach
                    qr = stance_q + a * (qb - stance_q)
                    qr[arm_ids] = stance_q[arm_ids] + min(1.0, 1.6 * a) * (qb - stance_q)[arm_ids]  # the arms lead
                    keys.append((stride * (1 + i), qr, k1_getup.rpy_quat(0.0, onset_pitch + a * (pitch_b - onset_pitch))))
                fb = stride * (1 + reach)
                keys.append((fb, qb, k1_getup.rpy_quat(0.0, pitch_b)))
                keys.append((fb + brace_hold, qb, k1_getup.rpy_quat(0.0, pitch_b)))
                f0 = keys[-1][0] - getups["front"][k1_getup.BRACE_KEYFRAME + 1][0]  # brace pose takes keyframe 3's slot
                keys += [(f0 + f, qk, k1_getup.rpy_quat(r, pt)) for f, qk, r, pt in rise]
                keys.append((keys[-1][0] + hold, stance_q, keys[-1][2]))
                tq, tquat = densify(keys)
                trp = ground(tq, tquat, np.zeros(2))
                fk = fk_bodies(trp, tquat, tq)
                pre = slice(0, fb - stride // 2)
                if touches(fk, hands)[pre].any() or not touches(fk, feet)[pre].all():
                    rejected.append((f"brace hip {hip_pitch} reach {reach}", "hands touch before the brace"))
                    continue
                n_brace += emit(f"k1_brace_hip{-hip_pitch * 10:.0f}_r{reach}_h{brace_hold}", tq, trp, tquat, stride, fb,
                                f0 + rise[-2][0], hand_support=True)

    arrays = {k: np.concatenate(v).astype(np.float32) for k, v in out.items() if k != "ref_contact"}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, args.name)
    np.savez_compressed(
        path,
        spec_version=np.array(spec.SPEC_VERSION),
        fps=np.array(fps),
        stride=np.array(stride),
        joint_names=np.array(names),
        body_names=np.array(body_names),
        contact_names=np.array(spec.REF_CONTACT_NAMES),
        ref_contact=np.concatenate(out["ref_contact"]),
        clip_names=np.array(meta["clip_names"]),
        clip_start=np.array(meta["clip_start"], dtype=np.int64),
        clip_length=np.array(meta["clip_length"], dtype=np.int64),
        fall_onset=np.array(meta["fall_onset"], dtype=np.int64),
        lowest=np.array(meta["lowest"], dtype=np.int64),
        getup_end=np.array(meta["getup_end"], dtype=np.int64),
        fall_dir=np.array(meta["fall_dir"], dtype=np.float32),
        clip_hand_support=np.array(meta["clip_hand_support"], dtype=bool),
        **arrays,
    )
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    lengths = np.array(meta["clip_length"]) / fps
    print(f"wrote {path}: {len(lengths)} clips (LAFAN1 falls + K1 get-up {kinds}, {n_brace} brace-and-rise), "
          f"{cursor} frames, {lengths.min():.1f}-{lengths.max():.1f}s, sha256 {digest[:12]}")
    for name, why in rejected:
        print(f"  rejected {name}: {why}")

    info_path = os.path.join(os.path.expanduser(args.retarget_dir), "retarget_info.json")
    info = json.load(open(info_path)) if os.path.isfile(info_path) else {}
    manifest_path = VIGOR_DIR / "data_manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {"datasets": {}}
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    manifest["datasets"][args.name] = {
        "sha256": digest,
        "size_bytes": os.path.getsize(path),
        "spec_version": spec.SPEC_VERSION,
        "segments": len(lengths),
        "arm_caught_segments": int(sum(meta["clip_hand_support"])),
        "frames": int(cursor),
        "fps": fps,
        "keyframe_stride": stride,
        "source": "Ubisoft LAFAN1 (CC BY-NC-ND 4.0) fallAndGetUp*/pushAndFall* falls retargeted with GMR, each continued with the K1 front/back get-up; plus synthetic forward brace-and-rise clips",
        "k1_getup_source": k1_getup.SOURCE,
        "getup_clips": kinds,
        "ik_config_sha256": info.get("ik_config_sha256"),
        "gmr_commit": info.get("gmr_commit"),
        "lafan1_zip_sha256": args.lafan1_zip_sha256,
        "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
        "build_args": vars(args) | {"retarget_dir": "<local>", "mesh_dir": "<local>", "out_dir": "<repo>/data/vigor"},
        "built_at_commit": commit,
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"updated {manifest_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
