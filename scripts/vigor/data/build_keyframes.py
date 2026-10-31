"""Build the VIGOR keyframe dataset from GMR-retargeted LAFAN1 clips (laptop side, no Isaac).

Steps per source clip:
  1. resample 30 -> 50 fps, map joints to spec.ALL_JOINT_NAMES, zero the (held) head,
     clamp to the Isaac soft joint limits (factor 0.9 as in BOOSTER_K1_CFG);
  2. ground every frame: shift the root so the lowest collision point touches z = 0;
  3. cut stand -> fall -> lie -> get-up -> stand segments;
  4. keep keyframes every ``stride`` frames and rebuild a *coarse* trajectory by
     interpolating between them (paper: sparse ~5 Hz structural priors);
  5. forward kinematics on the coarse trajectory -> keypoint poses and velocities.

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
import pathlib
import struct
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

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
    p.add_argument("--name", default="k1_fall_keyframes_v1.npz")
    p.add_argument("--out_dir", default=str(REPO / "data" / "vigor"))
    p.add_argument("--fall_z", type=float, default=0.25, help="trunk height that counts as fallen")
    p.add_argument("--stand_ratio", type=float, default=0.88)
    p.add_argument("--pre_roll", type=float, default=0.6)
    p.add_argument("--post_roll", type=float, default=1.2)
    p.add_argument("--min_duration", type=float, default=1.5)
    p.add_argument("--max_duration", type=float, default=7.0)
    p.add_argument("--max_joint_vel", type=float, default=25.0, help="reject segments with retarget glitches")
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
    body_names = spec.KEY_BODY_NAMES

    def fk_bodies(root_pos, root_quat, q):
        return model.fk(root_pos, root_quat, q, names)

    def touches(fk, idx: int, links: list[str], tol: float = 0.03) -> bool:
        for link in links:
            if link in contact:
                pos, rot = fk[link]
                if ((rot[idx] @ contact[link].T)[2] + pos[idx, 2]).min() < tol:
                    return True
        return False

    def lowest_point(fk):
        zmin = None
        for link, pts in contact.items():
            pos, rot = fk[link]
            z = (np.einsum("tij,pj->tpi", rot, pts)[..., 2] + pos[:, None, 2]).min(axis=1)
            zmin = z if zmin is None else np.minimum(zmin, z)
        return zmin

    # nominal standing trunk height (walking stance) for segmentation thresholds
    stand_z = spec.TRUNK_HEIGHT_NOMINAL

    out = {k: [] for k in ("joint_pos", "joint_vel", "root_pos", "root_quat", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel")}
    meta = {k: [] for k in ("clip_names", "clip_start", "clip_length", "fall_onset", "lowest", "getup_end", "fall_dir",
                            "clip_hand_support")}
    rejected, cursor = [], 0
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
        for k, (onset, lowest, getup, end) in enumerate(segment(root_pos[:, 2], trunk_up(root_quat), fps, stand_z, args)):
            start = max(0, onset - int(args.pre_roll * fps))
            sl = slice(start, end + 1)
            qs, rp, rq = q[sl], root_pos[sl], root_quat[sl]
            dq = np.abs(finite_diff(qs, dt)).max()
            if dq > args.max_joint_vel:
                rejected.append((f"{clip}_s{k}", f"joint vel {dq:.0f} rad/s"))
                continue
            n = len(qs)
            # sparse keyframes -> coarse reference (keeps the final frame as a keyframe too)
            key_idx = np.unique(np.append(np.arange(0, n, stride), n - 1))
            tk, tf = key_idx * dt, np.arange(n) * dt
            cq, crp, crq = resample(tk, tf, pos=qs[key_idx], lin=rp[key_idx], quat=rq[key_idx])
            fk = fk_bodies(crp, crq, cq)
            # re-ground the coarse poses (interpolation can dip below the floor)
            crp[:, 2] -= np.minimum(lowest_point(fk), 0.0)
            fk = fk_bodies(crp, crq, cq)
            bpos = np.stack([fk[b][0] for b in body_names], axis=1)
            bquat = matrix_to_quat(np.stack([fk[b][1] for b in body_names], axis=1))
            for i in range(1, n):  # sign continuity for finite differences
                flip = np.sum(bquat[i] * bquat[i - 1], axis=-1) < 0
                bquat[i, flip] *= -1
            # fall direction: horizontal trunk-axis lean when it first passes 45 deg, in the onset heading frame
            o, lw = onset - start, lowest - start
            ups = trunk_up(crq)
            j = o + int(np.argmax(ups[o : lw + 1] < np.cos(np.radians(45)))) if (ups[o : lw + 1] < np.cos(np.radians(45))).any() else lw
            zaxis = quat_to_matrix(crq[j])[:, 2]
            w, x, y, z = crq[o]
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
            meta["clip_names"].append(f"{clip}_s{k}")
            meta["clip_start"].append(cursor)
            meta["clip_length"].append(n)
            meta["fall_onset"].append(o)
            meta["lowest"].append(lw)
            meta["getup_end"].append(getup - start)
            meta["fall_dir"].append(float(np.arctan2(np.sin(lean), np.cos(lean))))
            # arm-caught fall: hands on the ground at the lowest pose, head and trunk off it
            meta["clip_hand_support"].append(
                touches(fk, lw, ["left_hand_link", "right_hand_link"]) and not touches(fk, lw, ["Trunk", "Head_1", "Head_2"])
            )
            cursor += n

    arrays = {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, args.name)
    np.savez_compressed(
        path,
        spec_version=np.array(spec.SPEC_VERSION),
        fps=np.array(fps),
        stride=np.array(stride),
        joint_names=np.array(names),
        body_names=np.array(body_names),
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
    print(f"wrote {path}: {len(lengths)} segments ({sum(meta['clip_hand_support'])} arm-caught), {cursor} frames, "
          f"{lengths.min():.1f}-{lengths.max():.1f}s, sha256 {digest[:12]}")
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
        "source": "Ubisoft LAFAN1 (CC BY-NC-ND 4.0) fallAndGetUp*/pushAndFall*, retargeted with GMR",
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
