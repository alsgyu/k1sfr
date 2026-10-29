"""Check that this machine is in sync with the VIGOR code, data and robot assets (no Isaac needed).

Run it on the GPU machine after every `git pull` / data sync and paste the summary back:

    python scripts/vigor/check_env.py                       # report only
    python scripts/vigor/check_env.py --vendor-walk-urdf    # also copy K1_jy_locomotion.urdf into the repo

Writes logs/vigor/env_report_<host>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import math
import os
import pathlib
import platform
import shutil
import socket
import subprocess
import xml.etree.ElementTree as ET

REPO = pathlib.Path(__file__).resolve().parents[2]
VIGOR = REPO / "source/JY_train/JY_train/tasks/manager_based/vigor"
URDF_DIR = REPO / "source/JY_train/JY_train/assets/robots/urdf"
FULL_URDF = URDF_DIR / "K1_22dof_jy.urdf"
WALK_URDF_NAME = "K1_jy_locomotion.urdf"

results: list[tuple[str, str, str]] = []


def report(name: str, status: str, detail: str = ""):
    results.append((name, status, detail))
    print(f"[{status:4s}] {name}: {detail}")


def git(*args, cwd=REPO) -> str:
    try:
        return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_blob_sha1(path) -> str:
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def load_spec():
    s = importlib.util.spec_from_file_location("vigor_spec", VIGOR / "spec.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def joints(urdf_path) -> dict:
    out = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        o = j.find("origin")
        ax = j.find("axis")
        out[j.get("name")] = {
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()],
            "rpy": [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()],
            "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else [1.0, 0.0, 0.0],
        }
    return out


def welded_angle(rpy, axis) -> tuple[float, bool]:
    """Angle of a welded joint's origin rotation about the revolute axis, and whether it is purely about that axis."""
    m = rpy_matrix(*rpy)
    tr = max(-1.0, min(1.0, (m[0][0] + m[1][1] + m[2][2] - 1) / 2))
    ang = math.acos(tr)
    if ang < 1e-6:
        return 0.0, True
    v = [m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1]]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    v = [x / n for x in v]
    dot = sum(a * b for a, b in zip(v, axis))
    return ang * (1 if dot >= 0 else -1), abs(abs(dot) - 1) < 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor-walk-urdf", action="store_true", help=f"copy {WALK_URDF_NAME} from booster_assets into the repo")
    args = ap.parse_args()
    spec = load_spec()
    info: dict = {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version()}

    # ---- code
    branch, head = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "HEAD")
    dirty = [l for l in git("status", "--porcelain").splitlines() if l.strip()]
    ahead_behind = git("rev-list", "--left-right", "--count", "@{u}...HEAD")
    info["git"] = {"branch": branch, "head": head, "dirty": dirty, "behind_ahead_upstream": ahead_behind}
    report("git", "OK" if not dirty else "WARN", f"{branch} @ {head[:10]}" + (f", {len(dirty)} uncommitted changes" if dirty else "")
           + (f", behind/ahead upstream: {ahead_behind}" if ahead_behind else ""))
    report("spec", "OK", spec.SPEC_VERSION)
    info["spec_version"] = spec.SPEC_VERSION

    # ---- packages
    pkgs = {}
    for p in ("isaacsim", "isaaclab", "isaaclab_rl", "isaaclab_tasks", "rsl-rl-lib", "torch", "gymnasium", "numpy",
              "onnx", "onnxruntime", "tensorboard", "booster_assets", "JY_train"):
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    info["packages"] = pkgs
    rsl = pkgs["rsl-rl-lib"]
    report("rsl-rl-lib", "OK" if rsl and rsl.startswith("2.3.") else "FAIL", f"{rsl} (VIGOR teacher targets 2.3.x)")
    report("isaac", "OK" if pkgs["isaaclab"] else "WARN", f"isaacsim {pkgs['isaacsim']}, isaaclab {pkgs['isaaclab']}, isaaclab_rl {pkgs['isaaclab_rl']}")
    jy = importlib.util.find_spec("JY_train")
    jy_path = pathlib.Path(jy.origin).resolve() if jy and jy.origin else None
    in_repo = jy_path is not None and REPO in jy_path.parents
    report("JY_train install", "OK" if in_repo else "FAIL", f"{jy_path} " + ("(this repo)" if in_repo else "(NOT this checkout: pip install -e source/JY_train)"))
    try:
        import torch

        cuda = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["cuda"] = {"available": torch.cuda.is_available(), "devices": cuda, "torch_cuda": torch.version.cuda}
        report("cuda", "OK" if cuda else "WARN", ", ".join(cuda) or "no CUDA device")
    except Exception as e:
        report("torch", "FAIL", str(e))

    # ---- keyframe data
    manifest = json.load(open(VIGOR / "data_manifest.json"))
    data_dir = pathlib.Path(os.environ.get("JY_VIGOR_DATA_DIR", REPO / "data" / "vigor"))
    info["datasets"] = {}
    for name, entry in manifest["datasets"].items():
        path = data_dir / name
        if not path.is_file():
            report(f"data {name}", "FAIL", f"missing at {path} (scripts/vigor/sync.sh push-data from the laptop)")
            continue
        digest = sha256(path)
        ok = digest == entry["sha256"] and entry["spec_version"] == spec.SPEC_VERSION
        info["datasets"][name] = digest
        report(f"data {name}", "OK" if ok else "FAIL", f"sha256 {digest[:12]} (manifest {entry['sha256'][:12]}), {entry['segments']} segments")

    # ---- robot assets
    try:
        import booster_assets

        assets = pathlib.Path(booster_assets.BOOSTER_ASSETS_DIR)
    except Exception as e:
        assets = None
        report("booster_assets", "FAIL", f"cannot import: {e}")
    if assets is not None:
        commit = git("rev-parse", "HEAD", cwd=assets) or git("rev-parse", "HEAD", cwd=assets.parent)
        info["booster_assets"] = {"dir": str(assets), "commit": commit}
        report("booster_assets", "OK", f"{assets} @ {commit[:10] or 'not a git checkout'}")
        mesh_dir = pathlib.Path(os.environ.get("JY_K1_MESH_DIR", assets / "robots" / "K1" / "meshes"))
        mm = json.load(open(URDF_DIR / "K1_22dof_jy.manifest.json"))["meshes"]
        bad = [n for n, e in mm.items() if not (mesh_dir / n).is_file() or git_blob_sha1(mesh_dir / n) != e["git_blob_sha1"]]
        report("K1 meshes", "OK" if not bad else "WARN", f"{len(mm) - len(bad)}/{len(mm)} match the vendored URDF's source"
               + (f"; differ/missing: {bad}" if bad else ""))

        walk = assets / "robots" / "K1" / WALK_URDF_NAME
        vendored_walk = URDF_DIR / WALK_URDF_NAME
        if walk.is_file():
            if args.vendor_walk_urdf:
                shutil.copy2(walk, vendored_walk)
                report("walk URDF", "OK", f"copied to {vendored_walk.relative_to(REPO)} - commit it so the laptop sees it")
            wj, fj = joints(walk), joints(FULL_URDF)
            info["walk_urdf_sha256"] = sha256(walk)
            # kinematics shared with the 22-DoF model must be identical
            diff = [n for n in fj if n in wj and (wj[n]["xyz"] != fj[n]["xyz"] or wj[n]["parent"] != fj[n]["parent"])]
            missing = [n for n in spec.LEG_JOINT_NAMES if n not in wj or wj[n]["type"] != "revolute"]
            report("walk URDF legs", "OK" if not diff and not missing else "FAIL",
                   "leg kinematics match K1_22dof_jy" if not diff and not missing else f"differ: {diff}, missing legs: {missing}")
            pose = {}
            for n in spec.ARM_JOINT_NAMES + spec.HEAD_JOINT_NAMES:
                if n in wj and wj[n]["type"] == "fixed" and n in fj:
                    ang, pure = welded_angle(wj[n]["rpy"], fj[n]["axis"])
                    pose[n] = round(ang, 4) if pure else f"rpy {wj[n]['rpy']} (not about the joint axis)"
            info["walk_welded_pose"] = pose
            default = {"Left_Shoulder_Roll": -1.3, "Right_Shoulder_Roll": 1.3}
            mismatch = {n: v for n, v in pose.items() if not isinstance(v, float) or abs(v - default.get(n, 0.0)) > 1e-3}
            report("walk arm pose", "OK" if not mismatch else "WARN",
                   "welded arms match K1_FULL_DEFAULT_JOINT_POS" if not mismatch else
                   f"welded pose {pose} differs from VIGOR default arms (shoulder roll -1.3/+1.3); see README 'arm pose'")
        else:
            report("walk URDF", "WARN", f"{walk} not found")

    out = REPO / "logs" / "vigor" / f"env_report_{socket.gethostname()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    info["checks"] = [{"name": n, "status": s, "detail": d} for n, s, d in results]
    out.write_text(json.dumps(info, indent=2))
    fails = [n for n, s, _ in results if s == "FAIL"]
    print(f"\n{'ALL GOOD' if not fails else 'FAILED: ' + ', '.join(fails)} - report: {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
