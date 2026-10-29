"""Retarget LAFAN1 fall/get-up BVH clips to K1 with GMR (headless, CPU).

Runs on the data-prep laptop (no Isaac).  Output per clip:
``<out_dir>/<clip>.npz`` with GMR's raw 30 fps result (root pose + 22 joints
in GMR's K1 model order) and the scaled human keypoints for inspection.

    python scripts/vigor/data/retarget_lafan1.py --bvh_dir ~/vigor_work/lafan1/bvh \
        --out_dir ~/vigor_work/retarget --clips fallAndGetUp pushAndFall
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pathlib
import time

import subprocess

import mujoco as mj
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
IK_CONFIG = HERE / "bvh_lafan1_to_k1.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--clips", nargs="+", default=["fallAndGetUp", "pushAndFall"], help="filename prefixes")
    parser.add_argument("--max_frames", type=int, default=None, help="debug: only the first N frames")
    args = parser.parse_args()

    from general_motion_retargeting import IK_CONFIG_DICT
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    IK_CONFIG_DICT["bvh_lafan1"]["booster_k1"] = IK_CONFIG
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for p in args.clips for f in glob.glob(os.path.join(os.path.expanduser(args.bvh_dir), f"{p}*.bvh")))
    if not files:
        raise SystemExit(f"no BVH files matching {args.clips} in {args.bvh_dir}")
    cfg_sha = hashlib.sha256(IK_CONFIG.read_bytes()).hexdigest()
    import general_motion_retargeting as gmr_pkg

    gmr_dir = pathlib.Path(gmr_pkg.__file__).resolve().parents[1]
    gmr_commit = subprocess.run(["git", "-C", str(gmr_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    for path in files:
        name = pathlib.Path(path).stem
        frames, human_height = load_bvh_file(path, format="lafan1")
        if args.max_frames:
            frames = frames[: args.max_frames]
        gmr = GMR(src_human="bvh_lafan1", tgt_robot="booster_k1", actual_human_height=human_height, verbose=False)
        joint_names = [mj.mj_id2name(gmr.model, mj.mjtObj.mjOBJ_JOINT, gmr.model.dof_jntid[i]) for i in range(6, gmr.model.nv)]
        qpos, human = [], []
        t0 = time.time()
        for f in frames:
            qpos.append(gmr.retarget(f))
            human.append(np.stack([np.asarray(gmr.scaled_human_data[k][0]) for k in HUMAN_KEYS]))
        qpos = np.asarray(qpos)
        np.savez_compressed(
            os.path.join(args.out_dir, f"{name}.npz"),
            fps=30,
            root_pos=qpos[:, :3],
            root_quat=qpos[:, 3:7],  # wxyz (mujoco)
            joint_pos=qpos[:, 7:],
            joint_names=np.array(joint_names),
            human_pos=np.asarray(human),
            human_keys=np.array(HUMAN_KEYS),
            source_bvh=os.path.basename(path),
            ik_config_sha256=cfg_sha,
        )
        print(f"{name}: {len(frames)} frames in {time.time() - t0:.0f}s")

    with open(os.path.join(args.out_dir, "retarget_info.json"), "w") as f:
        json.dump({"ik_config": str(IK_CONFIG), "ik_config_sha256": cfg_sha, "gmr_commit": gmr_commit,
                   "files": [os.path.basename(p) for p in files]}, f, indent=2)


# GMR only keeps (and needs IK offsets for) the bones listed in the IK config's scale table
HUMAN_KEYS = [
    "Hips", "LeftUpLeg", "LeftLeg", "LeftFootMod", "RightUpLeg", "RightLeg", "RightFootMod",
    "LeftArm", "LeftForeArm", "RightArm", "RightForeArm",
]

if __name__ == "__main__":
    main()
