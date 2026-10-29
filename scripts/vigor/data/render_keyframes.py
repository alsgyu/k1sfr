"""QA plots for a keyframe dataset: per-segment keyframe strips and summary statistics.

    python scripts/vigor/data/render_keyframes.py --out /tmp/vigor_qa [--name k1_fall_keyframes_v1.npz]
"""

from __future__ import annotations

import argparse
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from urdf_fk import URDFModel, quat_to_matrix  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
URDF = REPO / "source/JY_train/JY_train/assets/robots/urdf/K1_22dof_jy.urdf"


def lying_class(root_quat: np.ndarray) -> str:
    fwd = quat_to_matrix(root_quat)[:, 0]
    if fwd[2] > 0.5:
        return "supine"
    if fwd[2] < -0.5:
        return "prone"
    return "side"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="k1_fall_keyframes_v1.npz")
    p.add_argument("--data_dir", default=str(REPO / "data" / "vigor"))
    p.add_argument("--out", required=True)
    p.add_argument("--segments", type=int, default=12, help="segments per strip image")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(os.path.join(args.data_dir, args.name))
    model = URDFModel(str(URDF))
    names = [str(n) for n in d["joint_names"]]
    edges = [(model.joint_of_child[l].parent, l) for l in model.link_order[1:]]
    fps, stride = int(d["fps"]), int(d["stride"])
    starts, lengths = d["clip_start"], d["clip_length"]

    classes = [lying_class(d["root_quat"][s + lw]) for s, lw in zip(starts, d["lowest"])]
    stats = {c: classes.count(c) for c in ("prone", "supine", "side")}
    print("lying pose at lowest frame:", stats)
    print("fall direction (deg):", np.round(np.degrees(d["fall_dir"])).astype(int).tolist())

    fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    ax[0].hist(lengths / fps, bins=15)
    ax[0].set_title("segment length (s)")
    ax[1].hist(np.degrees(d["fall_dir"]), bins=16, range=(-180, 180))
    ax[1].set_title("fall direction vs heading (deg)")
    ax[2].bar(list(stats), list(stats.values()))
    ax[2].set_title("pose at lowest frame")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "summary.png"), dpi=70)
    plt.close()

    for page in range(0, len(starts), args.segments):
        ids = list(range(page, min(page + args.segments, len(starts))))
        cols = max(int(np.ceil(lengths[ids].max() / stride)) + 1, 2)
        fig = plt.figure(figsize=(1.3 * cols, 1.5 * len(ids)))
        for r, c in enumerate(ids):
            s, n = starts[c], lengths[c]
            key = np.unique(np.append(np.arange(0, n, stride), n - 1))
            g = s + key
            fk = model.fk(d["root_pos"][g], d["root_quat"][g], d["joint_pos"][g], names)
            for k, kk in enumerate(key):
                a = fig.add_subplot(len(ids), cols, r * cols + k + 1)
                center = d["root_pos"][g[k]]
                for pa, pb in edges:
                    xa, xb = fk[pa][0][k], fk[pb][0][k]
                    a.plot([xa[0] - center[0], xb[0] - center[0]], [xa[2], xb[2]], "b-", lw=1)
                a.axhline(0, color="k", lw=0.4)
                a.set_xlim(-0.6, 0.6)
                a.set_ylim(-0.05, 0.95)
                a.set_aspect("equal")
                a.axis("off")
                marks = []
                if kk >= d["fall_onset"][c] and (k == 0 or key[k - 1] < d["fall_onset"][c]):
                    marks.append("onset")
                if kk >= d["lowest"][c] and (k == 0 or key[k - 1] < d["lowest"][c]):
                    marks.append("low")
                if kk >= d["getup_end"][c] and (k == 0 or key[k - 1] < d["getup_end"][c]):
                    marks.append("up")
                a.set_title(("" if k else f"{str(d['clip_names'][c])[:22]} ") + " ".join(marks), fontsize=5, loc="left")
        plt.tight_layout(pad=0.1)
        out = os.path.join(args.out, f"keyframes_{page // args.segments:02d}.png")
        plt.savefig(out, dpi=60)
        plt.close()
        print("wrote", out)


if __name__ == "__main__":
    main()
