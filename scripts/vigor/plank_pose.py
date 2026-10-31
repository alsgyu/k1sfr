"""Hand-off plank poses of the safe fall (MuJoCo, CPU): find them, check them, export them.

The safe-fall policy ends every fall in one of two fixed planks (``safefall_math.PLANK_POSES``)
that the get-up starts from.  A pose qualifies when it is left-right symmetric, the K1 holds it
on its PD gains alone (placed on fists and feet, targets = pose, 3 s) with nothing but fists and
feet touching, and it survives 0.3 / 0.6 m/s pushes of the trunk in four directions.

    python scripts/vigor/plank_pose.py search front      # rank symmetric candidates
    python scripts/vigor/plank_pose.py settle            # CHOSEN -> PLANK_POSES lines (settled, symmetrised)
    python scripts/vigor/plank_pose.py check --out plank_poses.json --render plank_poses.png

``check`` writes the poses for the get-up team in the INHA-Recovery-Motion ``FallPose`` layout
(root height, root roll/pitch, 22 joint positions in K1_JOINT_ORDER, projected gravity).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import mujoco
import numpy as np

import safefall_mujoco as smj

sf, spec, k1_getup = smj.sf, smj.spec, smj.k1_getup
NAMES = spec.ALL_JOINT_NAMES
# INHA-Recovery-Motion K1_JOINT_ORDER: same order, these four names differ
RECOVERY_NAMES = {"AAHead_yaw": "Head_Yaw", "Head_pitch": "Head_Pitch",
                  "ALeft_Shoulder_Pitch": "Left_Shoulder_Pitch", "ARight_Shoulder_Pitch": "Right_Shoulder_Pitch"}
# symmetric pose parameters: shoulder pitch, |shoulder roll|, elbow pitch, |elbow yaw|, hip pitch,
# |hip roll| (legs apart), knee, ankle pitch
CHOSEN = {"front": (-1.2, 1.3, 0.3, 0.13, -0.2, 0.2, 0.12, 0.0), "back": (0.6, 1.3, 0.3, 0.13, -1.6, 0.2, 1.2, -0.2)}
GRIDS = {
    "front": ([-1.0, -1.2, -1.4, -1.6, -1.8], [1.3], [0.0, 0.3, 0.6], [0.13], [-0.3, -0.2, 0.0, 0.2, 0.4], [0.1, 0.2, 0.3],
              [0.12], [0.0, 0.28]),
    "back": ([0.6, 0.9, 1.1], [1.3], [0.0, 0.3], [0.13], [-1.6, -1.2, -0.8], [0.1, 0.2], [1.2, 1.6], [-0.2, 0.2]),
}
PUSHES = [(s, d) for s in (0.3, 0.6) for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
CORNERS = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])


class Bench:
    def __init__(self, mesh_dir: str):
        self.m = m = smj.build_model(mesh_dir)
        self.d = mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in NAMES])
        body_group = {m.body(b).id: g for g, bs in sf.CONTACT_GROUPS.items() for b in bs}
        self.geoms = {g: [i for i in range(m.ngeom) if body_group.get(m.geom_bodyid[i]) == g] for g in sf.CONTACT_GROUPS}
        self.bodies = {g: [m.body(b).id for b in bs] for g, bs in sf.CONTACT_GROUPS.items()}

    def low(self, groups) -> float:
        m, d = self.m, self.d
        z = np.inf
        for g in (i for grp in groups for i in self.geoms[grp]):
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            z = min(z, (d.geom_xpos[g] + (c + h * CORNERS) @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
        return z

    def place(self, q: np.ndarray, front: bool):
        """Pitch the trunk about world y until fists and feet are level and put them on the ground.
        Returns the clearance of everything else [m], or None when fists and feet never level."""
        m, d = self.m, self.d

        def gap(a):
            mujoco.mj_resetData(m, d)
            d.qpos[self.qadr] = q
            d.qpos[3:7] = [math.cos(a / 2), 0.0, math.sin(a / 2), 0.0]
            d.qpos[2] = 1.0
            mujoco.mj_kinematics(m, d)
            return self.low(["fist_l", "fist_r"]) - self.low(["feet"])

        lo, hi = (0.3, 1.9) if front else (-1.9, -0.1)
        if gap(lo) * gap(hi) > 0:
            return None
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if gap(lo) * gap(mid) > 0 else (lo, mid)
        gap((lo + hi) / 2)
        d.qpos[2] -= min(self.low(["fist_l", "fist_r"]), self.low(["feet"])) - 0.002
        mujoco.mj_kinematics(m, d)
        return min(self.low([g]) for g in sf.BAD_GROUPS)

    def hold(self, q: np.ndarray, seconds: float = 3.0, push=None) -> dict:
        """PD-hold q from the placed state; outcomes over the last second (push = (speed, dir) at 1 s)."""
        m, d = self.m, self.d
        d.ctrl[:] = q
        mujoco.mj_forward(m, d)
        steps, last = int(seconds / smj.DT), int(1.0 / smj.DT)
        bad = arm_tau = 0.0
        for k in range(steps):
            if push is not None and k == last:
                d.qvel[:3] += push[0] * np.array(push[1], dtype=float)
            mujoco.mj_step(m, d)
            if k >= steps - last and k % 5 == 0:
                mujoco.mj_rnePostConstraint(m, d)
                force = lambda g: sum(np.linalg.norm(d.cfrc_ext[b, 3:]) for b in self.bodies[g])  # noqa: E731
                bad = max(bad, max(force(g) for g in sf.BAD_GROUPS))
                arm_tau = max(arm_tau, np.abs(d.actuator_force[2:10]).max())
        fists = [np.linalg.norm(d.cfrc_ext[b, 3:]) > sf.CONTACT_THRESHOLD for b in self.bodies["fist_l"] + self.bodies["fist_r"]]
        ok = bad < sf.CONTACT_THRESHOLD and d.qpos[2] > sf.PLANK_MIN_TRUNK and all(fists)
        return {"ok": bool(ok), "trunk_h": float(d.qpos[2]), "arm_tau": float(arm_tau),
                "sag": float(np.abs(d.qpos[self.qadr] - q)[2:].max())}

    def pushes_survived(self, q: np.ndarray, front: bool) -> int:
        n = 0
        for p in PUSHES:
            self.place(q, front)
            n += self.hold(q, push=p)["ok"]
        return n


def pose(sp, sr, ep, ey, hp, hr, kp, ap) -> np.ndarray:
    """Symmetric 22-joint pose (head 0) from the CHOSEN parameters; roll/yaw flip sign on the right."""
    q = dict.fromkeys(NAMES, 0.0)
    for side, sg in (("Left", 1.0), ("Right", -1.0)):
        q[f"A{side}_Shoulder_Pitch"], q[f"{side}_Shoulder_Roll"] = sp, -sg * sr
        q[f"{side}_Elbow_Pitch"], q[f"{side}_Elbow_Yaw"] = ep, -sg * ey
        q[f"{side}_Hip_Pitch"], q[f"{side}_Hip_Roll"], q[f"{side}_Knee_Pitch"], q[f"{side}_Ankle_Pitch"] = hp, sg * hr, kp, ap
    return np.array([q[n] for n in NAMES])


def symmetrised(q: np.ndarray) -> np.ndarray:
    """Average each left/right pair with its mirror (removes settling noise); head yaw 0."""
    out = q.copy()
    for i, n in enumerate(NAMES):
        if "Left" in n:
            j = NAMES.index(n.replace("Left", "Right"))
            v = 0.5 * (q[i] + k1_getup.mirror_sign(n) * q[j])
            out[i], out[j] = v, k1_getup.mirror_sign(n) * v
    out[NAMES.index("AAHead_yaw")] = 0.0
    return out


def root_state(bench: Bench) -> dict:
    w, x, y, z = bench.d.qpos[3:7]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, bench.d.qpos[3:7])
    grav = rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    return {"root_height": round(float(bench.d.qpos[2]), 3), "root_rpy": [round(roll, 3), round(pitch, 3), 0.0],
            "projected_gravity": [round(float(v), 3) for v in grav],
            "head_height": round(float(bench.d.xpos[bench.m.body(sf.HEAD_BODY).id, 2]), 3)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["search", "settle", "check"])
    p.add_argument("side", nargs="?", choices=["front", "back"], default="front")
    p.add_argument("--mesh_dir", default=os.path.expanduser("~/vigor_work/GMR/assets/booster_k1/meshes"))
    p.add_argument("--out", default=None, help="check: JSON for the get-up team")
    p.add_argument("--render", default=None, help="check: PNG of both poses")
    a = p.parse_args()
    bench = Bench(a.mesh_dir)

    if a.mode == "search":
        rows = []
        for params in itertools.product(*GRIDS[a.side]):
            q = pose(*params)
            clear = bench.place(q, a.side == "front")
            if clear is None or clear < 0.02:
                continue
            h = bench.hold(q)
            if h["ok"]:
                rows.append((bench.pushes_survived(q, a.side == "front"), -h["sag"], params, h))
        rows.sort(key=lambda r: r[:2], reverse=True)
        print(f"{len(rows)} symmetric {a.side} candidates hold cleanly; best by pushes survived (of {len(PUSHES)}), then sag:")
        for n, _, params, h in rows[:15]:
            print(f"  {n}/{len(PUSHES)} {params}  trunk {h['trunk_h']:.3f} m, arm torque {h['arm_tau']:.1f} Nm, sag {h['sag']:.2f}")
        return

    if a.mode == "settle":
        for side, params in CHOSEN.items():
            q = pose(*params)
            bench.place(q, side == "front")
            bench.hold(q)
            settled = symmetrised(bench.d.qpos[bench.qadr])
            print(f'    "{side}": ({", ".join(f"{v:.3f}" for v in settled)}),  # {root_state(bench)}')
        return

    out, frames = {}, []
    for side, q in sf.PLANK_POSES.items():
        q = np.array(q)
        clear = bench.place(q, side == "front")
        placed = root_state(bench)  # the pose resting on fists and feet: root follows from the joints
        pushes = bench.pushes_survived(q, side == "front")
        bench.place(q, side == "front")
        h = bench.hold(q)  # with the pose as PD targets it sags a little further under load
        out[side] = placed | {
            "joint_pos": [round(float(v), 3) for v in q],
            "joint_names": [RECOVERY_NAMES.get(n, n) for n in NAMES],
            "holds": h["ok"], "pushes_survived": f"{pushes}/{len(PUSHES)}", "clearance": round(float(clear), 3),
            "arm_torque_nm": round(h["arm_tau"], 1),
        }
        print(side, {k: v for k, v in out[side].items() if k not in ("joint_pos", "joint_names")})
        if a.render:
            r = mujoco.Renderer(bench.m, 320, 320)
            cam = mujoco.MjvCamera()
            cam.distance, cam.elevation, cam.lookat[:] = 1.4, -12, [bench.d.qpos[0], bench.d.qpos[1], 0.2]
            for az in (90, 0):
                cam.azimuth = az
                r.update_scene(bench.d, cam)
                frames.append((f"{side} ({'side' if az == 90 else 'front'} view)", r.render().copy()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"source": "scripts/vigor/plank_pose.py check: joint_pos = safefall_math.PLANK_POSES (settled under "
                                 "load on the PD gains, K1_22dof_jy.urdf); root from resting that pose on fists and feet "
                                 "(geom bounding boxes, up to ~1 cm high)", "poses": out}, f, indent=1)
        print("wrote", a.out)
    if a.render:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3.2))
        for axis, (title, img) in zip(ax, frames):
            axis.imshow(img)
            axis.set_title(title, fontsize=8)
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(a.render, dpi=90)
        print("wrote", a.render)


if __name__ == "__main__":
    main()
