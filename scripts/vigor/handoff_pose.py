"""Hand-off poses of the safe fall (MuJoCo, CPU): find them, check them, export them.

The safe-fall policy ends every fall in one of two fixed multi-contact poses
(``safefall_math.HANDOFF_POSES``) that the get-up starts from.  A pose qualifies when it is
left-right symmetric and the K1, dropped onto the floor with the pose as PD targets, comes to
rest on both fists plus the other sturdy contacts of its side -- knees (and toes) in front, hips
(and heels) behind -- with head, upper arms, chest and back clear and the head high, and keeps
those contacts through 0.3 / 0.6 m/s pushes of the trunk in four directions.  Candidates rank
by the static load: the sum over bodies of (floor force / weight)^2, lowest = most evenly shared.

    python scripts/vigor/handoff_pose.py search front --n 2000   # random symmetric candidates, best first
    python scripts/vigor/handoff_pose.py poses                    # CHOSEN -> HANDOFF_POSES / HANDOFF_ROOTS lines
    python scripts/vigor/handoff_pose.py check --out handoff_poses.json --render handoff_poses.png

``check`` writes the poses for the get-up team in the INHA-Recovery-Motion ``FallPose`` layout
(root height, root roll/pitch, 22 settled joint positions in K1_JOINT_ORDER, projected gravity).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from multiprocessing import Pool

import mujoco
import numpy as np

import safefall_mujoco as smj

sf, spec, k1_getup = smj.sf, smj.spec, smj.k1_getup
NAMES = spec.ALL_JOINT_NAMES
MESH_DIR = os.path.expanduser("~/vigor_work/GMR/assets/booster_k1/meshes")
# INHA-Recovery-Motion K1_JOINT_ORDER: same order, these four names differ
RECOVERY_NAMES = {"AAHead_yaw": "Head_Yaw", "Head_pitch": "Head_Pitch",
                  "ALeft_Shoulder_Pitch": "Left_Shoulder_Pitch", "ARight_Shoulder_Pitch": "Right_Shoulder_Pitch"}
# symmetric pose parameters (shoulder pitch, |shoulder roll|, elbow pitch, |elbow yaw| (flexion), hip pitch,
# |hip roll| (legs apart), knee, ankle pitch) and the trunk pitch the drop starts from
CHOSEN = {"front": ((-1.2, 1.3, 0.3, 0.4, -1.05, 0.15, 1.65, 0.05), 1.10),
          "back": ((0.95, 1.15, 0.3, 0.2, -1.8, 0.15, 0.85, -0.35), -0.48)}
RANGES = {  # search: uniform ranges of the parameters above (elbow pitch 0.3, hip roll 0.15 fixed) and the drop pitch
    "front": {"sp": (-2.0, -0.8), "sr": (1.0, 1.5), "ey": (0.1, 1.0), "hp": (-1.8, -0.4), "kp": (1.0, 2.2), "ap": (-0.5, 0.34),
              "pitch": (0.9, 1.6)},
    "back": {"sp": (-0.6, 0.95), "sr": (1.0, 1.5), "ey": (0.1, 1.0), "hp": (-2.2, -0.6), "kp": (0.6, 2.2), "ap": (-0.8, 0.3),
             "pitch": (-1.0, -0.3)},
}
SUPPORT = {"front": ["Left_Shank", "Right_Shank"], "back": ["Left_Hip_Roll", "Right_Hip_Roll"]}  # besides the fists
NEVER = sf.CONTACT_GROUPS["head"] + sf.CONTACT_GROUPS["arms"]  # plus the trunk unless it is the pelvis
PUSHES = [s * np.array(d, dtype=float) for s in (0.3, 0.6) for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
CORNERS = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
TOUCH = 0.03  # body weights: a body counts as a contact of the pose


class Bench:
    def __init__(self, mesh_dir: str = MESH_DIR):
        self.m = m = smj.build_model(mesh_dir)
        self.d = mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in NAMES])
        self.bodies = np.array([m.body(b).id for b in sf.CONTACT_BODIES])
        lo, hi = m.jnt_range[[m.joint(n).id for n in NAMES]].T
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.9  # the robot config's soft_joint_pos_limit_factor
        self.soft = (mid - half, mid + half)

    def lowest(self, bodies=None) -> float:
        m, d = self.m, self.d
        z = np.inf
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b == 0 or (bodies is not None and m.body(b).name not in bodies):
                continue
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            z = min(z, (d.geom_xpos[g] + (c + h * CORNERS) @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
        return z

    def floor_forces(self) -> np.ndarray:
        """|force| from the floor per sf.CONTACT_BODIES entry (self-contacts left out)."""
        out, f6 = np.zeros(len(self.bodies)), np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 != 0 and c.geom2 != 0:
                continue
            j = np.where(self.bodies == self.m.geom_bodyid[c.geom2 if c.geom1 == 0 else c.geom1])[0]
            if len(j):
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                out[j[0]] += np.linalg.norm(f6[:3])
        return out

    def settle(self, q: np.ndarray, pitch: float, seconds: float = 3.0, push: np.ndarray | None = None) -> dict:
        """Drop the pose (trunk pitched about world y, lowest point 1 cm up) and hold it on the PD gains;
        loads [body weights] and torques over the last 0.5 s (a push lands 1 s before the end)."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = q
        d.qpos[3:7] = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
        d.qpos[2] = 1.0
        mujoco.mj_kinematics(m, d)
        d.qpos[2] -= self.lowest() - 0.01
        d.ctrl[:] = q
        mujoco.mj_forward(m, d)
        steps, last = int(seconds / smj.DT), int(0.5 / smj.DT)
        load, tau = np.zeros(len(self.bodies)), np.zeros(len(NAMES))
        for k in range(steps):
            if push is not None and k == steps - 2 * last:
                d.qvel[:3] += push
            mujoco.mj_step(m, d)
            if k >= steps - last:
                load += self.floor_forces() / last
                tau = np.maximum(tau, np.abs(d.actuator_force))
        w, x, y, z = d.qpos[3:7]
        return {"load": load / sf.WEIGHT, "tau": tau, "head_low": self.lowest(sf.CONTACT_GROUPS["head"]),
                "trunk_h": float(d.qpos[2]), "pitch": math.atan2(2 * (w * y - z * x), 1 - 2 * (x * x + y * y)),
                "speed": float(np.abs(d.qvel[:6]).max()), "q": d.qpos[self.qadr].copy()}


def contacts(r: dict) -> dict[str, float]:
    return {b: round(float(f), 3) for b, f in zip(sf.CONTACT_BODIES, r["load"]) if f > TOUCH}


def qualifies(r: dict, side: str, q: np.ndarray, bench: Bench) -> bool:
    """Contacts of the side (fists symmetric), head clear and high, at rest, targets inside the soft joint limits."""
    if ((q < bench.soft[0]) | (q > bench.soft[1])).any():
        return False
    c = contacts(r)
    never = NEVER + ([] if side == "back" and r["pitch"] > -math.radians(sf.PELVIS_TILT_DEG) else ["Trunk"])
    return (all(b in c for b in sf.HAND_BODIES + SUPPORT[side]) and not any(b in c for b in never)
            and abs(c["left_hand_link"] - c["right_hand_link"]) < 0.05 and r["head_low"] > 0.15 and r["speed"] < 0.05)


def pushes_survived(bench: Bench, q: np.ndarray, pitch: float) -> int:
    base = set(contacts(bench.settle(q, pitch)))
    n = 0
    for push in PUSHES:
        r = bench.settle(q, pitch, 4.0, push)
        n += set(contacts(r)) == base and r["head_low"] > 0.1 and r["speed"] < 0.05
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


def _search(args) -> list:
    side, n, seed = args
    bench, rng, out = Bench(), np.random.default_rng(seed), []
    for _ in range(n):
        p = {k: rng.uniform(*v) for k, v in RANGES[side].items()}
        params = (p["sp"], p["sr"], 0.3, p["ey"], p["hp"], 0.15, p["kp"], p["ap"])
        r = bench.settle(pose(*params), p["pitch"], 2.5)
        if qualifies(r, side, pose(*params), bench):
            out.append((float(np.square(r["load"]).sum()), params, p["pitch"], r))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["search", "poses", "check"])
    p.add_argument("side", nargs="?", choices=["front", "back"], default="front")
    p.add_argument("--n", type=int, default=2000, help="search: candidates")
    p.add_argument("--out", default=None, help="check: JSON for the get-up team")
    p.add_argument("--render", default=None, help="check: PNG of both poses")
    a = p.parse_args()

    if a.mode == "search":
        workers = os.cpu_count() or 4
        with Pool(workers) as pool:
            rows = [r for chunk in pool.map(_search, [(a.side, a.n // workers, s) for s in range(workers)]) for r in chunk]
        rows.sort(key=lambda r: r[0])
        bench = Bench()
        print(f"{len(rows)} of {a.n} symmetric {a.side} candidates qualify; most evenly loaded first:")
        for load, params, pitch, r in rows[:10]:
            n = pushes_survived(bench, pose(*params), pitch)
            print(f"  sum (F/W)^2 {load:.3f}  pushes {n}/{len(PUSHES)}  {tuple(round(x, 2) for x in params)} drop pitch {pitch:.2f}\n"
                  f"     head {r['head_low']:.2f} m, trunk {r['trunk_h']:.2f} m, pitch {math.degrees(r['pitch']):.0f} deg, "
                  f"arm torque {r['tau'][2:10].max():.1f} Nm, contacts {contacts(r)}")
        return

    bench = Bench()
    if a.mode == "poses":
        for side, (params, pitch) in CHOSEN.items():
            r = bench.settle(pose(*params), pitch)
            print(f'    "{side}": ({", ".join(f"{v:.3f}" for v in pose(*params))}),  # contacts {contacts(r)}')
            print(f'    root ({r["trunk_h"]:.3f}, {r["pitch"]:.3f})  qualifies: {qualifies(r, side, pose(*params), bench)}')
        return

    out, frames = {}, []
    rend = mujoco.Renderer(bench.m, 320, 320) if a.render else None  # one renderer: a second one renders noise
    for side, q in sf.HANDOFF_POSES.items():
        q, pitch = np.array(q), CHOSEN[side][1]
        pushes = pushes_survived(bench, q, pitch)
        r = bench.settle(q, pitch)
        out[side] = root_state(bench) | {
            "joint_pos": [round(float(v), 3) for v in symmetrised(r["q"])],  # settled under load
            "joint_names": [RECOVERY_NAMES.get(n, n) for n in NAMES],
            "target_joint_pos": [round(float(v), 3) for v in q],  # the PD targets (safefall_math.HANDOFF_POSES)
            "contacts_body_weights": contacts(r),
            "holds": bool(qualifies(r, side, q, bench)), "pushes_survived": f"{pushes}/{len(PUSHES)}",
            "head_clearance": round(float(r["head_low"]), 3), "arm_torque_nm": round(float(r["tau"][2:10].max()), 1),
        }
        print(side, {k: v for k, v in out[side].items() if k not in ("joint_pos", "joint_names", "target_joint_pos")})
        if rend is not None:
            cam = mujoco.MjvCamera()
            cam.distance, cam.elevation, cam.lookat[:] = 1.4, -12, [bench.d.qpos[0], bench.d.qpos[1], 0.2]
            for az in (90, 0):
                cam.azimuth = az
                rend.update_scene(bench.d, cam)
                frames.append((f"{side} ({'side' if az == 90 else 'front'} view)", rend.render().copy()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"poses": out}, f, indent=1)
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
