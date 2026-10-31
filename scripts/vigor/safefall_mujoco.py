"""Laptop-side check of the K1 safe-fall task in MuJoCo (CPU, no Isaac).

Builds the K1 from the same URDF and PD gains / torque limits as the Isaac task, starts
episodes from the same falls (``safefall_math.sample_starts``), scores them with the same
reward (``safefall_math.reward`` + the regularisation weights of the Isaac config), trains a
PPO policy with rsl_rl and compares it against the reflex alone (zero actions: reach into the hand-off plank).

    python scripts/vigor/safefall_mujoco.py train --envs 64 --iters 800 --out ~/vigor_work/safefall_mj
    python scripts/vigor/safefall_mujoco.py eval --ckpt ~/vigor_work/safefall_mj/model_800.pt --render ~/vigor_work/safefall_mj/rollouts.png
    python scripts/vigor/safefall_mujoco.py eval --passive
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
VIGOR = REPO / "source/JY_train/JY_train/tasks/manager_based/vigor"
sys.path.insert(0, str(VIGOR))
sys.path.insert(0, str(REPO / "scripts/vigor/data"))

import k1_getup  # noqa: E402  (walking stance)
import safefall_math as sf  # noqa: E402
import spec  # noqa: E402
import torch_math as tm  # noqa: E402

URDF = REPO / "source/JY_train/JY_train/assets/robots/urdf/K1_22dof_jy.urdf"
DT, DECIMATION = spec.SIM_DT, spec.DECIMATION
EPISODE_S = sf.EPISODE_S
HISTORY = 3

# BOOSTER_K1_FULL_CFG: (kp, kd, effort limit, armature) per joint pattern
GAINS = [
    (r".*_Hip_Pitch", 80.0, 2.0, 68.0, 0.0478125),
    (r".*_Hip_Roll", 80.0, 2.0, 76.0, 0.0339552),
    (r".*_Hip_Yaw", 80.0, 2.0, 38.3, 0.0282528),
    (r".*_Knee_Pitch", 80.0, 2.0, 112.0, 0.095625),
    (r".*_Ankle_.*", 30.0, 2.0, 38.3, 0.0565056),
    (r".*_(Shoulder|Elbow)_.*", 20.0, 1.0, 14.0, 0.001),
    (r"(AAHead_yaw|Head_pitch)", 10.0, 0.5, 6.0, 0.001),
]
# regularisation (per-second weights, as in the Isaac SafeFall config)
REG = {"action_rate": -0.01, "joint_pos_limit": -1.0, "joint_vel": -1.0e-4, "joint_acc": -2.5e-7, "torque": -1.0e-5}


def gains(name):
    return next(g[1:] for g in GAINS if re.fullmatch(g[0], name))


def build_model(mesh_dir: str) -> mujoco.MjModel:
    txt = open(URDF).read().replace('filename="meshes/', 'filename="')
    txt = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler meshdir="%s" discardvisual="true" balanceinertia="true" '
                 r'fusestatic="false"/></mujoco>' % os.path.abspath(mesh_dir), txt, count=1)
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(txt)
    s = mujoco.MjSpec.from_file(f.name)
    os.unlink(f.name)
    s.option.timestep = DT
    s.worldbody.first_body().add_freejoint()
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], friction=[0.8, 0.005, 0.0001])
    s.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    for name in spec.ALL_JOINT_NAMES:  # actuator i <-> canonical joint i
        kp, kd, effort, armature = gains(name)
        s.joint(name).armature = armature
        a = s.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0], a.biasprm[1], a.biasprm[2] = kp, -kp, -kd
        a.forcelimited, a.forcerange = True, [-effort, effort]
    return s.compile()


class SafeFallEnv:
    """rsl_rl VecEnv over independent MjData (threads step them; MuJoCo releases the GIL)."""

    def __init__(self, num_envs: int, mesh_dir: str, threads: int | None = None, seed: int = 0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.m = build_model(mesh_dir)
        self.num_envs, self.device, self.cfg = num_envs, "cpu", {}
        self.datas = [mujoco.MjData(self.m) for _ in range(num_envs)]
        self.pool = ThreadPoolExecutor(threads or os.cpu_count())
        self.chunks = np.array_split(np.arange(num_envs), threads or os.cpu_count())
        names = spec.ALL_JOINT_NAMES
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in names])
        self.vadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in names])
        self.act = np.array([names.index(n) for n in spec.ACTION_JOINT_NAMES])
        self.head_ids = np.array([names.index(n) for n in spec.HEAD_JOINT_NAMES])
        self.default = np.clip(k1_getup.stance(names), *self._soft_limits())
        self.default[[names.index("Left_Elbow_Yaw"), names.index("Right_Elbow_Yaw")]] = -0.13, 0.13  # as the Isaac task
        effort = np.array([gains(n)[2] for n in names])
        kp = np.array([gains(n)[0] for n in names])
        self.scale = (0.5 * effort / kp)[self.act]
        self.body = {g: np.array([self.m.body(b).id for b in bs]) for g, bs in sf.CONTACT_GROUPS.items()}
        self.head_body, self.trunk_body = self.m.body(sf.HEAD_BODY).id, self.m.body("Trunk").id
        self.hand_bodies = np.array([self.m.body(b).id for b in sf.HAND_BODIES])
        self.num_actions = len(self.act)
        self.max_episode_length = int(round(EPISODE_S / (DT * DECIMATION)))
        self.step_dt = DT * DECIMATION
        n = num_envs
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.targets = np.tile(self.default, (n, 1))
        self.prev_targets = self.targets.copy()
        self.delay = np.zeros(n, dtype=int)
        self.peak = np.zeros((n, self.m.nbody))
        self.last_action = np.zeros((n, self.num_actions))
        self.prev_qvel = np.zeros((n, len(names)))
        self.prev_z = np.zeros((n, 2))
        self.hist = np.zeros((n, HISTORY, 70))
        self.direction = np.zeros(n, dtype=int)
        self.facing = np.zeros(n)
        self.ep_sums: dict[str, np.ndarray] = {}
        self.tracker = sf.OutcomeTracker(n)
        self.passive = False
        self.finished: list[dict] = []
        self.reset(np.arange(n))
        self.obs, self.critic = self._observe()

    # ------------------------------------------------------------- helpers
    def _soft_limits(self):
        lo = np.array([self.m.jnt_range[self.m.joint(n).id][0] for n in spec.ALL_JOINT_NAMES])
        hi = np.array([self.m.jnt_range[self.m.joint(n).id][1] for n in spec.ALL_JOINT_NAMES])
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.9
        return mid - half, mid + half

    def _lowest(self, d) -> float:
        z = np.inf
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] == 0:
                continue
            c, h = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:]
            corners = c + h * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            z = min(z, (d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
        return z

    def reset(self, ids):
        if len(ids) == 0:
            return
        st = sf.sample_starts(len(ids))
        lo, hi = self._soft_limits()
        for k, i in enumerate(ids):
            d = self.datas[i]
            mujoco.mj_resetData(self.m, d)
            q = self.default + self.rng.uniform(-0.1, 0.1, len(self.default))
            q[self.head_ids] = self.default[self.head_ids]
            for leg, sgn in (("Left", 1), ("Right", -1)):  # mid-stride leg spread
                q[spec.ALL_JOINT_NAMES.index(f"{leg}_Hip_Pitch")] += sgn * self.rng.uniform(-0.25, 0.25)
            if st["plank"][k] >= 0:  # held in a hand-off plank (head stays at its default)
                body = len(self.head_ids)
                plank = np.array(sf.PLANK_POSES[("front", "back")[int(st["plank"][k])]])
                q[body:] = plank[body:] + self.rng.uniform(-0.05, 0.05, len(q) - body)
            q = np.clip(q, lo, hi)
            d.qpos[self.qadr] = q
            d.qpos[3:7] = st["root_quat"][k].numpy()
            d.qpos[:3] = st["root_rel"][k].numpy() + [0.0, 0.0, 1.0]
            mujoco.mj_kinematics(self.m, d)
            d.qpos[2] -= self._lowest(d) - 0.003  # lowest point 3 mm above the floor
            d.qvel[:3] = st["lin_vel"][k].numpy()
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, d.qpos[3:7])
            d.qvel[3:6] = rot.reshape(3, 3).T @ st["ang_vel"][k].numpy()  # free joint: body-frame angular velocity
            d.ctrl[:] = q
            mujoco.mj_forward(self.m, d)
            self.targets[i] = self.prev_targets[i] = q
            self.delay[i] = self.rng.integers(0, DECIMATION)
            self.direction[i] = int(st["direction"][k])
            self.facing[i] = float(st["facing"][k])
            self.prev_qvel[i] = d.qvel[self.vadr]
            self.prev_z[i] = d.xpos[self.head_body, 2], d.xpos[self.trunk_body, 2]
        self.last_action[ids] = 0.0
        self.episode_length_buf[torch.as_tensor(ids)] = 0
        for k in self.ep_sums:
            self.ep_sums[k][ids] = 0.0
        self.hist[ids] = 0.0

    def _substeps(self, chunk):
        for i in chunk:
            d = self.datas[i]
            peak = np.zeros(self.m.nbody)
            for k in range(DECIMATION):
                d.ctrl[:] = self.targets[i] if k >= self.delay[i] else self.prev_targets[i]
                mujoco.mj_step(self.m, d)
                mujoco.mj_rnePostConstraint(self.m, d)
                np.maximum(peak, np.linalg.norm(d.cfrc_ext[:, 3:], axis=1), out=peak)
            self.peak[i] = peak

    def _frame(self):
        qpos = np.stack([d.qpos for d in self.datas])
        qvel = np.stack([d.qvel for d in self.datas])
        quat = torch.as_tensor(qpos[:, 3:7], dtype=torch.float32)
        grav = tm.projected_gravity(quat).numpy()
        return qpos, qvel, quat, grav

    def _observe(self):
        qpos, qvel, quat, grav = self._frame()
        frame = np.concatenate([qvel[:, 3:6], grav, qpos[:, self.qadr] - self.default, qvel[:, self.vadr], self.last_action], 1)
        self.hist = np.concatenate([self.hist[:, 1:], frame[:, None]], 1)
        lin_b = tm.quat_apply_inverse(quat, torch.as_tensor(qvel[:, :3], dtype=torch.float32)).numpy()
        heights = np.stack([[d.xpos[self.trunk_body, 2], d.xpos[self.head_body, 2], d.xpos[self.hand_bodies, 2].min()]
                            for d in self.datas])
        forces = np.stack([self.peak[:, b].sum(1) for b in self.body.values()], 1) / sf.WEIGHT
        critic = np.concatenate([frame, lin_b, heights, forces, -grav[:, 2:3], self.facing[:, None]], 1)
        return (torch.as_tensor(self.hist.reshape(self.num_envs, -1), dtype=torch.float32),
                torch.as_tensor(critic, dtype=torch.float32))

    def get_observations(self):
        return self.obs, {"observations": {"critic": self.critic}}

    # ---------------------------------------------------------------- step
    def step(self, actions: torch.Tensor):
        a = actions.detach().cpu().numpy().astype(np.float64)
        if self.passive:
            a[:] = 0.0
        self.prev_targets[:] = self.targets
        reflex = sf.reflex_targets(tm.projected_gravity(torch.as_tensor(np.stack([d.qpos[3:7] for d in self.datas]), dtype=torch.float32)))
        self.targets[:, self.act] = reflex.numpy() + self.scale * a  # offsets from the hand-off plank (as Isaac)
        list(self.pool.map(self._substeps, self.chunks))
        self.episode_length_buf += 1

        qpos, qvel, quat, grav = self._frame()
        dt = self.step_dt
        head_z = np.array([d.xpos[self.head_body, 2] for d in self.datas])
        trunk_z = np.array([d.xpos[self.trunk_body, 2] for d in self.datas])
        hand_z = np.array([d.xpos[self.hand_bodies, 2].min() for d in self.datas])
        f = {g: torch.as_tensor(self.peak[:, b].sum(1), dtype=torch.float32) for g, b in self.body.items()}
        T = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
        state = sf.FallState(force=f, head_h=T(head_z), trunk_h=T(trunk_z), hand_h=T(hand_z),
                             head_vz=T((head_z - self.prev_z[:, 0]) / dt), trunk_vz=T((trunk_z - self.prev_z[:, 1]) / dt),
                             gravity_b=T(grav), q=T(qpos[:, self.qadr]), facing=T(self.facing))
        rew, parts = sf.reward(state, dt)
        lo, hi = self._soft_limits()
        q, qd = qpos[:, self.qadr], qvel[:, self.vadr]
        tau = np.stack([d.actuator_force for d in self.datas])
        reg = {
            "action_rate": np.square(a - self.last_action).sum(1),
            "joint_pos_limit": (np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None)).sum(1),
            "joint_vel": np.square(qd[:, self.act]).sum(1),
            "joint_acc": np.square((qd - self.prev_qvel) / dt)[:, self.act].sum(1),
            "torque": np.square(tau[:, self.act]).sum(1),
        }
        for k, v in reg.items():
            parts[k] = T(REG[k] * dt * v)
            rew = rew + parts[k]
        self.last_action, self.prev_qvel, self.prev_z = a, qd.copy(), np.stack([head_z, trunk_z], 1)

        self.tracker.update(state)
        self.last_state = state
        for k, v in parts.items():
            self.ep_sums.setdefault(k, np.zeros(self.num_envs))
            self.ep_sums[k] += v.numpy()

        bad = ~np.isfinite(qpos).all(1) | (np.abs(qvel).max(1) > 100.0)
        time_out = self.episode_length_buf.numpy() >= self.max_episode_length
        done = time_out | bad
        ids = np.where(done)[0]
        log = {}
        self.finished = self.tracker.close(torch.as_tensor(ids), torch.as_tensor(self.direction)) if len(ids) else []
        if len(ids):
            for k, v in self.ep_sums.items():
                log[f"Episode_Reward/{k}"] = float(v[ids].mean() / EPISODE_S)
            log |= {f"Metrics/{k}": v for k, v in sf.summarize(self.finished).items() if isinstance(v, float)}
        rew = torch.where(torch.as_tensor(bad), torch.zeros_like(rew), rew)
        self.reset(ids)
        self.obs, self.critic = self._observe()
        extras = {"observations": {"critic": self.critic}, "time_outs": torch.as_tensor(time_out)}
        if log:  # rsl_rl reads the keys of the first log of an iteration, so never hand it an empty one
            extras["log"] = log
        return self.obs, rew, torch.as_tensor(done), extras


# ----------------------------------------------------------------- PPO
def train_cfg(iters: int) -> dict:
    return {
        "num_steps_per_env": 24, "max_iterations": iters, "save_interval": 100, "empirical_normalization": True,
        "logger": "tensorboard", "experiment_name": "safefall_mj", "run_name": "",
        "policy": {"class_name": "ActorCritic", "init_noise_std": 0.5, "actor_hidden_dims": [256, 128, 64],
                   "critic_hidden_dims": [256, 128, 64], "activation": "elu"},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
                      "entropy_coef": 0.005, "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 1.0e-3,
                      "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "desired_kl": 0.01, "max_grad_norm": 1.0},
    }


def render(env: SafeFallEnv, policy, path: str, per_dir: int = 1):
    """Side-view frames every 0.1 s for one episode per fall direction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = env.m
    r = mujoco.Renderer(m, 240, 240)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 1.8, -10
    rows = []
    for want in range(len(sf.DIRECTIONS)):  # fall directions (not the plank starts)
        while True:  # resample env 0 until it starts in the wanted direction
            env.reset(np.array([0]))
            if env.direction[0] == want:
                break
        env.obs, env.critic = env._observe()
        d = env.datas[0]
        w = np.zeros(9)
        mujoco.mju_quat2Mat(w, d.qpos[3:7])
        yaw0 = math.atan2(w[3], w[0])
        frames = []
        for t in range(int(1.6 / env.step_dt) + 1):
            if t % 5 == 0:
                cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]
                cam.azimuth = math.degrees(yaw0) + 90.0 + (90.0 if want in (2, 3) else 0.0)
                r.update_scene(d, cam)
                frames.append(r.render().copy())
            with torch.inference_mode():
                a = policy(env.obs) if policy is not None else torch.zeros(env.num_envs, env.num_actions)
            env.step(a)
            if env.episode_length_buf[0] == 0:
                break
        rows.append((sf.DIRECTION_NAMES[want], frames))
    cols = max(len(f) for _, f in rows)
    fig, ax = plt.subplots(len(rows), cols, figsize=(cols * 1.3, len(rows) * 1.4))
    for i, (name, frames) in enumerate(rows):
        for j in range(cols):
            ax[i, j].axis("off")
            if j < len(frames):
                ax[i, j].imshow(frames[j])
                ax[i, j].set_title(f"{name} {j * 0.1:.1f}s" if j == 0 else f"{j * 0.1:.1f}s", fontsize=6)
    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=80)
    print("wrote", path)


def release_rounds(env: SafeFallEnv, policy, episodes: int, release_after: float):
    """All envs start together; at ``release_after`` every actuator loses its position gain (the robot's
    damping mode, actuator damping kept). Prints outcomes before and after the release."""
    m = env.m
    gain, bias = m.actuator_gainprm[:, 0].copy(), m.actuator_biasprm[:, 1].copy()
    before, after = sf.OutcomeTracker(env.num_envs), sf.OutcomeTracker(env.num_envs)
    eps_before, eps_after = [], []
    while len(eps_before) < episodes:
        env.reset(np.arange(env.num_envs))
        env.obs, env.critic = env._observe()
        for t in range(env.max_episode_length - 1):  # stop before the env's own time-out reset
            released = t * env.step_dt >= release_after
            m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = (0.0, 0.0) if released else (gain, bias)
            with torch.inference_mode():
                act = policy(env.obs) if policy is not None else torch.zeros(env.num_envs, env.num_actions)
            env.step(act)
            st = env.last_state
            (after if released else before).update(st)
        ids = torch.arange(env.num_envs)
        dirs = torch.as_tensor(env.direction)
        eps_before += before.close(ids, dirs)
        eps_after += after.close(ids, dirs)
    m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = gain, bias
    print(json.dumps({"before_release": sf.summarize(eps_before[:episodes]),
                      "after_release": sf.summarize(eps_after[:episodes])}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--mesh_dir", default=os.path.expanduser("~/vigor_work/GMR/assets/booster_k1/meshes"))
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--out", default=os.path.expanduser("~/vigor_work/safefall_mj"))
    p.add_argument("--ckpt", default=None)
    p.add_argument("--passive", action="store_true", help="eval zero actions: the reflex alone (reach into the hand-off plank)")
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", default=None, help="PNG with side-view frames per fall direction")
    p.add_argument("--release_after", type=float, default=None,
                   help="damping mode (position gains 0, damping kept) after this many seconds; all envs in step")
    a = p.parse_args()

    from rsl_rl.runners import OnPolicyRunner

    if a.mode == "eval":
        sf.PLANK_START_PROB = 0.0  # falls only
    env = SafeFallEnv(a.envs, a.mesh_dir, seed=a.seed)
    runner = OnPolicyRunner(env, train_cfg(a.iters), log_dir=a.out if a.mode == "train" else None, device="cpu")
    if a.mode == "train":
        runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
        return
    policy = None
    if not a.passive:
        runner.load(a.ckpt, load_optimizer=False)
        policy = runner.get_inference_policy(device="cpu")
    env.passive = a.passive
    if a.release_after is not None:
        release_rounds(env, policy, a.episodes, a.release_after)
        return
    eps = []
    obs, _ = env.get_observations()
    with torch.inference_mode():
        while len(eps) < a.episodes:
            act = policy(obs) if policy is not None else torch.zeros(env.num_envs, env.num_actions)
            obs, _, _, _ = env.step(act)
            eps += env.finished
    print(json.dumps(sf.summarize(eps[: a.episodes]), indent=1))
    if a.render:
        render(env, policy, a.render)


if __name__ == "__main__":
    main()
