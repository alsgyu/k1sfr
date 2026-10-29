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
    assert spec.PROPRIO_DIM == 70 and spec.GOAL_DIM == 68 and spec.TERRAIN_DIM == 226
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
    # heading-relative features are invariant to a global yaw rotation
    pos = torch.randn(8, 10, 3)
    root_q = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    root_p = torch.randn(8, 3)
    rot = tm.quat_from_yaw(torch.full((8,), 0.7))
    pos2 = tm.quat_apply(rot.unsqueeze(1).expand(-1, 10, -1), pos)
    a = tm.heading_relative_positions(pos, root_p, root_q)
    b = tm.heading_relative_positions(pos2, tm.quat_apply(rot, root_p), tm.quat_mul(rot, root_q))
    assert torch.allclose(a, b, atol=1e-5)
    g = tm.projected_gravity(tm.quat_from_axis_angle(torch.tensor([[1.0, 0, 0]]), torch.tensor([math.radians(20)])))
    assert abs(math.degrees(tm.tilt_angle(g).item()) - 20) < 1e-3


def test_dataset():
    keyframes = _load("keyframes", VIGOR / "keyframes.py")
    d = keyframes.KeyframeDataset("k1_fall_keyframes_v1.npz", strict_hash=True)
    print("   ", d.summary())
    assert d.rel_body_pos.shape == (d.num_frames, len(spec.KEY_BODY_NAMES), 3)
    assert d.rel_body_pos[:, 0].abs().max() < 1e-5  # root at origin
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
