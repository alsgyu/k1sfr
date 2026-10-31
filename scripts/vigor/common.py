"""Helpers shared by the VIGOR scripts.  Nothing here imports Isaac at module level."""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
VIGOR_DIR = REPO / "source/JY_train/JY_train/tasks/manager_based/vigor"


def load_vigor_module(name: str):
    """Import spec / networks without importing JY_train (which needs Isaac)."""
    if name == "networks":
        path = VIGOR_DIR / "modules/networks.py"
    else:
        path = VIGOR_DIR / f"{name}.py"
    key = f"_vigor_{name}"
    if key not in sys.modules:
        s = importlib.util.spec_from_file_location(key, path)
        m = importlib.util.module_from_spec(s)
        sys.modules[key] = m
        s.loader.exec_module(m)
    return sys.modules[key]


def resolve_checkpoint(log_root: str, run: str | None = None, checkpoint: str | None = None) -> str:
    """Latest ``model_<it>.pt`` in the latest (or given/regex) run under ``log_root``; absolute paths pass through."""
    if checkpoint and os.path.isfile(checkpoint):
        return os.path.abspath(checkpoint)
    runs = sorted(d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d)))
    if run:
        runs = [d for d in runs if re.search(run, d)]
    if not runs:
        raise FileNotFoundError(f"no runs matching {run!r} in {log_root}")
    run_dir = os.path.join(log_root, runs[-1])
    pattern = checkpoint or "model_*.pt"
    files = glob.glob(os.path.join(run_dir, pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {run_dir}")
    it = lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[-1]) if re.findall(r"(\d+)", os.path.basename(f)) else -1  # noqa: E731
    return max(files, key=it)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ teacher
class Teacher:
    """Inference wrapper around a VigorTeacherActorCritic rsl_rl checkpoint."""

    def __init__(self, path: str, num_obs: int, num_critic_obs: int, num_actions: int, policy_cfg: dict, device: str):
        from rsl_rl.modules import EmpiricalNormalization

        from JY_train.tasks.manager_based.vigor.modules import VigorTeacherActorCritic

        cfg = dict(policy_cfg)
        cfg.pop("class_name", None)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.policy = VigorTeacherActorCritic(num_obs, num_critic_obs, num_actions, **cfg).to(device).eval()
        self.policy.load_state_dict(ckpt["model_state_dict"])
        self.normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(device).eval()
        if "obs_norm_state_dict" in ckpt:
            self.normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
        else:
            print("[VIGOR] teacher checkpoint has no observation normalizer; using raw observations")
            self.normalizer = torch.nn.Identity()
        self.path = path

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.normalizer(obs)
        z = self.policy.actor.encode(x)
        return self.policy.actor(x), z

    def proprio_stats(self, proprio_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.normalizer, torch.nn.Identity):
            return torch.zeros(proprio_dim), torch.ones(proprio_dim)
        mean = self.normalizer._mean.reshape(-1)[:proprio_dim]
        std = self.normalizer._std.reshape(-1)[:proprio_dim] + self.normalizer.eps
        return mean.detach().cpu(), std.detach().cpu()


def student_layout_stats(mean: torch.Tensor, std: torch.Tensor, term_dims: list[int], history: int):
    """Per-frame proprio stats -> Isaac Lab term-major flattened history layout."""
    ms, ss, start = [], [], 0
    for d in term_dims:
        ms.append(mean[start : start + d].repeat(history))
        ss.append(std[start : start + d].repeat(history))
        start += d
    return torch.cat(ms), torch.cat(ss)


# ------------------------------------------------------------------ metrics
class RecoveryTracker:
    """Per-episode recovery metrics (paper Table II spirit), split by start regime.

    success  : upright (tilt < 25 deg, head above 0.75 x nominal) for the last ``stable_s`` of the episode
    full_fall: head went below ``spec.FULL_FALL_HEAD_HEIGHT`` -- the robot did not catch itself
    caught   : success without a full fall (the behaviour we want after a push / at hand-off)
    time     : start of the final upright stretch (0 when it never lost uprightness)
    head     : any contact on the camera head (paper Fig. 9 'unsafe' marker)
    """

    def __init__(self, num_envs: int, dt: float, device: str, full_fall_height: float, stable_s: float = 0.5):
        self.dt, self.stable_s, self.full_fall_height = dt, stable_s, full_fall_height
        z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        self.t, self.run, self.peak_force = z(), z(), z()
        self.min_head = torch.full((num_envs,), float("inf"), device=device)
        self.head_hit = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episodes: list[dict] = []

    def step(self, upright, head_height, head_contact, max_contact_force, done, regime_names=None):
        """Call right after env.step().  Envs flagged ``done`` were already reset inside that step, so
        their episode is closed from the accumulated values and the current reading starts the next one."""
        for i in done.nonzero(as_tuple=True)[0].tolist():
            ok = bool(self.run[i] >= self.stable_s)
            fell = bool(self.min_head[i] < self.full_fall_height)
            self.episodes.append({
                "regime": regime_names[i] if regime_names else "all",
                "success": ok,
                "full_fall": fell,
                "caught": ok and not fell,
                "time": (self.t[i] - self.run[i]).item() if ok else None,
                "min_head_height": self.min_head[i].item(),
                "head_contact": bool(self.head_hit[i]),
                "peak_contact_force": self.peak_force[i].item(),
            })
        self.t[done] = 0.0
        self.run[done] = 0.0
        self.peak_force[done] = 0.0
        self.min_head[done] = float("inf")
        self.head_hit[done] = False
        self.t += self.dt
        self.run = torch.where(upright, self.run + self.dt, torch.zeros_like(self.run))
        self.min_head = torch.minimum(self.min_head, head_height)
        self.head_hit |= head_contact
        self.peak_force = torch.maximum(self.peak_force, max_contact_force)

    @staticmethod
    def _summarise(e: list[dict]) -> dict:
        n = len(e)
        if n == 0:
            return {"episodes": 0}
        times = [x["time"] for x in e if x["success"]]
        return {
            "episodes": n,
            "SR": sum(x["success"] for x in e) / n,
            "caught": sum(x["caught"] for x in e) / n,
            "full_fall": sum(x["full_fall"] for x in e) / n,
            "time_s": sum(times) / len(times) if times else None,
            "head_contact_rate": sum(x["head_contact"] for x in e) / n,
            "peak_contact_force_kN": sum(x["peak_contact_force"] for x in e) / n / 1000.0,
        }

    def summary(self) -> dict:
        out = self._summarise(self.episodes)
        regimes = sorted({x["regime"] for x in self.episodes})
        if len(regimes) > 1:
            out["by_regime"] = {r: self._summarise([x for x in self.episodes if x["regime"] == r]) for r in regimes}
        return out


# ------------------------------------------------------------------- export
def export_student(ckpt_path: str, out_dir: str | None = None) -> dict:
    """Student checkpoint -> TorchScript + ONNX + policy_spec.json (Isaac-free)."""
    networks = load_vigor_module("networks")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    deploy, cfg = ckpt["deploy"], ckpt["distill_cfg"]
    student = networks.VigorStudent(
        deploy["term_dims"], deploy["history"], len(deploy["action_joint_names"]), cfg["latent_dim"],
        cfg["actor_hidden_dims"], cfg["fusion_hidden_dims"], cfg["activation"],
    )
    student.load_state_dict(ckpt["student_state_dict"])
    model = networks.NormalizedActor(student, ckpt["obs_mean"], ckpt["obs_std"]).eval()
    obs_dim = int(ckpt["obs_mean"].numel())
    example = torch.zeros(1, obs_dim)
    out_dir = out_dir or os.path.join(os.path.dirname(ckpt_path), "exported")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"vigor_student_{ckpt['iteration']}"
    pt, onnx_path = os.path.join(out_dir, f"{stem}.pt"), os.path.join(out_dir, f"{stem}.onnx")
    with torch.no_grad():
        torch.jit.trace(model, example).save(pt)
        kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(model, example, onnx_path, input_names=["obs"], output_names=["actions"],
                          dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17, **kwargs)
    spec = dict(deploy)
    spec.update({
        "spec_version": ckpt["spec_version"],
        "source_checkpoint": os.path.abspath(ckpt_path),
        "teacher_checkpoint": ckpt.get("teacher_checkpoint"),
        "obs_dim": obs_dim,
        "files": {os.path.basename(p): sha256_file(p) for p in (pt, onnx_path)},
        "obs_layout": "per term (spec.PROPRIO_TERMS order): history x dim row-major, oldest frame first; terms concatenated",
        "action": "joint target = default_joint_pos[action joint] + action_scale * action (action_joint_names order)",
    })
    spec_path = os.path.join(out_dir, f"{stem}_spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[VIGOR] exported {pt}\n[VIGOR] exported {onnx_path}\n[VIGOR] spec     {spec_path}")
    return {"pt": pt, "onnx": onnx_path, "spec": spec_path}
