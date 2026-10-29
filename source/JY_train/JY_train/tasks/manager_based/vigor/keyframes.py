"""Sparse-keyframe fall/recovery dataset (no Isaac Lab imports).

The ``.npz`` is produced offline by ``scripts/vigor/data/build_keyframes.py``.
All per-frame arrays hold the *coarse* reference: the retargeted human motion
re-interpolated from keyframes taken every ``stride`` frames, so tracking it is
only a structural prior (paper Sec. III-A).  Everything that does not depend on
the robot state is precomputed here once.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

try:
    from . import spec
    from .torch_math import (
        heading_relative_positions,
        heading_relative_quats,
        heading_relative_vectors,
        projected_gravity,
        yaw_from_quat,
    )
except ImportError:  # loaded by file path in laptop-side tests
    import spec
    from torch_math import (
        heading_relative_positions,
        heading_relative_quats,
        heading_relative_vectors,
        projected_gravity,
        yaw_from_quat,
    )

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_manifest.json")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6))
DATA_DIR = os.environ.get("JY_VIGOR_DATA_DIR", os.path.join(REPO_ROOT, "data", "vigor"))

REQUIRED_KEYS = (
    "spec_version", "fps", "stride", "joint_names", "body_names", "clip_names", "clip_start", "clip_length",
    "fall_onset", "lowest", "getup_end", "fall_dir", "joint_pos", "joint_vel", "root_pos", "root_quat",
    "body_pos", "body_quat", "body_lin_vel", "body_ang_vel",
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_dataset(name: str) -> tuple[str, str | None]:
    """Return (path, expected sha256 from the git-tracked manifest)."""
    path = name if os.path.isabs(name) else os.path.join(DATA_DIR, name)
    expected = None
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        entry = manifest.get("datasets", {}).get(os.path.basename(path))
        expected = entry["sha256"] if entry else None
    return path, expected


class KeyframeDataset:
    def __init__(self, name: str, device: str = "cpu", strict_hash: bool = True):
        path, expected = resolve_dataset(name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Keyframe dataset not found: {path}. Sync it from the data-prep machine "
                f"(scripts/vigor/sync.sh push-data) or set JY_VIGOR_DATA_DIR."
            )
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            msg = f"Keyframe dataset {path} sha256 {digest[:12]} != manifest {expected[:12]}; data and code are out of sync."
            if strict_hash:
                raise RuntimeError(msg)
            print(f"[WARN] {msg}")
        self.path, self.sha256 = path, digest

        data = np.load(path, allow_pickle=False)
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        if str(data["spec_version"]) != spec.SPEC_VERSION:
            raise RuntimeError(f"{path} was built for {data['spec_version']}, code expects {spec.SPEC_VERSION}")
        self.joint_names = [str(n) for n in data["joint_names"]]
        self.body_names = [str(n) for n in data["body_names"]]
        if self.joint_names != spec.ALL_JOINT_NAMES or self.body_names != spec.KEY_BODY_NAMES:
            raise RuntimeError("Dataset joint/body order differs from spec.ALL_JOINT_NAMES / KEY_BODY_NAMES")

        self.fps = int(data["fps"])
        self.stride = int(data["stride"])
        self.clip_names = [str(n) for n in data["clip_names"]]
        t = lambda k, dt=torch.float32: torch.as_tensor(np.asarray(data[k]), dtype=dt, device=device)  # noqa: E731
        self.clip_start = t("clip_start", torch.long)
        self.clip_length = t("clip_length", torch.long)
        self.fall_onset = t("fall_onset", torch.long)
        self.lowest = t("lowest", torch.long)
        self.getup_end = t("getup_end", torch.long)
        self.fall_dir = t("fall_dir")
        self.num_clips = len(self.clip_names)
        self.num_frames = int(data["joint_pos"].shape[0])

        self.joint_pos = t("joint_pos")
        self.joint_vel = t("joint_vel")
        self.root_pos = t("root_pos")
        self.root_quat = t("root_quat")
        body_pos, body_quat = t("body_pos"), t("body_quat")
        body_lin_vel, body_ang_vel = t("body_lin_vel"), t("body_ang_vel")

        # Heading-invariant reference quantities (paper: root-relative canonical frame).
        self.rel_body_pos = heading_relative_positions(body_pos, self.root_pos, self.root_quat)
        self.rel_body_quat = heading_relative_quats(body_quat, self.root_quat)
        self.rel_body_lin_vel = heading_relative_vectors(body_lin_vel, self.root_quat)
        self.rel_body_ang_vel = heading_relative_vectors(body_ang_vel, self.root_quat)
        self.root_gravity = projected_gravity(self.root_quat)
        self.root_yaw = yaw_from_quat(self.root_quat)
        # Pose-intrinsic root height: independent of where the clip was grounded.
        self.root_height = self.root_pos[:, 2:3] - body_pos[:, :, 2].min(dim=1, keepdim=True).values
        self.body_pos = body_pos

    def global_index(self, clip: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        local = torch.minimum(local.clamp_min(0), self.clip_length[clip] - 1)
        return self.clip_start[clip] + local

    def future_keyframes(self, clip: torch.Tensor, cursor: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Local frame indices of the next ``count`` keyframes after ``cursor`` (held at the clip end)."""
        base = torch.div(cursor.floor().long(), self.stride, rounding_mode="floor") + 1
        steps = torch.arange(count, device=cursor.device).unsqueeze(0)
        local = (base.unsqueeze(1) + steps) * self.stride
        local = torch.minimum(local, (self.clip_length[clip] - 1).unsqueeze(1))
        return local, self.clip_start[clip].unsqueeze(1) + local

    def summary(self) -> str:
        secs = self.clip_length.float() / self.fps
        return (
            f"{os.path.basename(self.path)} [{self.sha256[:12]}]: {self.num_clips} clips, {self.num_frames} frames, "
            f"{secs.min():.1f}-{secs.max():.1f}s per clip, keyframes every {self.stride / self.fps:.2f}s"
        )
