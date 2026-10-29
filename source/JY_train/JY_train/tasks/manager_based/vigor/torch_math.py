"""Quaternion helpers (w, x, y, z) without Isaac Lab imports.

Kept dependency-free so the same code runs in the env, the walk -> fall runner
and the laptop-side unit tests.
"""

from __future__ import annotations

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by ``q``; broadcasts over leading dims."""
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), v)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    return quat_from_yaw(yaw_from_quat(q))


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    half = 0.5 * angle.unsqueeze(-1)
    return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


def quat_error_magnitude(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation angle (rad) between ``a`` and ``b``."""
    d = quat_mul(quat_conjugate(a), b)
    return 2.0 * torch.atan2(d[..., 1:].norm(dim=-1), d[..., 0].abs())


def heading_relative_positions(body_pos: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """(N,B,3) world positions -> root-relative positions in the root heading (yaw) frame."""
    rel = body_pos - root_pos.unsqueeze(1)
    yq = yaw_quat(root_quat).unsqueeze(1).expand(-1, body_pos.shape[1], -1)
    return quat_apply_inverse(yq, rel)


def heading_relative_quats(body_quat: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """(N,B,4) world orientations -> orientations in the root heading frame."""
    yq = yaw_quat(root_quat).unsqueeze(1).expand(-1, body_quat.shape[1], -1)
    return quat_mul(quat_conjugate(yq), body_quat)


def heading_relative_vectors(vec: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """(N,B,3) world vectors (e.g. velocities) -> root heading frame."""
    yq = yaw_quat(root_quat).unsqueeze(1).expand(-1, vec.shape[1], -1)
    return quat_apply_inverse(yq, vec)


def projected_gravity(q: torch.Tensor) -> torch.Tensor:
    g = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    g[..., 2] = -1.0
    return quat_apply_inverse(q, g)


def tilt_angle(gravity_b: torch.Tensor) -> torch.Tensor:
    """Angle (rad) between the body z-axis and world up, from projected gravity."""
    return torch.acos((-gravity_b[..., 2]).clamp(-1.0, 1.0))
