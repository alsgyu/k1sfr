"""Minimal vectorised URDF forward kinematics (numpy only).

Used offline to compute keyframe body poses from the *same* URDF that Isaac
Lab spawns, so reference link frames match ``body_pos_w`` / ``body_quat_w``.
Quaternions are (w, x, y, z), matching Isaac Lab.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axis_angle_to_matrix(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """axis (3,), angle (T,) -> (T, 3, 3)."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    m = np.empty(angle.shape + (3, 3))
    m[..., 0, 0] = c + x * x * C
    m[..., 0, 1] = x * y * C - z * s
    m[..., 0, 2] = x * z * C + y * s
    m[..., 1, 0] = y * x * C + z * s
    m[..., 1, 1] = c + y * y * C
    m[..., 1, 2] = y * z * C - x * s
    m[..., 2, 0] = z * x * C - y * s
    m[..., 2, 1] = z * y * C + x * s
    m[..., 2, 2] = c + z * z * C
    return m


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) wxyz with w >= 0."""
    shape = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.empty((m.shape[0], 4))
    tr = np.trace(m, axis1=1, axis2=2)
    for i in range(m.shape[0]):
        a = m[i]
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * s, (a[2, 1] - a[1, 2]) / s, (a[0, 2] - a[2, 0]) / s, (a[1, 0] - a[0, 1]) / s]
        elif a[0, 0] > a[1, 1] and a[0, 0] > a[2, 2]:
            s = np.sqrt(1.0 + a[0, 0] - a[1, 1] - a[2, 2]) * 2
            q[i] = [(a[2, 1] - a[1, 2]) / s, 0.25 * s, (a[0, 1] + a[1, 0]) / s, (a[0, 2] + a[2, 0]) / s]
        elif a[1, 1] > a[2, 2]:
            s = np.sqrt(1.0 + a[1, 1] - a[0, 0] - a[2, 2]) * 2
            q[i] = [(a[0, 2] - a[2, 0]) / s, (a[0, 1] + a[1, 0]) / s, 0.25 * s, (a[1, 2] + a[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + a[2, 2] - a[0, 0] - a[1, 1]) * 2
            q[i] = [(a[1, 0] - a[0, 1]) / s, (a[0, 2] + a[2, 0]) / s, (a[1, 2] + a[2, 1]) / s, 0.25 * s]
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q.reshape(shape + (4,))


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin_pos: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class URDFModel:
    def __init__(self, urdf_path: str):
        root = ET.parse(urdf_path).getroot()
        self.links = [l.get("name") for l in root.findall("link")]
        self.joints: dict[str, Joint] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([1.0, 0, 0])
            lim = j.find("limit")
            lower = float(lim.get("lower", "-inf")) if lim is not None else -np.inf
            upper = float(lim.get("upper", "inf")) if lim is not None else np.inf
            self.joints[j.get("name")] = Joint(
                j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                xyz, rpy_to_matrix(rpy), axis, lower, upper,
            )
        children = {j.child for j in self.joints.values()}
        roots = [l for l in self.links if l not in children]
        assert len(roots) == 1, roots
        self.root = roots[0]
        self.joint_of_child = {j.child: j for j in self.joints.values()}
        # topological order of links
        order, frontier = [self.root], [self.root]
        while frontier:
            nxt = [j.child for j in self.joints.values() if j.parent in frontier]
            order += nxt
            frontier = nxt
        self.link_order = order
        self.actuated = [n for n, j in self.joints.items() if j.type in ("revolute", "continuous")]

    def limits(self, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([self.joints[n].lower for n in names]), np.array([self.joints[n].upper for n in names]))

    def fk(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray, joint_names: list[str]):
        """root_pos (T,3), root_quat (T,4 wxyz), joint_pos (T,J) -> {link: (pos (T,3), rotmat (T,3,3))}."""
        T = root_pos.shape[0]
        q = {n: joint_pos[:, i] for i, n in enumerate(joint_names)}
        out = {self.root: (root_pos.copy(), quat_to_matrix(root_quat))}
        for link in self.link_order[1:]:
            j = self.joint_of_child[link]
            p_pos, p_rot = out[j.parent]
            rot = p_rot @ j.origin_rot
            pos = p_pos + np.einsum("tij,j->ti", p_rot, j.origin_pos)
            if j.type in ("revolute", "continuous"):
                angle = q.get(j.name, np.zeros(T))
                rot = rot @ axis_angle_to_matrix(j.axis, angle)
            out[link] = (pos, rot)
        return out
