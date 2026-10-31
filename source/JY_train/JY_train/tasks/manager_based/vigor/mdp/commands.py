"""Sparse-keyframe recovery command: reference timeline, goal features and episode initialisation."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass

from .. import spec
from ..keyframes import KeyframeDataset
from ..torch_math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_yaw,
    quat_mul,
    quat_to_6d,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REGIME_FALL, REGIME_STANDUP, REGIME_HANDOVER, REGIME_PUSH = 0, 1, 2, 3
REGIME_NAMES = ["fall", "standup", "handover", "push"]


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


class VigorKeyframeCommand(CommandTerm):
    """Per env: a clip, a fractional frame cursor and a fixed yaw alignment of the reference.

    The reference is compared with the robot in the world frame after rotating it by
    the episode's yaw offset (paper: root-relative p_i - p_0), so no per-frame heading
    is ever extracted -- that is ill-defined when the trunk lies face up/down.

    ``command`` returns the goal features k_t of the next ``NUM_FUTURE_KEYFRAMES``
    keyframes in the robot's root frame.  Episodes start in one of four regimes:
    near fall onset of a clip (paper Fall-Recovery), around the lowest pose (paper
    Stand-Up), the walking stance tipped past ``ACTIVATION_TILT_DEG`` (hand-off from
    the walking policy) and the upright walking stance hit by a push.  The last two
    prefer "caught" demo segments so the policy learns to brace with its arms.
    """

    cfg: VigorKeyframeCommandCfg

    def __init__(self, cfg: VigorKeyframeCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.data = KeyframeDataset(cfg.dataset, device=self.device, strict_hash=cfg.strict_dataset_hash)
        print(f"[VIGOR] {self.data.summary()}")

        names = self.robot.joint_names
        missing = [n for n in spec.ALL_JOINT_NAMES if n not in names]
        if missing:
            raise RuntimeError(f"Robot is missing joints {missing}; expected the 22-DoF K1_22dof_jy model.")
        # dataset (canonical) -> articulation order
        self.joint_map = torch.tensor([spec.ALL_JOINT_NAMES.index(n) for n in names], device=self.device)
        self.head_ids = torch.tensor([names.index(n) for n in spec.HEAD_JOINT_NAMES], device=self.device)
        self.action_joint_ids = torch.tensor([names.index(n) for n in spec.ACTION_JOINT_NAMES], device=self.device)
        self.body_ids = torch.tensor(
            self.robot.find_bodies(spec.KEY_BODY_NAMES, preserve_order=True)[0], device=self.device
        )

        n = self.num_envs
        self.clip = torch.zeros(n, dtype=torch.long, device=self.device)
        self.cursor = torch.zeros(n, device=self.device)
        self.speed = torch.ones(n, device=self.device)
        self.q_off = torch.zeros(n, 4, device=self.device)
        self.q_off[:, 0] = 1.0
        self.regime = torch.zeros(n, dtype=torch.long, device=self.device)
        self._goal = torch.zeros(n, spec.GOAL_DIM, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(n, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(n, device=self.device)
        self.metrics["clip_progress"] = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------------ views
    @property
    def command(self) -> torch.Tensor:
        return self._goal

    @property
    def frame(self) -> torch.Tensor:
        """Global dataset index of the current reference frame."""
        return self.data.global_index(self.clip, self.cursor.floor().long())

    def _align(self, v: torch.Tensor) -> torch.Tensor:
        """Rotate (N, B, 3) reference vectors by the episode's yaw offset."""
        return quat_apply(self.q_off.unsqueeze(1).expand(-1, v.shape[1], -1), v)

    def ref_rel_body_pos(self) -> torch.Tensor:
        return self._align(self.data.rel_body_pos[self.frame])

    def ref_body_quat(self) -> torch.Tensor:
        q = self.data.body_quat[self.frame]
        return quat_mul(self.q_off.unsqueeze(1).expand(-1, q.shape[1], -1), q)

    def ref_body_lin_vel(self) -> torch.Tensor:
        return self._align(self.data.body_lin_vel[self.frame]) * self.speed.view(-1, 1, 1)

    def ref_body_ang_vel(self) -> torch.Tensor:
        return self._align(self.data.body_ang_vel[self.frame]) * self.speed.view(-1, 1, 1)

    def robot_rel_body_pos(self) -> torch.Tensor:
        d = self.robot.data
        return d.body_pos_w[:, self.body_ids] - d.root_pos_w.unsqueeze(1)

    def _compute_goal(self) -> None:
        d = self.data
        local, gidx = d.future_keyframes(self.clip, self.cursor, spec.NUM_FUTURE_KEYFRAMES)
        robot_q = self.robot.data.root_quat_w
        nb = len(spec.KEY_BODY_NAMES) - 1
        feats = []
        for k in range(spec.NUM_FUTURE_KEYFRAMES):
            g = gidx[:, k]
            target = self._align(d.rel_body_pos[g, 1:])
            target_b = quat_apply_inverse(robot_q.unsqueeze(1).expand(-1, nb, -1), target)
            ref_root = quat_mul(self.q_off, d.root_quat[g])
            dt = (local[:, k].float() - self.cursor).clamp_min(0.0) / (d.fps * self.speed)
            feats += [
                target_b.reshape(self.num_envs, -1),
                quat_to_6d(quat_mul(quat_conjugate(robot_q), ref_root)),
                d.root_height[g],
                dt.unsqueeze(-1),
            ]
        self._goal = torch.cat(feats, dim=-1)

    # -------------------------------------------------------------- manager API
    def _update_metrics(self):
        err = (self.ref_rel_body_pos() - self.robot_rel_body_pos()).norm(dim=-1).mean(dim=-1)
        self.metrics["error_body_pos"] = err
        ref_q = self.data.joint_pos[self.frame][:, self.joint_map]
        self.metrics["error_joint_pos"] = (ref_q - self.robot.data.joint_pos)[:, self.action_joint_ids].abs().mean(-1)
        self.metrics["clip_progress"] = self.cursor / self.data.clip_length[self.clip].float()

    def _update_command(self):
        self.cursor += self.speed * self.data.fps * self._env.step_dt
        self.cursor = torch.minimum(self.cursor, (self.data.clip_length[self.clip] - 1).float())
        self._compute_goal()

    def _sample_clips(self, regime: torch.Tensor) -> torch.Tensor:
        d, n = self.data, len(regime)
        clip = torch.randint(0, d.num_clips, (n,), device=self.device)
        if len(d.brace_clips):
            upright_start = (regime == REGIME_HANDOVER) | (regime == REGIME_PUSH)
            pick = upright_start & (torch.rand(n, device=self.device) < self.cfg.brace_clip_prob)
            idx = torch.randint(0, len(d.brace_clips), (n,), device=self.device)
            clip = torch.where(pick, d.brace_clips[idx], clip)
        return clip

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        cfg, d, n = self.cfg, self.data, len(env_ids)

        probs = torch.tensor(cfg.regime_probs, device=self.device)
        regime = torch.multinomial(probs / probs.sum(), n, replacement=True)
        clip = self._sample_clips(regime)
        speed = _uniform(*cfg.playback_speed_range, n, self.device)
        fps = float(d.fps)

        onset = d.fall_onset[clip].float()
        cursor = onset + fps * _uniform(*cfg.fall_onset_window_s, n, self.device)
        stand = regime == REGIME_STANDUP
        cursor[stand] = d.lowest[clip[stand]].float() + fps * _uniform(*cfg.lowest_window_s, int(stand.sum()), self.device)
        cursor[regime == REGIME_HANDOVER] = onset[regime == REGIME_HANDOVER]
        push = regime == REGIME_PUSH
        cursor[push] = onset[push] - fps * cfg.push_lead_s
        cursor = torch.minimum(cursor.clamp_min(0.0), (d.clip_length[clip] - 1).float())

        g = d.global_index(clip, cursor.floor().long())
        robot_yaw = _uniform(-math.pi, math.pi, n, self.device)
        q_off = quat_from_yaw(robot_yaw - d.root_yaw[g])

        # --- pose from the reference (fall / stand-up regimes); clips are grounded at z=0
        nb = len(spec.KEY_BODY_NAMES)
        root_quat = quat_mul(q_off, d.root_quat[g])
        root_lin_vel = quat_apply(q_off, d.body_lin_vel[g, 0]) * speed.unsqueeze(-1)
        root_ang_vel = quat_apply(q_off, d.body_ang_vel[g, 0]) * speed.unsqueeze(-1)
        joint_pos = d.joint_pos[g][:, self.joint_map]
        joint_vel = d.joint_vel[g][:, self.joint_map] * speed.unsqueeze(-1)
        # keypoints relative to the root, rotated into the env frame (terrain clearance check)
        body_rel_w = quat_apply(q_off.unsqueeze(1).expand(-1, nb, -1), d.rel_body_pos[g])
        root_z = d.root_pos[g, 2]
        root_xy_offset = torch.zeros(n, 2, device=self.device)

        # --- walking stance tipped about the feet (hand-off) or upright and pushed (push)
        up_start = (regime == REGIME_HANDOVER) | push
        if up_start.any():
            h = up_start.nonzero(as_tuple=True)[0]
            nh = len(h)
            is_push = push[h]
            lo = torch.where(is_push, cfg.push_tilt_deg[0], cfg.handover_tilt_deg[0])
            hi = torch.where(is_push, cfg.push_tilt_deg[1], cfg.handover_tilt_deg[1])
            tilt = torch.deg2rad(lo + (hi - lo) * torch.rand(nh, device=self.device))
            # tip / push in the direction this clip falls so the reference stays meaningful
            fall_dir = d.fall_dir[clip[h]] + torch.deg2rad(_uniform(-30.0, 30.0, nh, self.device))
            heading = quat_from_yaw(robot_yaw[h])
            zeros = torch.zeros_like(tilt)
            dir_w = quat_apply(heading, torch.stack((torch.cos(fall_dir), torch.sin(fall_dir), zeros), -1))
            axis = torch.stack((-dir_w[:, 1], dir_w[:, 0], zeros), -1)
            q_tilt = quat_from_axis_angle(axis, tilt)
            root_quat[h] = quat_mul(q_tilt, heading)
            up = torch.tensor([0.0, 0.0, spec.TRUNK_HEIGHT_NOMINAL], device=self.device).expand(nh, 3)
            pivot_to_root = quat_apply(q_tilt, up)
            root_z[h] = pivot_to_root[:, 2] + 0.09 * torch.sin(tilt)  # toe/heel dip below the pivot
            root_xy_offset[h] = pivot_to_root[:, :2]
            w_lo = torch.where(is_push, cfg.push_ang_speed[0], cfg.handover_ang_speed[0])
            w_hi = torch.where(is_push, cfg.push_ang_speed[1], cfg.handover_ang_speed[1])
            v_lo = torch.where(is_push, cfg.push_lin_speed[0], cfg.handover_lin_speed[0])
            v_hi = torch.where(is_push, cfg.push_lin_speed[1], cfg.handover_lin_speed[1])
            omega = axis * (w_lo + (w_hi - w_lo) * torch.rand(nh, device=self.device)).unsqueeze(-1)
            lin = dir_w * (v_lo + (v_hi - v_lo) * torch.rand(nh, device=self.device)).unsqueeze(-1)
            root_ang_vel[h] = omega
            root_lin_vel[h] = torch.cross(omega, pivot_to_root, dim=-1) + lin
            joint_pos[h] = self.robot.data.default_joint_pos[env_ids[h]]
            joint_vel[h] = 0.0
            body_rel_w[h] = -pivot_to_root.unsqueeze(1)  # clearance is checked under the feet

        # --- noise
        joint_pos = joint_pos + _uniform(-1.0, 1.0, n * spec.NUM_JOINTS, self.device).view(n, -1) * cfg.joint_pos_noise
        joint_pos[:, self.head_ids] = self.robot.data.default_joint_pos[env_ids][:, self.head_ids]
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
        joint_vel = joint_vel + _uniform(-1.0, 1.0, n * spec.NUM_JOINTS, self.device).view(n, -1) * cfg.joint_vel_noise
        root_lin_vel = root_lin_vel + _uniform(-1.0, 1.0, n * 3, self.device).view(n, 3) * cfg.root_lin_vel_noise
        root_ang_vel = root_ang_vel + _uniform(-1.0, 1.0, n * 3, self.device).view(n, 3) * cfg.root_ang_vel_noise

        # --- placement: env origin + xy noise, lifted above the terrain under every keypoint
        origin = self._env.scene.env_origins[env_ids]
        xy = origin[:, :2] + _uniform(-1.0, 1.0, n * 2, self.device).view(n, 2) * cfg.xy_noise + root_xy_offset
        body_xy = xy.unsqueeze(1) + body_rel_w[..., :2]
        terrain_rise = (self._terrain_height(body_xy) - origin[:, 2:3]).amax(dim=1).clamp_min(0.0)
        root_pos = torch.cat((xy, (origin[:, 2] + root_z + terrain_rise + cfg.spawn_margin).unsqueeze(-1)), dim=-1)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.cat((root_lin_vel, root_ang_vel), dim=-1), env_ids=env_ids)

        self.clip[env_ids], self.cursor[env_ids], self.speed[env_ids] = clip, cursor, speed
        self.q_off[env_ids], self.regime[env_ids] = q_off, regime
        self._compute_goal()

    # ------------------------------------------------------------------ helpers
    def _terrain_height(self, xy: torch.Tensor) -> torch.Tensor:
        """Terrain height under (N, K, 2) points via the height scanner's mesh; 0 if unavailable."""
        sensor: RayCaster | None = self._env.scene.sensors.get(self.cfg.height_sensor_name)
        meshes = getattr(sensor, "meshes", None) if sensor is not None else None
        if not meshes:
            return torch.zeros(xy.shape[:-1], device=self.device)
        from isaaclab.utils.warp import raycast_mesh

        mesh = next(iter(meshes.values()))
        starts = torch.cat((xy, torch.full_like(xy[..., :1], 20.0)), dim=-1)
        dirs = torch.zeros_like(starts)
        dirs[..., 2] = -1.0
        hits = raycast_mesh(starts.reshape(-1, 3).contiguous(), dirs.reshape(-1, 3).contiguous(), mesh, max_dist=100.0)[0]
        z = hits[..., 2].reshape(xy.shape[:-1])
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


@configclass
class VigorKeyframeCommandCfg(CommandTermCfg):
    class_type: type = VigorKeyframeCommand

    asset_name: str = MISSING
    dataset: str = "k1_fall_keyframes_v2.npz"
    strict_dataset_hash: bool = True
    height_sensor_name: str = "height_scanner"

    # fall / stand-up / hand-off from walking / push while standing
    regime_probs: tuple[float, float, float, float] = (0.2, 0.25, 0.35, 0.2)
    # hand-off and push starts draw a "caught" (brace) demo with this probability
    brace_clip_prob: float = 0.6
    fall_onset_window_s: tuple[float, float] = (-0.3, 0.6)
    lowest_window_s: tuple[float, float] = (-0.6, 0.4)
    handover_tilt_deg: tuple[float, float] = (12.0, 30.0)
    handover_ang_speed: tuple[float, float] = (0.5, 3.0)
    handover_lin_speed: tuple[float, float] = (-0.2, 0.8)
    push_tilt_deg: tuple[float, float] = (0.0, 6.0)
    push_ang_speed: tuple[float, float] = (0.0, 1.5)
    push_lin_speed: tuple[float, float] = (0.6, 1.8)
    push_lead_s: float = 0.2  # reference starts this long before its fall onset
    playback_speed_range: tuple[float, float] = (0.8, 1.1)

    joint_pos_noise: float = 0.1
    joint_vel_noise: float = 0.3
    root_lin_vel_noise: float = 0.2
    root_ang_vel_noise: float = 0.3
    xy_noise: float = 0.2
    spawn_margin: float = 0.03

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
