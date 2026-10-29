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
    heading_relative_positions,
    quat_apply,
    quat_from_axis_angle,
    quat_from_yaw,
    quat_mul,
    yaw_from_quat,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

REGIME_FALL, REGIME_STANDUP, REGIME_HANDOVER = 0, 1, 2


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


class VigorKeyframeCommand(CommandTerm):
    """Per env: a clip, a fractional frame cursor and a yaw offset of the reference.

    ``command`` returns the goal features k_t of the next ``NUM_FUTURE_KEYFRAMES``
    keyframes.  Episodes are initialised here (after reset events) in one of
    three regimes: near fall onset of a clip (paper Fall-Recovery), around the
    lowest pose (paper Stand-Up) and a tilted walking stance that mimics the
    hand-off from the walking policy at ``ACTIVATION_TILT_DEG``.
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
        self.head_body_id = self.robot.find_bodies("Head_2")[0][0]

        n = self.num_envs
        self.clip = torch.zeros(n, dtype=torch.long, device=self.device)
        self.cursor = torch.zeros(n, device=self.device)
        self.speed = torch.ones(n, device=self.device)
        self.yaw_offset = torch.zeros(n, device=self.device)
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

    @property
    def finished(self) -> torch.Tensor:
        """Reference has reached its final standing pose."""
        return self.cursor >= (self.data.getup_end[self.clip]).float()

    def robot_rel_body_pos(self) -> torch.Tensor:
        d = self.robot.data
        return heading_relative_positions(d.body_pos_w[:, self.body_ids], d.root_pos_w, d.root_quat_w)

    def _compute_goal(self) -> None:
        d = self.data
        local, gidx = d.future_keyframes(self.clip, self.cursor, spec.NUM_FUTURE_KEYFRAMES)
        robot_yaw = yaw_from_quat(self.robot.data.root_quat_w)
        feats = []
        for k in range(spec.NUM_FUTURE_KEYFRAMES):
            g = gidx[:, k]
            dyaw = d.root_yaw[g] + self.yaw_offset - robot_yaw
            dt = (local[:, k].float() - self.cursor).clamp_min(0.0) / (d.fps * self.speed)
            feats += [
                d.rel_body_pos[g, 1:].reshape(self.num_envs, -1),
                d.root_gravity[g],
                d.root_height[g],
                torch.stack((torch.cos(dyaw), torch.sin(dyaw)), dim=-1),
                dt.unsqueeze(-1),
            ]
        self._goal = torch.cat(feats, dim=-1)

    # -------------------------------------------------------------- manager API
    def _update_metrics(self):
        g = self.frame
        err = (self.data.rel_body_pos[g] - self.robot_rel_body_pos()).norm(dim=-1).mean(dim=-1)
        self.metrics["error_body_pos"] = err
        ref_q = self.data.joint_pos[g][:, self.joint_map]
        self.metrics["error_joint_pos"] = (ref_q - self.robot.data.joint_pos)[:, self.action_joint_ids].abs().mean(-1)
        self.metrics["clip_progress"] = self.cursor / self.data.clip_length[self.clip].float()

    def _update_command(self):
        self.cursor += self.speed * self.data.fps * self._env.step_dt
        self.cursor = torch.minimum(self.cursor, (self.data.clip_length[self.clip] - 1).float())
        self._compute_goal()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        cfg, d, n = self.cfg, self.data, len(env_ids)

        probs = torch.tensor(cfg.regime_probs, device=self.device)
        regime = torch.multinomial(probs / probs.sum(), n, replacement=True)
        clip = torch.randint(0, d.num_clips, (n,), device=self.device)
        speed = _uniform(*cfg.playback_speed_range, n, self.device)
        fps = float(d.fps)

        cursor = torch.empty(n, device=self.device)
        fall = regime == REGIME_FALL
        cursor[fall] = d.fall_onset[clip[fall]].float() + fps * _uniform(*cfg.fall_onset_window_s, int(fall.sum()), self.device)
        stand = regime == REGIME_STANDUP
        cursor[stand] = d.lowest[clip[stand]].float() + fps * _uniform(*cfg.lowest_window_s, int(stand.sum()), self.device)
        hand = regime == REGIME_HANDOVER
        cursor[hand] = d.fall_onset[clip[hand]].float()
        cursor = torch.minimum(cursor.clamp_min(0.0), (d.clip_length[clip] - 1).float())

        g = d.global_index(clip, cursor.floor().long())
        robot_yaw = _uniform(-math.pi, math.pi, n, self.device)
        yaw_offset = robot_yaw - d.root_yaw[g]
        q_off = quat_from_yaw(yaw_offset)

        # --- pose from the reference (fall / stand-up regimes); clips are grounded at z=0
        root_quat = quat_mul(q_off, d.root_quat[g])
        root_lin_vel = quat_apply(q_off, self._ref_root_lin_vel_w(g)) * speed.unsqueeze(-1)
        root_ang_vel = quat_apply(q_off, self._ref_root_ang_vel_w(g)) * speed.unsqueeze(-1)
        joint_pos = d.joint_pos[g][:, self.joint_map]
        joint_vel = d.joint_vel[g][:, self.joint_map] * speed.unsqueeze(-1)
        # keypoints relative to the root, rotated into the env frame (terrain clearance check)
        body_rel_w = quat_apply(q_off.unsqueeze(1).expand(-1, len(spec.KEY_BODY_NAMES), -1),
                                d.body_pos[g] - d.root_pos[g].unsqueeze(1))
        root_z = d.root_pos[g, 2]
        root_xy_offset = torch.zeros(n, 2, device=self.device)

        # --- walking stance tipped about the feet (hand-off regime)
        if hand.any():
            h = hand.nonzero(as_tuple=True)[0]
            nh = len(h)
            tilt = torch.deg2rad(_uniform(*cfg.handover_tilt_deg, nh, self.device))
            # tip in the direction this clip falls so the reference stays meaningful
            fall_dir = d.fall_dir[clip[h]] + torch.deg2rad(_uniform(-30.0, 30.0, nh, self.device))
            heading = quat_from_yaw(robot_yaw[h])
            dir_w = quat_apply(heading, torch.stack((torch.cos(fall_dir), torch.sin(fall_dir), torch.zeros_like(tilt)), -1))
            axis = torch.stack((-dir_w[:, 1], dir_w[:, 0], torch.zeros_like(tilt)), -1)
            q_tilt = quat_from_axis_angle(axis, tilt)
            root_quat[h] = quat_mul(q_tilt, heading)
            up = torch.tensor([0.0, 0.0, spec.TRUNK_HEIGHT_NOMINAL], device=self.device).expand(nh, 3)
            pivot_to_root = quat_apply(q_tilt, up)
            root_z[h] = pivot_to_root[:, 2] + 0.09 * torch.sin(tilt)  # toe/heel dip below the pivot
            root_xy_offset[h] = pivot_to_root[:, :2]
            omega = axis * _uniform(*cfg.handover_ang_speed, nh, self.device).unsqueeze(-1)
            walk_v = dir_w * _uniform(*cfg.handover_lin_speed, nh, self.device).unsqueeze(-1)
            root_ang_vel[h] = omega
            root_lin_vel[h] = torch.cross(omega, pivot_to_root, dim=-1) + walk_v
            joint_pos[h] = self.robot.data.default_joint_pos[env_ids[h]]
            joint_vel[h] = 0.0
            body_rel_w[h] = -pivot_to_root.unsqueeze(1)  # clearance is checked under the feet

        # yaw_offset maps reference headings onto the robot frame (also for the hand-off:
        # the robot starts facing along the reference heading at fall onset)

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
        self.yaw_offset[env_ids], self.regime[env_ids] = yaw_offset, regime
        self._compute_goal()

    # ------------------------------------------------------------------ helpers
    def _ref_root_lin_vel_w(self, g: torch.Tensor) -> torch.Tensor:
        # body_lin_vel is stored in the heading frame; rotate back with the ref heading
        return quat_apply(yaw_quat(self.data.root_quat[g]), self.data.rel_body_lin_vel[g, 0])

    def _ref_root_ang_vel_w(self, g: torch.Tensor) -> torch.Tensor:
        return quat_apply(yaw_quat(self.data.root_quat[g]), self.data.rel_body_ang_vel[g, 0])

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
    dataset: str = "k1_fall_keyframes_v1.npz"
    strict_dataset_hash: bool = True
    height_sensor_name: str = "height_scanner"

    # fall / stand-up / hand-off-from-walking initialisation
    regime_probs: tuple[float, float, float] = (0.4, 0.3, 0.3)
    fall_onset_window_s: tuple[float, float] = (-0.3, 0.6)
    lowest_window_s: tuple[float, float] = (-0.6, 0.4)
    handover_tilt_deg: tuple[float, float] = (12.0, 30.0)
    handover_ang_speed: tuple[float, float] = (0.5, 3.0)
    handover_lin_speed: tuple[float, float] = (-0.2, 0.8)
    playback_speed_range: tuple[float, float] = (0.8, 1.1)

    joint_pos_noise: float = 0.1
    joint_vel_noise: float = 0.3
    root_lin_vel_noise: float = 0.2
    root_ang_vel_noise: float = 0.3
    xy_noise: float = 0.2
    spawn_margin: float = 0.03

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
