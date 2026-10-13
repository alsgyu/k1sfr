from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

## 추가 scratch 용
class INHAWalkCommand(CommandTerm):
    cfg: INHAWalkCommandCfg

    def __init__(self, cfg: INHAWalkCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 10, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command
    @property
    def gait_frequency(self) -> torch.Tensor:
        return self._command[:, 3]
    
    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.play:
            self._command[env_ids, 0:3] = torch.tensor(self.cfg.play_velocity_command, device=self.device)
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)
            return
        
        vr = self.cfg.velocity_ranges
        self._command[env_ids, 0] = sample_uniform(*vr.lin_vel_x, (len(env_ids), ), device=self.device)
        self._command[env_ids, 1] = sample_uniform(*vr.lin_vel_y, (len(env_ids),), device=self.device)
        self._command[env_ids, 2] = sample_uniform(*vr.ang_vel_yaw, (len(env_ids),), device=self.device)

        if self.cfg.sample_internal_parameters:
            pr = self.cfg.parameter_ranges
            self._command[env_ids, 3] = sample_uniform(*pr.gait_frequency, (len(env_ids),), device=self.device)
            self._command[env_ids, 4] = sample_uniform(*pr.foot_yaw_L, (len(env_ids),), device=self.device)
            self._command[env_ids, 5] = sample_uniform(*pr.foot_yaw_R, (len(env_ids),), device=self.device)
            self._command[env_ids, 6] = sample_uniform(*pr.body_pitch_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 7] = sample_uniform(*pr.body_roll_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 8] = sample_uniform(*pr.feet_offset_x_target, (len(env_ids),), device=self.device)
            self._command[env_ids, 9] = sample_uniform(*pr.feet_offset_y_target, (len(env_ids),), device=self.device)
        else:
            self._command[env_ids, 3:10] = torch.tensor(self.cfg.default_internal_parameters, device=self.device)

        num_still = int(self.cfg.still_proportion * len(env_ids))
        if num_still > 0:
            still_ids = env_ids[torch.randperm(len(env_ids), device=self.device)[:num_still]]
            self._command[still_ids, 0:4] = 0.0
            self.gait_process[still_ids] = 0.0
        
    def _update_command(self): # 보행 주기에서 현재 몇 %까지 진행됐는지를 매 스텝마다 업데이트 
        self.gait_process[:] = torch.fmod(
            self.gait_process + self._env.step_dt * self.gait_frequency, 
            1.0,
        )
    
    def _update_metrics(self):
        pass
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass
    def _debug_vis_callback(self, event):
        pass

class StagedCombinationWalkCommand(INHAWalkCommand):
    """Sample pure commands first, then expand to every non-empty axis subset."""

    cfg: StagedCombinationWalkCommandCfg

    # [vx, vy, yaw].  Stage two retains the pure modes to avoid forgetting.
    _PURE_MODES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    _ALL_MODES = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    )

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if self.cfg.play:
            self._command[env_ids, 0:3] = torch.tensor(
                self.cfg.play_velocity_command, device=self.device
            )
            self._command[env_ids, 3:10] = torch.tensor(
                self.cfg.default_internal_parameters, device=self.device
            )
            return

        iteration = int(self._env.common_step_counter) // max(
            self.cfg.num_steps_per_iteration, 1
        )
        modes = (
            self._PURE_MODES
            if iteration < self.cfg.pure_command_iterations
            else self._ALL_MODES
        )
        mode_table = torch.tensor(
            modes, dtype=self._command.dtype, device=self.device
        )
        mode_ids = torch.randint(
            len(modes), (len(env_ids),), device=self.device
        )
        active_axes = mode_table[mode_ids]

        vr = self.cfg.velocity_ranges
        sampled = torch.stack(
            (
                sample_uniform(
                    *vr.lin_vel_x, (len(env_ids),), device=self.device
                ),
                sample_uniform(
                    *vr.lin_vel_y, (len(env_ids),), device=self.device
                ),
                sample_uniform(
                    *vr.ang_vel_yaw, (len(env_ids),), device=self.device
                ),
            ),
            dim=1,
        )
        self._command[env_ids, 0:3] = sampled * active_axes

        if self.cfg.sample_internal_parameters:
            pr = self.cfg.parameter_ranges
            ranges = (
                pr.gait_frequency,
                pr.foot_yaw_L,
                pr.foot_yaw_R,
                pr.body_pitch_target,
                pr.body_roll_target,
                pr.feet_offset_x_target,
                pr.feet_offset_y_target,
            )
            for command_index, value_range in enumerate(ranges, start=3):
                self._command[env_ids, command_index] = sample_uniform(
                    *value_range, (len(env_ids),), device=self.device
                )
        else:
            self._command[env_ids, 3:10] = torch.tensor(
                self.cfg.default_internal_parameters, device=self.device
            )

        # Keep the original explicit stand-command fraction in both stages.
        num_still = int(self.cfg.still_proportion * len(env_ids))
        if num_still > 0:
            still_ids = env_ids[
                torch.randperm(len(env_ids), device=self.device)[:num_still]
            ]
            self._command[still_ids, 0:4] = 0.0
            self.gait_process[still_ids] = 0.0


class WeightedStagedCombinationWalkCommand(StagedCombinationWalkCommand):
    """Keep the pure stage, then curriculum the 1/2/3-axis topology ratios.

    ``pure_command_iterations`` is measured from training iteration zero.  The
    topology curriculum begins only after that pure stage, and its stage
    boundaries are measured locally from the all-mode transition.
    """

    cfg: WeightedStagedCombinationWalkCommandCfg

    _DOUBLE_MODES = (
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    )

    def _topology_probabilities(
        self, all_mode_iteration: int
    ) -> tuple[float, float, float]:
        if all_mode_iteration < self.cfg.topology_stage_1_iterations:
            return (0.3, 0.4, 0.3)
        if all_mode_iteration < self.cfg.topology_stage_2_iterations:
            return (0.20, 0.4, 0.4)
        if all_mode_iteration < self.cfg.topology_stage_3_iterations:
            return (0.15, 0.3, 0.55)
        return (0.1, 0.3, 0.60)

    def _sample_active_axes(
        self,
        count: int,
        iteration: int,
    ) -> torch.Tensor:
        pure_modes = self._command.new_tensor(self._PURE_MODES)

        if iteration < self.cfg.pure_command_iterations:
            mode_ids = torch.randint(
                len(self._PURE_MODES),
                (count,),
                device=self.device,
            )
            return pure_modes[mode_ids]

        all_mode_iteration = iteration - self.cfg.pure_command_iterations
        one_axis_p, two_axis_p, _ = self._topology_probabilities(
            all_mode_iteration
        )
        topology_draw = torch.rand(count, device=self.device)
        one_axis_mask = topology_draw < one_axis_p
        two_axis_mask = (
            (topology_draw >= one_axis_p)
            & (topology_draw < one_axis_p + two_axis_p)
        )
        three_axis_mask = ~(one_axis_mask | two_axis_mask)

        active_axes = torch.zeros(
            count,
            3,
            dtype=self._command.dtype,
            device=self.device,
        )
        pure_choice = torch.randint(
            len(self._PURE_MODES),
            (count,),
            device=self.device,
        )
        double_modes = self._command.new_tensor(self._DOUBLE_MODES)
        double_choice = torch.randint(
            len(self._DOUBLE_MODES),
            (count,),
            device=self.device,
        )
        active_axes[one_axis_mask] = pure_modes[pure_choice[one_axis_mask]]
        active_axes[two_axis_mask] = double_modes[double_choice[two_axis_mask]]
        active_axes[three_axis_mask] = 1.0
        return active_axes

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if self.cfg.play:
            self._command[env_ids, 0:3] = self._command.new_tensor(
                self.cfg.play_velocity_command
            )
            self._command[env_ids, 3:10] = self._command.new_tensor(
                self.cfg.default_internal_parameters
            )
            return

        iteration = int(self._env.common_step_counter) // max(
            self.cfg.num_steps_per_iteration,
            1,
        )
        active_axes = self._sample_active_axes(len(env_ids), iteration)

        velocity_ranges = self.cfg.velocity_ranges
        sampled_velocity = torch.stack(
            (
                sample_uniform(
                    *velocity_ranges.lin_vel_x,
                    (len(env_ids),),
                    device=self.device,
                ),
                sample_uniform(
                    *velocity_ranges.lin_vel_y,
                    (len(env_ids),),
                    device=self.device,
                ),
                sample_uniform(
                    *velocity_ranges.ang_vel_yaw,
                    (len(env_ids),),
                    device=self.device,
                ),
            ),
            dim=1,
        )
        self._command[env_ids, 0:3] = sampled_velocity * active_axes

        if self.cfg.sample_internal_parameters:
            parameter_ranges = self.cfg.parameter_ranges
            ranges = (
                parameter_ranges.gait_frequency,
                parameter_ranges.foot_yaw_L,
                parameter_ranges.foot_yaw_R,
                parameter_ranges.body_pitch_target,
                parameter_ranges.body_roll_target,
                parameter_ranges.feet_offset_x_target,
                parameter_ranges.feet_offset_y_target,
            )
            for command_index, value_range in enumerate(ranges, start=3):
                self._command[env_ids, command_index] = sample_uniform(
                    *value_range,
                    (len(env_ids),),
                    device=self.device,
                )
        else:
            self._command[env_ids, 3:10] = self._command.new_tensor(
                self.cfg.default_internal_parameters
            )

        # Match StagedCombinationWalkCommand: stand is sampled after topology,
        # so the requested 1/2/3-axis ratios apply to moving commands.
        num_still = int(self.cfg.still_proportion * len(env_ids))
        if num_still > 0:
            still_ids = env_ids[
                torch.randperm(len(env_ids), device=self.device)[:num_still]
            ]
            self._command[still_ids, 0:4] = 0.0
            self.gait_process[still_ids] = 0.0
    
@configclass
class INHAWalkCommandCfg(CommandTermCfg):
    class_type: type = INHAWalkCommand

    @configclass
    class VelocityRanges:
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_yaw: tuple[float, float] = (-1.6, 1.6)

    @configclass
    class ParameterRanges:
        gait_frequency: tuple[float, float] = (1.5, 1.5)
        foot_yaw_L: tuple[float, float] = (-0.7, 0.7)
        foot_yaw_R: tuple[float, float] = (-0.7, 0.7)
        body_pitch_target: tuple[float, float] = (-0.1, 0.3)
        body_roll_target: tuple[float, float] = (-0.1, 0.1)
        feet_offset_x_target: tuple[float, float] = (-0.15, 0.15)
        feet_offset_y_target: tuple[float, float] = (-0.08, 0.15)

    asset_name: str = MISSING
    velocity_ranges: VelocityRanges = VelocityRanges()
    parameter_ranges: ParameterRanges = ParameterRanges()

    still_proportion: float = 0.1
    sample_internal_parameters: bool = False

    play: bool = False
    play_velocity_command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_internal_parameters: tuple[float, float, float, float, float, float, float] = (
        1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )

@configclass
class StagedCombinationWalkCommandCfg(INHAWalkCommandCfg):
    class_type: type = StagedCombinationWalkCommand
    pure_command_iterations: int = 10000
    num_steps_per_iteration: int = 24


@configclass
class WeightedStagedCombinationWalkCommandCfg(StagedCombinationWalkCommandCfg):
    class_type: type = WeightedStagedCombinationWalkCommand

    # Boundaries are local iterations after the pure-command stage finishes.
    topology_stage_1_iterations: int = 1000
    topology_stage_2_iterations: int = 2000
    topology_stage_3_iterations: int = 3000
