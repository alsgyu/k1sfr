# Scratch symmetry tasks

This package is an independent copy of `scratch` with K1 left-right mirror
training enabled through RSL-RL's `RslRlSymmetryCfg`.

| Task ID | Mirror loss | Mirror augmentation |
|---|---:|---:|
| `jy_scratch_symmetry_loss_v0` | on (`0.01`) | off |
| `jy_scratch_symmetry_aug_v0` | off | on |
| `jy_scratch_symmetry_v0` | on (`0.005`) | on |
| `jy_scratch_symmetry_v0-Play` | on (`0.005`) | on |

The reflection swaps the left and right leg joints, changes the signs of
roll/yaw joint channels, mirrors lateral velocity and yaw-rate commands, and
shifts gait phase by half a cycle:

```text
(cos(2πφ), sin(2πφ)) -> -(cos(2πφ), sin(2πφ))
                      = (cos(2π(φ+0.5)), sin(2π(φ+0.5)))
```

Policy observations, privileged critic observations, previous actions, and
current actions use the same reflection convention. The augmentation callback
returns the original batch first and the mirrored batch second, as required by
RSL-RL 2.3.1.

Empirical observation normalization remains enabled. `scripts/rsl_rl/train.py`
attaches the runner-owned policy and critic normalizers to this task after the
runner is created. The callback then applies:

```text
normalized observation -> raw observation -> mirror -> normalized observation
```

This bridge is necessary because per-channel empirical normalization does not
generally commute with left-right signed permutations. Training through a
different entry script must provide the same bridge.

## Command and foot-trajectory curriculum

For the first 10,000 PPO iterations, each moving environment receives one pure
velocity axis: `vx`, `vy`, or `yaw`. Afterwards, the sampler expands to all
seven non-empty subsets of the three axes, retaining the pure modes while
adding `vx+vy`, `vx+yaw`, `vy+yaw`, and `vx+vy+yaw`. The existing 10% explicit
stand-command fraction remains active in both stages.

A single paper-style `footswing_trajectory_tracking` cost is enabled:

```text
r_fs = sum_i ||p_i,xy - p*_i,xy||^2
       + 5 sum_i (h_i - h*_i)^2
```

The left/right height targets are periodic profiles centered at gait phases
0.25 and 0.75 (`H_max=0.06 m`, `kappa=0.04`). The XY targets are generated around
each hip using the Raibert heuristic, stance time, and
current-versus-commanded planar velocity. The function returns a positive
cost and therefore uses reward weight `-1.0`. It is disabled below the
configured moving-command threshold and is invariant under the task's
left-right mirror transform. The earlier separate exponential height and XY
rewards remain in `mdp/rewards.py` but are not configured.

Run the mapping test from an Isaac Lab Python environment:

```bash
./isaaclab.sh -p scripts/test_scratch_symmetry.py
```

The test checks observation/action involution, command signs, joint swapping,
phase shifting, and augmented batch order.
