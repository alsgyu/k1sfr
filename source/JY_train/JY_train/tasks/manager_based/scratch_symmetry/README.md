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

Run the mapping test from an Isaac Lab Python environment:

```bash
./isaaclab.sh -p scripts/test_scratch_symmetry.py
```

The test checks observation/action involution, command signs, joint swapping,
phase shifting, and augmented batch order.
