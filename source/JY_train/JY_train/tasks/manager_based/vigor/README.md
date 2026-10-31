# VIGOR fall safety for K1 (walk -> fall -> recover -> walk)

Implementation of *Unified Perceptive Fall Safety for Humanoids* (Azulay et al., RA-L 2026,
arXiv:2602.16511) for the 22-DoF Booster K1, wired to take over from the walking policy
(`curriculum2/walk_001`, `jy_curriculum_v1`).

| Paper component | Here |
|---|---|
| Human fall/recovery demos, retargeted, 5 Hz sparse keyframes | LAFAN1 `fallAndGetUp*` / `pushAndFall*` -> GMR -> `data/vigor/k1_fall_keyframes_v2.npz` (80 segments, 38 caught on the hands) |
| Motion-terrain factorisation (vertical terrain projection) | `mdp/commands.py` (spawn clearance), rewards are root-relative so projection never costs reward |
| Privileged teacher: goal-in-context latent `z_g = F(k_t, h_t)`, PPO, flat -> rough | `modules/networks.py:GoalInContextActor`, `modules/teacher.py`, tasks `...Teacher-Flat/Rough-v0` |
| Reward table (imitation / regularisation / post-recovery) | `mdp/rewards.py`, weights in `vigor_env_cfg.py:RewardsCfg` |
| Free-fall torque suppression, pushes, dropouts | `mdp/events.py` |
| Student: latent + action distillation, BC + DAgger mixture | `scripts/vigor/distill.py` (proprio-history student) |
| 20 deg activation, real-time switching | `scripts/vigor/play_walk_fall.py` |

Episodes start in four regimes (`mdp/commands.py`): near a demo's fall onset, around its lowest
pose, the walking stance tipped past 12-30 deg (hand-off from the walking policy) and the upright
walking stance hit by a 0.6-1.8 m/s push. Push and hand-off starts prefer arm-caught demos, and a
dense head-height reward plus trunk/head contact penalties favour catching the fall on the hands.

### v2 (2026-10-31) - what changed after the first flat teacher failed (0 % recoveries)

- tracking used per-frame *heading* frames, which are undefined when the trunk lies face up/down
  (exactly when the robot must get up); now the reference is aligned once per episode by a yaw
  offset and compared in the world frame (paper formula), goal features are in the robot root frame
- the torque-limit penalty (-0.1, paper value) outweighed all imitation rewards because the K1 PD
  saturates in any forceful stand-up -> -0.005; action scale 0.25 -> 0.5 (init noise 1.0 -> 0.5)
- the dataset only contained full falls; partial falls caught on hands/knees are now included and
  flagged (`clip_hand_support`)
- eval reports success / caught (no full fall) / full-fall rate per start regime; the v1 tracker
  also scored finished episodes with the next episode's first frame
- experiments are `k1_vigor_teacher_v2` / `k1_vigor_student_v2`

Not yet: the egocentric **depth** branch of the student (this first version distils the
paper's "w.o Vision" student so the full loop can be tested without camera rendering),
the foothold penalty for stairs, and symmetry augmentation.

## Robot model

`assets/robots/booster_k1_full.py` spawns the vendored `urdf/K1_22dof_jy.urdf` (booster_assets
commit `508cbee`, the pre-rename naming the walking tasks use). Legs/feet reuse the walking
actuators unchanged; arms get kp 20 / kd 1, the head is held. Meshes come from
`booster_assets` and are checked by `check_env.py`.

**Arm pose while walking.** The walking policy was trained with `K1_jy_locomotion.urdf`, whose
arms are welded. That file only exists on the GPU machine. VIGOR's default (and the arm
target used while walking in `play_walk_fall.py`) is shoulder roll -1.3/+1.3 (arms down).
Run `check_env.py --vendor-walk-urdf`, commit the copied URDF, and if the welded pose differs,
change `K1_FULL_DEFAULT_JOINT_POS` before training the teacher.

## Keeping the two machines in sync

| What | How | Guard |
|---|---|---|
| Code | git branch `feat/vigor-fall-safety` | `check_env.py` prints branch/HEAD/dirty/behind |
| Keyframe data (not in git, LAFAN1 is CC BY-NC-ND) | `scripts/vigor/sync.sh push-data <host> <repo>` | sha256 in `data_manifest.json`, verified on load (`KeyframeDataset`, strict) |
| Interfaces (names, dims, timing) | `spec.py`, `SPEC_VERSION` stored in datasets and exported policies | load-time asserts in command, distill, play |
| Robot assets | vendored URDF + `K1_22dof_jy.manifest.json` (mesh git-blob hashes) | `check_env.py` |
| Walking policy interface | `export_walk_spec.py` -> `logs/vigor/walk_spec.json` | `play_walk_fall.py` checks obs layout, dt, defaults |
| Results back to the laptop | `scripts/vigor/sync.sh pull-results <host> <repo>` | exported `*_spec.json` carries file hashes + source checkpoint |

## GPU machine runbook

```bash
git fetch && git checkout feat/vigor-fall-safety && git pull
python -m pip install -e source/JY_train                 # once; keeps the repo-relative data path
python scripts/vigor/check_env.py --vendor-walk-urdf    # paste the summary back
python scripts/vigor/test_offline.py               # Isaac-free sanity checks

# 1) teacher, flat then rough (rough resumes the flat run). Check a short run before a long one:
python scripts/rsl_rl/train.py --task Booster-K1-Vigor-Teacher-Flat-v0 --headless --run_name flat --max_iterations 3000
python scripts/vigor/eval.py --policy teacher --task Booster-K1-Vigor-Teacher-Flat-Play-v0 --run flat --headless --video
python scripts/rsl_rl/train.py --task Booster-K1-Vigor-Teacher-Flat-v0 --headless --resume --load_run '.*_flat' --run_name flat --max_iterations 7000
python scripts/rsl_rl/train.py --task Booster-K1-Vigor-Teacher-Rough-v0 --headless --resume --load_run '.*_flat' --run_name rough --max_iterations 15000
python scripts/vigor/eval.py --policy teacher --task Booster-K1-Vigor-Teacher-Rough-Play-v0 --run rough --headless --video

# 2) student distillation (512 envs) + export
python scripts/vigor/distill.py --task Booster-K1-Vigor-Student-Rough-v0 --teacher_run rough --headless
python scripts/vigor/eval.py --policy student --task Booster-K1-Vigor-Student-Rough-Play-v0 --headless --video

# 3) walk -> fall -> walk in simulation
python scripts/rsl_rl/play.py --task jy_curriculum_v1-Play --load_run <walk run> --headless      # exports walking .pt
python scripts/vigor/export_walk_spec.py --task jy_curriculum_v1-Play --headless
python scripts/vigor/play_walk_fall.py --headless --video \
    --walk_policy logs/rsl_rl/JY_walk_001_symmetry/<walk run>/exported/<name>.pt \
    --student_policy logs/rsl_rl/k1_vigor_student/<run>/exported/vigor_student_<it>.pt
```

Iteration counts are starting points, not tuned values (an RTX 5080 laptop runs ~3.4 s/iteration at
4096 envs). Watch `Episode_Reward/track_body_pos`, `Episode_Reward/head_height`,
`Metrics/motion/error_body_pos`, and in the eval summary `SR` (ends upright), `caught` (upright
without a full fall; the arm-bracing behaviour, mostly in the push / handover regimes) and
`head_contact_rate`. Paper teacher: SR 98 % stand-up / 96 % fall-recovery.

## Laptop (data prep, no Isaac)

```bash
conda create -n vigor_data python=3.11 && conda activate vigor_data
git clone https://github.com/YanjieZe/GMR && pip install -e GMR && pip install rsl-rl-lib==2.3.1 onnx onnxruntime matplotlib
python scripts/vigor/data/retarget_lafan1.py --bvh_dir <lafan1>/bvh --out_dir <work>/retarget
python scripts/vigor/data/build_keyframes.py --retarget_dir <work>/retarget --mesh_dir GMR/assets/booster_k1/meshes
python scripts/vigor/data/render_keyframes.py --out <work>/qa
python scripts/vigor/test_offline.py
git add source/JY_train/JY_train/tasks/manager_based/vigor/data_manifest.json && git commit
scripts/vigor/sync.sh push-data <gpu-host> <gpu-repo-path>
```

Rebuilding the dataset changes its sha256: commit the manifest and push the data together,
otherwise the GPU side refuses to load it.
