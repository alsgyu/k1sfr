# VIGOR fall safety for K1 (walk -> fall -> recover -> walk)

Implementation of *Unified Perceptive Fall Safety for Humanoids* (Azulay et al., RA-L 2026,
arXiv:2602.16511) for the 22-DoF Booster K1, wired to take over from the walking policy
(`curriculum2/walk_001`, `jy_curriculum_v1`).

| Paper component | Here |
|---|---|
| Human fall/recovery demos, retargeted, 5 Hz sparse keyframes | LAFAN1 `fallAndGetUp*` / `pushAndFall*` falls -> GMR, continued with the K1-native front/back get-up, plus synthetic brace-and-rise clips -> `data/vigor/k1_fall_keyframes_v3.npz` (`scripts/vigor/data/k1_getup.py`) |
| Motion-terrain factorisation (vertical terrain projection) | `mdp/commands.py` (spawn clearance), rewards are root-relative so projection never costs reward |
| Privileged teacher: goal-in-context latent `z_g = F(k_t, h_t)`, PPO, flat -> rough | `modules/networks.py:GoalInContextActor`, `modules/teacher.py`, tasks `...Teacher-Flat/Rough-v0` |
| Reward table (imitation / regularisation / post-recovery) | `mdp/rewards.py`, weights in `vigor_env_cfg.py:RewardsCfg` |
| Free-fall torque suppression, pushes, dropouts | `mdp/events.py` |
| Student: latent + action distillation, BC + DAgger mixture | `scripts/vigor/distill.py` (proprio-history student) |
| 20 deg activation, real-time switching | `scripts/vigor/play_walk_fall.py` |

Episodes start in four regimes (`mdp/commands.py`): near a demo's fall onset, around its lowest
pose, the walking stance tipped past 12-30 deg (hand-off from the walking policy) and the upright
walking stance hit by a 0.6-1.8 m/s push. Push and hand-off starts mostly draw brace-and-rise
clips (catch the fall on both hands, stand up from there without lying down).

## Safe fall (`Booster-K1-SafeFall-Flat-v0`) - impact mitigation only

Stand-up is handled elsewhere; this task only asks the K1 to fall safely on flat ground: land
on the fists, let nothing else touch, and hold a fist plank (fists + feet, stiff arms) until
the end of the 4 s episode, on the side it fell to (forward falls chest down, backward falls
chest up). Lowering down afterwards is the robot's damping mode (kp = 0),
checked with `--release_after`. No demonstrations: v1-v3 tracked references the K1 could not
follow inside a fall ("arms up, face first"). The reward is written on the fall itself
(`vigor/safefall_math.py`, shared with the MuJoCo check):

| term | weight /s | meaning |
|---|---|---|
| `plank` | +1.5 | 0.5 per fist on the ground while nothing but fists and feet touches, trunk held up |
| `fist_share` | +2 | share of the ground load (feet aside) on the fists |
| `hands_lead` | +0.5 | while falling: fists below head and trunk before they touch |
| `head_contact` / `head_impact` | -30 / -10 | head (stereo camera) on the ground / its force in body weights |
| `trunk_contact` / `leg_contact` / `arm_contact` | -5 / -15 / -2 | contact force on trunk / hips-thighs-knees / upper arms-forearms |
| `fist_impact` | -1 | fist force above 2 body weights (let the elbows give) |
| `head_descent` / `trunk_descent` | -1 / -0.5 | falling fast close to the ground |
| `facing` | -5 | turned round mid-fall (forward fall chest up, backward chest down), scaled by cos of the fall direction; also voids `plank` and `fist_share` |
| `upright` | +3 | still standing (not falling stays the best outcome) |

plus action-rate / joint-limit / joint-velocity / acceleration / torque regularisation. Impacts
last a few 20 ms steps while the plank pays for seconds, hence the large contact weights (one head
hit costs more than a whole episode of plank). Contact forces are the peak over the 10 physics
substeps of a policy step. Episodes start from the walking stance tipping 10-30 deg (0.5-2.5 rad/s) or pushed (0.8-2.0 m/s), 45 % forward, 20 %
backward, 35 % sideways. The actor sees only IMU + joints + last action (3-step history), so it
deploys without distillation; the critic also gets velocities, heights, contact forces and the fall direction.

```bash
# laptop, MuJoCo (same URDF, gains, starts and reward), ~15 min for 1500 iterations
python scripts/vigor/safefall_mujoco.py train --envs 64 --iters 1500 --out ~/vigor_work/safefall_mj/run2
python scripts/vigor/safefall_mujoco.py eval --ckpt ~/vigor_work/safefall_mj/run2/model_1499.pt --render rollouts.png
python scripts/vigor/safefall_mujoco.py eval --passive
# GPU, Isaac
python scripts/rsl_rl/train.py --task Booster-K1-SafeFall-Flat-v0 --headless --run_name flat --max_iterations 1500
python scripts/vigor/eval_safefall.py --run flat --headless
python scripts/vigor/eval_safefall.py --run flat --direction forward --follow --real_time --num_envs 16 --release_after 2.5
```

Outcomes (training `Metrics/monitor/*`, eval `summary.json`): `head_hit` / `trunk_hit` /
`knee_hit`, their peak forces (body weights), `fist_share`, `fists_first`, `plank` (fraction of
the time after touch-down spent on the fists only), `upright_end`, `flipped` (turned round by the
end: forward fall chest up / backward fall chest down); with `--release_after` also
`release_*` for the lowering phase.

### v3 (2026-10-31) - K1-native get-up and brace (flat ground)

The v2 flat teacher touched down with the arms first but then lay down and never stood up.
In the LAFAN1 get-up frames 75 % pin some K1 joint at its soft limit (deep knee bends, arms
overhead): human get-ups do not fit the K1, so the stand-up part could not be tracked.

- every LAFAN1 fall is cut at its lowest pose and continued with the B-Human fast front or back
  get-up (K1-native: straight legs, hips folded, hands-and-feet pike, ~2 s), taken from the team
  repo INHA-Recovery-Motion; source keyframes are quantised to the 0.2 s keyframe grid, the head is
  held at 0, and the keyframes that lean on the pitched head get their trunk pitch re-solved
- 16 synthetic forward brace-and-rise clips: from the walking stance fold into that get-up's
  hands-and-feet pike (both hands and feet on the ground, head 12 cm and trunk 22 cm up, held
  0.2/0.4 s) and stand up from it; push and hand-off starts draw them with p = 0.75 and push in
  their direction
- the dataset records per-frame reference contacts of hands / head / trunk: `hand_contact`
  rewards planting the hands when the reference does; head and trunk are penalised only where
  the reference keeps them off the ground (the K1 get-ups lean on the head)
- the pose-imitation terms are multiplied by a height gate (B-Human): no reward while the root is
  more than 5 cm below the reference's, so lying down with matching limbs no longer pays
- a full fall is now the trunk dropping below 0.25 m (the brace pike puts the head ~12 cm off the
  ground with the trunk ~0.37 m up), and the head-height reward only pays once upright (paper)
- experiments are `k1_vigor_teacher_v3` / `k1_vigor_student_v3`

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
4096 envs). The command term logs the eval outcome definitions while training, as rates over the
last 1000 full-length episodes of each start regime: `Metrics/motion/<regime>/success` (ends
upright), `.../full_fall` (trunk below 0.25 m) and `.../caught` (upright without a full fall; the
arm-bracing behaviour, mostly in the push / handover regimes). Print them with the tracking
error and reward terms as a text table, also while the run is still going:

```bash
python scripts/vigor/summarize_run.py --run flat
```

The eval summary reports the same keys per regime plus `head_contact_rate`. Paper teacher:
SR 98 % stand-up / 96 % fall-recovery.

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
