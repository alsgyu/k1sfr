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

Stand-up is handled elsewhere; this task only asks the K1 to fall safely on flat ground: spread the
landing over several sturdy contacts and over time, keep the head (least of all), chest, back and upper
arms off the ground, and end - on the side it fell to - in a fixed multi-contact pose the get-up starts
from: kneeling on fists, knees and toes (forward falls, chest down) or sitting on the hips with the fists
behind and the heels down (backward falls, chest up). If the get-up is not ready, lowering down is the
robot's damping mode (kp = 0), checked with `--release_after`. No demonstrations: v1-v3 tracked references
the K1 could not follow inside a fall ("arms up, face first"). The reward is written on the fall itself
(`vigor/safefall_math.py`, shared with the MuJoCo check):

| term | weight /s | meaning |
|---|---|---|
| `impact` | -5 | sum over bodies of fragility x (step-mean ground force in body weights - 0.25)^2 (below) |
| `support` | +1.5 | 0.25 per fist and per contact of the latched side's pose (knees in front, hips or pelvis behind) on the ground while head, upper arms, chest and back stay off it, trunk held up |
| `hands_lead` | +0.5 | while falling: fists below head and trunk before they touch |
| `handoff_pose` | +4 | `support` x exp(-mean squared joint error to the hand-off pose of the latched side / 0.5) |
| `head_contact` / `head_impact` | -50 / -20 | head (stereo camera) on the ground / its force in body weights |
| `head_low` | -5 | head origin below 0.15 m (about to touch), shaping before contact |
| `trunk_contact` | -5 | force on the chest or back (trunk tilted past 70 deg; less tilted, only the pelvis can touch: sitting) |
| `head_descent` / `trunk_descent` | -1 / -0.5 | falling fast close to the ground |
| `fist_descent` | -2 | fist tips (0.2 m out along the hand link) moving down within a few cm of the ground, (m/s)^2 |
| `facing` | -5 | turned round mid-fall (forward fall chest up, backward chest down), scaled by cos of the fall direction; also voids `support` and `handoff_pose` |
| `upright` | +3 | still standing (not falling stays the best outcome) |

plus action-rate / joint-limit / joint-velocity / acceleration / torque regularisation. One head hit
costs more than a whole episode of the hand-off pose. Contact detection uses the peak force over the 10
physics substeps of a policy step; `impact` uses the mean (impulse / 20 ms) per body.

`impact` is what the landing minimises. A landing's total impulse is fixed by the fall (momentum), so a
linear penalty per link only moves it to the cheapest link: with the fists the only allowed contact that
was the arms (the fist-plank policy took 99 % of the landing impulse on the arms, 2.5-3.5 body weights
through one arm, 14 Nm shoulder and elbow motors). The square makes sharing pay (n bodies taking equal
parts cost 1/n) and stretching the impulse in time too (joints that give while they take load; constant
force over the longest stroke is the optimum). Fragility weights by the joints behind the link: chest /
back 3, upper arms 3, forearm + fist 1.5, pelvis 1, hips / thighs / knees 1 (68-112 Nm motors), feet 0.5;
each body carries 0.25 body weights for free (the hand-off contacts hold 0.1-0.3 each). The legs are 5-8x
stronger than the arms, so the policy brakes with them: knees down in front, hips down behind.

The rest of the arm peak is one 20 ms step at the first fist touch-down: swung down by the reflex, the
fist tips arrived at 2.3-2.8 m/s (forward) to 4.8 m/s (backward) while the trunk fell at 1.3-1.5.
`fist_descent` asks for the fists to reach the ground early and touch softly. MuJoCo, 2000 iterations,
mean of seeds 0 and 1 (256 falls each, `fist_descent` 0 / -2 / -5): impact 0.94 / 0.81 / 0.91, chest or
back contact 13 / 8 / 11 %, `handoff_end` 0.71 / 0.91 / 0.74, `arm_peak` 3.0 / 2.8 / 2.7 body weights,
`arm_share` 0.76 / 0.73 / 0.74, fist speed at touch-down about 3.5 / 3.4 / 3.1 m/s, head contact 4.5 / 5.7
/ 6.7 % (mostly backward falls). The fist-plank reward before it: `arm_share` 0.99.

Episodes follow the robot's fall handling: the K1 stands on the ground (half of the episodes walking:
0.2-0.6 m/s forward with one leg lifted mid-swing) and is pushed (0.6-1.8 m/s on the trunk; 45 %
forward, 20 % backward, 35 % sideways, +-45 deg). Until the IMU tilt reaches 20 deg (the fall detector,
`TRIGGER_TILT_DEG`) the start pose is held and the policy's actions are ignored and unrewarded - on the
robot the walking policy runs until then; from the trigger on the policy acts (MuJoCo: every push
triggers, after 0.2-0.5 s). The facing of the fall is taken from the tilt direction at the trigger. 20 %
of the episodes instead start resting in a hand-off pose (direction `handoff`) so the policy learns to
hold the target poses. Eval scripts measure falls only (`HANDOFF_START_PROB = 0`); training
`Metrics/monitor/*` include the hand-off starts. The actor sees only IMU + joints + last action (3-step
history), so it deploys without distillation; the critic also gets velocities, heights, contact forces,
the fall direction and whether the trigger has fired.

Domain randomisation, matched to the team's walking task (`scratch_symmetry` walk_001): static / dynamic
friction 0-2 / 0.1-2, restitution 0.1-0.9, trunk mass x0.8-1.2 and CoM +-5 cm, leg encoder offsets
+-0.03 rad (joints settle at target - offset and read position + offset), start joints +-0.05 rad and
+-0.1 rad/s, PD gains +-10 %, a +-0.5 m/s kick every 4 s (lands while the hand-off pose is held), uniform
noise on the actor's observations (angular velocity 0.2, gravity 0.05, joint position 0.01, joint velocity
0.5); on top of that every other link mass +-10 % and the actuator model's command delay 0-36 ms. The
MuJoCo check shares one model across envs: it randomises the encoder offsets, noise, kick, delay and start
pose, not friction or masses.

On the robot: run the walking policy; once the IMU tilt reaches 20 deg switch to this policy for good
(until the get-up). Its observation history should be warm at the switch, as in training where the policy
already runs (ignored) before the trigger: run it in shadow mode while walking, or fill the history with
the current frame and a zero last action.

Actions are offsets from a reflex target, not from the walking stance (`ReflexJointPositionAction`,
`safefall_math.reflex_targets`): the hand-off pose of a latched side, set from the tilt direction at the
trigger and switched only once the chest clearly faces the other way (|chest_z| > 0.25, `update_side`).
Choosing by the sign of chest_z every step flipped the legs between the poses whenever a side fall rolled
through chest-sideways, and the policy flailed them; `residual` (-0.01 x sum of squared actions) keeps the
policy on the reflex unless the landing needs otherwise. Zero action therefore reaches for the ground and
ends in the hand-off pose; the policy learns the landing (when the elbows and knees give) and balance on
top. Measured from the stance, holding a plank took actions of -3.4 on shoulder or hip pitch, and the
policy left it within 0.5 s even when the episode started in it. On the robot the same reflex function of
the IMU's projected gravity must be added to the policy output, and the observations scaled as in training
(angular velocity x 0.25, joint velocity x 0.05) (`--passive` in the eval scripts is the reflex alone).

Left-right symmetry augmentation (`safefall_symmetry.py`, rsl_rl `symmetry_cfg`): PPO also trains on
the mirror image of every sample, so a straight forward fall gets a symmetric landing. Without it the
policy broke symmetry at random (run8: elbows 0.66 rad apart, right fist always further up). The
augmentation reads the observation layout from the observation manager; a new observation term needs
an entry in `TERM_KINDS`. rsl_rl stores normalised observations before augmenting, so the task uses
fixed observation scales instead of empirical normalisation.

```bash
# laptop, MuJoCo (same URDF, gains, starts and reward), ~25 min for 2000 iterations
python scripts/vigor/safefall_mujoco.py train --envs 64 --iters 2000 --out ~/vigor_work/safefall_mj/run16s1
python scripts/vigor/safefall_mujoco.py eval --ckpt ~/vigor_work/safefall_mj/run16s1/model_1999.pt --render rollouts.png
python scripts/vigor/safefall_mujoco.py eval --passive
# GPU, Isaac
python scripts/rsl_rl/train.py --task Booster-K1-SafeFall-Flat-v0 --headless --run_name share --max_iterations 6000
python scripts/vigor/eval_safefall.py --run share --sweep 500 --num_envs 256 --headless
python scripts/vigor/eval_safefall.py --run share --direction forward --follow --real_time --num_envs 16
```

Outcomes (training `Metrics/monitor/*`, eval `summary.json`): `head_hit` / `trunk_hit` (chest or back)
and their peak forces (body weights); `impact` (the `impact` term integrated over the episode); `arm_peak`
(largest step-mean ground force through one arm, body weights); `max_peak` (same for any single body but
the feet); `arm_share` (the arms' share of the ground impulse, feet aside, in the 0.5 s after the first
touch-down); `fist_speed` (fastest downward speed of a fist tip within 5 cm of touching, m/s);
`fists_first`; `support` (fraction of the time after touch-down spent supported);
`upright_end`; `flipped` (turned round by the end: forward fall chest up / backward fall chest down);
`handoff_end` (supported on both fists, head, chest, back and upper arms clear, at the last step);
`pose_rms` (RMS joint error to the hand-off pose at the last step, rad); `asym_rms` (RMS left-right
asymmetry of the end pose, rad); with `--release_after` also `release_*` for the lowering phase. Which
checkpoint is best: `eval_safefall.py --sweep 500` evaluates every 500th checkpoint of the run on
identical falls and ranks them by `safefall_math.score` (handoff_end - 3 head_hit - trunk_hit - 0.5 impact
- pose_rms).

Hand-off poses (`safefall_math.HANDOFF_POSES`): symmetric PD targets the K1 rests in on several contacts
-- front: fists 0.21, knees 0.18, toes 0.14 body weights each; back: hips 0.30, heels 0.11, fists 0.09 --
on its PD gains alone (arms < 2 Nm), keeping those contacts through 0.3 / 0.6 m/s pushes of the trunk in
four directions, head 0.31-0.33 m up, inside the soft joint limits. The old fist planks put 0.31 body
weights on each arm. `scripts/vigor/handoff_pose.py` drops random symmetric candidates on the floor with
the pose as PD targets and ranks those that qualify by the static load (sum of squared contact forces);
`check` writes `vigor/handoff_poses.json` for the get-up team (22 settled joint positions in
INHA-Recovery-Motion `K1_JOINT_ORDER`, the PD targets, root height, roll/pitch, projected gravity, contact
loads). `eval_safefall.py` also writes `end_poses.json`: the poses the falls actually ended in, per side,
with their spread and every end state as `samples` for the get-up to start from (run it without
`--release_after`).

```bash
python scripts/vigor/handoff_pose.py search back --n 4000
python scripts/vigor/handoff_pose.py check --out source/JY_train/JY_train/tasks/manager_based/vigor/handoff_poses.json --render handoff_poses.png
```

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
