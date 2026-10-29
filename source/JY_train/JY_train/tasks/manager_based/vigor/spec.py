"""Shared K1 constants for the VIGOR fall-safety tasks.

This module is pure Python on purpose: the env configs (GPU machine), the
offline keyframe builder (laptop, no Isaac), the distillation/export scripts
and the walk -> fall runner all import names and sizes from here, so the
interfaces cannot silently drift between machines.  Bump ``SPEC_VERSION``
whenever a layout below changes; datasets and exported policies record it.
"""

SPEC_VERSION = "k1-vigor-v1"

# ---------------------------------------------------------------------------
# Kinematic names (pre-rename booster_assets naming, same as the walking tasks)
# ---------------------------------------------------------------------------
HEAD_JOINT_NAMES = ["AAHead_yaw", "Head_pitch"]
ARM_JOINT_NAMES = [
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]
LEG_JOINT_NAMES = [
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]
# Canonical order used by keyframe datasets (NOT the articulation order).
ALL_JOINT_NAMES = HEAD_JOINT_NAMES + ARM_JOINT_NAMES + LEG_JOINT_NAMES
# Policy action order (head is held, see HoldJointPositionAction).
ACTION_JOINT_NAMES = ARM_JOINT_NAMES + LEG_JOINT_NAMES

ROOT_BODY_NAME = "Trunk"
# Keypoints for sparse keyframes; index 0 must be the root.
KEY_BODY_NAMES = [
    "Trunk",
    "Head_2",
    "Left_Arm_2",
    "left_hand_link",
    "Right_Arm_2",
    "right_hand_link",
    "Left_Shank",
    "Right_Shank",
    "left_foot_link",
    "right_foot_link",
]
HEAD_BODY_NAMES = ["Head_1", "Head_2"]
FOOT_BODY_NAMES = ["left_foot_link", "right_foot_link"]

# ---------------------------------------------------------------------------
# Timing (identical to the walking tasks so actuator delays mean the same)
# ---------------------------------------------------------------------------
SIM_DT = 0.002
DECIMATION = 10
POLICY_DT = SIM_DT * DECIMATION  # 50 Hz
EPISODE_LENGTH_S = 7.5
KEYFRAME_FPS = 50
KEYFRAME_STRIDE = 10  # 5 Hz sparse keyframes
NUM_FUTURE_KEYFRAMES = 2

# Controller hand-off rule from the paper: policy engages at 20 deg tilt.
ACTIVATION_TILT_DEG = 20.0

# Default standing heights of the K1_22dof_jy model in the walking stance,
# measured from the sole plane (computed with scripts/vigor/data/urdf_fk.py).
TRUNK_HEIGHT_NOMINAL = 0.532
HEAD_HEIGHT_NOMINAL = 0.780  # Head_2 link origin
SOLE_OFFSET = 0.027  # foot link origin above the sole (Left_Foot.STL)

# ---------------------------------------------------------------------------
# Observation layouts
# ---------------------------------------------------------------------------
NUM_JOINTS = len(ALL_JOINT_NAMES)  # 22
NUM_ACTIONS = len(ACTION_JOINT_NAMES)  # 20
# (term name, dim) in the order they appear in the policy/student groups.
PROPRIO_TERMS = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", NUM_JOINTS),
    ("joint_vel", NUM_JOINTS),
    ("last_action", NUM_ACTIONS),
]
PROPRIO_DIM = sum(d for _, d in PROPRIO_TERMS)  # 70

# Per future keyframe: root-relative heading-frame positions of the 9 non-root
# keypoints (27) + reference root gravity (3) + reference root height above its
# lowest keypoint (1) + heading offset cos/sin (2) + time to keyframe (1).
KEYFRAME_FEATURE_DIM = (len(KEY_BODY_NAMES) - 1) * 3 + 3 + 1 + 2 + 1  # 34
GOAL_DIM = NUM_FUTURE_KEYFRAMES * KEYFRAME_FEATURE_DIM  # 68

HEIGHT_SCAN_SIZE = (1.4, 1.4)
HEIGHT_SCAN_RESOLUTION = 0.1
HEIGHT_SCAN_POINTS = 15 * 15
# terrain shape relative to the terrain under the trunk + trunk height above it
TERRAIN_DIM = HEIGHT_SCAN_POINTS + 1  # 226

TEACHER_OBS_DIM = PROPRIO_DIM + GOAL_DIM + TERRAIN_DIM  # 364
LATENT_DIM = 32
STUDENT_HISTORY = 25  # 0.5 s of proprioception
