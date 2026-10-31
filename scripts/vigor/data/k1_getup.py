"""K1-native brace and get-up references for the VIGOR keyframe dataset (flat ground).

Retargeted human get-ups do not fit the K1: in the LAFAN1 segments 75 % of the get-up
frames pin some joint at its soft limit (deep knee bends, arms overhead), so the policy
cannot follow them and learns to stay down.  The B-Human fast get-up is native to the K1:
legs nearly straight, hips folded to about -2.3 rad, a hands-and-feet pike, and standing
after about 1.7 s.  Its seven keyframes are taken from the team repo INHA-Recovery-Motion
(captured from B-Human's K1_fast.onnx in MuJoCo) and used here in two ways:

* every LAFAN1 fall is cut at its lowest pose and continued with the front or back get-up;
* forward pushes get synthetic clips that fold straight into the hands-and-feet pike
  (front keyframe 3) and rise from there, so the robot catches itself and never lies down.

Only joint angles and torso roll/pitch come from the source.  Root height and position come
from forward kinematics on our URDF (build_keyframes.py grounds every frame).  The head is
held at 0 as in the VIGOR env, so the keyframes that lean on the pitched head get their
trunk pitch re-solved to keep head + feet on the ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SOURCE = (
    "Inha-united-soccer/INHA-Recovery-Motion@d5fc894850e39fa229e6dca83e0c3f5683aeb8b7 "
    "src/inha_mjlab/tasks/getup/reference_poses.py (B-Human fast get-up, K1_fast.onnx sha256 adae2f00...)"
)
SOURCE_JOINT_ORDER = (
    "Head_Yaw", "Head_Pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
_RENAME = {"Head_Yaw": "AAHead_yaw", "Head_Pitch": "Head_pitch",
           "Left_Shoulder_Pitch": "ALeft_Shoulder_Pitch", "Right_Shoulder_Pitch": "ARight_Shoulder_Pitch"}


@dataclass(frozen=True)
class GetupMotion:
    joint_pos: tuple[tuple[float, ...], ...]  # SOURCE_JOINT_ORDER
    torso_roll_pitch: tuple[tuple[float, float], ...]
    duration: tuple[float, ...]  # seconds from the previous keyframe


GETUPS = {
    "front": GetupMotion(
        joint_pos=(
            (0.0, 0.0, 0.4, -1.4, 0.0, 0.0, 0.4, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.017512, -0.348294, -0.337887, -0.578933, -0.348964, -0.060147, -0.230405, 0.360901, 0.506238, 0.030684, -1.156408, 0.235201, 0.026394, 0.506031, -0.15686, 0.077313, -1.149098, -0.268432, 0.117616, 0.448326, -0.348677, 0.330403),
            (-0.065976, -0.0879, 0.041601, 0.555826, -0.298458, 0.002182, 0.146234, -0.512809, 0.579264, 0.028578, -2.207362, 0.485911, -0.042371, 0.210993, -0.195311, -0.100786, -1.870557, -0.547721, 0.186672, 0.073287, -0.197848, 0.346022),
            (-0.013712, 0.856903, 0.177454, 0.659189, -0.329553, -0.021274, 0.551547, -0.918465, 0.675883, 0.066578, -2.396742, 0.58311, -0.30044, 0.230036, -0.075079, -0.346267, -2.182666, -0.480351, -0.049965, 0.042257, 0.013558, 0.346265),
            (-0.062561, 0.848169, 0.079651, 0.064118, -0.57384, -0.010431, 0.673236, -0.707007, 0.571759, 0.000878, -1.573013, 0.494081, -0.235199, 0.574481, -0.077373, -0.34387, -1.311164, -0.502816, 0.031334, 0.244433, 0.012728, 0.317902),
            (-0.097305, 0.629887, 0.077693, -0.67397, -0.675326, 0.000388, 0.681147, 0.00182, 0.524519, -0.000322, -0.560471, 0.395825, -0.025324, 0.543674, -0.216142, -0.306641, -0.492325, -0.392475, -0.035867, 0.364802, -0.108779, 0.127116),
            (-0.061651, -0.029697, 0.121952, -1.515371, -0.385524, -0.025055, 0.176244, 1.286141, 0.305697, 0.005167, -0.146614, 0.110281, -0.065217, 0.37013, -0.199553, -0.090962, -0.260332, -0.110871, 0.039378, 0.381523, -0.092483, -0.072708),
        ),
        torso_roll_pitch=((0.0, 1.57), (0.014801, 2.118679), (-0.079599, 2.527997), (0.04588, 2.150139), (0.0313, 1.113107), (0.029151, 0.339172), (0.017015, -0.033278)),
        duration=(0.02, 0.14, 0.14, 0.52, 0.18, 0.18, 0.56),
    ),
    "back": GetupMotion(
        joint_pos=(
            (0.0, 0.35, 0.0, -1.4, 0.0, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (-0.082604, 0.914214, -0.193249, -0.415494, -0.316672, -0.019232, 0.077819, 0.507077, 0.474169, 0.006438, 0.597643, 0.171049, 0.070393, 0.489347, -0.073151, -0.066643, 0.429179, -0.086694, -0.28504, 0.920447, -0.642958, 0.215602),
            (-0.036475, 0.577612, -0.599402, 0.173883, -0.207995, -0.087751, -0.158422, -0.266821, 0.362045, 0.019923, 1.238773, 0.224006, -0.095306, 0.604943, 0.158911, -0.294826, 1.334556, -0.068071, -0.49648, 1.451112, -0.881335, 0.310579),
            (0.081626, 0.088722, -0.914274, 0.802645, -0.264684, -0.000555, -0.203975, -0.229023, 0.077692, 0.023662, 1.176719, 0.070063, -0.313432, 0.647674, -0.190046, -0.3449, 1.879667, -0.22762, -0.328378, 1.09277, -0.874244, -0.039135),
            (0.046753, 0.007092, -0.83887, 0.734144, -0.316685, -0.012085, 0.035836, 0.428066, 0.046879, 0.017418, 0.74026, 0.081713, -0.335179, 0.5547, -0.260885, -0.336805, 0.979226, -0.204169, -0.056414, 0.838242, -0.694135, 0.073547),
            (-0.076609, -0.055638, -0.622455, 0.326877, -0.38752, -0.020926, 0.044749, 0.907646, 0.091584, 0.009205, 0.117871, 0.152916, -0.25283, 0.386983, -0.206325, -0.264625, 0.12954, -0.118386, 0.016884, 0.595979, -0.331306, 0.137419),
            (-0.050539, -0.094136, -0.049091, -1.344375, -0.356, 5.8e-05, 0.135986, 1.430883, 0.181648, 0.002353, -0.142407, 0.091152, 0.029911, 0.405993, -0.264193, -0.064101, -0.19376, -0.076989, -0.007159, 0.379802, -0.185319, 0.054561),
        ),
        torso_roll_pitch=((0.0, -1.57), (0.023569, -2.150694), (0.141993, -2.542432), (-0.201507, -2.107837), (-0.119138, -1.119392), (-0.020882, -0.393868), (0.017196, 0.004376)),
        duration=(0.02, 0.12, 0.12, 0.18, 0.14, 0.14, 0.48),
    ),

}
# Source keyframes that rest on the (pitched) head; re-solved for head = 0.
HEAD_SUPPORT = {"front": (1, 2), "back": (1, 2)}
# Front keyframe 3: both hands and feet on the ground, hips up -- the brace after a push.
BRACE_KEYFRAME = 3

# Walking stance; must equal booster_k1_full.K1_FULL_DEFAULT_JOINT_POS (test_offline checks it).
STANCE = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0, "Left_Shoulder_Roll": -1.3, "Left_Elbow_Pitch": 0.0, "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.0, "Right_Shoulder_Roll": 1.3, "Right_Elbow_Pitch": 0.0, "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": -0.2, "Left_Hip_Roll": 0.0, "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.4, "Left_Ankle_Pitch": -0.2, "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.2, "Right_Hip_Roll": 0.0, "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.4, "Right_Ankle_Pitch": -0.2, "Right_Ankle_Roll": 0.0,
}


def mirror_sign(name: str) -> float:
    """Left <-> right joint mapping on this URDF: pitch joints keep their sign, roll/yaw joints flip."""
    return 1.0 if "Pitch" in name or "pitch" in name else -1.0


def canonical(q_src, names: list[str]) -> np.ndarray:
    """Source keyframe -> ``names`` order, head held at 0."""
    src = {_RENAME.get(n, n): v for n, v in zip(SOURCE_JOINT_ORDER, q_src)}
    q = np.array([src[n] for n in names], dtype=np.float64)
    q[[i for i, n in enumerate(names) if "Head" in n]] = 0.0
    return q


def stance(names: list[str]) -> np.ndarray:
    return np.array([STANCE[n] for n in names], dtype=np.float64)


def mirrored(q: np.ndarray, names: list[str], from_side: str = "Right") -> np.ndarray:
    """Copy one side onto the other (symmetric pose)."""
    other = "Left" if from_side == "Right" else "Right"
    out = q.copy()
    for i, n in enumerate(names):
        if other in n:
            j = names.index(n.replace(other, from_side))
            out[i] = mirror_sign(n) * q[j]
    return out


def rpy_quat(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    """wxyz quaternion of R = Rz(yaw) Ry(pitch) Rx(roll) (the source's torso roll/pitch convention)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr])


def quantize(duration_s: float, fps: int, stride: int) -> int:
    """Frames for one source segment, rounded to whole keyframe strides so every source keyframe
    lands on the sparse-keyframe grid (otherwise the 5 Hz resampling would skip the 0.14 s phases)."""
    return max(stride, stride * round(duration_s * fps / stride))
