"""22-DoF K1 (head + arms + legs) for whole-body tasks such as VIGOR fall safety.

The walking tasks use ``K1_jy_locomotion.urdf`` (arms and head welded).  Fall
safety needs the arms, so this config spawns the vendored pre-rename
``urdf/K1_22dof_jy.urdf`` instead.  Joint/link names are identical to the
walking model, and the leg/feet actuators are copied from ``booster.py`` so the
leg dynamics do not change at the walk -> fall-safety handover.

Mesh files are not vendored: they are resolved against ``booster_assets`` at
import time (override with ``JY_K1_MESH_DIR``).  ``scripts/vigor/check_env.py``
verifies them against ``urdf/K1_22dof_jy.manifest.json``.
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings

from JY_train.assets.robots import actuator
from JY_train.assets.robots.actuator import BoosterDelayedPDActuatorCfg
from JY_train.assets.robots.booster import BOOSTER_K1_CFG

from booster_assets import BOOSTER_ASSETS_DIR

URDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdf")
K1_FULL_URDF_SRC = os.path.join(URDF_DIR, "K1_22dof_jy.urdf")
K1_FULL_URDF_MANIFEST = os.path.join(URDF_DIR, "K1_22dof_jy.manifest.json")
K1_MESH_DIR = os.environ.get("JY_K1_MESH_DIR", os.path.join(BOOSTER_ASSETS_DIR, "robots", "K1", "meshes"))
URDF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "JY_train", "urdf")


def resolve_urdf_meshes(src: str = K1_FULL_URDF_SRC, mesh_dir: str = K1_MESH_DIR, cache_dir: str = URDF_CACHE_DIR) -> str:
    """Write a copy of ``src`` whose ``meshes/`` paths point to ``mesh_dir`` and return its path."""
    with open(src) as f:
        text = f.read()
    names = sorted(set(re.findall(r'filename="meshes/([^"]+)"', text)))
    missing = [n for n in names if not os.path.isfile(os.path.join(mesh_dir, n))]
    if missing:
        raise FileNotFoundError(f"K1 meshes missing in {mesh_dir}: {missing}. Set JY_K1_MESH_DIR or update booster_assets.")
    resolved = text.replace('filename="meshes/', f'filename="{os.path.abspath(mesh_dir)}/')
    digest = hashlib.sha1(resolved.encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"K1_22dof_jy_{digest}.urdf")
    if not os.path.isfile(out):
        tmp = f"{out}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(resolved)
        os.replace(tmp, out)
    return out


try:
    K1_FULL_URDF_PATH = resolve_urdf_meshes()
except Exception as err:  # do not break registration of the walking tasks
    warnings.warn(f"[booster_k1_full] {err}. VIGOR tasks will fail to spawn the robot until this is fixed.")
    K1_FULL_URDF_PATH = K1_FULL_URDF_SRC


# Walking stance (booster.py init_state) with arms lowered.  The arm values must
# match the pose that is welded in K1_jy_locomotion.urdf / used on the real
# robot while walking; check_env.py prints the welded pose for comparison.
K1_FULL_DEFAULT_JOINT_POS = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.3,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.3,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.0,
    ".*_Hip_Pitch": -0.2,
    ".*_Hip_Roll": 0.0,
    ".*_Hip_Yaw": 0.0,
    ".*_Knee_Pitch": 0.4,
    ".*_Ankle_Pitch": -0.2,
    ".*_Ankle_Roll": 0.0,
}

BOOSTER_K1_FULL_CFG = BOOSTER_K1_CFG.replace(
    spawn=BOOSTER_K1_CFG.spawn.replace(asset_path=K1_FULL_URDF_PATH),
    init_state=BOOSTER_K1_CFG.init_state.replace(joint_pos=K1_FULL_DEFAULT_JOINT_POS),
    actuators={
        "legs": BOOSTER_K1_CFG.actuators["legs"],
        "feet": BOOSTER_K1_CFG.actuators["feet"],
        # R14 at its 10 Hz default gives kp~4, far too soft to arrest a fall or
        # push the trunk up; 20/1 matches the order of booster_deploy's prepare gains.
        "arms": BoosterDelayedPDActuatorCfg(
            max_delay=18,
            min_delay=0,
            joint_names_expr=[".*_Shoulder_Pitch", ".*_Shoulder_Roll", ".*_Elbow_Pitch", ".*_Elbow_Yaw"],
            stiffness=20.0,
            damping=1.0,
            booster_joint_cfgs=actuator.BoosterJointR14(),
        ),
        # The head is held at its default pose (camera extrinsics stay fixed).
        "head": BoosterDelayedPDActuatorCfg(
            max_delay=18,
            min_delay=0,
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            stiffness=10.0,
            damping=0.5,
            booster_joint_cfgs=actuator.BoosterJointHT4438(),
        ),
    },
)


def _action_scale(cfg, factor: float) -> dict[str, float]:
    scale = {}
    for a in cfg.actuators.values():
        e, s = a.effort_limit_sim, a.stiffness
        names = a.joint_names_expr
        e = e if isinstance(e, dict) else {n: e for n in names}
        s = s if isinstance(s, dict) else {n: s for n in names}
        for n in names:
            if n in e and n in s and s[n]:
                scale[n] = factor * e[n] / s[n]
    return scale


# Same convention as K1_ACTION_SCALE (BeyondMimic): 0.25 * effort / stiffness.
K1_FULL_ACTION_SCALE = _action_scale(BOOSTER_K1_FULL_CFG, 0.25)
