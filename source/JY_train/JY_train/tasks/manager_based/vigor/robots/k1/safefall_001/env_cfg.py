"""K1 safe fall on flat ground: land on the fists and hold a fist plank.

No demonstrations and no stand-up (another team handles getting up).  Episodes start
from the walking stance tipping or pushed in a random direction and last 4 s; the reward
(``safefall_math``) lets only fists and feet touch the ground, rewards holding the body up
on the fists with stiff arms, penalises head, trunk, knee/thigh and forearm contact, and
keeps forward falls chest down and backward falls chest up.
The same task runs on the laptop in MuJoCo (scripts/vigor/safefall_mujoco.py).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

import JY_train.tasks.manager_based.vigor.mdp as mdp
from JY_train.assets.robots.booster_k1_full import BOOSTER_K1_FULL_CFG, K1_FULL_ACTION_SCALE
from JY_train.tasks.manager_based.vigor import safefall_math as sf
from JY_train.tasks.manager_based.vigor import spec

EPISODE_LENGTH_S = sf.EPISODE_S
HISTORY = 3
ACTION_SCALE = {k: v for k, v in K1_FULL_ACTION_SCALE.items() if k not in spec.HEAD_JOINT_NAMES}
# the walking stance has elbow yaw 0, on the hard limit and outside the 0.9 soft range
ROBOT = BOOSTER_K1_FULL_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=BOOSTER_K1_FULL_CFG.init_state.replace(
        joint_pos=BOOSTER_K1_FULL_CFG.init_state.joint_pos | {"Left_Elbow_Yaw": -0.13, "Right_Elbow_Yaw": 0.13}
    ),
)


@configclass
class SafeFallSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = ROBOT
    # history spans the policy step's physics substeps, so the peak impact force is seen
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=spec.DECIMATION, track_air_time=False)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0))


@configclass
class CommandsCfg:
    monitor = mdp.SafeFallMonitorCfg()


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=spec.ACTION_JOINT_NAMES, scale=ACTION_SCALE, use_default_offset=True, preserve_order=True
    )
    head = mdp.HoldJointPositionActionCfg(asset_name="robot", joint_names=spec.HEAD_JOINT_NAMES)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable: IMU + joint encoders + last action, last HISTORY steps."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(std=0.1), history_length=HISTORY)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.02), history_length=HISTORY)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(std=0.01), history_length=HISTORY)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Gnoise(std=0.1), history_length=HISTORY)
        last_action = ObsTerm(func=mdp.last_action, history_length=HISTORY)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        heights = ObsTerm(func=mdp.body_heights)
        contact_forces = ObsTerm(func=mdp.contact_force_groups)
        facing = ObsTerm(func=mdp.fall_facing)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )
    trunk_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk"), "mass_distribution_params": (0.9, 1.1), "operation": "scale"},
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    fall_start = EventTerm(func=mdp.reset_to_fall_start, mode="reset")


def _sf(name: str) -> RewTerm:
    return RewTerm(func=mdp.safefall_term, weight=sf.WEIGHTS[name], params={"name": name})


@configclass
class RewardsCfg:
    # task (safefall_math.WEIGHTS)
    upright = _sf("upright")
    plank = _sf("plank")
    fist_share = _sf("fist_share")
    hands_lead = _sf("hands_lead")
    head_contact = _sf("head_contact")
    head_impact = _sf("head_impact")
    trunk_contact = _sf("trunk_contact")
    leg_contact = _sf("leg_contact")
    arm_contact = _sf("arm_contact")
    fist_impact = _sf("fist_impact")
    head_descent = _sf("head_descent")
    trunk_descent = _sf("trunk_descent")
    facing = _sf("facing")
    # regularisation (same weights as scripts/vigor/safefall_mujoco.py REG)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_pos_limit = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_state = DoneTerm(func=mdp.bad_state, params={"max_lin_vel": 15.0, "max_ang_vel": 40.0})


@configclass
class SafeFallEnvCfg(ManagerBasedRLEnvCfg):
    scene: SafeFallSceneCfg = SafeFallSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = spec.DECIMATION
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = spec.SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.origin_type = "world"
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SafeFallPlayEnvCfg(SafeFallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
