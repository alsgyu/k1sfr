"""K1 safe fall on flat ground: spread the landing over fists, knees / hips and feet and end in a
multi-contact hand-off pose.

No demonstrations and no stand-up (another team handles getting up).  Episodes start standing
or walking, pushed in a random direction, and hand over to the policy at a 20 deg tilt; the
reward (``safefall_math``) minimises a fragility-weighted sum of squared impacts over all links
(so the load is shared and the joints give), keeps head, chest, back and upper arms off the
ground, rewards resting in one of two fixed hand-off poses for the get-up (kneeling on fists,
knees and toes; sitting on the hips with the fists behind), and keeps forward falls chest down
and backward falls chest up.
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
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import JY_train.tasks.manager_based.vigor.mdp as mdp
from JY_train.assets.robots.booster_k1_full import BOOSTER_K1_FULL_CFG, K1_FULL_ACTION_SCALE
from JY_train.tasks.manager_based.vigor import safefall_math as sf
from JY_train.tasks.manager_based.vigor import spec

EPISODE_LENGTH_S = sf.EPISODE_S
HISTORY = 3
# observation scaling and bounds (no empirical normalisation; as the MuJoCo check); clips are in raw units
ANG_VEL_SCALE, JOINT_VEL_SCALE, CONTACT_CLIP = 0.25, 0.05, 5.0
ANG_VEL_CLIP, JOINT_VEL_CLIP, LIN_VEL_CLIP = (-40.0, 40.0), (-40.0, 40.0), (-15.0, 15.0)
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
    # ground contacts only (force_matrix_w_history): the K1 collides with itself, and the net forces counted a fist
    # pressed against the trunk as a chest / back hit and a fist on the thigh as a fist on the ground
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=spec.DECIMATION, track_air_time=False,
                                      filter_prim_paths_expr=[mdp.GROUND_COLLIDER])
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0))


@configclass
class CommandsCfg:
    monitor = mdp.SafeFallMonitorCfg()


@configclass
class ActionsCfg:
    # offsets from the hand-off pose of the latched side (safefall_math.reflex_targets)
    joint_pos = mdp.ReflexJointPositionActionCfg(
        asset_name="robot", joint_names=spec.ACTION_JOINT_NAMES, scale=ACTION_SCALE, preserve_order=True
    )
    head = mdp.HoldJointPositionActionCfg(asset_name="robot", joint_names=spec.HEAD_JOINT_NAMES)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable: IMU + joint encoders + last action, last HISTORY steps.  Fixed scales (no empirical
        normalisation) so that left-right mirroring commutes with them (safefall_symmetry)."""

        # noise as the team's walking task (scratch_symmetry walk_001); joint positions with encoder offsets
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=ANG_VEL_CLIP,
                               scale=ANG_VEL_SCALE, history_length=HISTORY)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=HISTORY)
        joint_pos = ObsTerm(func=mdp.joint_pos_measured, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=HISTORY)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), clip=JOINT_VEL_CLIP,
                            scale=JOINT_VEL_SCALE, history_length=HISTORY)
        last_action = ObsTerm(func=mdp.last_action, history_length=HISTORY)
        fall_time = ObsTerm(func=mdp.fall_time, history_length=HISTORY)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=ANG_VEL_CLIP, scale=ANG_VEL_SCALE)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=JOINT_VEL_CLIP, scale=JOINT_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        fall_time = ObsTerm(func=mdp.fall_time)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=LIN_VEL_CLIP)
        heights = ObsTerm(func=mdp.body_heights)
        contact_forces = ObsTerm(func=mdp.contact_force_groups, clip=(0.0, CONTACT_CLIP))
        facing = ObsTerm(func=mdp.fall_facing)
        triggered = ObsTerm(func=mdp.fall_triggered)

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
            "static_friction_range": (0.0, 2.0),  # ranges as the team's walking task
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (0.1, 0.9),
            "num_buckets": 64,
        },
    )
    trunk_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk"), "mass_distribution_params": (0.8, 1.2), "operation": "scale"},
    )
    link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="(?!Trunk$).*"), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )
    trunk_com = EventTerm(
        func=mdp.randomize_trunk_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}},
    )
    joint_bias = EventTerm(func=mdp.randomize_joint_bias, mode="startup")  # leg encoder offsets +-0.03 rad
    # kick every 4 s as the team's walking task: here it lands while the hand-off pose is held
    kick = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
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


def _reg(term: str, weight: float) -> RewTerm:
    return RewTerm(func=mdp.guarded_reward, weight=weight, params={"term": term})


@configclass
class RewardsCfg:
    # task (safefall_math.WEIGHTS)
    upright = _sf("upright")
    support = _sf("support")
    hands_lead = _sf("hands_lead")
    handoff_pose = _sf("handoff_pose")
    impact = _sf("impact")
    head_contact = _sf("head_contact")
    head_impact = _sf("head_impact")
    head_low = _sf("head_low")
    trunk_contact = _sf("trunk_contact")
    head_descent = _sf("head_descent")
    trunk_descent = _sf("trunk_descent")
    fist_descent = _sf("fist_descent")
    early_contact = _sf("early_contact")
    foot_lift = _sf("foot_lift")
    leg_spread = _sf("leg_spread")
    arm_width = _sf("arm_width")
    fist_relift = _sf("fist_relift")
    facing = _sf("facing")
    # regularisation (same weights as scripts/vigor/safefall_mujoco.py REG), guarded against blow-ups
    action_rate = _reg("action_rate_l2", -0.01)
    joint_pos_limit = _reg("joint_pos_limits", -1.0)
    joint_vel = _reg("joint_vel_l2", -1.0e-4)
    joint_acc = _reg("joint_acc_l2", -2.5e-7)
    torque = _reg("joint_torques_l2", -1.0e-5)
    residual = RewTerm(func=mdp.safefall_residual, weight=sf.RESIDUAL_WEIGHT)


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
        self.events.kick = None  # eval scores the fall itself (comparable across versions); training keeps it
