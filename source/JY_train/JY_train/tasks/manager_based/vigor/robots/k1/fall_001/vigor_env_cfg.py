"""VIGOR fall-safety environment for the 22-DoF K1 (teacher, student and walk -> fall play).

Timing, leg/feet actuators, default stance and joint names are shared with the
walking tasks (curriculum2/walk_001) so a policy can take over mid-fall.
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

import JY_train.tasks.manager_based.vigor.mdp as mdp
from JY_train.assets.robots.booster_k1_full import BOOSTER_K1_FULL_CFG, K1_FULL_ACTION_SCALE
from JY_train.tasks.manager_based.vigor import spec

ACTION_SCALE = {k: v for k, v in K1_FULL_ACTION_SCALE.items() if k not in spec.HEAD_JOINT_NAMES}


@configclass
class VigorSceneCfg(InteractiveSceneCfg):
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
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = BOOSTER_K1_FULL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # privileged h_t: gravity-aligned grid that stays meaningful while the trunk rotates
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=spec.HEIGHT_SCAN_RESOLUTION, size=list(spec.HEIGHT_SCAN_SIZE)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class CommandsCfg:
    motion = mdp.VigorKeyframeCommandCfg(asset_name="robot", debug_vis=False)


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=spec.ACTION_JOINT_NAMES,
        scale=ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
    )
    head = mdp.HoldJointPositionActionCfg(asset_name="robot", joint_names=spec.HEAD_JOINT_NAMES)


def _proprio_terms(history: int = 0, noisy: bool = True) -> dict[str, ObsTerm]:
    """o_t = [omega, g, q, qdot, a_{t-1}] in spec.PROPRIO_TERMS order."""
    kw = {"history_length": history, "flatten_history_dim": True} if history else {}
    n = (lambda std: Gnoise(mean=0.0, std=std)) if noisy else (lambda std: None)
    return {
        "base_ang_vel": ObsTerm(func=mdp.base_ang_vel, noise=n(0.1), **kw),
        "projected_gravity": ObsTerm(func=mdp.projected_gravity, noise=n(0.02), **kw),
        "joint_pos": ObsTerm(func=mdp.joint_pos_rel, noise=n(0.01), **kw),
        "joint_vel": ObsTerm(func=mdp.joint_vel_rel, noise=n(0.1), **kw),
        "last_action": ObsTerm(func=mdp.last_action, **kw),
    }


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Teacher actor input [o_t | k_t | h_t]."""

        base_ang_vel = _proprio_terms()["base_ang_vel"]
        projected_gravity = _proprio_terms()["projected_gravity"]
        joint_pos = _proprio_terms()["joint_pos"]
        joint_vel = _proprio_terms()["joint_vel"]
        last_action = _proprio_terms()["last_action"]
        keyframes = ObsTerm(func=mdp.keyframe_goal, params={"command_name": "motion"})
        terrain = ObsTerm(func=mdp.terrain_context, params={"sensor_cfg": SceneEntityCfg("height_scanner")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        keypoints = ObsTerm(func=mdp.robot_keypoints, params={"command_name": "motion"})
        contacts = ObsTerm(
            func=mdp.contact_flags,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=spec.KEY_BODY_NAMES, preserve_order=True)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class StudentObsCfg(ObsGroup):
    """Deployable student input: noisy proprioceptive history o_{t-T:t} (term-major, oldest first)."""

    base_ang_vel = _proprio_terms(spec.STUDENT_HISTORY)["base_ang_vel"]
    projected_gravity = _proprio_terms(spec.STUDENT_HISTORY)["projected_gravity"]
    joint_pos = _proprio_terms(spec.STUDENT_HISTORY)["joint_pos"]
    joint_vel = _proprio_terms(spec.STUDENT_HISTORY)["joint_vel"]
    last_action = _proprio_terms(spec.STUDENT_HISTORY)["last_action"]

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 64,
        },
    )
    trunk_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    # order matters: clear old dropouts -> randomise gains -> start a new free-fall
    torque_dropout_clear = EventTerm(func=mdp.torque_dropout_clear, mode="reset")
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
    free_fall = EventTerm(
        func=mdp.torque_dropout_on_reset,
        mode="reset",
        params={"probability": 0.2, "joint_fraction": (0.3, 1.0), "duration_s": (0.1, 0.4)},
    )
    torque_dropout = EventTerm(
        func=mdp.torque_dropout_step,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"probability_per_s": 0.05, "joint_fraction": (0.05, 0.2), "duration_s": (0.1, 0.3)},
    )
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Paper Table I."""

    # motion imitation (root-relative, heading-invariant); the pose terms pay nothing while the root
    # sits more than 5 cm below the reference's (B-Human get-up), so lying with matching limbs does not pay
    track_body_pos = RewTerm(
        func=mdp.track_body_pos_exp, weight=1.25, params={"command_name": "motion", "std": 0.3, "height_sigma": 0.1}
    )
    track_body_rot = RewTerm(
        func=mdp.track_body_rot_exp, weight=0.5, params={"command_name": "motion", "std": 0.4, "height_sigma": 0.1}
    )
    track_body_lin_vel = RewTerm(
        func=mdp.track_body_lin_vel_exp, weight=0.125, params={"command_name": "motion", "std": 1.0}
    )
    track_body_ang_vel = RewTerm(
        func=mdp.track_body_ang_vel_exp, weight=0.125, params={"command_name": "motion", "std": 3.14}
    )
    track_joint_pos = RewTerm(
        func=mdp.track_joint_pos_exp, weight=0.5, params={"command_name": "motion", "std": 0.5, "height_sigma": 0.1}
    )
    track_joint_vel = RewTerm(func=mdp.track_joint_vel_exp, weight=0.125, params={"command_name": "motion", "std": 3.0})
    # regularization & safety
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    # paper uses -0.1; the K1 PD saturates in any forceful stand-up, so at -0.1 this term
    # outweighed all imitation rewards and the first teacher learned to stay down
    torque_limit = RewTerm(func=mdp.torque_limit_excess, weight=-0.005)
    joint_pos_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    joint_vel_limit = RewTerm(func=mdp.joint_vel_limits, weight=-5.0, params={"soft_ratio": 1.0})
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    momentum_change = RewTerm(func=mdp.momentum_change, weight=-5.0e-3)
    body_yank = RewTerm(
        func=mdp.body_yank, weight=-2.0e-6, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")}
    )
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # contacts against the reference: plant the hands when it braces; head (stereo camera) and trunk
    # may only touch where the reference does (the K1 get-ups lean on the head while pushing up)
    hand_contact = RewTerm(
        func=mdp.ref_contact_match,
        weight=0.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_hand_link", "right_hand_link"], preserve_order=True),
            "contact_ids": [spec.REF_CONTACT_NAMES.index("left_hand"), spec.REF_CONTACT_NAMES.index("right_hand")],
        },
    )
    head_contact = RewTerm(
        func=mdp.unexpected_contact,
        weight=-1.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=spec.HEAD_BODY_NAMES),
            "contact_id": spec.REF_CONTACT_NAMES.index("head"),
        },
    )
    trunk_contact = RewTerm(
        func=mdp.unexpected_contact,
        weight=-0.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"),
            "contact_id": spec.REF_CONTACT_NAMES.index("trunk"),
        },
    )
    # TODO(stairs): foothold edge penalty (BeamDojo) once stair terrains are used in training.
    # keep the head up: dense here (paper: post-recovery only) so catching a fall pays off
    head_height = RewTerm(func=mdp.head_height_exp, weight=0.5, params={"std": 0.3, "gated": True})
    # post-recovery stabilization (active once upright)
    base_lin_vel = RewTerm(func=mdp.base_lin_vel_l2_upright, weight=-0.01)
    base_ang_vel = RewTerm(func=mdp.base_ang_vel_l2_upright, weight=-0.05)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_state = DoneTerm(func=mdp.bad_state, params={"max_lin_vel": 15.0, "max_ang_vel": 40.0})


@configclass
class VigorEnvCfg(ManagerBasedRLEnvCfg):
    scene: VigorSceneCfg = VigorSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = spec.DECIMATION
        self.episode_length_s = spec.EPISODE_LENGTH_S
        self.sim.dt = spec.SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # whole-body ground contact during falls needs more contact patches than walking
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.viewer.origin_type = "world"
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
