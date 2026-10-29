import copy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab.terrains as terrain_gen

import JY_train.tasks.manager_based.vigor.mdp as mdp
from JY_train.tasks.manager_based.vigor import spec

from .vigor_env_cfg import StudentObsCfg, VigorEnvCfg

# Paper Fig. 4 terrain families, heights scaled ~0.7x for the 0.95 m K1.
VIGOR_ROUGH_TERRAINS = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.02, 0.06), num_waves=4, border_width=0.25),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.02), noise_step=0.005, border_width=0.25
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.35), platform_width=1.0, border_width=0.25
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.04, 0.12), step_width=0.3, platform_width=1.0, border_width=1.0
        ),
        "inverted_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15, step_height_range=(0.04, 0.12), step_width=0.3, platform_width=1.0, border_width=1.0
        ),
    },
)


@configclass
class TeacherFlatEnvCfg(VigorEnvCfg):
    """Stage 1: basic fall mitigation and stand-up on flat ground."""


@configclass
class TeacherRoughEnvCfg(TeacherFlatEnvCfg):
    """Stage 2: randomized non-flat terrains (resume from the flat checkpoint)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = copy.deepcopy(VIGOR_ROUGH_TERRAINS)
        self.scene.terrain.max_init_terrain_level = None
        # spread starts over slopes / stair edges instead of the tile centre
        self.commands.motion.xy_noise = 1.0


def _play(cfg: VigorEnvCfg) -> None:
    cfg.scene.num_envs = 32
    cfg.scene.env_spacing = 3.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.push = None
    cfg.events.torque_dropout = None
    cfg.commands.motion.debug_vis = True


@configclass
class TeacherFlatPlayEnvCfg(TeacherFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class TeacherRoughPlayEnvCfg(TeacherRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 6


@configclass
class StudentFlatEnvCfg(TeacherFlatEnvCfg):
    """Teacher env + the deployable 'student' observation group (scripts/vigor/distill.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.observations.student = StudentObsCfg()


@configclass
class StudentRoughEnvCfg(TeacherRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.observations.student = StudentObsCfg()


@configclass
class StudentFlatPlayEnvCfg(StudentFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.observations.student.enable_corruption = False


@configclass
class StudentRoughPlayEnvCfg(StudentRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.observations.student.enable_corruption = False


# --------------------------------------------------------------------------
# Walk -> fall-safety hand-off (scripts/vigor/play_walk_fall.py)
# --------------------------------------------------------------------------
@configclass
class _WalkFallObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # the runner builds both policies' inputs itself; this group only satisfies the manager
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

    policy: PolicyCfg = PolicyCfg()


@configclass
class _WalkFallActionsCfg:
    # absolute joint targets for all 22 joints, written by the runner
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=spec.ALL_JOINT_NAMES, scale=1.0, use_default_offset=False, preserve_order=True
    )


@configclass
class _WalkFallEventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"yaw": (-3.14, 3.14)}, "velocity_range": {}},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class _WalkFallTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class _EmptyCfg:
    pass


@configclass
class WalkFallPlayEnvCfg(TeacherFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 120.0
        self.observations = _WalkFallObservationsCfg()
        self.actions = _WalkFallActionsCfg()
        self.events = _WalkFallEventsCfg()
        self.terminations = _WalkFallTerminationsCfg()
        self.commands = _EmptyCfg()
        self.rewards = _EmptyCfg()


@configclass
class WalkFallRoughPlayEnvCfg(WalkFallPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = copy.deepcopy(VIGOR_ROUGH_TERRAINS)
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 6
