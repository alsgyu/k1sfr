from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from booster_assets import BOOSTER_ASSETS_DIR
from JY_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from JY_train.tasks.manager_based.scratch_symmetry.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from .tracking_env_cfg import TrackingEnvCfg


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = 1.0 # htwk 형태로 1.0 변경 -> default_joint_pos + action * 1.0 이 될 것

@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()


@configclass
class RoughWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg): # 평지용 환경 설정을 상속 후, 지형만 랜덤 rough terrain으로 바꾸는 설정
    def __post_init__(self):
        super().__post_init__()
        #self.events.kick_robot = None
        #self.events.external_push = None
        self.scene.terrain.terrain_type = "generator" # 지형 타입 변경 → 설정값에 따라 높낮이가 있는 지형 생성
        self.scene.terrain.debug_vis = False        # 지형 시각화 끄기
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),            # 각 terrain tile 하나의 크기 m
            border_width=20.0,            # 전체 지형 바깥쪽에 추가되는 경계 영역 폭
            num_rows=5,                   # 地形网格行数
            num_cols=10,                  # 地形网格列数
            horizontal_scale=0.1,         # 水平分辨率
            vertical_scale=0.005,         # 垂直分辨率
            slope_threshold=0.75,         # 网格简化阈值
            use_cache=False,              # 每次重新生成地形
            curriculum=False,              # terrain 난이도 curriculum을 사용하지 않음
            sub_terrains={
                # 80%接近平面的地形（非常平滑）
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(0.0, 0.005),    # 高度波动0-0.5cm（几乎平坦）
                    noise_step=0.005,            # 噪声步长0.5cm
                    border_width=0.25,
                ),
                # 20%随机粗糙地形
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.0,
                    noise_range=(-0.015, 0.015),    # 高度波动±1.5cm
                    noise_step=0.005,               # 噪声步长0.5cm
                    border_width=0.25,
                ),
            },
        )


@configclass
class PlayFlatWoStateEstimationEnvCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.kick_robot = None
        self.events.external_push = None

        self.commands.motion.play_velocity_command = (1.5, 0.0, 0.0)
        self.commands.motion.default_internal_parameters = (1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
