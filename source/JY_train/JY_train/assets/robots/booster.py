import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import *
from JY_train.assets.robots import actuator
from JY_train.assets.robots.actuator import (
    BoosterDelayedImplicitActuatorCfg,
    BoosterDelayedPDActuatorCfg,
    DelayedImplicitActuatorCfg
)

from booster_assets import BOOSTER_ASSETS_DIR

BOOSTER_K1_CFG = ArticulationCfg( # 로봇을 어떻게 생성할지 설정하는 spawn 
    spawn=sim_utils.UrdfFileCfg( # urdf 이용
        fix_base=False, # base 몸통이 자유롭게 움직임 여부 -> 월드 고정 여부
        replace_cylinders_with_capsules=False, # 원통 충돌체를 캡슐로 바꿀지
        asset_path=f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_jy_locomotion.urdf",
        activate_contact_sensors=True, # 접촉 센서 활성화
        rigid_props=sim_utils.RigidBodyPropertiesCfg( # 로봇을 구성하는 각 강체 링크의 물리 특성 설정
            disable_gravity=False, # 중력 적용
            retain_accelerations=False, # 가속도 유지 여부 -> 물리 시뮬 단계 끝난 후 계산된 가속도 값을 유지할지
            linear_damping=0.0, # 선형 감쇠
            angular_damping=0.0, # 각속도 감쇠
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0, # 최대 침투 해소 속도
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg( # 로봇 전체 관절 연결 구조와 물리 솔버 설정
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg( # urdf -> sim 형식으로 변환할 때 조인트에 생성되는 drive 설정
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0) # 조인트 pd gain
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57), # 초기 base 위치 
        joint_pos={
            #"Left_Shoulder_Roll": 0.0,
            #"Right_Shoulder_Roll": 0.0,
            #"Left_Shoulder_Yaw" : -1.57,
            #"Right_Shoulder_Yaw" : 1.57,
            # 추가
            ".*_Hip_Pitch": -0.2,
            ".*_Knee_Pitch": 0.4,
            ".*_Ankle_Pitch": -0.2,
        },
        joint_vel={".*": 0.0}, # 초기 관절 속도
    ),
    soft_joint_pos_limit_factor=0.9, # urdf에 정의된 실제 관절 위치 제한보다 조금 좁은 범위를 정책이 사용하도록

    actuators={
        "legs": BoosterDelayedPDActuatorCfg(
            max_delay=18,
            min_delay=0,
            joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            stiffness={
                ".*_Hip_Pitch": 80.0,
                ".*_Hip_Roll": 80.0,
                ".*_Hip_Yaw": 80.0,
                ".*_Knee_Pitch": 80.0,
            },
            damping={
                ".*_Hip_Pitch": 2.0,
                ".*_Hip_Roll": 2.0,
                ".*_Hip_Yaw": 2.0,
                ".*_Knee_Pitch": 2.0,
            },
            booster_joint_cfgs={
                ".*_Hip_Pitch": actuator.BoosterJointE6408(),
                ".*_Hip_Roll": actuator.BoosterJointE4315(),
                ".*_Hip_Yaw": actuator.BoosterJointE4310(),
                ".*_Knee_Pitch": actuator.BoosterJointE6416(),
            },
        ),
        "feet": BoosterDelayedPDActuatorCfg(
            max_delay=9,
            min_delay=0,
            joint_names_expr=[
                ".*_Ankle_Pitch",
                ".*_Ankle_Roll",
            ],
            stiffness={
                ".*_Ankle_Pitch": 30.0,
                ".*_Ankle_Roll": 30.0,
            },
            damping={
                ".*_Ankle_Pitch": 2.0,
                ".*_Ankle_Roll": 2.0,
            },
            booster_joint_cfgs={
                ".*_Ankle_Pitch": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=0,
                ),
                ".*_Ankle_Roll": actuator.BoosterK1AnkleParaWrapperCfg(
                    base_joint_cfg=actuator.BoosterJointE4310(),
                    serial_index=1,
                ),
            },
        ),

    }
)

K1_ACTION_SCALE = {}
for a in BOOSTER_K1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            K1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

print(f'{BOOSTER_K1_CFG.actuators=}')
print(f'{K1_ACTION_SCALE=}')
