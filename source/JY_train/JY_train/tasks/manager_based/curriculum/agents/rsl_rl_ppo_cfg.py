from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 # 정책 업데이트 한 번 전에 24step의 경험을 수집
    max_iterations = 20000  # [100%] 20k curriculum 기준
    save_interval = 100
    experiment_name = "locomotion"
    empirical_normalization = True # 관측값의 평균 분산을 학습 중에 계산해 정규화
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.15,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, # critic value loss 계수
        use_clipped_value_loss=True, # critic 예측값이 한번 업데이트에서 너무 크게 바뀌지 않도록 제한
        clip_param=0.2, # 정책이 한 번에 너무 크게 변하는 걸 막음
        entropy_coef=0.01, # 행동 분포가 너무 빨리 좁아지는 것을 막는 탐색 보너스 -> 높이면 행동이 다양해짐
        num_learning_epochs=5, # rollout 데이터 5번 반복 학습
        num_mini_batches=4,
        learning_rate=1e-5,
        schedule="adaptive", # learning rate를 KL divergence에 따라 조정
        gamma=0.99, # discount factor
        lam=0.95, # GAE lambda 
        desired_kl=0.01,
        max_grad_norm=1.0, # gradient clipping
    )


LOW_FREQ_SCALE = 0.5 # 정책 실행 주파수를 절반으로 낮춤


@configclass
class BaseLowFreqPPORunnerCfg(BasePPORunnerCfg): # config 객체가 생성된 뒤 실행되는 후처리 함수
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
