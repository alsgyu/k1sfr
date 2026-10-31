from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from JY_train.tasks.manager_based.vigor import spec


@configclass
class VigorTeacherActorCriticCfg(RslRlPpoActorCriticCfg):
    """Extra fields are forwarded to VigorTeacherActorCritic.__init__ by OnPolicyRunner."""

    class_name: str = "VigorTeacherActorCritic"
    proprio_dim: int = spec.PROPRIO_DIM
    latent_dim: int = spec.LATENT_DIM
    encoder_hidden_dims: list[int] = [256, 128]


@configclass
class VigorTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "k1_vigor_teacher_v2"  # v1 (heading-frame tracking) runs stay in k1_vigor_teacher
    empirical_normalization = True
    policy = VigorTeacherActorCriticCfg(
        init_noise_std=0.5,  # with the doubled action scale this keeps joint-space noise unchanged
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class VigorDistillCfg:
    """Student distillation (scripts/vigor/distill.py): BC + DAgger with latent supervision."""

    experiment_name: str = "k1_vigor_student_v2"
    teacher_experiment_name: str = "k1_vigor_teacher_v2"
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 6000
    save_interval: int = 250
    learning_rate: float = 5.0e-4
    num_learning_epochs: int = 4
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0
    # rho = alpha * rho_teacher + (1 - alpha) * rho_student, alpha: 1 -> 0
    alpha_anneal_fraction: float = 0.4
    lambda_latent: float = 1.0
    lambda_action: float = 1.0
    history: int = spec.STUDENT_HISTORY
    latent_dim: int = spec.LATENT_DIM
    actor_hidden_dims: list[int] = [512, 256, 128]
    fusion_hidden_dims: list[int] = [256]
    activation: str = "elu"
    init_head_from_teacher: bool = True
