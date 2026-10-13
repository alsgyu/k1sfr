"""PPO configurations for K1 mirror loss and symmetry augmentation."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from JY_train.tasks.manager_based.scratch_symmetry.agents.rsl_rl_ppo_cfg import (
    BasePPORunnerCfg,
)

from .symmetry import data_augmentation_func


def _symmetry_cfg(*, augmentation: bool, mirror_loss: bool, coeff: float):
    return RslRlSymmetryCfg(
        use_data_augmentation=augmentation,
        use_mirror_loss=mirror_loss,
        data_augmentation_func=data_augmentation_func,
        mirror_loss_coeff=coeff,
    )


@configclass
class _BaseSymmetryPPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 30000
    experiment_name = "JY_walk_001_symmetry"


@configclass
class MirrorLossPPORunnerCfg(_BaseSymmetryPPORunnerCfg):
    """PPO with auxiliary policy-mean mirror loss only."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = _symmetry_cfg(
            augmentation=False, mirror_loss=True, coeff=0.01
        )


@configclass
class MirrorAugPPORunnerCfg(_BaseSymmetryPPORunnerCfg):
    """PPO with original+mirrored policy and critic samples."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = _symmetry_cfg(
            augmentation=True, mirror_loss=False, coeff=0.0
        )


@configclass
class SymmetryPPORunnerCfg(_BaseSymmetryPPORunnerCfg):
    """PPO with mirror augmentation and a small auxiliary mirror loss."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = _symmetry_cfg(
            augmentation=True, mirror_loss=True, coeff=0.005
        )
