"""VIGOR (Unified Perceptive Fall Safety) for the 22-DoF Booster K1.

Importing the package registers ``VigorTeacherActorCritic`` with rsl_rl, so the
stock scripts/rsl_rl/train.py can train the teacher.
"""

from . import modules  # noqa: F401
