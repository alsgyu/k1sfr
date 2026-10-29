"""Export a student checkpoint to TorchScript/ONNX + policy_spec.json.  Runs without Isaac (GPU box or laptop).

    python scripts/vigor/export_policy.py logs/rsl_rl/k1_vigor_student/<run>/model_6000.pt [--out_dir DIR]
"""

import argparse

from common import export_student

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()
    export_student(a.checkpoint, a.out_dir)
