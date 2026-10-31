"""Print a training run's TensorBoard scalars as a compact text table (to paste from the GPU box).

    python scripts/vigor/summarize_run.py                       # latest k1_vigor_teacher_v2 run
    python scripts/vigor/summarize_run.py --run flat --points 10
    python scripts/vigor/summarize_run.py --experiment k1_vigor_student_v2

Each column is the mean over an equal slice of the iterations logged so far, labelled with the
slice's last iteration.  Safe to run while training is still writing the run.
"""

import argparse
import os
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# printed in this order; the per-regime outcome rates come first
GROUPS = [
    "Metrics/motion/push/",
    "Metrics/motion/handover/",
    "Metrics/motion/fall/",
    "Metrics/motion/standup/",
    "Metrics/",
    "Train/",
    "Episode_Reward/",
    "Episode_Termination/",
    "Policy/",
    "Loss/",
    "Perf/",
]


def group_of(tag: str) -> str:
    return next((g for g in GROUPS if tag.startswith(g)), tag.split("/")[0] + "/")


def fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if 0.001 <= abs(v) < 1000 or v == 0 else f"{v:.2e}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_root", default="logs/rsl_rl")
    p.add_argument("--experiment", default="k1_vigor_teacher_v2")
    p.add_argument("--run", default=None, help="run folder regex (latest if omitted)")
    p.add_argument("--points", type=int, default=8)
    p.add_argument("--tags", default=None, help="only tags matching this regex")
    a = p.parse_args()

    root = os.path.join(a.log_root, a.experiment)
    runs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and (not a.run or re.search(a.run, d)))
    if not runs:
        raise SystemExit(f"no runs matching {a.run!r} in {root}")
    run_dir = os.path.join(root, runs[-1])
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    tags = [t for t in acc.Tags()["scalars"] if not a.tags or re.search(a.tags, t)]
    series = {t: acc.Scalars(t) for t in tags}
    series = {t: s for t, s in series.items() if s}
    if not series:
        raise SystemExit(f"no scalars in {run_dir} yet")

    first = min(s[0].step for s in series.values())
    last = max(s[-1].step for s in series.values())
    points = max(1, min(a.points, last - first + 1))
    edges = [first + (last - first + 1) * i / points for i in range(points + 1)]
    wall = [e.wall_time for s in series.values() for e in s]
    hours = (max(wall) - min(wall)) / 3600
    print(f"run: {run_dir}")
    print(f"iterations {first}..{last}, {hours:.2f} h" + (f", {hours * 3600 / (last - first):.2f} s/it" if last > first else ""))

    def binned(s) -> list[float | None]:
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            v = [e.value for e in s if lo <= e.step < hi]
            out.append(sum(v) / len(v) if v else None)
        return out

    order = sorted(series, key=lambda t: (GROUPS.index(g) if (g := group_of(t)) in GROUPS else len(GROUPS), g, t))
    width = max(len(t) - len(group_of(t)) for t in order) + 2
    header = " " * width + "".join(f"{'<=' + str(int(hi) - 1):>10}" for hi in edges[1:])
    current = None
    for t in order:
        g = group_of(t)
        if g != current:
            current = g
            print(f"\n[{g.rstrip('/')}]")
            print(header)
        name = t[len(g):]
        print(f"{name:<{width}}" + "".join(f"{fmt(v):>10}" for v in binned(series[t])))


if __name__ == "__main__":
    main()
