"""Screen many lighting presets for ones that degrade the visual PPO experts, as a grid of success
rates (one row per preset, one column per task) plus a grouped bar chart with SEM error bars.

    python expert_preset_screen.py [--presets NAME ...] [--episodes N] [--seed N] [--gpus 0 1] [--workers-per-gpu 2]

Episodes, success metric and per-condition subprocesses are as in `expert_lighting_success.py`;
conditions are spread over `--gpus`, `--workers-per-gpu` at a time on each.
"""

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path
from queue import Queue

import numpy as np

HERE = Path(__file__).resolve().parent

from environment_screenshots import TASKS  # noqa: E402

DEFAULT_PRESETS = (
    "default",
    # exposure
    "dim", "very-dim",
    # colour temperature
    "warm", "very-warm", "warm-set-0.5",
    "cool", "very-cool", "cool-set-1", "cool-set-1.6",
    # key-light direction and shadows
    "side-set-0.5", "side", "shadows",
    # table appearance
    "table-set-0.6", "dark-table", "table-set-0.2",
    # object appearance
    "object-hue-15", "object-hue-60", "object-hue-90", "object-hue-180",
)


def run(task_id, lighting, episodes, seed, device):
    proc = subprocess.run(
        [sys.executable, str(HERE / "expert_lighting_success.py"), "--one", task_id, lighting,
         "--episodes", str(episodes), "--seed", str(seed), "--device", device],
        capture_output=True, text=True,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if proc.returncode != 0 or not lines:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-4000:])
        raise RuntimeError(f"{task_id} under {lighting} failed (exit {proc.returncode})")
    return json.loads(lines[-1][len("RESULT "):])


def plot(results, presets, episodes, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 0.42 * len(presets) + 1.2))
    height = 0.8 / len(TASKS)
    y = np.arange(len(presets))
    for i, ((title, task_id), color) in enumerate(zip(TASKS, ("#4c72b0", "#dd8452", "#55a868"))):
        outcomes = [np.asarray(results[task_id][p], dtype=float) for p in presets]
        means = [o.mean() for o in outcomes]
        # SEM of a Bernoulli sample: sample std (ddof=1) over sqrt(n)
        sems = [o.std(ddof=1) / math.sqrt(len(o)) for o in outcomes]
        ax.barh(y + (i - 1) * height, means, height, xerr=sems, capsize=2, color=color,
                label=title, error_kw=dict(lw=0.8))
    ax.set_yticks(y, presets)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"Success rate ({episodes} episodes, ±SEM)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    for path in (out, out.with_suffix(".pdf")):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    parser.add_argument("--episodes", type=int, default=20, help="episodes per condition, run as parallel envs")
    parser.add_argument("--seed", type=int, default=0, help="reset seed, shared by every condition")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--out", type=Path, default=HERE / "expert_preset_screen.png")
    args = parser.parse_args()

    devices = Queue()
    for _ in range(args.workers_per_gpu):
        for gpu in args.gpus:
            devices.put(f"cuda:{gpu}")

    def job(task_id, lighting):
        device = devices.get()
        try:
            outcomes = run(task_id, lighting, args.episodes, args.seed, device)
        finally:
            devices.put(device)
        print(f"{task_id:22s} {lighting:22s} {sum(outcomes):2d}/{args.episodes}", flush=True)
        return task_id, lighting, outcomes

    conditions = [(task_id, p) for _, task_id in TASKS for p in args.presets]
    results = {task_id: {} for _, task_id in TASKS}
    with ThreadPoolExecutor(devices.qsize()) as pool:
        for task_id, lighting, outcomes in pool.map(lambda c: job(*c), conditions):
            results[task_id][lighting] = outcomes

    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(
        dict(episodes=args.episodes, seed=args.seed, presets=list(args.presets), results=results), indent=2
    ))
    print(f"wrote {json_path.resolve()}")

    print(f"\n{'preset':22s}" + "".join(f"{title:>22s}" for title, _ in TASKS))
    for p in args.presets:
        print(f"{p:22s}" + "".join(f"{sum(results[t][p]):>19d}/{args.episodes}" for _, t in TASKS))
    plot(results, list(args.presets), args.episodes, args.out)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1] / "tools"))
    main()
