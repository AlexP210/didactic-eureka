"""Success rate of each task's visual PPO expert as the scene is made brighter, to find where the
brightness perturbation starts to break it.

    python expert_brightness_sweep.py [--scales 1 2.5 4 ...] [--episodes N] [--seed N] [--device cuda:N]

Brightness scale k is the preset `bright-set-<0.3k>-<k>`: the default's ambient (0.3) and key/fill
lights (1.0) both multiplied by k, so k=1 is the default scene and k=2.5 is the `bright-set-0.75-2.5`
of `environment_screenshots.py`. Episodes, success metric and per-condition subprocesses are as in
`expert_lighting_success.py`.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from environment_screenshots import TASKS  # noqa: E402

DEFAULT_AMBIENT = 0.3
DEFAULT_SCALES = (1, 2.5, 4, 6, 8, 12, 16, 24)


def preset(scale):
    return f"bright-set-{round(DEFAULT_AMBIENT * scale, 4):g}-{scale:g}"


def plot(results, scales, episodes, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (title, task_id), color in zip(TASKS, ("#4c72b0", "#dd8452", "#55a868")):
        outcomes = [np.asarray(results[task_id][preset(s)], dtype=float) for s in scales]
        means = np.array([o.mean() for o in outcomes])
        # SEM of a Bernoulli sample: sample std (ddof=1) over sqrt(n)
        sems = np.array([o.std(ddof=1) / math.sqrt(len(o)) for o in outcomes])
        ax.plot(scales, means, "o-", color=color, label=title)
        ax.fill_between(scales, means - sems, np.minimum(means + sems, 1.0), color=color, alpha=0.2)
    ax.axvline(1, color="grey", ls=":", lw=1)
    ax.axvline(2.5, color="grey", ls="--", lw=1)
    ax.text(1, 1.07, " default", fontsize=8, color="grey")
    ax.text(2.5, 1.07, " Bright", fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xticks(scales, [f"{s:g}" for s in scales])
    ax.minorticks_off()
    ax.set_xlabel("Brightness scale k  (ambient 0.3k, lights k)")
    ax.set_ylabel(f"Success rate ({episodes} episodes, ±SEM)")
    ax.set_ylim(-0.03, 1.12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    for path in (out, out.with_suffix(".pdf")):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--episodes", type=int, default=20, help="episodes per condition, run as parallel envs")
    parser.add_argument("--seed", type=int, default=0, help="reset seed, shared by every condition")
    parser.add_argument("--device", default="cuda:0", help="torch device; also pins physx and sapien")
    parser.add_argument("--out", type=Path, default=HERE / "expert_brightness_sweep.png")
    args = parser.parse_args()

    runner = HERE / "expert_lighting_success.py"
    results = {}
    for _, task_id in TASKS:
        results[task_id] = {}
        for scale in args.scales:
            lighting = preset(scale)
            proc = subprocess.run(
                [sys.executable, str(runner), "--one", task_id, lighting, "--episodes", str(args.episodes),
                 "--seed", str(args.seed), "--device", args.device],
                capture_output=True, text=True,
            )
            lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if proc.returncode != 0 or not lines:
                sys.stderr.write(proc.stdout + proc.stderr)
                raise RuntimeError(f"{task_id} under {lighting} failed (exit {proc.returncode})")
            results[task_id][lighting] = json.loads(lines[-1][len("RESULT "):])
            print(f"{task_id:22s} {lighting:22s} {sum(results[task_id][lighting])}/{args.episodes}", flush=True)

    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(
        dict(episodes=args.episodes, seed=args.seed, scales=list(args.scales), results=results), indent=2
    ))
    print(f"wrote {json_path.resolve()}")
    plot(results, list(args.scales), args.episodes, args.out)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1] / "tools"))
    main()
