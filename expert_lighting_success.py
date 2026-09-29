"""Success rate of each task's visual PPO expert under the default lighting and the three visual
perturbations of `environment_screenshots.py`, as one bar chart per task with SEM error bars.

    python expert_lighting_success.py [--episodes N] [--seed N] [--device cuda:N] [--out PATH]

Each (task, lighting) condition runs in a fresh subprocess (`--one TASK LIGHTING`), because SAPIEN
caches edited textures for the whole process and would carry one condition's edits into the next.
Every condition resets from the same seed on the same backend, so they all see the same N starting
configurations and differ only in appearance.

The expert acts deterministically (`actor_mean`), and an episode counts as a success if the task's
`evaluate()` reports success at any step of its 50-step horizon (`success_once`, the metric the
experts were trained against).
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from environment_screenshots import LIGHTING_COLUMNS, TASKS, expert_folder, load_expert_config  # noqa: E402

CONDITIONS = (("Default", "default"),) + LIGHTING_COLUMNS


def run_condition(task_id, lighting, episodes, seed, device_name):
    """Per-episode success_once of the expert on `task_id` under `lighting`, as a list of bools."""
    import torch
    from mani_skill.utils import gym_utils
    from make_expert_demos import build_env, load_agent, resolve_backends
    from ppo_visual_expert_fast import as_obs_td

    device = torch.device(device_name)
    backends = resolve_backends(device)
    cfg = load_expert_config(task_id)
    env = build_env(cfg, episodes, dict(backends, lighting=lighting))
    try:
        obs, _ = env.reset(seed=seed)
        agent = load_agent(
            as_obs_td(obs, episodes),
            math.prod(env.get_wrapper_attr("single_action_space").shape),
            expert_folder(task_id) / "best_ckpt.pt",
            device,
        )
        horizon = gym_utils.find_max_episode_steps_value(env)
        success = torch.zeros(episodes, dtype=torch.bool, device=device)
        for _ in range(horizon):
            with torch.no_grad():
                action = agent.get_action(as_obs_td(obs, episodes))
            obs, _, _, _, info = env.step(action)
            success |= info["success"].to(device)
        return success.cpu().tolist()
    finally:
        env.close()


def plot(results, episodes, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TASKS), figsize=(5 * len(TASKS), 4.2), sharey=True)
    colors = ("#4c72b0", "#dd8452", "#c44e52", "#55a868")
    x = np.arange(len(CONDITIONS))
    for ax, (title, task_id) in zip(axes, TASKS):
        outcomes = [np.asarray(results[task_id][lighting], dtype=float) for _, lighting in CONDITIONS]
        means = [o.mean() for o in outcomes]
        # SEM of a Bernoulli sample: sample std (ddof=1) over sqrt(n)
        sems = [o.std(ddof=1) / math.sqrt(len(o)) for o in outcomes]
        ax.bar(x, means, yerr=sems, capsize=5, color=colors, edgecolor="black", linewidth=0.6)
        # above the error bar rather than inside the bar, so a zero-height bar still gets a label
        for xi, m, s in zip(x, means, sems):
            ax.text(xi, min(m + s, 1.0) + 0.02, f"{m:.1f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x, [name for name, _ in CONDITIONS], rotation=15)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"Success rate ({episodes} episodes, ±SEM)")
    fig.tight_layout()
    for path in (out, out.with_suffix(".pdf")):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10, help="episodes per condition, run as parallel envs")
    parser.add_argument("--seed", type=int, default=0, help="reset seed, shared by every condition")
    parser.add_argument("--device", default="cuda:0", help="torch device; also pins physx and sapien")
    parser.add_argument("--out", type=Path, default=HERE / "expert_lighting_success.png")
    parser.add_argument("--one", nargs=2, metavar=("TASK", "LIGHTING"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        outcomes = run_condition(*args.one, args.episodes, args.seed, args.device)
        print("RESULT " + json.dumps(outcomes))
        return

    results = {}
    for _, task_id in TASKS:
        results[task_id] = {}
        for _, lighting in CONDITIONS:
            proc = subprocess.run(
                [sys.executable, __file__, "--one", task_id, lighting, "--episodes", str(args.episodes),
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
    json_path.write_text(json.dumps(dict(episodes=args.episodes, seed=args.seed, results=results), indent=2))
    print(f"wrote {json_path}")
    plot(results, args.episodes, args.out)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1] / "tools"))
    main()
