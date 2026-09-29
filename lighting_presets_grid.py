"""Render the pairwise lighting-preset grids: every pair of `PRESETS` stacked ("a+b"), in the
upper triangle, with each preset alone on the diagonal. Two figures from the same envs -- the
task's three-quarter `render_camera` view, and the wrist `hand_camera` a policy observes.

    python lighting_presets_grid.py [--task ID] [--distractors] [--out-prefix PATH] [--seed N]

`--distractors` adds a "distractors" row and column (the task's `distractors=True` scene, see
`custom_maniskill_tasks.DistractorsMixin`) under every preset. Only tasks that take the kwarg
support it -- currently LiftPegUpright-v1.1.

Needs a real render backend (SAPIEN/Vulkan), so run it inside the project's apptainer container
with --nv on a GPU node.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from custom_maniskill_tasks import make_env

HERE = Path(__file__).resolve().parent

PRESETS = (
    "very-dim", "dim", "bright", "very-bright", "very-cool",
    "cool", "warm", "very-warm", "side", "shadows",
)
DISTRACTORS = "distractors"
"""Not a lighting preset: the scene with `distractors=True`, under default lighting on the
diagonal and under the other preset off it."""

HAND_CAMERA = "hand_camera"


def render_cell(task, names, seed):
    """(render_camera frame, hand_camera frame) for the stack of `names` at reset."""
    lighting = [n for n in names if n != DISTRACTORS]
    extra = {"distractors": True} if DISTRACTORS in names else {}
    env = make_env(
        task,
        obs_mode="rgb",
        camera_view="wrist",
        lighting="+".join(lighting) if lighting else "default",
        sim_backend="physx_cpu",
        **extra,
    )
    obs, _ = env.reset(seed=seed)
    hand = np.asarray(obs["sensor_data"][HAND_CAMERA]["rgb"].cpu())[0]
    view = np.asarray(env.render().cpu())[0]
    env.close()
    return view, hand


def plot_grid(frames, labels, title, path):
    n = len(labels)
    fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < row:
                ax.set_facecolor("#eeeeee")
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.imshow(frames[row, col])
                for spine in ax.spines.values():
                    spine.set_visible(col == row)
                    spine.set_color("grey")
                    spine.set_linewidth(2)
            if row == 0:
                ax.set_title(labels[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(labels[row], rotation=0, ha="right", va="center", fontsize=11)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", default="PushCube-v1.1")
    parser.add_argument("--distractors", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=HERE / "lighting_presets")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels = PRESETS + ((DISTRACTORS,) if args.distractors else ())
    n = len(labels)
    views = np.empty((n, n), dtype=object)
    hands = np.empty((n, n), dtype=object)
    for row in range(n):
        for col in range(row, n):
            names = (labels[row],) if row == col else (labels[row], labels[col])
            views[row, col], hands[row, col] = render_cell(args.task, names, args.seed)
            print(f"  {'+'.join(names)}")

    subject = f"{args.task}{', with distractors' if args.distractors else ''}"
    plot_grid(
        views, labels,
        f"Lighting presets, stacked pairwise, {subject} (diagonal = single preset alone)",
        args.out_prefix.with_name(args.out_prefix.name + ".png"),
    )
    plot_grid(
        hands, labels,
        f"Lighting presets, stacked pairwise, {subject}, {HAND_CAMERA} view "
        "(diagonal = single preset alone)",
        args.out_prefix.with_name(args.out_prefix.name + "_hand_camera.png"),
    )


if __name__ == "__main__":
    main()
