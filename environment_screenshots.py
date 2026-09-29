"""Render a grid of env screenshots: one row per task, and six columns -- the episode's start
state, the wrist camera the policy observes at that start, the goal state the task's visual PPO
expert reaches from it, and the same start state under the three visual perturbations evaluated
(brighter lights, warmer lights, and the task objects' colours shifted round the hue wheel).

    python environment_screenshots.py [--out PATH] [--seed N] [--device cuda:N]

Needs a real render backend (SAPIEN/Vulkan), so this has to run where that's available --
inside the project's apptainer container with --nv on a GPU node, not on the login node.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mani_skill.utils import gym_utils

HERE = Path(__file__).resolve().parent

# the expert's env and network are rebuilt by the script that recorded its demos, so the rollout
# here cannot drift from the one the datasets came from
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from make_expert_demos import build_env, load_agent, resolve_backends  # noqa: E402
from ppo_visual_expert_fast import as_obs_td  # noqa: E402

EXPERTS_DIR = Path("/path/to/datasets/maniskill")

# row title, task id.
TASKS = (
    ("PushCube-v1.1", "PushCube-v1.1"),
    ("LiftPegUpright-v1.1", "LiftPegUpright-v1.1"),
    ("PickCube-v1.1", "PickCube-v1.1"),
)

# column title, lighting preset. The first three columns are under the stock ("default")
# condition; these are the start frame again, under each perturbation.
LIGHTING_COLUMNS = (
    ("Bright", "bright-set-0.75-2.5"),
    ("Warm", "warm-set-1.1"),
    ("Colour-Shift", "object-hue-30"),
)
FONT_SIZE = 38
"""Column titles and row labels. The row labels break before the task version, since a one-line
"LiftPegUpright-v1.1" at this size is taller than its row."""
COLUMN_TITLES = ("Start", "Visual Obs", "Goal") + tuple(title for title, _ in LIGHTING_COLUMNS)
"""All but "Visual Obs" come from `env.render()`, i.e. each task's own `render_camera`
(`_default_human_render_camera_configs`): the 512x512 three-quarter view ManiSkill's docs and
videos show, which no `camera_view` here touches. "Visual Obs" is the wrist camera a policy
actually observes -- a different camera, which is why the viewpoint changes."""


def expert_folder(task_id):
    return EXPERTS_DIR / f"{task_id}-ppo-visual-wrist-224-pd_ee_delta_pos"


def load_expert_config(task_id):
    """The env the expert was trained and rolled out on, read back from its demo summary.

    `demos/demo_summary.json` rather than `training_summary.json`, because only the former is in
    every one of these folders; it carries the same env fields, copied from the training run.
    """
    summary = json.loads((expert_folder(task_id) / "demos" / "demo_summary.json").read_text())
    assert summary["env_id"] == task_id, f"{expert_folder(task_id)} holds a {summary['env_id']} run"
    return dict(
        env_id=summary["env_id"],
        camera_view=summary["camera_view"],
        camera_resolution=summary["camera_resolution"],
        include_state=summary["include_state"],
        control_mode=summary["control_mode"],
    )


def _render(env):
    return env.unwrapped.render_rgb_array()[0].cpu().numpy()


def render_expert_episode(task_id, cfg, seed, backends, device):
    """The (start, wrist, goal) frames of one expert episode, as (H, W, 3) uint8 arrays.

    The expert acts deterministically (`actor_mean`, no noise) from the seeded reset, and the goal
    frame is the first step the task's own `evaluate()` reports success on. An episode that never
    succeeds fails loudly rather than passing off its last frame as a goal.
    """
    env = build_env(cfg, 1, backends)
    try:
        obs, _ = env.reset(seed=seed)
        start = _render(env)
        # a copy: the rgb leaf is ManiSkill's live camera buffer, overwritten by the next step
        wrist = obs["rgb"][0].clone().cpu().numpy()
        agent = load_agent(
            as_obs_td(obs, 1),
            math.prod(env.get_wrapper_attr("single_action_space").shape),
            expert_folder(task_id) / "best_ckpt.pt",
            device,
        )
        horizon = gym_utils.find_max_episode_steps_value(env)
        for _ in range(horizon):
            with torch.no_grad():
                action = agent.get_action(as_obs_td(obs, 1))
            obs, _, _, _, info = env.step(action)
            if bool(info["success"][0]):
                return start, wrist, _render(env)
        raise AssertionError(
            f"the {task_id} expert did not reach success within {horizon} steps from seed {seed}; "
            "try another --seed"
        )
    finally:
        env.close()


def render_relit_start(cfg, seed, backends, lighting):
    """The render-camera frame at the start of the same episode, under `lighting`.

    Built through the same `build_env`, camera view and sim backend as the expert episode: the
    reset randomization draws from the sim device's torch generator, so a different backend would
    lay out a different scene from the same seed.
    """
    env = build_env(cfg, 1, dict(backends, lighting=lighting))
    try:
        env.reset(seed=seed)
        return _render(env)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=HERE / "environment_screenshots.png",
        help="path to write the grid image to; a PDF copy is written alongside it, with the "
        "same name and a .pdf suffix",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="reset seed, shared across every task/lighting condition so each row's frames show "
        "the same starting configuration",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="torch device, which also pins the physx sim and the sapien renderer",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    backends = resolve_backends(device)
    frames = {}
    for row, (_, task_id) in enumerate(TASKS):
        cfg = load_expert_config(task_id)
        frames[row, 0], frames[row, 1], frames[row, 2] = render_expert_episode(
            task_id, cfg, args.seed, backends, device
        )
        for col, (_, lighting) in enumerate(LIGHTING_COLUMNS, start=3):
            frames[row, col] = render_relit_start(cfg, args.seed, backends, lighting)

    fig, axes = plt.subplots(
        len(TASKS), len(COLUMN_TITLES), figsize=(4 * len(COLUMN_TITLES), 4 * len(TASKS))
    )

    for row, (row_title, _) in enumerate(TASKS):
        for col, title in enumerate(COLUMN_TITLES):
            ax = axes[row, col]
            ax.imshow(frames[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=FONT_SIZE)
            if col == 0:
                ax.set_ylabel(row_title.replace("-v", "\n-v"), fontsize=FONT_SIZE)

    fig.tight_layout()
    # the frames are raster either way; dpi sets the resolution they are embedded at in the PDF
    for path in (args.out, args.out.with_suffix(".pdf")):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
