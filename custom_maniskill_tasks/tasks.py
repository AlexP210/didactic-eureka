"""The task ids this project's datasets and policies actually use.

Every offline dataset here was collected by `tools/ppo_stages_fast.py`, which builds the stock
ManiSkill task and then wraps it in `ManiSkillVectorEnv(..., ignore_terminations=True)`: an episode
is never cut short by reaching the goal, so it always runs the full task horizon, and because the
dense reward keeps accruing while the goal is held, reaching it sooner is worth strictly more
return. That is a property of the task the data describes rather than of the training loop that
happened to collect it, so it is a registered task id here -- `gym.make("PushCube-v1.1")` cannot
forget the flag, and the `env_id` in a recorded trajectory's json says which convention produced it.

Registered by importing `custom_maniskill_tasks`. Note that other processes have to import it too:
a tool that only does `import mani_skill.envs` (`tools/replay_trajectory.py`,
`tools/ppo_stages_fast.py`) will not find these ids.

These ids also take a `lighting` kwarg (`LightingMixin`, see `lighting`) naming the condition the
scene is rendered under. It defaults to the stock lighting, so it changes nothing unless asked;
being an ordinary env kwarg rather than something applied around `gym.make`, it is recorded into
any trajectory collected under it and replays from that metadata on its own.

Everything else is the stock task: same scene, reward, success predicate and 50-step horizon, and
the same `normalized_dense` default reward mode. The one baked-in default is
`control_mode="pd_ee_delta_pos"`, which every recording in this project used; it is a registration
default, so an explicit `gym.make(..., control_mode=...)` still wins.

`LiftPegUpright-v1.1` alone also takes `distractors=True`, which adds four objects borrowed from the
other tasks (red and blue cubes and spheres) off to the sides of the table; see `DistractorsMixin`.
Off by default, so the stock scene is still what a bare `gym.make` builds.
"""

from __future__ import annotations

from dataclasses import dataclass

import sapien
import torch

from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.pick_single_ycb import PickSingleYCBEnv
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.envs.tasks.tabletop.poke_cube import PokeCubeEnv
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import REGISTERED_ENVS, register_env
from mani_skill.utils.structs.pose import Pose

from custom_maniskill_tasks.lighting import LightingMixin

CONTROL_MODE = "pd_ee_delta_pos"

FULL_HORIZON_TASKS = {
    "PushCube-v1.1": "PushCube-v1",
    "PlaceSphere-v1.1": "PlaceSphere-v1",
    "LiftPegUpright-v1.1": "LiftPegUpright-v1",
    "PokeCube-v1.1": "PokeCube-v1",
    "PickCube-v1.1": "PickCube-v1",
    "PickSingleYCB-v1.1": "PickSingleYCB-v1",
}
"""Each registered variant and the stock task it derives from. The `-v1.1` suffix is not a
gymnasium version (gymnasium only parses integer versions, so these are unversioned ids whose name
happens to contain a dot) -- it reads as "our revision of -v1"."""


def _horizon(base_task: str) -> int:
    """The stock task's registered time limit, so the variant cannot drift from it."""
    return REGISTERED_ENVS[base_task].max_episode_steps


def _asset_download_ids(base_task: str) -> list[str]:
    """The assets the stock task declares, so the variant prompts to fetch them too.

    `register_env` records these and `gym.make` checks them, so a variant that dropped them would
    fail deep inside the task's `__init__` (a missing json) instead of offering the download.
    """
    return REGISTERED_ENVS[base_task].asset_download_ids


PUSH_CUBE_BLUE = (12 / 255, 42 / 255, 160 / 255, 1.0)
"""PushCube's cube and PlaceSphere's sphere -- also the blue half of LiftPegUpright's peg."""
PICK_CUBE_RED = (1.0, 0.0, 0.0, 1.0)
"""PickCube's cube."""


@dataclass(frozen=True)
class Distractor:
    """One task-irrelevant object: the other tasks' own assets, parked off to the side."""

    name: str
    shape: str  # "cube" (half_size) or "sphere" (radius)
    size: float
    color: tuple[float, float, float, float]
    xy: tuple[float, float]


DISTRACTORS = (
    # PickCube's red cube and PushCube's blue one, PlaceSphere's blue sphere and a red one of the
    # same size. At |y| = 0.22 they clear the peg's whole spawn footprint (|y| <= 0.125 lying
    # down: 0.1 of spawn jitter plus its 0.025 half width) with room for an open gripper, and sit
    # inside the wrist camera's view from the rest pose. Each colour appears on both sides, so
    # "the red/blue thing" never picks out the peg by position alone.
    Distractor("distractor_red_cube", "cube", 0.02, PICK_CUBE_RED, (0.1, 0.22)),
    Distractor("distractor_blue_sphere", "sphere", 0.02, PUSH_CUBE_BLUE, (-0.1, 0.22)),
    Distractor("distractor_blue_cube", "cube", 0.02, PUSH_CUBE_BLUE, (0.1, -0.22)),
    Distractor("distractor_red_sphere", "sphere", 0.02, PICK_CUBE_RED, (-0.1, -0.22)),
)


class DistractorsMixin:
    """Gives a task a `distractors` kwarg: when true, `DISTRACTORS` are added to its scene.

    Off by default, so the scene every existing dataset and checkpoint was made in is unchanged.
    Like `lighting` it is an ordinary env kwarg, so it lands in `env.spec.kwargs` and any trajectory
    json recorded under it, and a replay rebuilds the same scene.

    The objects are dynamic -- the robot can knock them -- but nothing reads them: observations,
    reward and success are the task's own. A state observation is unchanged too, since tasks put
    only their own objects into `_get_obs_extra`. They are reset to the same poses each episode
    and draw nothing from the rng, so the task's own reset distribution is untouched.

    Mixed in after `LightingMixin`, so they are built inside its `super()._load_scene` and count
    as task objects to an `object_hue` shift, exactly as the peg does.
    """

    def __init__(self, *args, distractors: bool = False, **kwargs):
        self._distractors = DISTRACTORS if distractors else ()
        super().__init__(*args, **kwargs)

    @property
    def distractors(self) -> dict:
        """The distractor actors in this scene, by name; empty when built without them."""
        return self._distractor_actors

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._distractor_actors = {}
        for d in self._distractors:
            build = actors.build_cube if d.shape == "cube" else actors.build_sphere
            size_kwarg = {"half_size": d.size} if d.shape == "cube" else {"radius": d.size}
            self._distractor_actors[d.name] = build(
                self.scene,
                **size_kwarg,
                color=list(d.color),
                name=d.name,
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[*d.xy, d.size]),
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        by_name = {d.name: d for d in self._distractors}
        for name, actor in self._distractor_actors.items():
            d = by_name[name]
            p = torch.tensor([*d.xy, d.size], device=self.device).expand(len(env_idx), 3)
            actor.set_pose(Pose.create_from_pq(p=p))
            # a knocked sphere would otherwise still be rolling after the reset
            actor.set_linear_velocity(torch.zeros_like(p))
            actor.set_angular_velocity(torch.zeros_like(p))


class FullHorizonMixin:
    """Never terminate: every episode runs to its time limit.

    ManiSkill sets `terminated` from `info["success"]`/`info["fail"]` recomputed from the current
    state on every step (`BaseEnv.step`), so it is a momentary predicate rather than an absorbing
    state -- it flips back to False when the goal stops being satisfied. Suppressing it here, in
    the task, means nothing downstream has to remember to: `ManiSkillVectorEnv` has no termination
    left to auto-reset on, and the `IgnoreTerminations` wrapper becomes a no-op (`make_env` still
    applies it, harmlessly, since it also serves the stock `-v1` ids).

    The signal itself is untouched and still reported as `info["success"]`.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, torch.zeros_like(terminated), truncated, info


@register_env(
    "PushCube-v1.1", max_episode_steps=_horizon("PushCube-v1"), control_mode=CONTROL_MODE
)
class PushCubeFullHorizonEnv(FullHorizonMixin, LightingMixin, PushCubeEnv):
    """PushCube-v1 with no early termination and a `lighting` kwarg."""


@register_env(
    "PlaceSphere-v1.1", max_episode_steps=_horizon("PlaceSphere-v1"), control_mode=CONTROL_MODE
)
class PlaceSphereFullHorizonEnv(FullHorizonMixin, LightingMixin, PlaceSphereEnv):
    """PlaceSphere-v1 with no early termination and a `lighting` kwarg."""


@register_env(
    "LiftPegUpright-v1.1",
    max_episode_steps=_horizon("LiftPegUpright-v1"),
    control_mode=CONTROL_MODE,
)
class LiftPegUprightFullHorizonEnv(
    FullHorizonMixin, LightingMixin, DistractorsMixin, LiftPegUprightEnv
):
    """LiftPegUpright-v1 with no early termination, a `lighting` kwarg and a `distractors` one
    (`gym.make("LiftPegUpright-v1.1", distractors=True)`; see `DistractorsMixin`)."""


@register_env(
    "PokeCube-v1.1", max_episode_steps=_horizon("PokeCube-v1"), control_mode=CONTROL_MODE
)
class PokeCubeFullHorizonEnv(FullHorizonMixin, LightingMixin, PokeCubeEnv):
    """PokeCube-v1 with no early termination and a `lighting` kwarg."""


@register_env(
    "PickCube-v1.1", max_episode_steps=_horizon("PickCube-v1"), control_mode=CONTROL_MODE
)
class PickCubeFullHorizonEnv(FullHorizonMixin, LightingMixin, PickCubeEnv):
    """PickCube-v1 with no early termination and a `lighting` kwarg."""


@register_env(
    "PickSingleYCB-v1.1",
    max_episode_steps=_horizon("PickSingleYCB-v1"),
    asset_download_ids=_asset_download_ids("PickSingleYCB-v1"),
    control_mode=CONTROL_MODE,
)
class PickSingleYCBFullHorizonEnv(FullHorizonMixin, LightingMixin, PickSingleYCBEnv):
    """PickSingleYCB-v1 with no early termination and a `lighting` kwarg.

    The only one of these variants whose scene is not fixed: which YCB object is in it is drawn
    per parallel env at reconfiguration, so the stock task defaults `reconfiguration_freq` to 1 at
    `num_envs=1` (a new object every reset) and to 0 above it (one draw, held for the run). That
    default is the task's own and is untouched here.
    """
