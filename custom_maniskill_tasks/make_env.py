"""The one place a ManiSkill task in this project gets built."""

from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym

import mani_skill.envs  # noqa: F401  (registers the task ids with gymnasium)

from custom_maniskill_tasks.backends import backend_kwargs
from custom_maniskill_tasks.cameras import (
    DEFAULT_CAMERA_RESOLUTION,
    FOCUSED_CAMERA_UID,
    build_sensor_configs,
    camera_view_applied,
    canonical_camera_view,
    check_camera_view,
)
from custom_maniskill_tasks.lighting import (
    DEFAULT_LIGHTING,
    DEFAULT_LIGHTING_PRESET,
    canonical_lighting,
    check_lighting,
    supports_lighting,
)
from custom_maniskill_tasks.wrappers import FrameSkip, FrameStack, IgnoreTerminations

TASKS_IN_USE = (
    # the stock tasks minus early termination, registered by importing this package (see `tasks`);
    # this project's datasets are collected with these
    "PushCube-v1.1",
    "PlaceSphere-v1.1",
    "LiftPegUpright-v1.1",
    "PokeCube-v1.1",
    "PickCube-v1.1",
    "PickSingleYCB-v1.1",
    # stock ManiSkill tasks that older datasets and configs still name
    "PushCube-v1",
    "PickCube-v1",
    "PegInsertionSide-v1",
    "PlaceSphere-v1",
    "LiftPegUpright-v1",
    "PokeCube-v1",
)
"""The tasks this project has datasets and configs for. Documentation, not a restriction:
`make_env` works for any registered ManiSkill task id."""


def make_env(
    task_name: str,
    *,
    obs_mode: str = "rgb",
    control_mode: str | None = "pd_ee_delta_pos",
    num_envs: int = 1,
    camera_view: str = "default",
    lighting: str | dict = DEFAULT_LIGHTING_PRESET,
    camera_resolution: int | None = DEFAULT_CAMERA_RESOLUTION,
    wrist_only: bool = True,
    focused_camera_uid: str = FOCUSED_CAMERA_UID,
    frame_skip: int = 1,
    n_frames: int | None = None,
    frame_axis: int = 1,
    ignore_terminations: bool = True,
    sim_backend: str | None = None,
    render_backend: str | None = None,
    render_mode: str | None = "rgb_array",
    reward_wrappers: Sequence[Callable[[gym.Env], gym.Env]] = (),
    obs_wrappers: Sequence[Callable[[gym.Env], gym.Env]] = (),
    **env_kwargs,
) -> gym.Env:
    """Build `task_name` with the camera view, action chunking and frame stacking asked for.

    The env never resets itself: `gym.make` applies no auto-reset (only `make_vec` /
    `ManiSkillVectorEnv` do), `ignore_terminations` stops a momentary success from ending the
    episode, and nothing here calls `reset`, so the caller owns every reset and the RNG stream that
    seeds it. Truncation from the task's time limit is still reported, as the signal to reset.

    Wrapper order is `IgnoreTerminations` -> `FrameSkip` -> `reward_wrappers` -> `obs_wrappers` ->
    `FrameStack`, so a stacked observation spans `n_frames` macro steps of `frame_skip` primitive
    steps each, and an observation adapter placed in `obs_wrappers` sees single unstacked frames.

    Args:
        task_name: a registered ManiSkill task id, e.g. "PushCube-v1.1"; see `TASKS_IN_USE`. The
            `-v1.1` ids are this project's own (registered by importing this package): the stock
            tasks with early termination removed, which is how every offline dataset here was
            collected. On those, `ignore_terminations` is already true of the task itself.
        obs_mode, control_mode, num_envs: passed to `gym.make`. `control_mode` defaults to the
            one every recorded dataset in this project uses; a policy run under a different
            control mode than its data was recorded with is a silent failure. `control_mode=None`
            means "whatever the task id already defaults to" and is not forwarded, since
            `gym.make(control_mode=None)` would override a registration default (the `-v1.1` ids
            carry one) rather than defer to it.
        lighting: which lighting condition the scene is rendered under -- a name from
            `LIGHTING_PRESETS` ("default" is the stock lighting every dataset here was recorded
            with; "dim", "bright", "warm", "cool", "side" and "shadows" are named shifts away
            from it, each of "dim", "bright", "warm" and "cool" also has a more extreme
            "very-<preset>" sibling, and "random" draws one condition per parallel env),
            "bright-set-<ambient>-<lights>" to set the ambient and both lights' levels outright
            (e.g. "bright-set-0.45-1.5"), "side-set-<amount>" to turn the key light that
            fraction (0 to 1) of the way from "default"'s direction to "side"'s,
            "warm-set-<amount>" / "cool-set-<amount>" to tint the lights that fraction of the way
            along the blackbody curve from 6500 K to 2700 K / 12000 K (and past 1, up to 1.6:
            about 2000 K / 24400 K), "dark-table" / "very-dark-table" or "table-set-<scale>" /
            "table-set-<r>-<g>-<b>" to recolour the table and leave the lights alone (e.g.
            "table-set-0.4"), "object-hue-<degrees>" to turn the task objects' colours that far
            round the hue wheel (e.g. "object-hue-60"), "domain-randomization" (or a list of
            preset names) to draw a fresh condition per parallel env on every reset -- one of
            `DOMAIN_RANDOMIZATION_PRESETS` (or of the list) at a severity uniform between the
            default and that preset, see `LightingMixin` -- a "+"-joined stack of those shift names for a condition that combines several at once
            (e.g. "very-dim+very-warm+side"), or a config dict for a one-off condition. Only the
            project's own `-v1.1` ids support it. It is a real env kwarg, so a non-default
            condition is recorded into any trajectory collected through it; a "default" one is not
            passed at all, leaving existing configs' recorded metadata byte-identical to what they
            produced before this argument existed.
        camera_view: which camera the observations come from -- "default" (the task's own camera,
            also accepted as "standard"), "focused" (that camera re-posed onto the tabletop
            workspace through a narrow fov) or "wrist" (a hand-mounted fisheye).
        camera_resolution: square resolution for every camera; None keeps each task's own. Note
            `ManiSkillTask.make_env` used to leave the default view at ManiSkill's 128, so
            reproducing an old "standard" env means passing `camera_resolution=None`.
        wrist_only: under `camera_view="wrist"`, drop the task's own camera instead of rendering
            and recording both.
        focused_camera_uid: which camera `camera_view="focused"` re-poses.
        frame_skip: act in chunks of this many primitive actions -- see `FrameSkip`. The task's
            time limit is unchanged and still counted in primitive steps, so an episode lasts
            `max_episode_steps / frame_skip` macro steps.
        n_frames: stack this many past observations -- see `FrameStack`. None leaves observations
            exactly as ManiSkill returns them; any integer (1 included) adds the frame axis.
        frame_axis: where `FrameStack` inserts that axis. 1 for raw ManiSkill observations, whose
            leading axis is `num_envs`; 0 if an `obs_wrappers` entry squeezed that axis away.
        ignore_terminations: never report `terminated` -- see `IgnoreTerminations`.
        sim_backend, render_backend: resolved and pinned to the same cuda device by
            `backend_kwargs`; None means ManiSkill's own choice (physx_cpu at `num_envs=1`).
        render_mode: for `env.render()`, which uses the separate human render camera and is
            unaffected by `camera_view`.
        reward_wrappers: callables applied in order between `FrameSkip` and `obs_wrappers`, for
            wrappers that rewrite the reward (`DINORewardWrapper`). Underneath the observation
            adapters, so they still see ManiSkill's own observation dict; outside `FrameSkip`, so
            they are called once per macro step rather than once per primitive step.
        obs_wrappers: callables applied in order between `reward_wrappers` and `FrameStack`, for
            project-specific observation adapters (S2P's TensorDict view of the observation dict,
            DINO-WM's flat state/proprio one).
        **env_kwargs: anything else `gym.make` takes -- `reward_mode`, `reconfiguration_freq`,
            `max_episode_steps`, `shader_dir`, a `sensor_configs` dict merged over the one this
            builds, ...
    """
    view = canonical_camera_view(camera_view)
    lighting_config = canonical_lighting(lighting)

    sensor_configs = build_sensor_configs(view, camera_resolution, focused_camera_uid)
    sensor_configs.update(env_kwargs.pop("sensor_configs", {}) or {})

    if control_mode is not None:
        env_kwargs["control_mode"] = control_mode

    if lighting_config != DEFAULT_LIGHTING:
        if not supports_lighting(task_name):
            raise ValueError(
                f"lighting={lighting!r} was asked for, but {task_name} is not one of this "
                f"project's own task ids and its class takes no `lighting` kwarg. Use the "
                f"corresponding -v1.1 id (see `TASKS_IN_USE`), which does."
            )
        # a plain list, so a domain-randomization list from a hydra ListConfig is recorded into
        # env.spec.kwargs (and any trajectory json) as json
        env_kwargs["lighting"] = (
            list(lighting) if isinstance(lighting, Sequence) and not isinstance(lighting, str)
            else lighting
        )

    with camera_view_applied(view, task_name, wrist_only=wrist_only):
        env = gym.make(
            id=task_name,
            obs_mode=obs_mode,
            num_envs=num_envs,
            sensor_configs=sensor_configs,
            render_mode=render_mode,
            **backend_kwargs(sim_backend, render_backend, num_envs),
            **env_kwargs,
        )
    check_camera_view(env, view, focused_camera_uid)
    check_lighting(env, lighting_config)

    if ignore_terminations:
        env = IgnoreTerminations(env)
    if frame_skip > 1:
        env = FrameSkip(env, frame_skip)
    for wrapper in reward_wrappers:
        env = wrapper(env)
    for wrapper in obs_wrappers:
        env = wrapper(env)
    if n_frames is not None:
        env = FrameStack(env, n_frames, frame_axis=frame_axis)
    return env
