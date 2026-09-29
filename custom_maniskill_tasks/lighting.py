"""The lighting conditions these tasks can be rendered under.

One `lighting` preset selects the lights for the whole scene, the same way one `camera_view`
selects the observation camera, and everything about a condition lives here: the ambient colour,
the directional lights, and the ranges a randomizing condition draws from.

The `"default"` preset is not a choice -- it is `BaseEnv._load_lighting` transcribed, which is
what every dataset in this project was rendered under. Treat those numbers as data. The other
presets are deliberate *shifts* away from it, meant to be named in an experiment ("evaluated under
`dim`") rather than tuned per run, so a reported number says which condition produced it. Several
have a more extreme `"very-<preset>"` sibling, and any of these shift names can be joined with
`"+"` (e.g. `"very-dim+very-warm+side"`) to stack their effects on top of one another; see
`canonical_lighting`.

Applied through `LightingMixin` on the registered `-v1.1` task classes (see `tasks`) rather than
around the `gym.make` call the way the camera views are. That makes `lighting` an ordinary env
kwarg: `gym.make` records it in `env.spec.kwargs`, `RecordEpisode` writes it into the trajectory
json, and a replay of that dataset rebuilds the same condition without being told. A shifted
recording is therefore distinguishable from a default-lit one on disk, which a scene patched from
the outside would not be.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import sapien

from mani_skill import logger
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import REGISTERED_ENVS

Color = tuple[float, float, float]


@dataclass(frozen=True)
class DirectionalLight:
    """One directional light, in the terms `ManiSkillScene.add_directional_light` takes.

    `direction` is the direction the light travels in and is not normalized (sapien normalizes it
    when it builds the pose), so it reads the same way as ManiSkill's own `[1, 1, -1]`: a light
    from above, behind and to one side. `shadow=None` means "follow the env's `enable_shadow`",
    which is how the base task's key light behaves; True or False pins it regardless.
    """

    direction: Color
    color: Color
    shadow: bool | None = None
    shadow_scale: float = 10.0
    shadow_map_size: int = 2048


@dataclass(frozen=True)
class LightingRandomization:
    """Per-env draws applied on top of a `LightingConfig`, as multipliers on it.

    Multipliers rather than absolute ranges so a randomizing condition can sit on top of any base
    preset and stay recognisably that preset. Every range is `(low, high)` and drawn uniformly.

    `tint` is drawn once per channel, so it produces colour casts rather than pure brightness
    changes; `direction_jitter` is added to each component of a light's direction, which moves
    where the shading and any shadows fall.
    """

    ambient: tuple[float, float] = (1.0, 1.0)
    brightness: tuple[float, float] = (1.0, 1.0)
    tint: tuple[float, float] = (1.0, 1.0)
    direction_jitter: float = 0.0


@dataclass(frozen=True)
class SceneProp:
    """One static box, visual only, added to the scene as part of a lighting condition.

    Not a physics object -- no collision shape, so it can never obstruct the robot or the task
    regardless of where it sits. Its entire purpose is to give a directional light something to
    cast a shadow of: a light itself is invisible, so a preset built around `shadow=True` needs an
    occluder in the scene before it reads as anything more than a slightly darker render.
    """

    position: Color
    half_size: Color
    color: Color = (0.25, 0.25, 0.25)


@dataclass(frozen=True)
class LightingConfig:
    """A complete lighting condition: what `_load_lighting` (and `_load_scene`) will build.

    `randomization=None` is a fixed condition -- every parallel env is lit identically and the
    condition is fully described by the preset name. With a `LightingRandomization` each parallel
    env draws its own variation at reconfiguration; see `apply_lighting` for what that means and
    does not mean.
    """

    ambient: Color
    lights: tuple[DirectionalLight, ...]
    randomization: LightingRandomization | None = None
    props: tuple[SceneProp, ...] = ()
    table_tint: Color = (1.0, 1.0, 1.0)
    """Multiplied into the table's material colours (and so its wood texture); see `_tint_table`.
    Not light at all, strictly, but an appearance shift of the same kind -- what the wrist camera
    sees of the tabletop -- so it rides on the same `lighting` kwarg rather than a second one."""
    object_hue: float = 0.0
    """Degrees the hue of every task object's colours is rotated by (the table, ground, robot and
    `props` excluded); see `_shift_object_hues`. Appearance again, not light, for the same reason."""
    domain_randomization: tuple[tuple[str, "LightingConfig"], ...] = ()
    """Per-episode domain randomization, as (preset name, that preset's config) pairs. Non-empty
    means every reset rebuilds the scene and each parallel env draws its own condition for the
    episode: one of these presets, uniformly, at a severity drawn uniformly from [0, 1] between
    this config (the default) at 0 and the preset at 1. See `domain_randomized` and
    `LightingMixin`."""


DEFAULT_AMBIENT: Color = (0.3, 0.3, 0.3)

DEFAULT_LIGHTING = LightingConfig(
    ambient=DEFAULT_AMBIENT,
    lights=(
        # the key light, whose shadow the `enable_shadow` env kwarg switches on
        DirectionalLight(
            direction=(1.0, 1.0, -1.0), color=(1.0, 1.0, 1.0), shadow_scale=5.0
        ),
        # the fill from straight above, which the base task never shadows
        DirectionalLight(direction=(0.0, 0.0, -1.0), color=(1.0, 1.0, 1.0), shadow=False),
    ),
)
"""`BaseEnv._load_lighting` as data, down to the shadow parameters it passes. Building this
config has to be indistinguishable from not overriding `_load_lighting` at all -- every dataset in
this project was recorded under it, so a policy evaluated at `lighting="default"` has to be
looking at the same scene its training frames came from."""


def _tinted(config: LightingConfig, tint: Color) -> LightingConfig:
    """`config` with `tint` multiplied into every colour, the ambient included.

    Multiplying the ambient too is what makes an exposure preset read as one: leaving it at 0.3
    while halving the lights would darken the lit faces and leave the shadowed ones alone, which
    is a change of contrast rather than of light level.
    """
    return replace(
        config,
        ambient=tuple(channel * scale for channel, scale in zip(config.ambient, tint)),
        lights=tuple(
            replace(light, color=tuple(c * s for c, s in zip(light.color, tint)))
            for light in config.lights
        ),
    )


SIDE_KEY_DIRECTION: Color = (-1.0, -1.0, -0.35)
"""Where `side` puts the key light: crossed to the other side of the table and raking lower."""


def _sided(config: LightingConfig) -> LightingConfig:
    """`config` with the key light crossed to the other side, raking lower.

    Shading (and, under `enable_shadow`, shadows) then falls the opposite way at the same
    exposure. Only ever touches `lights[0]`, so it composes with a tint effect regardless of
    which runs first.
    """
    return _with_key_direction(config, SIDE_KEY_DIRECTION)


def _with_key_direction(config: LightingConfig, direction: Color) -> LightingConfig:
    return replace(
        config,
        lights=(replace(config.lights[0], direction=direction), *config.lights[1:]),
    )


def _key_direction_towards_side(amount: float) -> Color:
    """The key light direction `amount` of the way from the default's to `SIDE_KEY_DIRECTION`.

    A slerp between the two, not a lerp: both lie in the vertical plane x == y, and the short arc
    between them passes straight overhead, so equal steps in `amount` turn the light through equal
    angles (about 13 degrees per 0.1). A lerp would bunch the steps up near the overhead point.
    The two endpoints come back as the presets' own vectors rather than their normalized forms, so
    `side-set-0` builds exactly `default` and `side-set-1` exactly `side`.
    """
    start = DEFAULT_LIGHTING.lights[0].direction
    if amount == 0.0:
        return start
    if amount == 1.0:
        return SIDE_KEY_DIRECTION
    a = np.asarray(start) / np.linalg.norm(start)
    b = np.asarray(SIDE_KEY_DIRECTION) / np.linalg.norm(SIDE_KEY_DIRECTION)
    angle = np.arccos(np.clip(a @ b, -1.0, 1.0))
    direction = (np.sin((1.0 - amount) * angle) * a + np.sin(amount * angle) * b) / np.sin(angle)
    return tuple(float(component) for component in direction)


def _tint_effect(tint: Color) -> Callable[[LightingConfig], LightingConfig]:
    return lambda config: _tinted(config, tint)


def _levels_set(config: LightingConfig, ambient: float, lights: float) -> LightingConfig:
    """`config` with the ambient set to `ambient` and every light's colour to `lights`, all grey.

    Absolute where `_tinted` is relative: whatever colours `config` had are replaced, not scaled,
    so a tint stacked *before* this is overwritten, while one stacked after it still applies.
    """
    return replace(
        config,
        ambient=(ambient, ambient, ambient),
        lights=tuple(replace(light, color=(lights, lights, lights)) for light in config.lights),
    )


def _table_set(config: LightingConfig, tint: Color) -> LightingConfig:
    """`config` with the table tinted by `tint`, outright: absolute like `_levels_set`, so of two
    in one stack the last one wins, and the lights are left alone."""
    return replace(config, table_tint=tint)


def _object_hue_set(config: LightingConfig, degrees: float) -> LightingConfig:
    """`config` with the task objects' hue rotated by `degrees`, outright: of two in one stack the
    last one wins, and the lights and table are left alone."""
    return replace(config, object_hue=degrees)


DEFAULT_KELVIN = 6500.0
"""The colour temperature `"default"`'s white lights are taken to be; the hue sliders' zero."""
HUE_SET_KELVIN = {"warm": 2700.0, "cool": 12000.0}
"""Where `"warm-set-1"` and `"cool-set-1"` end: a household incandescent bulb, and open shade
under a clear blue sky. Both about as far as real light goes before it stops reading as white
light at all. For scale, the hand-picked `"very-warm"` is roughly 3400 K and `"very-cool"` is bluer
than 20000 K, i.e. bluer than any daylight."""
HUE_SET_MAX_AMOUNT = 1.6
"""How far past 1 the hue sliders run. For warm that is about 2000 K, candlelight, just short of
where a blackbody's blue channel goes negative in linear sRGB (about 1917 K), which no light colour
can be; for cool about 24400 K, just inside the 25000 K where `_planckian_linear_rgb`'s fit stops
being valid."""

# linear sRGB from CIE XYZ, D65 white
_XYZ_TO_LINEAR_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])
_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


def _planckian_linear_rgb(kelvin: float) -> np.ndarray:
    """A blackbody's colour at `kelvin`, in linear sRGB at luminance 1.

    From the cubic-spline fit to the Planckian locus of Kim et al. (2002), valid 1667-25000 K.
    Linear rather than gamma-encoded because light colours multiply radiance in the renderer.
    """
    t = kelvin
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return _XYZ_TO_LINEAR_SRGB @ np.array([x / y, 1.0, (1.0 - x - y) / y])


def _hue_kelvin(hue: str, amount: float) -> float:
    """The colour temperature `amount` of the way from `DEFAULT_KELVIN` to `HUE_SET_KELVIN[hue]`,
    stepped in mired (1e6 / kelvin), the scale colour-temperature differences are perceived
    evenly on. Past 1 it carries on at the same rate."""
    start, end = 1e6 / DEFAULT_KELVIN, 1e6 / HUE_SET_KELVIN[hue]
    return 1e6 / (start + amount * (end - start))


def _hue_tint(hue: str, amount: float) -> Color:
    """The tint `amount` of the way from `DEFAULT_KELVIN` to `HUE_SET_KELVIN[hue]`.

    Relative to the default's blackbody so that 0 is exactly no tint (see `_hue_kelvin` for the
    stepping), and rescaled to luminance 1, so the slider moves the hue and not the exposure: that
    is left to `bright-set`.
    """
    if amount == 0.0:
        return (1.0, 1.0, 1.0)
    kelvin = _hue_kelvin(hue, amount)
    tint = _planckian_linear_rgb(kelvin) / _planckian_linear_rgb(DEFAULT_KELVIN)
    return tuple(float(channel) for channel in tint / (_LUMINANCE @ tint))


_NUMBER = r"(\d+(?:\.\d+)?)"
_BRIGHT_SET = re.compile(rf"bright-set-{_NUMBER}-{_NUMBER}")
"""`"bright-set-<ambient>-<lights>"`, e.g. `"bright-set-0.45-1.5"` (the levels `"bright"` works out
to): an exposure condition given by its levels rather than as a multiple of the default's."""
_SIDE_SET = re.compile(rf"side-set-{_NUMBER}")
"""`"side-set-<amount>"`, `amount` in [0, 1]: the key light turned that fraction of the way from
where `"default"` has it to where `"side"` does; see `_key_direction_towards_side`."""
_HUE_SET = re.compile(rf"(warm|cool)-set-{_NUMBER}")
_TABLE_SET = re.compile(rf"table-set-{_NUMBER}(?:-{_NUMBER}-{_NUMBER})?")
"""`"table-set-<scale>"` or `"table-set-<r>-<g>-<b>"`: the table's colours multiplied by `scale` (a
grey tint, so below 1 darkens the wood without changing its hue) or per channel; see `_tint_table`."""
_OBJECT_HUE = re.compile(rf"object-hue-{_NUMBER}")
"""`"object-hue-<degrees>"`, degrees in [0, 360): every task object's colours turned that far round
the hue wheel (red -> yellow -> green -> blue at 0 -> 60 -> 120 -> 240); greys and white stay put.
See `_shift_object_hues`."""
"""`"warm-set-<amount>"` / `"cool-set-<amount>"`: every light tinted along the blackbody curve from
`DEFAULT_KELVIN` at 0 to `HUE_SET_KELVIN` at 1, and on up to `HUE_SET_MAX_AMOUNT`; see
`_hue_tint`."""


_SHADOW_CASTER = (
    # a floor lamp with a broad overhead shade: a thin pole and a wide flat panel, planted off to
    # the side of the table (see `LightingMixin._load_scene`) where it is nowhere near the robot's
    # own reach and well outside the wrist camera's fov. The key light travels in direction
    # (1, 1, -1), so a point at height h casts its shadow (h, h) away horizontally -- the pole
    # position is chosen so that offset lands the shade's shadow on the patch of table the wrist
    # camera looks down at, which is the one place `shadow=True` alone never reaches (see
    # `_shadows`'s docstring: the robot's own shadow rarely falls under its own wrist camera).
    # The shade also has to be *wide*: a directional light's shadow is the same size as the object
    # casting it regardless of distance, so a small occluder only darkens a small patch, easy to
    # miss if the gripper is not exactly where the aim assumed. Wide enough here to blanket the
    # wrist camera's field of view with margin for the gripper having moved.
    SceneProp(position=(-0.5, -0.6, 0.3), half_size=(0.035, 0.035, 0.3)),
    SceneProp(position=(-0.5, -0.6, 0.65), half_size=(0.3, 0.3, 0.05)),
)


def _shadows(config: LightingConfig) -> LightingConfig:
    """`config` with every light that follows `enable_shadow` (`shadow=None`) pinned to cast one,
    plus an occluder in the scene for it to cast.

    Ordinarily whether the key light casts a shadow is a session setting -- the env's
    `enable_shadow` kwarg, off by default -- so the same preset can render shadowed or not
    depending on how the env was built. Here the *condition* decides instead, so the scene always
    renders with those shadows regardless of what the caller passed. A light that already opts
    out (`shadow=False`, the fill light's own choice) is left alone -- this forces shadows on, it
    does not turn the ones already off back on.

    A shadow needs something to fall across, though: pinning `shadow=True` on a scene with
    nothing but the robot and a cube mostly changes what the *robot* looks like (self-shadowing),
    not what a downward-looking camera sees on the table -- the wrist camera especially, which
    sits close enough above the workspace that the robot's own shadow rarely reaches it. Hence
    `_SHADOW_CASTER`, added to the scene by `LightingMixin._load_scene` whenever it appears in
    `props`.
    """
    return replace(
        config,
        lights=tuple(
            replace(light, shadow=True) if light.shadow is None else light
            for light in config.lights
        ),
        props=config.props + _SHADOW_CASTER,
    )


# One effect per stackable preset name, each a `LightingConfig -> LightingConfig` shift away from
# whatever it is applied to. `LIGHTING_PRESETS` below is these applied once to `DEFAULT_LIGHTING`;
# `canonical_lighting` applies a "+"-joined chain of them in sequence to the same starting point,
# so "very-dim+very-warm+side" is not a name that has to be pre-declared to exist.
_PRESET_EFFECTS: dict[str, Callable[[LightingConfig], LightingConfig]] = {
    # exposure shifts: the scene is lit the same way, less or more of it
    "dim": _tint_effect((0.5, 0.5, 0.5)),
    "very-dim": _tint_effect((0.25, 0.25, 0.25)),
    "bright": _tint_effect((1.5, 1.5, 1.5)),
    "very-bright": _tint_effect((2.25, 2.25, 2.25)),
    # colour-temperature shifts: same geometry and roughly the same exposure, different cast
    "warm": _tint_effect((1.15, 0.9, 0.65)),
    "very-warm": _tint_effect((1.3, 0.8, 0.4)),
    "cool": _tint_effect((0.7, 0.85, 1.2)),
    "very-cool": _tint_effect((0.5, 0.75, 1.4)),
    # a geometric shift: no "very" variant, there being only one other side to cross to
    "side": _sided,
    # a shadow shift: no "very" variant either, a shadow being either cast or not
    "shadows": _shadows,
    # appearance shifts: the lights are untouched, the tabletop they fall on is darker
    "dark-table": lambda config: _table_set(config, (0.4, 0.4, 0.4)),
    "very-dark-table": lambda config: _table_set(config, (0.2, 0.2, 0.2)),
}


def _stacked(name: str) -> LightingConfig:
    """`name` as a "+"-joined chain of `_PRESET_EFFECTS`, applied in order to `DEFAULT_LIGHTING`.

    Each effect is a plain function of a `LightingConfig`, so stacking them is just folding: the
    tint effects multiply together regardless of order, and `side` only ever touches `lights[0]`'s
    direction, so "very-dim+very-warm+side" and "side+very-warm+very-dim" build the same config.
    The exception is `"bright-set-<ambient>-<lights>"`, which sets the levels outright and so
    overwrites any tint before it: "bright-set-0.3-1+warm" is warm, "warm+bright-set-0.3-1" is not.
    `side` and `"side-set-<amount>"` likewise both set the key light's direction outright, so of
    two of them in one stack the last one wins, as do two table tints (`dark-table`,
    `"table-set-..."`), which touch nothing but the table, and two `"object-hue-..."`s, which touch
    nothing but the task objects. `"warm-set-<amount>"` / `"cool-set-<amount>"` are
    tints like `warm`, so they multiply with the others in any order.
    """
    parts = name.split("+")
    config = DEFAULT_LIGHTING
    for part in parts:
        effect = _effect(part)
        if effect is None:
            raise ValueError(
                f"Unknown lighting preset {part!r} in stacked preset {name!r}; a stack can only "
                f"combine {', '.join(repr(preset) for preset in _PRESET_EFFECTS)}, "
                f"'bright-set-<ambient>-<lights>', 'side-set-<amount>', 'warm-set-<amount>', "
                f"'cool-set-<amount>', 'table-set-<scale>' / 'table-set-<r>-<g>-<b>' and "
                f"'object-hue-<degrees>'"
            )
        config = effect(config)
    return config


def _effect(part: str) -> Callable[[LightingConfig], LightingConfig] | None:
    """The effect one stack element names: a `_PRESET_EFFECTS` entry, or a `_BRIGHT_SET`,
    `_SIDE_SET`, `_HUE_SET`, `_TABLE_SET` or `_OBJECT_HUE` match."""
    effect = _PRESET_EFFECTS.get(part)
    if effect is not None:
        return effect
    match = _BRIGHT_SET.fullmatch(part)
    if match is not None:
        ambient, lights = (float(level) for level in match.groups())
        return lambda config: _levels_set(config, ambient, lights)
    match = _SIDE_SET.fullmatch(part)
    if match is not None:
        amount = float(match.group(1))
        # past 1 the arc runs on towards the horizon and, at about 1.1, under the table
        if amount > 1.0:
            raise ValueError(
                f"Lighting preset {part!r} is past 'side': side-set takes an amount in [0, 1]"
            )
        direction = _key_direction_towards_side(amount)
        return lambda config: _with_key_direction(config, direction)
    match = _HUE_SET.fullmatch(part)
    if match is not None:
        hue, amount = match.group(1), float(match.group(2))
        if amount > HUE_SET_MAX_AMOUNT:
            raise ValueError(
                f"Lighting preset {part!r} is past {hue}-set-{HUE_SET_MAX_AMOUNT} "
                f"({_hue_kelvin(hue, HUE_SET_MAX_AMOUNT):.0f} K): {hue}-set takes an amount from "
                f"0 to {HUE_SET_MAX_AMOUNT}"
            )
        return _tint_effect(_hue_tint(hue, amount))
    match = _TABLE_SET.fullmatch(part)
    if match is not None:
        scale, green, blue = match.groups()
        tint = (float(scale),) * 3 if green is None else (float(scale), float(green), float(blue))
        return lambda config: _table_set(config, tint)
    match = _OBJECT_HUE.fullmatch(part)
    if match is not None:
        degrees = float(match.group(1))
        if degrees >= 360.0:
            raise ValueError(
                f"Lighting preset {part!r} is a full turn or more: object-hue takes degrees in "
                "[0, 360)"
            )
        return lambda config: _object_hue_set(config, degrees)
    return None


DOMAIN_RANDOMIZATION = "domain-randomization"
"""The name `lighting` takes for per-episode domain randomization over `DOMAIN_RANDOMIZATION_PRESETS`;
a list of preset names randomizes over those instead."""
DOMAIN_RANDOMIZATION_PRESETS = (
    "default",
    "bright-set-0.75-2.5",
    "warm-set-1.1",
    "object-hue-30",
)
"""The visual perturbations this project evaluates under, each at its full ("official") severity;
`"default"` among them keeps a share of episodes unperturbed."""


def domain_randomized(presets: Sequence[str]) -> LightingConfig:
    """`DEFAULT_LIGHTING`, randomized per episode and per parallel env over `presets`.

    Each preset has to be one a severity can be dialled between: a fixed condition whose lights
    match the default's one for one, and which changes only the lights, the ambient and the
    object hue. Those are the parts of a scene that can differ between parallel envs -- a table
    tint cannot (every env's table shares one textured material, see `_tint_table`), a prop
    cannot be half-built, and a preset that already randomizes has no single severity-1 form.
    Each is refused by name rather than quietly dropped from the draw.
    """
    if isinstance(presets, (str, bytes)) or not isinstance(presets, Sequence) or not presets:
        raise ValueError(
            f"domain randomization needs a non-empty list of preset names, got {presets!r}"
        )
    endpoints = []
    for name in presets:
        if not isinstance(name, str):
            raise TypeError(f"domain randomization presets must be names, got {name!r}")
        if name == DOMAIN_RANDOMIZATION:
            raise ValueError(f"{DOMAIN_RANDOMIZATION!r} cannot be one of its own presets")
        config = canonical_lighting(name)
        problems = []
        if config.randomization is not None or config.domain_randomization:
            problems.append("it randomizes already")
        if config.props:
            problems.append("it adds props to the scene")
        if config.table_tint != DEFAULT_LIGHTING.table_tint:
            problems.append("it tints the table, which every parallel env shares")
        if len(config.lights) != len(DEFAULT_LIGHTING.lights) or any(
            light.shadow != base.shadow for light, base in zip(config.lights, DEFAULT_LIGHTING.lights)
        ):
            problems.append("its lights are not the default's lights re-coloured or re-aimed")
        if problems:
            raise ValueError(
                f"Lighting preset {name!r} cannot be domain-randomized: {'; '.join(problems)}"
            )
        endpoints.append((name, config))
    return replace(DEFAULT_LIGHTING, domain_randomization=tuple(endpoints))


def _slerp(start: Color, end: Color, t: float) -> Color:
    """The direction `t` of the way from `start` to `end` along the arc between them, returned
    as `start` / `end` themselves at the endpoints (see `_key_direction_towards_side`)."""
    if t == 0.0 or start == end:
        return start
    if t == 1.0:
        return end
    a = np.asarray(start) / np.linalg.norm(start)
    b = np.asarray(end) / np.linalg.norm(end)
    angle = np.arccos(np.clip(a @ b, -1.0, 1.0))
    if angle < 1e-9:
        return start
    direction = (np.sin((1.0 - t) * angle) * a + np.sin(t * angle) * b) / np.sin(angle)
    return tuple(float(component) for component in direction)


def _interpolated(start: LightingConfig, end: LightingConfig, t: float) -> LightingConfig:
    """The condition `t` of the way from `start` (at 0) to `end` (at 1): colours and levels
    linearly, light directions along the arc, the object hue in degrees. Linear in the tint is
    what keeps a hue slider's constant luminance at every severity, luminance being linear in it.
    """
    lerp = lambda a, b: tuple(float((1.0 - t) * x + t * y) for x, y in zip(a, b))
    return replace(
        start,
        ambient=lerp(start.ambient, end.ambient),
        lights=tuple(
            replace(
                light,
                color=lerp(light.color, target.color),
                direction=_slerp(light.direction, target.direction, t),
            )
            for light, target in zip(start.lights, end.lights)
        ),
        object_hue=(1.0 - t) * start.object_hue + t * end.object_hue,
        domain_randomization=(),
    )


@dataclass(frozen=True)
class EpisodeLighting:
    """What one parallel env drew for its episode under domain randomization."""

    preset: str
    severity: float
    config: LightingConfig


def _drawn_episode_lighting(config: LightingConfig, episode_rng, num_scenes: int) -> list[EpisodeLighting]:
    """One `EpisodeLighting` per parallel env, off the batched episode rng (see `_drawn_configs`
    for why batched): a preset uniformly from `config.domain_randomization`, and a severity
    uniformly from [0, 1)."""
    if episode_rng.batch_size != num_scenes:
        raise AssertionError(
            f"episode rng is batched over {episode_rng.batch_size} envs but the scene has "
            f"{num_scenes} sub-scenes"
        )
    presets = config.domain_randomization
    choice = np.minimum((episode_rng.uniform(0.0, 1.0) * len(presets)).astype(int), len(presets) - 1)
    severity = episode_rng.uniform(0.0, 1.0)
    drawn = []
    for i in range(num_scenes):
        name, endpoint = presets[choice[i]]
        drawn.append(EpisodeLighting(
            preset=name,
            severity=float(severity[i]),
            config=_interpolated(replace(config, domain_randomization=()), endpoint, float(severity[i])),
        ))
    return drawn


LIGHTING_PRESETS: dict[str, LightingConfig] = {
    "default": DEFAULT_LIGHTING,
    **{name: effect(DEFAULT_LIGHTING) for name, effect in _PRESET_EFFECTS.items()},
    # training-time domain randomization: each parallel env draws its own condition, held for
    # the life of the env unless it reconfigures -- see `apply_lighting`
    "random": replace(
        DEFAULT_LIGHTING,
        randomization=LightingRandomization(
            ambient=(0.5, 1.6),
            brightness=(0.5, 1.5),
            tint=(0.75, 1.0),
            direction_jitter=0.5,
        ),
    ),
}
"""The named conditions, and the whole interface: a preset name is what an experiment reports and
what a recording carries in its metadata. A dict of the same shape as `LightingConfig` is accepted
wherever a name is, for a one-off condition that has not earned a name yet -- and so is a
"+"-joined chain of these names (e.g. `"very-dim+very-warm+side"`), for a stack that has not
earned one either; see `canonical_lighting`."""

DEFAULT_LIGHTING_PRESET = "default"


def _color(value, where: str) -> Color:
    """Three numbers, or a loud error naming the field that was wrong."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"lighting {where} must be a sequence of 3 numbers, got {value!r}")
    if len(value) != 3:
        raise ValueError(f"lighting {where} must have 3 components, got {len(value)}: {value!r}")
    return tuple(float(channel) for channel in value)


def _range(value, where: str) -> tuple[float, float]:
    """A `(low, high)` pair, low first."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"lighting {where} must be a (low, high) pair, got {value!r}")
    low, high = (float(bound) for bound in value)
    if low > high:
        raise ValueError(f"lighting {where} has low > high: {value!r}")
    return low, high


def _fields(entry, required: set[str], optional: set[str], where: str) -> dict:
    """`entry` as a plain dict, with a missing or unknown key raising rather than defaulting.

    A silently ignored key is the failure mode this guards: `{"colour": ...}` for `"color"` would
    otherwise build a perfectly working env lit exactly the way the caller was trying to change.
    """
    if not isinstance(entry, Mapping):
        raise TypeError(f"lighting {where} must be a mapping, got {entry!r}")
    keys = set(entry)
    missing = required - keys
    unknown = keys - required - optional
    # both at once, because a misspelled key produces both halves and each alone is misleading
    problems = []
    if missing:
        problems.append(f"is missing {sorted(missing)}")
    if unknown:
        problems.append(f"has unknown keys {sorted(unknown)}")
    if problems:
        raise ValueError(
            f"lighting {where} {' and '.join(problems)}; "
            f"expected {sorted(required | optional)}"
        )
    return dict(entry)


def _light_from_dict(entry, index: int) -> DirectionalLight:
    where = f"lights[{index}]"
    fields = _fields(
        entry,
        required={"direction", "color"},
        optional={"shadow", "shadow_scale", "shadow_map_size"},
        where=where,
    )
    light = DirectionalLight(
        direction=_color(fields["direction"], f"{where}.direction"),
        color=_color(fields["color"], f"{where}.color"),
    )
    shadow = fields.get("shadow", light.shadow)
    return replace(
        light,
        shadow=None if shadow is None else bool(shadow),
        shadow_scale=float(fields.get("shadow_scale", light.shadow_scale)),
        shadow_map_size=int(fields.get("shadow_map_size", light.shadow_map_size)),
    )


def _randomization_from_dict(entry) -> LightingRandomization:
    fields = _fields(
        entry,
        required=set(),
        optional={"ambient", "brightness", "tint", "direction_jitter"},
        where="randomization",
    )
    base = LightingRandomization()
    return LightingRandomization(
        ambient=_range(fields.get("ambient", base.ambient), "randomization.ambient"),
        brightness=_range(fields.get("brightness", base.brightness), "randomization.brightness"),
        tint=_range(fields.get("tint", base.tint), "randomization.tint"),
        direction_jitter=float(fields.get("direction_jitter", base.direction_jitter)),
    )


def lighting_config_from_dict(spec: Mapping) -> LightingConfig:
    """A `LightingConfig` from the json-shaped dict form, for a condition with no preset name."""
    fields = _fields(
        spec, required={"ambient", "lights"}, optional={"randomization", "table_tint", "object_hue"},
        where="config",
    )
    lights = fields["lights"]
    if isinstance(lights, (str, bytes)) or not isinstance(lights, Sequence) or not lights:
        raise ValueError(f"lighting config.lights must be a non-empty sequence, got {lights!r}")
    randomization = fields.get("randomization")
    return LightingConfig(
        ambient=_color(fields["ambient"], "config.ambient"),
        lights=tuple(_light_from_dict(entry, i) for i, entry in enumerate(lights)),
        randomization=(
            None if randomization is None else _randomization_from_dict(randomization)
        ),
        table_tint=_color(fields.get("table_tint", (1.0, 1.0, 1.0)), "config.table_tint"),
        object_hue=float(fields.get("object_hue", 0.0)),
    )


def canonical_lighting(lighting: str | Mapping | LightingConfig) -> LightingConfig:
    """Resolve a preset name or a dict into a `LightingConfig`, rejecting unknown ones early.

    A name is looked up in `LIGHTING_PRESETS` first, so every named condition (`"default"`,
    `"random"` included) resolves exactly as it always has. Failing that, a "+"-joined name (e.g.
    `"very-dim+very-warm+side"`) is folded through `_stacked` instead of requiring every
    combination to be pre-declared there, and so is `"bright-set-<ambient>-<lights>"`, which sets
    the ambient to `<ambient>` and the key and fill lights to `<lights>` (each grey, all channels
    equal) rather than scaling the default's, and `"side-set-<amount>"`, which turns the key light
    `<amount>` (in [0, 1]) of the way from `"default"`'s direction to `"side"`'s, and
    `"warm-set-<amount>"` / `"cool-set-<amount>"`, which tint every light `<amount>` of the way
    along the blackbody curve from 6500 K to 2700 K / 12000 K at constant luminance, and past 1 on
    to 1.6 (about 2000 K / 24400 K), and `"table-set-<scale>"` / `"table-set-<r>-<g>-<b>"`, which
    multiply the table's colours by that tint and leave the lights alone, and
    `"object-hue-<degrees>"`, which turns every task object's colours that far round the hue wheel.

    `"domain-randomization"`, or a list of preset names, randomizes per episode and per parallel
    env over `DOMAIN_RANDOMIZATION_PRESETS` or that list; see `domain_randomized`.

    Worth doing before `gym.make` for the same reason `canonical_camera_view` is: a misspelled
    preset that fell through to "no override" would build a perfectly working env showing the
    unshifted scene, and nothing downstream would ever say so.
    """
    if isinstance(lighting, LightingConfig):
        return lighting
    if isinstance(lighting, str):
        if lighting == DOMAIN_RANDOMIZATION:
            return domain_randomized(DOMAIN_RANDOMIZATION_PRESETS)
        try:
            return LIGHTING_PRESETS[lighting]
        except KeyError:
            pass
        if "+" in lighting or _effect(lighting) is not None:
            return _stacked(lighting)
        raise ValueError(
            f"Unknown lighting preset {lighting!r}, expected one of "
            f"{', '.join(repr(name) for name in LIGHTING_PRESETS)}, "
            f"'bright-set-<ambient>-<lights>' (e.g. 'bright-set-0.45-1.5'), 'side-set-<amount>' "
            f"(e.g. 'side-set-0.5'), 'warm-set-<amount>' / 'cool-set-<amount>' (e.g. "
            f"'warm-set-0.5'), 'table-set-<scale>' / 'table-set-<r>-<g>-<b>' (e.g. "
            f"'table-set-0.4'), 'object-hue-<degrees>' (e.g. 'object-hue-60'), or a \"+\"-joined "
            f"stack of "
            f"{', '.join(repr(name) for name in _PRESET_EFFECTS)} and those sliders"
        )
    if isinstance(lighting, Mapping):
        return lighting_config_from_dict(lighting)
    if isinstance(lighting, Sequence) and not isinstance(lighting, bytes):
        return domain_randomized(list(lighting))
    raise TypeError(
        f"lighting must be a preset name, a list of them (domain randomization), a config dict "
        f"or a LightingConfig, got {lighting!r}"
    )


def _drawn_configs(config: LightingConfig, episode_rng, num_scenes: int) -> list[LightingConfig]:
    """One config per parallel env, drawn from `config.randomization`.

    Every draw is one batched call on `_batched_episode_rng`, which advances each env's own
    numpy RandomState in lockstep, so a given episode seed produces the same lighting under CPU
    and GPU simulation. Drawing per scene in a python loop would not: each scene would consume a
    different number of values from its stream.
    """
    random = config.randomization
    if episode_rng.batch_size != num_scenes:
        raise AssertionError(
            f"episode rng is batched over {episode_rng.batch_size} envs but the scene has "
            f"{num_scenes} sub-scenes"
        )

    ambient_scale = episode_rng.uniform(*random.ambient)
    ambient_tint = episode_rng.uniform(*random.tint, size=(3,))
    per_light = [
        (
            episode_rng.uniform(*random.brightness),
            episode_rng.uniform(*random.tint, size=(3,)),
            episode_rng.uniform(
                -random.direction_jitter, random.direction_jitter, size=(3,)
            ),
        )
        for _ in config.lights
    ]

    drawn = []
    for i in range(num_scenes):
        lights = []
        for light, (brightness, tint, jitter) in zip(config.lights, per_light):
            lights.append(
                replace(
                    light,
                    direction=tuple(np.asarray(light.direction) + jitter[i]),
                    color=tuple(np.asarray(light.color) * brightness[i] * tint[i]),
                )
            )
        drawn.append(
            replace(
                config,
                ambient=tuple(np.asarray(config.ambient) * ambient_scale[i] * ambient_tint[i]),
                lights=tuple(lights),
                randomization=None,
            )
        )
    return drawn


def _add_light(scene, light: DirectionalLight, enable_shadow: bool, scene_idxs=None) -> None:
    scene.add_directional_light(
        [float(component) for component in light.direction],
        [float(channel) for channel in light.color],
        shadow=enable_shadow if light.shadow is None else light.shadow,
        shadow_scale=light.shadow_scale,
        shadow_map_size=light.shadow_map_size,
        scene_idxs=scene_idxs,
    )


def apply_lighting(
    scene,
    config: LightingConfig,
    enable_shadow: bool,
    episode_rng=None,
    per_scene: Sequence[LightingConfig] | None = None,
) -> None:
    """Build `config`'s lights into `scene`. The body of a `_load_lighting` override.

    `per_scene` is one already-drawn config per parallel env, which is how a domain-randomized
    config arrives (`LightingMixin` draws it in `_load_scene`, which runs first, since the object
    hues it draws have to be applied there); each sub-scene is then lit by its own.

    A fixed config lights every parallel env identically. A randomizing one draws per sub-scene,
    which is per parallel env and *not* per episode: `_load_lighting` runs only inside
    `_reconfigure`, so an env keeps the condition it drew until it reconfigures. Under the default
    `reconfiguration_freq=0` that is once, at construction, off ManiSkill's fixed `2022 + i` seeds
    -- so a randomizing preset gives the same `num_envs` conditions on every run, and a reset seed
    does not move them. `reconfiguration_freq=1` redraws every reset, at the cost of rebuilding
    the scene.

    That makes randomization the right tool for training-time domain randomization (`num_envs`
    conditions at once) and the wrong one for measuring performance under a distribution of
    conditions -- for that, build the env once per named preset.
    """
    if config.domain_randomization and per_scene is None:
        raise ValueError("A domain-randomized lighting config needs its per-env draws.")
    if config.randomization is None and per_scene is None:
        scene.set_ambient_light([float(channel) for channel in config.ambient])
        for light in config.lights:
            _add_light(scene, light, enable_shadow)
        return

    if scene.parallel_in_single_scene:
        raise ValueError(
            "A randomizing lighting config needs one lighting condition per parallel env, but "
            "`parallel_in_single_scene=True` puts every env in one sapien scene, where "
            "`add_directional_light` adds a single light for all of them. Use a fixed preset "
            "here, or build the env with parallel_in_single_scene=False."
        )
    if per_scene is None:
        if episode_rng is None:
            raise ValueError("A randomizing lighting config needs the env's batched episode rng.")
        per_scene = _drawn_configs(config, episode_rng, len(scene.sub_scenes))
    if len(per_scene) != len(scene.sub_scenes):
        raise AssertionError(
            f"{len(per_scene)} per-env lighting configs for {len(scene.sub_scenes)} sub-scenes"
        )

    for scene_idx, drawn in enumerate(per_scene):
        # per-sub-scene, because `set_ambient_light` writes the same colour to all of them
        scene.sub_scenes[scene_idx].render_system.ambient_light = [
            float(channel) for channel in drawn.ambient
        ]
        for light in drawn.lights:
            _add_light(scene, light, enable_shadow, scene_idxs=[scene_idx])


def _add_scene_props(scene, props: tuple[SceneProp, ...]) -> None:
    """Build `props` into `scene` as static, visual-only boxes. The body of `_load_scene`'s addon.

    `build_static` with no collision shape added: invisible to physics, so a prop can never
    obstruct the robot regardless of where it sits. `props` defaults to `()`, so this is a no-op
    for every condition but the ones that ask for one (currently just `"shadows"`).
    """
    for i, prop in enumerate(props):
        builder = scene.create_actor_builder()
        builder.add_box_visual(
            pose=sapien.Pose(p=[float(c) for c in prop.position]),
            half_size=[float(c) for c in prop.half_size],
            material=sapien.render.RenderMaterial(
                base_color=[*(float(c) for c in prop.color), 1.0]
            ),
        )
        builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 0.0]))
        builder.build_static(name=f"lighting_prop_{i}")


def _srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def _remapped_texture(texture, remap: Callable[[np.ndarray], np.ndarray]):
    """A new texture holding `texture`'s pixels with `remap` applied to their linear colours.

    New rather than `texture.upload`-ed in place: sapien caches a model's textures for the whole
    process, so rewriting one would recolour that model in every env built after this one too,
    `"default"` included. `remap` takes and returns an (..., 3) array of linear RGB -- an sRGB
    texture is decoded before and re-encoded after, so a 0.4 tint reflects 40% of the light the way
    a 0.4 light colour would -- and alpha is untouched.
    """
    pixels = texture.download()
    if pixels.dtype != np.uint8:
        raise TypeError(f"expected a uint8 texture, got {pixels.dtype}")
    colour = pixels[..., :3].astype(np.float64) / 255.0
    if texture.is_srgb:
        colour = _linear_to_srgb(np.clip(remap(_srgb_to_linear(colour)), 0.0, 1.0))
    else:
        colour = remap(colour)
    tinted = pixels.copy()
    tinted[..., :3] = np.clip(np.round(colour * 255.0), 0, 255).astype(np.uint8)
    return sapien.render.RenderTexture2D(
        tinted,
        texture.format,
        texture.mipmap_levels,
        texture.filter_mode,
        texture.address_mode,
        texture.is_srgb,
    )


def _tint_table(env, tint: Color) -> None:
    """Multiply `tint` into the colour of `env`'s table: the wood texture of its top, and the
    `base_color` of its untextured parts.

    Only takes effect before the scene first renders, so it runs from `_load_scene`: a material
    edited after that is stored but never reaches the renderer. And the texture has to be replaced
    rather than scaled through `base_color`, which sapien ignores on a textured material.

    Every parallel env's table is the same model, so the tinted materials are worked out once, from
    the first env's, and assigned to all of them. That is also what keeps the tint from
    compounding: envs can share one material, and tinting each env's in turn would tint a shared
    one again. A task with no `TableSceneBuilder` table fails loudly rather than rendering the
    untinted scene under a condition that says otherwise.
    """
    table_scene = getattr(env, "table_scene", None)
    if table_scene is None:
        raise ValueError(
            f"{type(env).__name__} has no `table_scene`, so a lighting condition with a "
            f"table_tint of {tint} has no table to tint."
        )

    def parts(entity):
        body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        return [part for shape in body.render_shapes for part in shape.parts]

    entities = table_scene.table._objs
    template = parts(entities[0])
    tinted = []
    for part in template:
        texture = part.material.base_color_texture
        r, g, b, a = part.material.base_color
        tinted.append((
            None if texture is None else _remapped_texture(texture, lambda c: c * np.asarray(tint)),
            [r * tint[0], g * tint[1], b * tint[2], a],
        ))
    for entity in entities:
        entity_parts = parts(entity)
        if len(entity_parts) != len(template):
            raise AssertionError(
                f"table entity {entity.name} has {len(entity_parts)} render parts, the first has "
                f"{len(template)}: the parallel envs' tables are not the same model"
            )
        for part, (texture, base_color) in zip(entity_parts, tinted):
            if texture is None:
                part.material.base_color = base_color
            else:
                part.material.base_color_texture = texture


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """`colorsys.rgb_to_hsv` over an (..., 3) array, hue in [0, 1)."""
    r, g, b = np.moveaxis(rgb, -1, 0)
    value = rgb.max(axis=-1)
    delta = value - rgb.min(axis=-1)
    safe = np.where(delta > 0, delta, 1.0)
    hue = np.select(
        [delta == 0, value == r, value == g],
        [0.0, ((g - b) / safe) % 6.0, (b - r) / safe + 2.0],
        default=(r - g) / safe + 4.0,
    ) / 6.0
    saturation = np.where(value > 0, delta / np.where(value > 0, value, 1.0), 0.0)
    return np.stack([hue % 1.0, saturation, value], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """`colorsys.hsv_to_rgb` over an (..., 3) array."""
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    sector = np.floor(hue * 6.0)
    f = hue * 6.0 - sector
    p, q, t = value * (1 - saturation), value * (1 - saturation * f), value * (1 - saturation * (1 - f))
    sector = sector.astype(int) % 6
    channels = [
        np.choose(sector, [value, q, p, p, t, value]),
        np.choose(sector, [t, value, value, q, p, p]),
        np.choose(sector, [p, p, t, value, value, q]),
    ]
    return np.stack(channels, axis=-1)


def _rotate_hue(linear_rgb: np.ndarray, degrees: float) -> np.ndarray:
    """`linear_rgb` (..., 3) turned `degrees` round the HSV hue wheel of its sRGB encoding, the
    space "hue" is ordinarily meant in. Saturation and value are kept, so greys stay grey."""
    hsv = _rgb_to_hsv(_linear_to_srgb(np.clip(linear_rgb, 0.0, 1.0)))
    hsv[..., 0] = (hsv[..., 0] + degrees / 360.0) % 1.0
    return _srgb_to_linear(_hsv_to_rgb(hsv))


_MATERIAL_TEXTURES = (
    "base_color_texture",
    "emission_texture",
    "metallic_texture",
    "normal_texture",
    "roughness_texture",
    "transmission_texture",
)


def _copied_material(material):
    """A new `RenderMaterial` with every one of `material`'s values and textures."""
    copy = sapien.render.RenderMaterial(
        emission=list(material.emission),
        base_color=list(material.base_color),
        specular=material.specular,
        roughness=material.roughness,
        metallic=material.metallic,
        transmission=material.transmission,
        ior=material.ior,
        transmission_roughness=material.transmission_roughness,
    )
    for name in _MATERIAL_TEXTURES:
        texture = getattr(material, name)
        if texture is not None:
            setattr(copy, name, texture)
    return copy


@contextmanager
def _own_materials_per_entity(scene):
    """Within this, every actor built through `scene` gets its own copy of each visual material
    per parallel env, where ManiSkill's `ActorBuilder.build` would hand one material to all of
    them. Materials have no setter once built, so a per-env colour (a domain-randomized object
    hue) needs them apart from the start. Identical values, so the render is unchanged.

    A mesh loaded from a file (the table's glb) carries its own materials, not the builder's, and
    stays shared -- which is why a table tint cannot be domain-randomized.
    """
    create_actor_builder = scene.create_actor_builder

    def create_actor_builder_with_own_materials(*args, **kwargs):
        builder = create_actor_builder(*args, **kwargs)
        build_entity = builder.build_entity

        def build_entity_with_own_materials():
            for record in builder.visual_records:
                if record.material is not None:
                    record.material = _copied_material(record.material)
            return build_entity()

        builder.build_entity = build_entity_with_own_materials
        return builder

    scene.create_actor_builder = create_actor_builder_with_own_materials
    try:
        yield
    finally:
        del scene.create_actor_builder


def _shift_object_hues(env, degrees: float | Sequence[float]) -> None:
    """Turn the colours of every task object in `env` `degrees` round the hue wheel: each scene
    actor but the table scene's own (table and ground) and the lighting `props`, so the cube, the
    goal target and site, the peg -- whatever the task built. The robot is an articulation, not an
    actor, and is untouched.

    Like `_tint_table`, only effective from `_load_scene`, before the scene first renders. Every
    original colour is read before any is written, and each written as a function of its original,
    so a material shared between parallel envs is rotated once, not once per env; a textured
    object gets a new, rotated texture per entity rather than an in-place upload (see
    `_remapped_texture`). Objects are not assumed identical across envs, since some tasks draw a
    different one per env.

    `degrees` is one angle for every env, or one per parallel env (domain randomization). Per env
    only works on materials each env has to itself (`_own_materials_per_entity`), so every
    colour is read back afterwards: one that is not what its env asked for means another env's
    write landed on a shared material, and raises rather than rendering the wrong condition.
    """
    num_envs = env.num_envs
    per_env = np.broadcast_to(np.asarray(degrees, dtype=np.float64), (num_envs,))
    table_scene = getattr(env, "table_scene", None)
    fixtures = set() if table_scene is None else {obj.name for obj in table_scene.scene_objects}
    edits = []
    for name, actor in env.scene.actors.items():
        if name in fixtures or name.startswith("lighting_prop_"):
            continue
        for entity, scene_idx in zip(actor._objs, np.asarray(actor._scene_idxs.cpu()).tolist()):
            body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if body is None:
                continue
            for shape in body.render_shapes:
                for part in shape.parts:
                    material = part.material
                    edits.append((
                        material,
                        list(material.base_color),
                        material.base_color_texture,
                        float(per_env[scene_idx]),
                    ))
    if not edits:
        raise ValueError(
            f"{type(env).__name__} has no task objects to recolour, so a lighting condition with an "
            f"object_hue of {degrees} would render the unshifted scene."
        )
    rotated = [
        (
            material,
            [*_rotate_hue(np.asarray(base_color[:3]), angle).tolist(), base_color[3]]
            if angle != 0.0 else base_color,
            None if texture is None or angle == 0.0
            else _remapped_texture(texture, lambda c, angle=angle: _rotate_hue(c, angle)),
            angle,
        )
        for material, base_color, texture, angle in edits
    ]
    for material, base_color, texture, angle in rotated:
        if angle == 0.0:
            continue
        material.base_color = base_color
        if texture is not None:
            material.base_color_texture = texture
    for material, base_color, _, _ in rotated:
        if not np.allclose(list(material.base_color), base_color, atol=1e-5):
            raise AssertionError(
                f"a task object's material reads {list(material.base_color)} where its env asked "
                f"for {base_color}: parallel envs share this material, so they cannot be given "
                "different object hues"
            )


class LightingMixin:
    """Gives a task a `lighting` kwarg naming the condition it renders under.

    Mixed in ahead of the task class, so its `_load_lighting` wins. That is what makes the shift
    reach a task at all, and also means a task that lights its own scene would be overridden
    rather than shifted -- `__init__` says so loudly if it finds one, since the result would look
    like a lighting shift while actually being a different scene. `_load_scene` is mixed in the
    same way but *adds* to the task's own scene rather than replacing it -- see the method.

    The config is resolved and stored before `super().__init__`, because `BaseEnv.__init__`
    reconfigures (and so calls `_load_lighting` and `_load_scene`) before it returns.

    Under domain randomization (`lighting="domain-randomization"` or a list of preset names)
    every reset reconfigures -- `reconfiguration_freq=1`, the only way to change the lights,
    which the renderer does not pick up once built -- and each reconfiguration draws a fresh
    condition per parallel env, seeded by that reset's episode seed: the same reset seed redraws
    the same conditions, an unseeded reset draws new ones. ManiSkill cannot reconfigure on a
    partial reset, so every env has to reset together, which the `-v1.1` ids' never-terminating
    episodes guarantee. `episode_lighting` says what each env drew.
    """

    def __init__(
        self,
        *args,
        lighting: str | Sequence[str] | Mapping | LightingConfig = DEFAULT_LIGHTING_PRESET,
        **kwargs,
    ):
        self._lighting = canonical_lighting(lighting)
        self._episode_lighting: tuple[EpisodeLighting, ...] | None = None
        if self._lighting.domain_randomization:
            frequency = kwargs.setdefault("reconfiguration_freq", 1)
            if frequency != 1:
                raise ValueError(
                    f"Domain-randomized lighting redraws on every reset, which needs "
                    f"reconfiguration_freq=1, but reconfiguration_freq={frequency} was passed."
                )
        self._warn_if_task_lights_itself()
        super().__init__(*args, **kwargs)

    @property
    def lighting(self) -> LightingConfig:
        """The condition this env was built with; `DEFAULT_LIGHTING` unless asked otherwise."""
        return self._lighting

    @property
    def episode_lighting(self) -> tuple[EpisodeLighting, ...] | None:
        """Under domain randomization, what each parallel env drew for the current episode
        (preset, severity, and the config built from them); None otherwise."""
        return self._episode_lighting

    def _warn_if_task_lights_itself(self) -> None:
        mro = type(self).__mro__
        after_mixin = mro[mro.index(LightingMixin) + 1 :]
        for klass in after_mixin:
            if "_load_lighting" not in vars(klass):
                continue
            if klass is BaseEnv:
                return  # the default, which is exactly what DEFAULT_LIGHTING reproduces
            logger.warning(
                f"{klass.__name__} defines its own `_load_lighting`, which LightingMixin "
                "overrides: this env is lit by its `lighting` config alone, not by the task's "
                "own lights."
            )
            return

    def _load_lighting(self, options: dict):
        apply_lighting(
            self.scene,
            self._lighting,
            enable_shadow=self.enable_shadow,
            episode_rng=self._batched_episode_rng,
            per_scene=None if self._episode_lighting is None
            else [drawn.config for drawn in self._episode_lighting],
        )

    def _load_scene(self, options: dict):
        # unlike `_load_lighting`, this adds to the task's own scene rather than replacing it --
        # the task still builds its table, robot workspace and objects; a condition with `props`
        # (currently just "shadows") gets its occluder added alongside them, and one with a
        # `table_tint` has that table recoloured, and one with an `object_hue` the task's objects.
        # A domain-randomized one draws each env's condition here, after the task's own scene
        # (so the task's draws off the episode rng are the same as without it) and before
        # `_load_lighting`, which builds the lights from it.
        randomized = bool(self._lighting.domain_randomization)
        with _own_materials_per_entity(self.scene) if randomized else nullcontext():
            super()._load_scene(options)
        if randomized:
            self._episode_lighting = tuple(_drawn_episode_lighting(
                self._lighting, self._batched_episode_rng, self.num_envs
            ))
        _add_scene_props(self.scene, self._lighting.props)
        if self._lighting.table_tint != (1.0, 1.0, 1.0):
            _tint_table(self, self._lighting.table_tint)
        if randomized:
            hues = [drawn.config.object_hue for drawn in self._episode_lighting]
            if any(hue != 0.0 for hue in hues):
                _shift_object_hues(self, hues)
        elif self._lighting.object_hue != 0.0:
            _shift_object_hues(self, self._lighting.object_hue)


def supports_lighting(task_name: str) -> bool:
    """Whether `task_name`'s registered class takes a `lighting` kwarg."""
    spec = REGISTERED_ENVS.get(task_name)
    return spec is not None and issubclass(spec.cls, LightingMixin)


def check_lighting(env, config: LightingConfig) -> None:
    """Assert the condition `config` asked for is the one actually in the built scene.

    Two silent failures to catch. A scene that cannot render never has `_load_lighting` called at
    all (`BaseEnv._reconfigure`), so a shift asked for on a state-only env would simply not
    happen. And a light count that disagrees with the config means something else lit the scene.

    The default condition is not checked: it is the absence of a shift, so there is nothing it
    could silently fail to do, and every env in this project that predates `lighting` builds it.
    """
    if config == DEFAULT_LIGHTING:
        return

    scene = env.unwrapped.scene
    task = env.unwrapped.spec.id if env.unwrapped.spec else type(env.unwrapped).__name__
    if not scene.can_render():
        raise ValueError(
            f"A non-default lighting config was requested for {task}, but its scene cannot "
            "render, so ManiSkill never loaded any lighting. These observations carry no shift."
        )

    built = sum(entity.name == "directional_light" for entity in scene.sub_scenes[0].entities)
    if built != len(config.lights):
        raise ValueError(
            f"Lighting config for {task} declares {len(config.lights)} directional lights but "
            f"the built scene has {built}. Something other than this config lit the scene."
        )

    if config.randomization is None and not config.domain_randomization:
        ambient = np.asarray(scene.sub_scenes[0].render_system.ambient_light)[:3]
        if not np.allclose(ambient, config.ambient, atol=1e-5):
            raise ValueError(
                f"Lighting config for {task} asks for ambient {tuple(config.ambient)} but the "
                f"built scene has {tuple(ambient)}."
            )
