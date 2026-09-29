"""Lighting presets against the real simulator: `python tests/test_lighting.py`.

Everything here runs on physx_cpu at num_envs=1, for the reason `test_make_env.py` gives: sapien
can only enable GPU PhysX once per process. That rules out building a randomizing preset, whose
whole point is one condition per parallel env, so the draw itself is tested against the batched
rng directly and the scene it would build against a stand-in scene.

The test that matters most is `test_default_is_byte_identical`: every dataset in this project was
recorded under `BaseEnv`'s own lighting, so `lighting="default"` has to be indistinguishable from
not having this feature at all.
"""

import json
from dataclasses import replace

import numpy as np
import sapien
import torch

from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG

from custom_maniskill_tasks import lighting
from custom_maniskill_tasks import (
    DEFAULT_LIGHTING,
    DOMAIN_RANDOMIZATION,
    DOMAIN_RANDOMIZATION_PRESETS,
    LIGHTING_PRESETS,
    LightingConfig,
    apply_lighting,
    canonical_lighting,
    check_lighting,
    make_env,
)

TASK = "PushCube-v1.1"
STOCK_TASK = "PushCube-v1"


def _frame(**kwargs):
    """The rendered reset frame of a freshly built env, plus what `gym.make` recorded."""
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", **kwargs)
    env.reset(seed=7)
    image = env.render()
    image = np.asarray(image.cpu() if isinstance(image, torch.Tensor) else image)
    spec_kwargs = dict(env.unwrapped.spec.kwargs)
    env.close()
    return image, spec_kwargs


def _hand_frame(**kwargs):
    """The `hand_camera` observation of a freshly built env, wrist-mounted rather than rendered."""
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist", **kwargs)
    obs, _ = env.reset(seed=7)
    image = obs["sensor_data"]["hand_camera"]["rgb"]
    image = np.asarray(image.cpu() if isinstance(image, torch.Tensor) else image)[0]
    env.close()
    return image


def test_default_is_byte_identical():
    """`lighting="default"` renders the same pixels as an env built before `lighting` existed.

    And it stays out of the env kwargs, so a recording made through it carries the same metadata
    it always did -- an existing dataset must not become distinguishable from a new one.
    """
    baseline, baseline_kwargs = _frame()
    explicit, explicit_kwargs = _frame(lighting="default")
    assert np.array_equal(baseline, explicit), "the default preset is not the stock lighting"
    assert "lighting" not in baseline_kwargs
    assert "lighting" not in explicit_kwargs, "a default condition should not reach gym.make"


def test_every_preset_shifts_the_image():
    """A named shift has to actually change what the camera sees, and say so in the env kwargs."""
    baseline, _ = _frame()
    for name in LIGHTING_PRESETS:
        if LIGHTING_PRESETS[name].randomization is not None:
            continue  # needs one sub-scene per env; see the module docstring
        if name == "default":
            continue
        image, spec_kwargs = _frame(lighting=name)
        difference = np.abs(image.astype(np.int16) - baseline.astype(np.int16)).mean()
        assert difference > 1.0, f"preset {name!r} barely changes the image ({difference:.3f})"
        assert spec_kwargs["lighting"] == name, f"preset {name!r} was not recorded in env kwargs"


def test_stacked_presets_compose():
    """A "+"-joined stack folds its presets' effects together, order-independent, into the scene."""
    stacked = canonical_lighting("very-dim+very-warm+side")
    reordered = canonical_lighting("side+very-warm+very-dim")
    assert stacked == reordered, "stacking is not order-independent"
    assert stacked != LIGHTING_PRESETS["very-dim"], "the stack collapsed to just one of its parts"
    assert stacked != canonical_lighting("very-dim+very-warm"), "'side' did not change the stack"

    image, spec_kwargs = _frame(lighting="very-dim+very-warm+side")
    baseline, _ = _frame()
    difference = np.abs(image.astype(np.int16) - baseline.astype(np.int16)).mean()
    assert difference > 1.0, f"stacked preset barely changes the image ({difference:.3f})"
    assert spec_kwargs["lighting"] == "very-dim+very-warm+side", "the stack name was not recorded"


def test_shadows_forces_only_the_lights_that_follow_enable_shadow():
    """The preset pins whichever lights follow `enable_shadow`, leaving an opted-out one alone."""
    config = LIGHTING_PRESETS["shadows"]
    assert config.lights[0].shadow is True, "the key light was not forced to cast a shadow"
    assert config.lights[1].shadow is False, "the fill light's own no-shadow choice was overridden"
    assert config.ambient == DEFAULT_LIGHTING.ambient, "shadows changed more than the shadow flags"

    # the env this preset builds always shows the shadow, even though `enable_shadow` defaults to
    # False -- that default-off session setting is exactly what this preset overrides
    image, _ = _frame(lighting="shadows")
    baseline, _ = _frame()
    difference = np.abs(image.astype(np.int16) - baseline.astype(np.int16)).mean()
    assert difference > 1.0, f"shadows barely changes the image ({difference:.3f})"

    stacked = canonical_lighting("shadows+side")
    assert stacked.lights[0].shadow is True, "shadows did not survive stacking with side"
    assert stacked.lights[0].direction == (-1.0, -1.0, -0.35), "side did not survive the stack"


def test_shadows_prop_is_out_of_view_but_casts_onto_the_hand_camera():
    """The occluder `shadows` adds to the scene sits out of frame, but its shadow does not.

    The wrist camera sits close above the workspace looking straight down (see `cameras.py`), so
    the robot's own shadow rarely reaches it -- forcing `shadow=True` alone barely changes what it
    sees (self-shadowing on the fingers and the target's raised rim, no real shadow on the table).
    `shadows` earns its name there because `LightingMixin._load_scene` also adds `_SHADOW_CASTER`.

    The comparison has to isolate that occluder's own contribution rather than compare against the
    true baseline: a first attempt at `_SHADOW_CASTER`'s placement passed a `shadow=True`-vs-
    default comparison (self-shadowing alone is a measurable, if invisible-in-practice, change)
    while its own shadow fell entirely outside the wrist camera's tiny field of view.
    """
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", lighting="shadows")
    scene = env.unwrapped.scene.sub_scenes[0]
    prop_names = {entity.name for entity in scene.entities if "lighting_prop_" in entity.name}
    assert prop_names, "no lighting_prop_* entity was built into the scene"
    env.close()

    shadow_without_prop = replace(LIGHTING_PRESETS["shadows"], props=())
    with_prop = _hand_frame(lighting="shadows")
    without_prop = _hand_frame(lighting=shadow_without_prop)
    difference = np.abs(with_prop.astype(np.int16) - without_prop.astype(np.int16)).mean()
    assert difference > 3.0, (
        f"_SHADOW_CASTER barely changes the hand_camera image on its own ({difference:.3f}) -- "
        "its shadow is likely missing the wrist camera's field of view"
    )


def test_bright_set_sets_the_levels_outright():
    """"bright-set-A-B" puts the ambient at A and both lights at B, alone or inside a stack."""
    config = canonical_lighting("bright-set-0.6-2.5")
    assert config.ambient == (0.6, 0.6, 0.6)
    assert [light.color for light in config.lights] == [(2.5, 2.5, 2.5), (2.5, 2.5, 2.5)]
    assert [light.direction for light in config.lights] == [
        light.direction for light in DEFAULT_LIGHTING.lights
    ]
    assert canonical_lighting("bright-set-0.3-1") == DEFAULT_LIGHTING

    # absolute, so a tint before it is overwritten and one after it still applies
    assert canonical_lighting("warm+bright-set-0.6-2.5") == config
    warm_after = canonical_lighting("bright-set-0.6-2.5+warm")
    assert warm_after.lights[0].color == (2.5 * 1.15, 2.5 * 0.9, 2.5 * 0.65)
    assert canonical_lighting("side+bright-set-0.6-2.5").lights[0].direction == (-1.0, -1.0, -0.35)

    for bad in ["bright-set-0.6", "bright-set-0.6-", "bright-set--1-2", "bright-set-a-2"]:
        try:
            canonical_lighting(bad)
        except ValueError as error:
            assert bad in str(error), f"{bad!r} raised {error!r}"
        else:
            raise AssertionError(f"{bad!r} was accepted")


def test_side_set_turns_the_key_light_from_default_to_side():
    """"side-set-t" slerps the key light from default's direction to side's, touching nothing else."""
    assert canonical_lighting("side-set-0") == DEFAULT_LIGHTING
    assert canonical_lighting("side-set-1") == LIGHTING_PRESETS["side"]

    def unit(v):
        return np.asarray(v) / np.linalg.norm(v)

    start = unit(DEFAULT_LIGHTING.lights[0].direction)
    end = unit(LIGHTING_PRESETS["side"].lights[0].direction)
    total = np.degrees(np.arccos(start @ end))
    for t in (0.1, 0.25, 0.5, 0.9):
        config = canonical_lighting(f"side-set-{t}")
        key = unit(config.lights[0].direction)
        assert np.isclose(key[0], key[1]), "left the x == y plane both endpoints lie in"
        assert np.isclose(np.degrees(np.arccos(start @ key)), t * total), "steps are not equal angles"
        assert np.isclose(np.degrees(np.arccos(key @ end)), (1 - t) * total)
        assert config.ambient == DEFAULT_LIGHTING.ambient
        assert config.lights[1:] == DEFAULT_LIGHTING.lights[1:]
        assert config.lights[0].color == DEFAULT_LIGHTING.lights[0].color

    # the short arc goes over the top, so somewhere in between the key light points straight down
    overhead = np.degrees(np.arccos(start @ np.array([0.0, 0.0, -1.0]))) / total
    assert np.allclose(unit(canonical_lighting(f"side-set-{overhead:.6f}").lights[0].direction),
                       [0.0, 0.0, -1.0], atol=1e-5)

    # composes with an exposure change, and a later direction setter wins
    stacked = canonical_lighting("side-set-0.5+bright-set-0.6-2.5")
    assert stacked.lights[0].direction == canonical_lighting("side-set-0.5").lights[0].direction
    assert stacked.ambient == (0.6, 0.6, 0.6)
    assert canonical_lighting("side+side-set-0.5") == canonical_lighting("side-set-0.5")

    for bad in ["side-set-1.5", "side-set-", "side-set--0.5", "side-set-x"]:
        try:
            canonical_lighting(bad)
        except ValueError as error:
            assert bad in str(error), f"{bad!r} raised {error!r}"
        else:
            raise AssertionError(f"{bad!r} was accepted")


def test_hue_set_slides_along_the_blackbody_curve():
    """"warm-set-t" / "cool-set-t" tint every light, from no tint at 0 to 2700 K / 12000 K at 1."""
    assert canonical_lighting("warm-set-0") == DEFAULT_LIGHTING
    assert canonical_lighting("cool-set-0") == DEFAULT_LIGHTING
    luminance = np.array([0.2126, 0.7152, 0.0722])

    def tint(name):
        config = canonical_lighting(name)
        ratio = np.asarray(config.ambient) / np.asarray(DEFAULT_LIGHTING.ambient)
        # one tint on everything, ambient included, same as the warm/cool presets
        for light, base in zip(config.lights, DEFAULT_LIGHTING.lights):
            assert np.allclose(np.asarray(light.color) / np.asarray(base.color), ratio)
            assert light.direction == base.direction
        return ratio

    for hue, redder in (("warm", True), ("cool", False)):
        previous = 1.0
        for t in (0.25, 0.5, 0.75, 1):
            ratio = tint(f"{hue}-set-{t}")
            assert np.isclose(luminance @ ratio, 1.0), "the hue slider changed the exposure"
            blue_over_red = ratio[2] / ratio[0]
            assert (blue_over_red < previous) if redder else (blue_over_red > previous)
            previous = blue_over_red
    # the endpoints are real light sources, not saturated primaries
    assert (np.asarray(tint("warm-set-1")) > 0.1).all() and (np.asarray(tint("cool-set-1")) > 0.5).all()

    stacked = canonical_lighting("warm-set-0.5+bright-set-0.3-1")
    assert stacked == DEFAULT_LIGHTING, "bright-set after a tint should overwrite it"
    assert canonical_lighting("warm-set-0.5+side") == canonical_lighting("side+warm-set-0.5")

    # past 1 keeps going the same way, until the limits where a light colour stops making sense
    for hue, redder in (("warm", True), ("cool", False)):
        at_one, past = tint(f"{hue}-set-1"), tint(f"{hue}-set-1.6")
        assert (past[2] / past[0] < at_one[2] / at_one[0]) if redder else (past[2] / past[0] > at_one[2] / at_one[0])
        assert np.isclose(luminance @ past, 1.0) and (past >= 0).all()

    for bad in ["warm-set-1.61", "cool-set-1.61", "cool-set-", "cool-set--0.5", "hot-set-0.5"]:
        try:
            canonical_lighting(bad)
        except ValueError as error:
            assert bad in str(error), f"{bad!r} raised {error!r}"
        else:
            raise AssertionError(f"{bad!r} was accepted")


def test_table_set_tints_only_the_table():
    """"table-set-s" / "table-set-r-g-b" recolour the table, leave the lights alone, and darken it
    in the wrist camera's view."""
    assert canonical_lighting("table-set-1") == DEFAULT_LIGHTING
    config = canonical_lighting("table-set-0.4")
    assert config.table_tint == (0.4, 0.4, 0.4)
    assert replace(config, table_tint=DEFAULT_LIGHTING.table_tint) == DEFAULT_LIGHTING
    assert canonical_lighting("table-set-0.2-0.3-0.5").table_tint == (0.2, 0.3, 0.5)
    assert config == LIGHTING_PRESETS["dark-table"]

    # stacks with the light shifts in any order, and the last table tint wins
    assert canonical_lighting("dark-table+warm") == canonical_lighting("warm+dark-table")
    assert canonical_lighting("dark-table+very-dark-table") == LIGHTING_PRESETS["very-dark-table"]
    assert canonical_lighting("bright-set-0.6-2+dark-table").lights[0].color == (2.0, 2.0, 2.0)

    # the wrist camera looks almost entirely at the tabletop, so a 0.4 tint has to darken it a lot
    baseline = _hand_frame().astype(np.float64).mean()
    darkened = _hand_frame(lighting="dark-table").astype(np.float64).mean()
    assert darkened < 0.8 * baseline, f"dark-table only took the wrist view from {baseline:.1f} to {darkened:.1f}"
    # sapien caches the table's texture process-wide, so a tint written into it would recolour
    # every env built afterwards; the default has to come back exactly as it was
    assert np.array_equal(_hand_frame(), _hand_frame()), "the hand camera frame is not deterministic"
    before, _ = _frame()
    _frame(lighting="very-dark-table")
    after, _ = _frame()
    assert np.array_equal(before, after), "a table tint leaked into an env built after it"

    for bad in ["table-set-", "table-set-0.4-0.5", "table-set--0.4", "table-set-a"]:
        try:
            canonical_lighting(bad)
        except ValueError as error:
            assert bad in str(error), f"{bad!r} raised {error!r}"
        else:
            raise AssertionError(f"{bad!r} was accepted")


def test_object_hue_turns_only_the_task_objects():
    """"object-hue-d" rotates the task objects' hues, touches nothing else in the config, changes
    the wrist view, and does not leak into an env built after it."""
    assert canonical_lighting("object-hue-0") == DEFAULT_LIGHTING
    config = canonical_lighting("object-hue-120")
    assert config.object_hue == 120.0
    assert replace(config, object_hue=0.0) == DEFAULT_LIGHTING
    assert canonical_lighting("object-hue-60+dark-table") == canonical_lighting("dark-table+object-hue-60")
    assert canonical_lighting("object-hue-60+object-hue-120") == config

    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", lighting="object-hue-120")
    unwrapped = env.unwrapped
    colours = {}
    for name in ("cube", "table-workspace"):
        body = unwrapped.scene.actors[name]._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
        colours[name] = [list(part.material.base_color) for shape in body.render_shapes for part in shape.parts]
    env.close()
    # PushCube's cube is blue (0.047, 0.165, 0.627); 120 degrees on is red-dominant
    r, g, b, _ = colours["cube"][0]
    assert r > g and r > b, f"the cube was not turned from blue towards red: {colours['cube'][0]}"
    assert colours["table-workspace"][1][:3] == [0.800000011920929] * 3, "the table was recoloured"

    baseline = _hand_frame()
    shifted = _hand_frame(lighting="object-hue-120")
    difference = np.abs(shifted.astype(np.int16) - baseline.astype(np.int16)).mean()
    assert difference > 1.0, f"object-hue-120 barely changes the wrist view ({difference:.3f})"
    assert np.array_equal(_hand_frame(), baseline), "an object hue leaked into an env built after it"

    for bad in ["object-hue-", "object-hue-360", "object-hue--30", "object-hue-a"]:
        try:
            canonical_lighting(bad)
        except ValueError as error:
            assert bad in str(error), f"{bad!r} raised {error!r}"
        else:
            raise AssertionError(f"{bad!r} was accepted")


def test_domain_randomization_resolves_to_its_presets():
    """"domain-randomization" is the project's four perturbations; a list names others; and every
    preset that cannot differ between parallel envs, or has no single severity-1 form, is refused."""
    config = canonical_lighting(DOMAIN_RANDOMIZATION)
    assert [name for name, _ in config.domain_randomization] == list(DOMAIN_RANDOMIZATION_PRESETS)
    assert replace(config, domain_randomization=()) == DEFAULT_LIGHTING
    assert canonical_lighting(["warm-set-1.1", "object-hue-30"]).domain_randomization == (
        ("warm-set-1.1", canonical_lighting("warm-set-1.1")),
        ("object-hue-30", canonical_lighting("object-hue-30")),
    )

    # severity 0 is the default and 1 the preset itself, for every preset in the list
    for name, endpoint in config.domain_randomization:
        assert lighting._interpolated(DEFAULT_LIGHTING, endpoint, 0.0) == DEFAULT_LIGHTING
        assert lighting._interpolated(DEFAULT_LIGHTING, endpoint, 1.0) == endpoint, name
    halfway = lighting._interpolated(DEFAULT_LIGHTING, canonical_lighting("bright-set-0.75-2.5"), 0.5)
    assert np.allclose(halfway.ambient, 0.525) and np.allclose(halfway.lights[0].color, 1.75)
    assert lighting._interpolated(DEFAULT_LIGHTING, canonical_lighting("object-hue-30"), 0.5).object_hue == 15.0

    for bad in (["table-set-0.4"], ["shadows"], ["random"], [DOMAIN_RANDOMIZATION], [], ["warm", 3]):
        try:
            canonical_lighting(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"{bad!r} was accepted for domain randomization")
    try:
        canonical_lighting(DOMAIN_RANDOMIZATION + "+warm")
    except ValueError:
        pass
    else:
        raise AssertionError("domain randomization was accepted inside a stack")


def _randomized_reset(env, seed):
    obs, _ = env.reset(seed=seed)
    image = obs["sensor_data"]["hand_camera"]["rgb"]
    image = np.asarray(image.cpu() if isinstance(image, torch.Tensor) else image)[0]
    (drawn,) = env.unwrapped.episode_lighting
    return image, drawn


def test_domain_randomization_redraws_every_reset():
    """Each reset rebuilds the scene under a fresh draw, the same seed redraws the same condition,
    and the draw reaches the rendered frame -- lights, ambient and object hue alike."""
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist", lighting=DOMAIN_RANDOMIZATION)
    try:
        assert env.unwrapped.reconfiguration_freq == 1
        draws = {seed: _randomized_reset(env, seed) for seed in range(12)}
        presets = {drawn.preset for _, drawn in draws.values()}
        assert len(presets) >= 3, f"12 resets only ever drew {presets}"
        severities = [drawn.severity for _, drawn in draws.values()]
        assert len(set(severities)) == len(severities) and all(0.0 <= s < 1.0 for s in severities)

        again, drawn_again = _randomized_reset(env, 3)
        assert drawn_again == draws[3][1], "the same reset seed drew a different condition"
        assert np.array_equal(again, draws[3][0]), "the same draw rendered a different frame"

        # an unseeded reset draws anew rather than repeating the last condition
        env.reset(seed=0)
        unseeded = [env.unwrapped.episode_lighting[0] for _ in range(4) if env.reset() is not None]
        assert len({(d.preset, d.severity) for d in unseeded}) == 4, "unseeded resets repeated a draw"

        # the drawn object hue is what the cube is built with
        seed, (_, hue_draw) = next((k, v) for k, v in draws.items() if v[1].preset == "object-hue-30")
        _randomized_reset(env, seed)
        cube = env.unwrapped.scene.actors["cube"]._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
        colour = np.asarray(cube.render_shapes[0].parts[0].material.base_color[:3])
        expected = lighting._rotate_hue(np.array([0.047, 0.165, 0.627]), 30.0 * hue_draw.severity)
        assert np.allclose(colour, expected, atol=2e-3), f"cube is {colour}, the draw asked for {expected}"
    finally:
        env.close()

    try:
        make_env(TASK, num_envs=1, sim_backend="physx_cpu", lighting=DOMAIN_RANDOMIZATION, reconfiguration_freq=0)
    except ValueError:
        pass
    else:
        raise AssertionError("domain randomization accepted reconfiguration_freq=0")


def test_domain_randomization_over_default_alone_is_the_stock_scene():
    """Randomizing over just "default" goes through every per-env path (a light per sub-scene,
    its own material per entity) and has to come out byte-identical to the default scene."""
    baseline = _hand_frame()
    env = make_env(TASK, num_envs=1, sim_backend="physx_cpu", camera_view="wrist", lighting=["default"])
    try:
        image, drawn = _randomized_reset(env, 7)
    finally:
        env.close()
    assert drawn.preset == "default"
    assert np.array_equal(image, baseline), "per-env lighting of the default does not render the default"


def test_unknown_preset_in_a_stack_is_rejected():
    """A misspelling inside a stack has to raise, same as a misspelled lone preset name."""
    try:
        canonical_lighting("very-dim+bogus")
    except ValueError as error:
        assert "bogus" in str(error) and "very-dim+bogus" in str(error)
    else:
        raise AssertionError("a stack with an unknown preset name was accepted")


def test_unknown_conditions_are_rejected_before_the_env_is_built():
    """A misspelling has to raise, not fall through to the unshifted scene."""
    for bad, expected in [
        ("dimm", "Unknown lighting preset"),
        ({"ambient": [1, 1, 1]}, "missing ['lights']"),
        ({"ambient": [1, 1, 1], "lights": [], "randomization": None}, "non-empty"),
        (
            {"ambient": [1, 1, 1], "lights": [{"direction": [0, 0, -1], "colour": [1, 1, 1]}]},
            "unknown keys ['colour']",
        ),
        ({"ambient": [1, 1], "lights": [{"direction": [0, 0, -1], "color": [1, 1, 1]}]}, "3 components"),
    ]:
        try:
            canonical_lighting(bad)
        except (ValueError, TypeError) as error:
            assert expected in str(error), f"{bad!r} raised {error!r}, expected {expected!r}"
        else:
            raise AssertionError(f"{bad!r} was accepted")


def test_a_task_without_the_mixin_says_so():
    """Stock ids take no `lighting` kwarg, so asking for a shift on one has to fail loudly."""
    try:
        make_env(STOCK_TASK, lighting="dim", num_envs=1, sim_backend="physx_cpu")
    except ValueError as error:
        assert "takes no `lighting` kwarg" in str(error)
    else:
        raise AssertionError("a stock -v1 id accepted a lighting shift")

    # but the default condition is not a shift, so it must not lock stock ids out
    env = make_env(STOCK_TASK, lighting="default", num_envs=1, sim_backend="physx_cpu")
    assert "lighting" not in env.unwrapped.spec.kwargs
    env.close()


def test_dict_conditions_reach_the_scene_and_survive_json():
    """A one-off condition builds, and is still readable in a recorded trajectory's metadata."""
    spec = {
        "ambient": [0.05, 0.05, 0.08],
        "lights": [{"direction": [0.2, -1.0, -0.6], "color": [1.4, 1.2, 0.9]}],
    }
    env = make_env(TASK, lighting=spec, num_envs=1, sim_backend="physx_cpu")
    scene = env.unwrapped.scene.sub_scenes[0]
    ambient = np.asarray(scene.render_system.ambient_light)[:3]
    lights = sum(entity.name == "directional_light" for entity in scene.entities)
    assert np.allclose(ambient, spec["ambient"], atol=1e-5), ambient
    assert lights == 1, f"expected the config's one light, found {lights}"
    assert env.unwrapped.lighting == canonical_lighting(spec)
    assert json.loads(json.dumps(env.unwrapped.spec.kwargs))["lighting"] == spec
    env.close()


def test_check_lighting_catches_a_scene_lit_by_something_else():
    """The guard `make_env` runs after the build, against a condition the scene does not have."""
    env = make_env(TASK, lighting="dim", num_envs=1, sim_backend="physx_cpu")
    check_lighting(env, LIGHTING_PRESETS["dim"])  # the truth: must not raise
    try:
        check_lighting(env, LIGHTING_PRESETS["bright"])
    except ValueError as error:
        assert "ambient" in str(error)
    else:
        raise AssertionError("check_lighting accepted the wrong condition")
    env.close()


class _StandInScene:
    """The two calls `apply_lighting` makes, recorded rather than rendered."""

    def __init__(self, num_scenes, parallel_in_single_scene=False):
        self.parallel_in_single_scene = parallel_in_single_scene
        self.sub_scenes = [type("S", (), {"render_system": type("R", (), {})()})() for _ in range(num_scenes)]
        self.lights = []

    def set_ambient_light(self, color):
        for scene in self.sub_scenes:
            scene.render_system.ambient_light = color

    def add_directional_light(self, direction, color, scene_idxs=None, **kwargs):
        self.lights.append((tuple(direction), tuple(color), scene_idxs))


def test_randomization_draws_one_condition_per_env():
    """Per parallel env, inside the configured ranges, and reproducible from the episode seeds."""
    config = LIGHTING_PRESETS["random"]
    num_envs = 4

    def build():
        scene = _StandInScene(num_envs)
        rng = BatchedRNG.from_seeds([2022 + i for i in range(num_envs)])
        apply_lighting(scene, config, enable_shadow=False, episode_rng=rng)
        return scene

    scene = build()
    ambients = [tuple(s.render_system.ambient_light) for s in scene.sub_scenes]
    assert len(set(ambients)) == num_envs, f"envs share a condition: {ambients}"
    assert len(scene.lights) == num_envs * len(config.lights)
    assert {light[2][0] for light in scene.lights} == set(range(num_envs)), "lights are not per-env"

    low, high = config.randomization.ambient
    tint_low, tint_high = config.randomization.tint
    for ambient in ambients:
        for channel, base in zip(ambient, DEFAULT_LIGHTING.ambient):
            assert base * low * tint_low - 1e-9 <= channel <= base * high * tint_high + 1e-9

    again = [tuple(s.render_system.ambient_light) for s in build().sub_scenes]
    assert again == ambients, "the same episode seeds drew different lighting"


def test_randomization_refuses_a_single_shared_scene():
    """One sapien scene cannot hold one condition per env, and must not pretend to."""
    scene = _StandInScene(4, parallel_in_single_scene=True)
    rng = BatchedRNG.from_seeds([2022 + i for i in range(4)])
    try:
        apply_lighting(scene, LIGHTING_PRESETS["random"], enable_shadow=False, episode_rng=rng)
    except ValueError as error:
        assert "parallel_in_single_scene" in str(error)
    else:
        raise AssertionError("a randomizing config was accepted in a single shared scene")


def test_fixed_conditions_light_every_env_the_same():
    """No draws, no per-env split: a named preset is one condition, which is what makes it a name."""
    scene = _StandInScene(3)
    apply_lighting(scene, LIGHTING_PRESETS["side"], enable_shadow=False)
    assert len({tuple(s.render_system.ambient_light) for s in scene.sub_scenes}) == 1
    assert all(light[2] is None for light in scene.lights), "a fixed condition drew per-env lights"
    assert len(scene.lights) == len(LIGHTING_PRESETS["side"].lights)


if __name__ == "__main__":
    tests = [
        test_unknown_conditions_are_rejected_before_the_env_is_built,
        test_randomization_draws_one_condition_per_env,
        test_randomization_refuses_a_single_shared_scene,
        test_fixed_conditions_light_every_env_the_same,
        test_default_is_byte_identical,
        test_every_preset_shifts_the_image,
        test_stacked_presets_compose,
        test_shadows_forces_only_the_lights_that_follow_enable_shadow,
        test_shadows_prop_is_out_of_view_but_casts_onto_the_hand_camera,
        test_bright_set_sets_the_levels_outright,
        test_side_set_turns_the_key_light_from_default_to_side,
        test_hue_set_slides_along_the_blackbody_curve,
        test_table_set_tints_only_the_table,
        test_object_hue_turns_only_the_task_objects,
        test_domain_randomization_resolves_to_its_presets,
        test_domain_randomization_redraws_every_reset,
        test_domain_randomization_over_default_alone_is_the_stock_scene,
        test_unknown_preset_in_a_stack_is_rejected,
        test_a_task_without_the_mixin_says_so,
        test_dict_conditions_reach_the_scene_and_survive_json,
        test_check_lighting_catches_a_scene_lit_by_something_else,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
