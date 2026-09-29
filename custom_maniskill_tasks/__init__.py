"""Central definitions of the ManiSkill tasks this project trains and plans on.

Importing this package registers the task ids this project's datasets were collected with --
`PushCube-v1.1`, `PlaceSphere-v1.1`, `LiftPegUpright-v1.1`, `PokeCube-v1.1`, `PickCube-v1.1` and
`PickSingleYCB-v1.1`, the stock tasks with early termination removed (see `tasks`). They are then
available to plain `gym.make`, not only to `make_env`.

`make_env` is the entry point; everything else is exported for the callers that build part of a
stack themselves (a dataset replay that needs `sensor_configs` for `RecordEpisode`, an agent that
inserts its own observation adapter between `FrameSkip` and `FrameStack`).
"""

from custom_maniskill_tasks.backends import backend_kwargs, resolve_sim_backend
from custom_maniskill_tasks.cameras import (
    CAMERA_VIEWS,
    DEFAULT_CAMERA_RESOLUTION,
    FOCUSED_CAMERA_EYE,
    FOCUSED_CAMERA_FOV,
    FOCUSED_CAMERA_TARGET,
    FOCUSED_CAMERA_UID,
    WRIST_CAMERA_FOV,
    WRIST_CAMERA_UID,
    PandaHandCam,
    build_sensor_configs,
    camera_view_applied,
    canonical_camera_view,
    check_camera_view,
)
from custom_maniskill_tasks.lighting import (
    DEFAULT_LIGHTING,
    DEFAULT_LIGHTING_PRESET,
    DOMAIN_RANDOMIZATION,
    DOMAIN_RANDOMIZATION_PRESETS,
    LIGHTING_PRESETS,
    DirectionalLight,
    EpisodeLighting,
    LightingConfig,
    LightingMixin,
    LightingRandomization,
    SceneProp,
    apply_lighting,
    canonical_lighting,
    check_lighting,
    domain_randomized,
    lighting_config_from_dict,
    supports_lighting,
)
from custom_maniskill_tasks.make_env import TASKS_IN_USE, make_env

# imported for its side effect: registering the -v1.1 task ids with ManiSkill and gymnasium
from custom_maniskill_tasks.tasks import (
    CONTROL_MODE,
    DISTRACTORS,
    FULL_HORIZON_TASKS,
    Distractor,
    DistractorsMixin,
    FullHorizonMixin,
    LiftPegUprightFullHorizonEnv,
    PickCubeFullHorizonEnv,
    PickSingleYCBFullHorizonEnv,
    PlaceSphereFullHorizonEnv,
    PokeCubeFullHorizonEnv,
    PushCubeFullHorizonEnv,
)
from custom_maniskill_tasks.wrappers import (
    DINORewardWrapper,
    FrameSkip,
    FrameStack,
    IgnoreTerminations,
    clone_observations,
    stack_observations,
    stack_space,
)

__all__ = [
    "CAMERA_VIEWS",
    "CONTROL_MODE",
    "DEFAULT_CAMERA_RESOLUTION",
    "DEFAULT_LIGHTING",
    "DEFAULT_LIGHTING_PRESET",
    "DINORewardWrapper",
    "DISTRACTORS",
    "DOMAIN_RANDOMIZATION",
    "DOMAIN_RANDOMIZATION_PRESETS",
    "DirectionalLight",
    "Distractor",
    "DistractorsMixin",
    "EpisodeLighting",
    "FULL_HORIZON_TASKS",
    "FOCUSED_CAMERA_EYE",
    "FOCUSED_CAMERA_FOV",
    "FOCUSED_CAMERA_TARGET",
    "FOCUSED_CAMERA_UID",
    "FrameSkip",
    "FrameStack",
    "FullHorizonMixin",
    "IgnoreTerminations",
    "LIGHTING_PRESETS",
    "LightingConfig",
    "LightingMixin",
    "LightingRandomization",
    "LiftPegUprightFullHorizonEnv",
    "PandaHandCam",
    "PickCubeFullHorizonEnv",
    "PickSingleYCBFullHorizonEnv",
    "PlaceSphereFullHorizonEnv",
    "PokeCubeFullHorizonEnv",
    "PushCubeFullHorizonEnv",
    "SceneProp",
    "TASKS_IN_USE",
    "WRIST_CAMERA_FOV",
    "WRIST_CAMERA_UID",
    "apply_lighting",
    "backend_kwargs",
    "build_sensor_configs",
    "camera_view_applied",
    "canonical_camera_view",
    "canonical_lighting",
    "check_camera_view",
    "check_lighting",
    "clone_observations",
    "domain_randomized",
    "lighting_config_from_dict",
    "make_env",
    "resolve_sim_backend",
    "stack_observations",
    "stack_space",
    "supports_lighting",
]
