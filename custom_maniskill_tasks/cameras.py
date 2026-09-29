"""The camera views these tasks can be observed through.

One `camera_view` selects the *observation* camera for the whole task family, and everything
about that view lives here: the sensor overrides `gym.make` needs, plus the two things
`sensor_configs` cannot express (adding a camera, removing one), which are applied around the
`gym.make` call by `camera_view_applied`.

Previously this logic existed three times over -- in `ManiSkillTask.make_env`, in
`tools/replay_trajectory.py` and (as hardcoded constants) in DINO-WM's `PushCubeWrapper` -- which
matters because a dataset recorded through one view is useless to a policy running in another. The
numbers below are the ones every recorded dataset in this project was rendered with, so treat them
as data, not as tunables.
"""

from __future__ import annotations

import contextlib

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill import logger
from mani_skill.agents.registration import REGISTERED_AGENTS, register_agent
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import REGISTERED_ENVS

DEFAULT_CAMERA_RESOLUTION = 224
"""Square resolution every view renders at unless told otherwise. 224 because that is what the
DINO(v2/v3) encoders consuming these frames want; ManiSkill's own task defaults are 128."""

WRIST_CAMERA_FOV = 0.6 * np.pi
"""Deliberately much wider than the realsense ManiSkill models, for the fisheye look.

Applied through `sensor_configs`, not baked into an agent, because which Panda a task builds is
the task's own choice: PushCube-v1 defaults to `panda`, while the wristcam-only family
(PegInsertionSide-v1 and friends) declares `SUPPORTED_ROBOTS = ["panda_wristcam"]`. Overriding
the camera rather than the robot leaves each task on the robot it was recorded with.
"""

FOCUSED_CAMERA_EYE = [0.3, 0.0, 0.9]
FOCUSED_CAMERA_TARGET = [-0.1, 0.0, -0.3]
FOCUSED_CAMERA_FOV = np.pi / 5
"""A narrow (36 degree) view from above and in front, framed on the tabletop workspace."""

FOCUSED_CAMERA_UID = "base_camera"
"""Which of the task's own cameras `camera_view="focused"` re-poses. Every tabletop task in this
project registers exactly one external camera under this uid; `make_env` fails loudly rather than
silently rendering the task default if a task does not."""

WRIST_CAMERA_UID = "hand_camera"
"""ManiSkill's uid for the wrist realsense, on `panda_wristcam` and on `PandaHandCam` below."""

CAMERA_VIEWS = ("default", "focused", "wrist")

_CAMERA_VIEW_ALIASES = {
    # `ManiSkillTask` and `replay_trajectory.py` both called the default view "standard"
    "standard": "default",
}


def canonical_camera_view(camera_view: str) -> str:
    """Normalize a `camera_view` string, rejecting unknown ones early.

    Worth doing before `gym.make` rather than after: an unrecognized view that fell through to an
    empty `sensor_configs` would build a perfectly working env showing the wrong thing.
    """
    view = _CAMERA_VIEW_ALIASES.get(camera_view, camera_view)
    if view not in CAMERA_VIEWS:
        raise ValueError(
            f"Unknown camera_view {camera_view!r}, expected one of "
            f"{', '.join(repr(v) for v in CAMERA_VIEWS)}"
        )
    return view


class PandaHandCam(Panda):
    """The stock Panda with the wrist realsense that ManiSkill only ships on `panda_wristcam`.

    Only needed for tasks that default to plain `panda`, which has no wrist camera at all:
    `sensor_configs` can retune a camera but cannot add one. Tasks already on `panda_wristcam`
    need none of this and are left alone -- `WRIST_CAMERA_FOV` reaches their existing hand_camera
    through `sensor_configs`.

    The uid stays "panda" deliberately. TableSceneBuilder picks the starting qpos off the robot
    uid string, and its `panda_wristcam` branch rolls joint 7 by -90 degrees relative to the
    `panda` one, which changes the task. Overriding `panda` instead keeps the original starting
    pose and just adds the camera. Registered from `camera_view_applied`, not at import, so only
    the wrist view pays to render it.
    """

    uid = "panda"

    @property
    def _sensor_configs(self):
        # panda_v3 hangs the camera off panda_hand through two fixed joints; panda_v2 has the
        # same panda_hand, so composing the two puts the camera in exactly the same place
        pose = sapien.Pose(
            p=[0.035, 0, 0.036], q=euler2quat(0, -1.5707, 3.1415)
        ) * sapien.Pose(p=[0, 0.02, 0.0115])
        return [
            CameraConfig(
                uid=WRIST_CAMERA_UID,
                pose=pose,
                width=DEFAULT_CAMERA_RESOLUTION,
                height=DEFAULT_CAMERA_RESOLUTION,
                fov=WRIST_CAMERA_FOV,
                near=0.01,
                far=100,
                mount=self.robot.links_map["panda_hand"],
            )
        ]


def build_sensor_configs(
    camera_view: str,
    resolution: int | None = DEFAULT_CAMERA_RESOLUTION,
    focused_camera_uid: str = FOCUSED_CAMERA_UID,
) -> dict:
    """The `sensor_configs` kwarg for `gym.make` that realizes `camera_view`.

    `resolution=None` leaves every camera at whatever the task registered it with; any integer
    applies to all of them. ManiSkill reads a key that is not a camera uid as a global override
    and a key that is as a per-camera one (see `update_sensor_configs_from_dict`), which is why
    width/height sit at the top level while fov/pose are nested under a uid: re-posing globally
    would drag every camera a task owns onto the same viewpoint.
    """
    view = canonical_camera_view(camera_view)
    configs: dict = {}
    if resolution is not None:
        configs.update(width=resolution, height=resolution)

    if view == "default":
        # the task's own camera, left at its registered pose and fov
        return configs
    if view == "wrist":
        # The camera itself comes from the robot (see camera_view_applied) and is mounted at a
        # fixed pose relative to panda_hand, so the pose needs no override. The fov does: tasks
        # that default to `panda_wristcam` bring their own hand_camera at the stock pi/2 and
        # never consult the `panda` uid that PandaHandCam overrides.
        configs[WRIST_CAMERA_UID] = dict(fov=WRIST_CAMERA_FOV)
        return configs
    if view == "focused":
        camera_pose = sapien_utils.look_at(
            eye=FOCUSED_CAMERA_EYE, target=FOCUSED_CAMERA_TARGET
        )
        configs[focused_camera_uid] = dict(
            fov=FOCUSED_CAMERA_FOV,
            # [px, py, pz, qw, qx, qy, qz] rather than a Pose: ManiSkill converts the list form
            # back into a Pose for camera-specific entries, and a plain list keeps env kwargs
            # json-serializable (recorded into trajectory metadata) and picklable (the replay
            # tool ships them to spawned worker processes)
            pose=camera_pose.p[0].tolist() + camera_pose.q[0].tolist(),
        )
        return configs
    raise AssertionError(f"unhandled camera view {view!r}")  # canonical_camera_view guards this


def _register_wrist_camera_panda():
    """Point the `panda` uid at `PandaHandCam`, once per process.

    Process-wide and permanent, unlike the task-class swap below, because the agent is rebuilt
    from this registry on every reconfiguration (`BaseEnv._load_agent`), not just at construction:
    restoring the stock Panda afterwards would silently drop the wrist camera the next time a
    dataset-recording env reconfigures. The cost is that a *later* `camera_view="default"` env
    for a `panda` task in the same process also gets a hand_camera, and hence a wider observation
    space than it would have alone -- see the warning in `camera_view_applied`.
    """
    spec = REGISTERED_AGENTS.get(PandaHandCam.uid)
    if spec is not None and spec.agent_cls is PandaHandCam:
        return
    register_agent(override=True)(PandaHandCam)


@contextlib.contextmanager
def _task_without_default_sensors(task_name: str):
    """Build `task_name` from a subclass whose `_default_sensor_configs` is empty.

    `BaseEnv._setup_sensors` builds its sensor set as the task's `_default_sensor_configs` plus
    the agent's, so emptying the former is the only way to drop a task's own camera --
    `sensor_configs` can retune a camera but never remove one.

    Done by swapping the class the registry instantiates for the duration of the `gym.make` call
    rather than by patching the task class in place (which is what `replay_trajectory.py` used to
    do). The instance is left permanently on the subclass, so a later reconfiguration still sees
    the empty set, while other envs of the same task id in this process are untouched.
    """
    spec = REGISTERED_ENVS.get(task_name)
    if spec is None:
        # let gym.make raise its own error about the unknown id
        yield
        return
    original_cls = spec.cls
    spec.cls = type(
        f"{original_cls.__name__}NoDefaultSensors",
        (original_cls,),
        {"_default_sensor_configs": property(lambda self: [])},
    )
    try:
        yield
    finally:
        spec.cls = original_cls


_wrist_agent_registered = False


@contextlib.contextmanager
def camera_view_applied(camera_view: str, task_name: str, wrist_only: bool = True):
    """Hold the side effects `camera_view` needs, for the duration of a `gym.make` call.

    Only the wrist view needs any: a camera has to be added to the robot (the link to mount it on
    does not exist until `gym.make` has built the scene, so it cannot come from `sensor_configs`)
    and, under `wrist_only`, the task's own camera has to be removed or `BaseEnv` renders it and
    `RecordEpisode` writes it into every episode alongside the wrist view -- roughly doubling both
    the render cost and the dataset size for frames nothing reads. `render_mode="rgb_array"` is
    unaffected either way: it renders from the separate human render camera.

    Re-entered per process by design, since the replay tool's CPU path uses `mp.Pool` with the
    spawn start method and its children inherit neither the agent registry nor the class swap.
    """
    global _wrist_agent_registered
    view = canonical_camera_view(camera_view)

    if view != "wrist":
        if _wrist_agent_registered:
            logger.warning(
                f"Building a camera_view={view!r} env after a camera_view='wrist' one in the same "
                f"process: the `panda` uid still resolves to PandaHandCam, so this env will also "
                f"carry a {WRIST_CAMERA_UID} (and a correspondingly wider observation space). "
                "Build the two views in separate processes if that matters."
            )
        yield
        return

    _register_wrist_camera_panda()
    _wrist_agent_registered = True
    if wrist_only:
        with _task_without_default_sensors(task_name):
            yield
    else:
        yield


def check_camera_view(
    env, camera_view: str, focused_camera_uid: str = FOCUSED_CAMERA_UID
) -> None:
    """Assert the camera `camera_view` promised is actually in the built scene.

    Both non-default views address a camera by uid, and ManiSkill ignores a `sensor_configs` entry
    whose uid no camera matches. Without this check, a task whose external camera is not called
    `base_camera`, or one whose robot never got a wrist camera, would render its own default view
    and report nothing -- the exact failure that is invisible until the dataset is trained on.
    """
    view = canonical_camera_view(camera_view)
    required = {"focused": focused_camera_uid, "wrist": WRIST_CAMERA_UID}.get(view)
    if required is None:
        return
    sensors = getattr(env.unwrapped, "_sensors", {})
    if required not in sensors:
        raise ValueError(
            f"camera_view={view!r} configures the camera {required!r}, but "
            f"{env.unwrapped.spec.id if env.unwrapped.spec else type(env.unwrapped).__name__} "
            f"built sensors {sorted(sensors)}. Nothing was overridden, so these observations are "
            "the task's own default view."
        )
