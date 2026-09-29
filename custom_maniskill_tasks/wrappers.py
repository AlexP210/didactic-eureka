"""Wrappers shared by everything that runs these tasks: online RL, planning, dataset replay.

`FrameSkip` and `FrameStack` are written to compose in that order (skip inside, stack outside),
which is what makes a stacked observation span `n_frames` *macro* steps rather than `n_frames`
primitive sim steps -- see `FrameStack`.

Both are agnostic about the observation layout so that they can sit on either side of a
project-specific observation adapter: ManiSkill's nested dicts of batched torch tensors, a
`TensorDict`, or a single tensor/array all work.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

try:  # tensordict is a S2P dependency, not a ManiSkill one
    from tensordict import TensorDict, TensorDictBase
except ImportError:  # pragma: no cover
    TensorDict = TensorDictBase = None


# --------------------------------------------------------------------------------------------- #
# observation-tree helpers
# --------------------------------------------------------------------------------------------- #


def _is_mapping(x) -> bool:
    return isinstance(x, dict) or (
        TensorDictBase is not None and isinstance(x, TensorDictBase)
    )


def _rebuild_mapping(template, items: dict):
    if TensorDictBase is not None and isinstance(template, TensorDictBase):
        return TensorDict(items)
    return dict(items)


def clone_observations(obs):
    """A private copy of `obs`, safe to hold on to across steps.

    Necessary, not defensive: ManiSkill hands back its camera capture buffer itself rather than a
    copy, so a `sensor_data/*/rgb` tensor (and `info["elapsed_steps"]`) is *overwritten in place*
    on the next step -- verified by `data_ptr` staying constant while the contents change. A frame
    buffer that stored the tensor instead of a copy would therefore hold N references to the
    current frame and stack it with itself. Rewards, termination flags, state observations and
    other extras are freshly allocated each step and need no copy.
    """
    if _is_mapping(obs):
        return _rebuild_mapping(
            obs, {key: clone_observations(value) for key, value in obs.items()}
        )
    if isinstance(obs, torch.Tensor):
        return obs.clone()
    if isinstance(obs, np.ndarray):
        return obs.copy()
    return obs


def stack_observations(frames, axis: int):
    """Stack a sequence of like-structured observations along a new axis at `axis`."""
    first = frames[0]
    if _is_mapping(first):
        return _rebuild_mapping(
            first,
            {key: stack_observations([f[key] for f in frames], axis) for key in first.keys()},
        )
    if isinstance(first, torch.Tensor):
        return torch.stack(list(frames), dim=axis)
    return np.stack([np.asarray(f) for f in frames], axis=axis)


def stack_space(space: gym.Space, n_frames: int, axis: int) -> gym.Space:
    """The observation space of `stack_observations` applied to samples from `space`."""
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {key: stack_space(sub, n_frames, axis) for key, sub in space.spaces.items()}
        )
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.stack([space.low] * n_frames, axis=axis),
            high=np.stack([space.high] * n_frames, axis=axis),
            dtype=space.dtype,
        )
    raise TypeError(
        f"FrameStack can only describe Dict and Box observation spaces, got {type(space)}"
    )


def _resolve_device(device):
    """`device` as a `torch.device` with an explicit cuda index, or None.

    An index-less `"cuda"` is refused on a multi-gpu host, because it does not mean one thing
    here. Two ways it goes wrong, both silent until a conv deep inside the encoder complains that
    its weights and its input disagree:

    - `Tensor.to("cuda")` is a *no-op* on a tensor already on cuda:1 -- an index-less spec matches
      any device of that type -- so it would not move ManiSkill's frames off the renderer's device
      at all.
    - `"cuda"` resolves to whatever `torch.cuda.current_device()` happens to be, and building a
      ManiSkill env *changes* that: it is 0 before `make_env` and 1 after. A module moved with
      `.to("cuda")` before the env is built and frames sent to `"cuda"` after it therefore end up
      on different gpus from the same spelling.

    `backends.py` pins the sim and render backends by index for the same reason; ask the caller to
    do the same for the encoder rather than guessing on their behalf.
    """
    if device is None:
        return None
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        if torch.cuda.device_count() > 1:
            raise ValueError(
                f"device={device!r} is ambiguous on a {torch.cuda.device_count()}-gpu host, and "
                "an index-less 'cuda' will not move a frame the renderer already put on another "
                "gpu. Pass the encoder's device explicitly, e.g. 'cuda:0' -- note that "
                "torch.cuda.current_device() is 0 before make_env and 1 after it, so read the "
                "index off the encoder (next(model.parameters()).device) rather than off torch."
            )
        return torch.device("cuda", 0)
    return device


def _wrapper_attr(env, name: str, default=None):
    """`env.<name>` from anywhere in the wrapper stack, or `default`.

    gymnasium >= 1.0 dropped `Wrapper.__getattr__`, so a plain `getattr` only sees the outermost
    wrapper; `get_wrapper_attr` is the replacement that walks inwards.
    """
    try:
        return env.get_wrapper_attr(name)
    except (AttributeError, TypeError):
        return getattr(env, name, default)


class _AttributeForwarding:
    """Forward unknown public attributes inwards, the way pre-1.0 gymnasium wrappers did.

    Everything in this project reaches through the wrapper stack for ManiSkill's own API
    (`base_env`, `control_freq`, `get_state_dict`, ...), which gymnasium >= 1.0 only supports via
    the explicit `get_wrapper_attr`. Private names are deliberately not forwarded: letting
    `__deepcopy__`/`__getstate__`/`_frames` style lookups fall through hides real bugs.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "env":
            raise AttributeError(name)
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        try:
            return env.get_wrapper_attr(name)
        except (AttributeError, TypeError):
            raise AttributeError(
                f"{type(self).__name__} and the envs it wraps have no attribute {name!r}"
            ) from None


# --------------------------------------------------------------------------------------------- #
# episode-boundary helpers, shared by FrameSkip
# --------------------------------------------------------------------------------------------- #


def _episode_over(terminated, truncated):
    if isinstance(terminated, torch.Tensor) or isinstance(truncated, torch.Tensor):
        return torch.as_tensor(terminated) | torch.as_tensor(truncated)
    return bool(terminated) or bool(truncated)


def _all(flag) -> bool:
    if isinstance(flag, torch.Tensor):
        return bool(flag.all())
    return bool(np.all(flag))


def _keep_running(done):
    """The complement of `done`: which envs' next sub-step counts."""
    if isinstance(done, torch.Tensor):
        return ~done
    return not done


def _select(new, old, keep):
    """`new` where `keep`, `old` elsewhere, elementwise over the leading (num_envs) axis.

    Leaves that are not tensors of a maskable shape are taken from `new` wholesale -- a non-tensor
    info entry, or one that is not batched per env, carries no per-env structure to mask.
    """
    if keep is True or keep is False:
        return new if keep else old
    if _all(keep):
        return new
    if _is_mapping(new) and _is_mapping(old):
        return _rebuild_mapping(
            new,
            {
                key: _select(value, old[key], keep) if key in old else value
                for key, value in new.items()
            },
        )
    if (
        isinstance(new, torch.Tensor)
        and isinstance(old, torch.Tensor)
        and new.shape == old.shape
        and new.ndim > 0
        and new.shape[0] == keep.shape[0]
    ):
        mask = keep.reshape((-1,) + (1,) * (new.ndim - 1))
        return torch.where(mask, new, old)
    return new


# --------------------------------------------------------------------------------------------- #
# wrappers
# --------------------------------------------------------------------------------------------- #


class IgnoreTerminations(_AttributeForwarding, gym.Wrapper):
    """Report `terminated=False` always, so episodes end only when the caller says so.

    ManiSkill sets `terminated` to `info["success"]` recomputed from the current state on every
    step (`BaseEnv.step`), so it is a momentary predicate, not a true absorbing state -- it flips
    back to False if the goal stops being satisfied. Honouring it would end the episode the first
    time the goal is met, turning "reach the goal and hold it" into "reach it once" and forfeiting
    every subsequent reward (ManiSkill's dense rewards are positive at every step, so an early
    cutoff lowers the return). The tabletop tasks used here have no genuine absorbing states, so
    episodes end only on truncation (the time limit). Nothing is lost: the signal is still
    available as `info["success"]`.

    ManiSkill offers no construction-time equivalent that is usable underneath other wrappers:
    `BaseEnv` takes no termination-related kwarg, `ManiSkillVectorEnv(ignore_terminations=True)`
    also brings auto-reset and the gym vector API, and `CPUGymWrapper(ignore_terminations=True)`
    converts observations to unbatched numpy and must be outermost.

    Note this wrapper is *not* what keeps the env from resetting itself: plain `gym.make` never
    auto-resets (only `ManiSkillVectorEnv`/`make_vec` does), so a truncated episode stays in its
    final state until the caller calls `reset`.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(terminated, torch.Tensor):
            terminated = torch.zeros_like(terminated)
        else:
            terminated = False
        return obs, reward, terminated, truncated, info


class FrameSkip(_AttributeForwarding, gym.Wrapper):
    """Act in the space of `frame_skip` consecutive primitive actions.

    One `step` takes the concatenation of the next `frame_skip` actions -- shape `(frame_skip * D,)`
    where the wrapped env takes `(D,)`, or `(num_envs, frame_skip * D)` batched -- applies them in
    order, and reports the observation after the last one. This is the open-loop-chunk action space
    DINO-WM plans in (`rearrange(actions, "b (t f) d -> b t (f d)")` in its planner and
    `TrajSlicerDataset`'s `action_dim * frameskip`); as a wrapper, the env itself now speaks it, so
    the planner, the online trainer and the dataset all see the same interface.

    Semantics across the skipped steps:

    - reward is the **sum** over the sub-steps, which keeps returns comparable to the unskipped env
    - `terminated`/`truncated` are OR-ed
    - the observation and info are those of the last sub-step actually taken

    Sub-stepping stops early once every env reports done, so an episode cannot run past its time
    limit just because the limit fell in the middle of a chunk. With `num_envs > 1` and envs
    finishing at different sub-steps, data for an env that has already finished is frozen at its
    final step (the masking `mani_skill.utils.wrappers.ActionRepeatWrapper` does for the same
    reason), so a finished env contributes no further reward and no later observation.

    `frame_skip=1` is a pass-through apart from the (unchanged) action space.
    """

    def __init__(self, env: gym.Env, frame_skip: int = 1):
        super().__init__(env)
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
        self.frame_skip = frame_skip

        inner_action_space = env.action_space
        self._action_dim = int(inner_action_space.shape[-1])
        self.action_space = self._repeat_action_space(inner_action_space)
        single = _wrapper_attr(env, "single_action_space")
        if single is not None:
            # ManiSkill's per-env action space, which stays meaningful when num_envs > 1
            self.single_action_space = self._repeat_action_space(single)

    def _repeat_action_space(self, space: gym.Space) -> gym.Space:
        if self.frame_skip == 1:
            return space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(
                f"FrameSkip needs a Box action space to concatenate, got {type(space)}. "
                "Flatten dict actions first (mani_skill.utils.wrappers.FlattenActionSpaceWrapper)."
            )
        return gym.spaces.Box(
            low=np.concatenate([space.low] * self.frame_skip, axis=-1),
            high=np.concatenate([space.high] * self.frame_skip, axis=-1),
            dtype=space.dtype,
        )

    def _split_action(self, action):
        if self.frame_skip == 1:
            return [action]
        if not isinstance(action, (torch.Tensor, np.ndarray)):
            action = np.asarray(action, dtype=np.float32)
        width = action.shape[-1]
        if width != self.frame_skip * self._action_dim:
            raise ValueError(
                f"FrameSkip(frame_skip={self.frame_skip}) expects {self.frame_skip} x "
                f"{self._action_dim} = {self.frame_skip * self._action_dim} action values, got "
                f"{width}"
            )
        if isinstance(action, torch.Tensor):
            return list(torch.split(action, self._action_dim, dim=-1))
        return list(np.split(action, self.frame_skip, axis=-1))

    def step(self, action):
        sub_actions = self._split_action(action)
        obs, reward, terminated, truncated, info = self.env.step(sub_actions[0])
        for sub_action in sub_actions[1:]:
            done = _episode_over(terminated, truncated)
            if _all(done):
                break
            keep = _keep_running(done)
            if not _all(keep):
                # Some env has finished, so its observation has to survive the sub-steps the
                # others still take -- and ManiSkill would overwrite it in place (see
                # clone_observations). Only reached with num_envs > 1 and staggered episode ends,
                # so the common path pays nothing.
                obs = clone_observations(obs)
                info = clone_observations(info)
            new_obs, new_reward, new_terminated, new_truncated, new_info = self.env.step(
                sub_action
            )
            obs = _select(new_obs, obs, keep)
            info = _select(new_info, info, keep)
            # `+`/`|` rather than in-place ops: the env owns the tensors it returned
            reward = reward + new_reward * keep
            terminated = _select(new_terminated | terminated, terminated, keep)
            truncated = _select(new_truncated | truncated, truncated, keep)
        return obs, reward, terminated, truncated, info

    def rand_act(self):
        """A uniform random macro action, as a float32 torch tensor.

        Here rather than only on an inner wrapper because a `rand_act` that sampled the *primitive*
        action space would silently produce actions `frame_skip` times too short.
        """
        return torch.from_numpy(np.asarray(self.action_space.sample(), dtype=np.float32))


class FrameStack(_AttributeForwarding, gym.Wrapper):
    """Return the last `n_frames` observations stacked along a new axis.

    The axis is inserted at `frame_axis`, which defaults to 1 because ManiSkill observations carry
    a leading `num_envs` axis: `(num_envs, ...)` becomes `(num_envs, n_frames, ...)`. Pass
    `frame_axis=0` when an observation adapter underneath has already squeezed the batch axis away
    (as both the S2P and DINO-WM adapters do at `num_envs=1`), giving `(n_frames, ...)`.

    Stacked at whatever rate the env underneath steps at, so putting this outside `FrameSkip` --
    which `make_env` does -- stacks frames `frame_skip` primitive steps apart, matching how
    `TrajSlicerDataset` reads observations at the frameskip stride. Putting it inside would stack
    consecutive sim frames instead.

    The frame axis is added even at `n_frames=1`, so that observations keep the same rank (and a
    model keeps the same input shape) as `n_frames` is varied.

    After `reset` the buffer holds `n_frames` copies of the initial observation.
    """

    def __init__(self, env: gym.Env, n_frames: int = 1, frame_axis: int = 1):
        super().__init__(env)
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        self.n_frames = n_frames
        self.frame_axis = frame_axis
        self._frames: deque = deque([], maxlen=n_frames)
        self.observation_space = stack_space(env.observation_space, n_frames, frame_axis)

    def _stacked(self, frame, is_reset: bool = False):
        # copied because ManiSkill mutates its image buffers in place; see clone_observations
        frame = clone_observations(frame)
        for _ in range(self.n_frames if is_reset else 1):
            self._frames.append(frame)
        return stack_observations(list(self._frames), self.frame_axis)

    def reset(self, **kwargs):
        options = kwargs.get("options")
        num_envs = _wrapper_attr(self.env, "num_envs", 1)
        if (
            isinstance(options, dict)
            and "env_idx" in options
            and len(options["env_idx"]) < num_envs
        ):
            raise RuntimeError(
                "FrameStack cannot honour a partial reset: it keeps one frame buffer for the whole "
                "batch, so the envs left running and the ones just reset would share a history."
            )
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            frame, info = result
            return self._stacked(frame, is_reset=True), info
        # the S2P observation adapters return the observation alone
        return self._stacked(result, is_reset=True)

    def step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)
        return self._stacked(frame), reward, terminated, truncated, info


class DINORewardWrapper(_AttributeForwarding, gym.Wrapper):
    """Replace the task's reward with progress towards a goal *image*, measured in DINO space.

    With the default `shaping="delta"` the reward is the negative change in the distance to the
    goal encoding::

        d_t = ||encode(frame_t) - encode(goal)||_2
        r_t = d_{t-1} - d_t

    which is potential-based shaping with `Phi = -d`. Two consequences worth designing around:

    - The episode return telescopes to `d_0 - d_T`, so it is invariant to the path taken and says
      only how much closer the episode finished than it started.
    - Reward is *zero* once the goal is reached, so this rewards reaching and not holding. For
      tasks scored on the final frame (which is every task here, since `IgnoreTerminations` runs
      episodes to the time limit) pass `shaping="negative_distance"` for `r_t = -d_t` instead,
      which does pay for staying put. Both are available; `d_t` itself is always reported as
      `info["dino_distance"]`, so the other can be recovered offline either way.

    The env's own reward is not discarded, only demoted: it is moved to `info["env_reward"]`, and
    `info["success"]` is untouched, so the real task metric still reports normally.

    Where to put it in the stack
    ---------------------------
    Between `FrameSkip` and any observation adapter -- what `make_env(reward_wrappers=...)` does.
    There the observation is still ManiSkill's nested dict, so this reads
    `obs["sensor_data"][<camera>]["rgb"]` and works whichever adapter (S2P's TensorDict view,
    TD-MPC2's CHW tensor, DINO-WM's flat dict) is layered on afterwards. Outside `FrameSkip` rather
    than inside it, so a macro step costs one backbone forward instead of `frame_skip` of them --
    free, because the shaping telescopes and the two give the same total over a chunk either way.

    The encoder
    -----------
    `encoder` is supplied by the caller rather than built here, for two reasons: the reward then
    provably lives in the same feature space as whatever encoder the agent uses (a mismatched
    backbone produces the right *shape* and silently wrong features -- see the note in
    `s2p/models/dinov3_encoder_model.py`), and one frozen backbone is loaded instead of two.

    Its contract is deliberately narrow: it takes the raw camera batch, a `(num_envs, H, W, 3)`
    uint8 tensor exactly as ManiSkill returns it, and returns a float tensor batched the same way,
    `(num_envs, ...)`. *All* preprocessing -- resize, scale to [0, 1], ImageNet normalization --
    therefore belongs inside the callable, next to the backbone it was chosen for, and the goal
    image is put through the identical callable. For `DINOV3EncoderModel` (which wants
    `(T, B, S, 3, H, W)` and applies its own resize/normalize) that adapter is::

        def encode(images):                    # (N, H, W, 3) uint8
            x = images.permute(0, 3, 1, 2)     # (N, 3, H, W) -- still uint8, see below
            return model.encode(x[None, :, None])[0, :, 0]   # (N, num_tokens, token_length)

    Note the frames stay uint8: that model's `make_transform` reaches [0, 1] with
    `v2.ToDtype(scale=True)`, and `scale=True` is a *no-op* on an input that is already float, so
    handing it `.float()` frames in [0, 255] would ImageNet-normalize the wrong range and quietly
    encode into a different feature space -- the same class of failure the docstring in
    `s2p/models/dinov3_encoder_model.py` describes.

    The call is always made under `torch.no_grad()`: a reward carrying a graph back into a frozen
    backbone would be kept alive by every replay buffer that stores it.

    Distance
    --------
    Over the encoder's output flattened per env, i.e. the whole patch grid, which is the same
    quantity the DINO-WM/TC-WM planners minimize (`planning/objectives.py`). `normalize="rms"`
    (the default) divides by `sqrt(D)` so the scale does not move when the backbone, token mode or
    patch count changes; `normalize="none"` leaves the raw L2. `reward_scale` multiplies on top.

    The goal image
    --------------
    Set at construction, later via `set_goal`, or per episode via
    `reset(options={"goal_image": ...})` -- the key is consumed here and never forwarded to
    ManiSkill. It is uint8 `(H, W, 3)`, broadcast to every env, or `(num_envs, H, W, 3)` for a
    different goal per env, and it is copied on the way in: a frame captured from this env aliases
    the capture buffer ManiSkill overwrites in place (see `clone_observations`).

    Its encoding is computed once and cached -- not per step -- and deliberately not until the
    first frame arrives, because only then is the device ManiSkill renders on known. A goal built
    on the cpu (loaded from a file, or captured from a `physx_cpu` probe env) is therefore usable
    against a `physx_cuda` env without the caller having to know which cuda index `backend_kwargs`
    pinned the sim to.

    Partial resets need no special handling: ManiSkill's `reset(options={"env_idx": ...})` returns
    the whole batch's observation, and an env that was left running has the frame -- and therefore
    the distance -- it already had, so recomputing `d_0` for all of them is a no-op for those.

    Args:
        env: the wrapped env, in `rgb` obs mode.
        encoder: `(num_envs, H, W, 3)` uint8 -> `(num_envs, ...)` float; see above.
        goal_image: the initial goal, if there is one at construction time. `None` means one must
            arrive via `set_goal` or `reset(options=...)` before the first `reset`.
        camera_uid: which camera to encode. `None` resolves the scene's only rgb camera and fails
            if there are several, rather than guessing -- a reward computed against the wrong view
            is a silent failure.
        device: where to move frames before encoding; must be the encoder's device. `None` leaves
            them where the sim put them, which is right when both are already pinned to the same
            cuda device (`backend_kwargs`) or both on cpu. Must carry an explicit index on a
            multi-gpu host -- see `_resolve_device` for why a bare `"cuda"` is two different
            devices depending on when it is read.
        shaping: `"delta"` for `d_{t-1} - d_t`, `"negative_distance"` for `-d_t`.
        normalize: `"rms"` or `"none"`, as above.
        reward_scale: multiplies the reward, after normalization.
        expose_features: also put the frame's encoding in `info["dino_features"]`. Off by default
            because it is ~300 KB per env per step for a patch grid, but worth turning on when the
            agent's encoder is this same frozen backbone: the replayed frames then never have to be
            re-encoded during training.
    """

    SHAPINGS = ("delta", "negative_distance")
    NORMALIZATIONS = ("rms", "none")

    def __init__(
        self,
        env: gym.Env,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        goal_image=None,
        *,
        camera_uid: str | None = None,
        device=None,
        shaping: str = "delta",
        normalize: str = "rms",
        reward_scale: float = 1.0,
        expose_features: bool = False,
    ):
        super().__init__(env)
        if shaping not in self.SHAPINGS:
            raise ValueError(f"shaping must be one of {self.SHAPINGS}, got {shaping!r}")
        if normalize not in self.NORMALIZATIONS:
            raise ValueError(
                f"normalize must be one of {self.NORMALIZATIONS}, got {normalize!r}"
            )
        if not callable(encoder):
            raise TypeError(f"encoder must be callable, got {type(encoder)}")

        self.encoder = encoder
        self.device = _resolve_device(device)
        self.shaping = shaping
        self.normalize = normalize
        self.reward_scale = float(reward_scale)
        self.expose_features = expose_features

        self.num_envs = _wrapper_attr(env, "num_envs", 1)
        self.camera_uid = camera_uid or self._resolve_camera_uid(env)
        self._image_shape = self._resolve_image_shape(env)

        self._goal_image = None
        self._goal_features = None
        self._prev_distance = None
        if goal_image is not None:
            self.set_goal(goal_image)

    # ------------------------------------------------------------------ #
    # construction-time resolution
    # ------------------------------------------------------------------ #

    def _sensor_space(self, env):
        space = _wrapper_attr(env, "single_observation_space", None)
        if space is None:
            space = env.observation_space
        if not isinstance(space, gym.spaces.Dict) or "sensor_data" not in space.spaces:
            raise TypeError(
                f"{type(self).__name__} needs an env in obs_mode='rgb', whose observation is a "
                "dict with a 'sensor_data' entry; got an observation space of "
                f"{type(space).__name__}. Place this wrapper underneath any observation adapter "
                "-- make_env(reward_wrappers=...) does."
            )
        return space["sensor_data"]

    def _resolve_camera_uid(self, env) -> str:
        """The scene's one rgb camera, or a refusal to guess.

        Same reasoning as `CustomManiSkillWrapper._resolve_camera_uid` in the TD-MPC2 adapter:
        normally there is exactly one (the camera views add none and `wrist_only` drops the task's
        own), and where there are several, picking one silently would compute the reward against a
        view the policy never sees.
        """
        sensors = self._sensor_space(env)
        uids = [uid for uid, space in sensors.spaces.items() if "rgb" in space.spaces]
        if len(uids) == 1:
            return uids[0]
        raise ValueError(
            f"{type(self).__name__} cannot tell which camera to measure the goal against: the "
            f"scene renders {uids}. Pass camera_uid= explicitly."
        )

    def _resolve_image_shape(self, env):
        space = self._sensor_space(env)[self.camera_uid]["rgb"]
        return tuple(space.shape[-3:])  # (H, W, 3), whether or not num_envs leads

    # ------------------------------------------------------------------ #
    # the goal
    # ------------------------------------------------------------------ #

    @property
    def goal_image(self):
        """The current goal, as a `(1 or num_envs, H, W, 3)` uint8 tensor, or None."""
        return self._goal_image

    @property
    def goal_features(self):
        """The flattened encoding of `goal_image`, or None until the first frame has arrived.

        Not computed at `set_goal` time: see the class docstring on why the device it lives on is
        only knowable once ManiSkill has handed over a frame.
        """
        return self._goal_features

    def set_goal(self, goal_image) -> None:
        """Adopt `goal_image`. Does not reset the running distance.

        Call before `reset` -- or through `reset(options={"goal_image": ...})`, which sets it in
        the right order for you. Calling it mid-episode leaves `d_{t-1}` measured against the old
        goal, which makes exactly one step's reward meaningless.
        """
        self._goal_image = self._as_image_batch(goal_image)
        self._goal_features = None  # encoded on first use, see `_goal_reference`

    def _as_image_batch(self, goal_image) -> torch.Tensor:
        if isinstance(goal_image, (str, Path)):
            from PIL import Image  # lazy: only a file goal needs Pillow

            with Image.open(goal_image) as handle:
                goal_image = np.asarray(handle.convert("RGB"))
        if isinstance(goal_image, np.ndarray):
            goal_image = torch.from_numpy(goal_image)
        if not isinstance(goal_image, torch.Tensor):
            raise TypeError(
                "goal_image must be a torch tensor, a numpy array or a path to an image file, got "
                f"{type(goal_image)}"
            )

        if goal_image.dtype != torch.uint8:
            raise ValueError(
                f"goal_image must be uint8, like the camera frames it is compared against, got "
                f"{goal_image.dtype}. Convert it deliberately: whether the encoder sees [0, 255] "
                "or [0, 1] changes the features, and the goal has to travel the same path as the "
                "observations."
            )
        if goal_image.ndim == 3:
            goal_image = goal_image.unsqueeze(0)
        if goal_image.ndim != 4 or goal_image.shape[-1] != 3:
            raise ValueError(
                f"goal_image must be (H, W, 3) or (num_envs, H, W, 3), got "
                f"{tuple(goal_image.shape)}"
            )
        if goal_image.shape[0] not in (1, self.num_envs):
            raise ValueError(
                f"goal_image batches {goal_image.shape[0]} goals, which is neither 1 (shared by "
                f"every env) nor num_envs={self.num_envs}"
            )
        if tuple(goal_image.shape[-3:]) != self._image_shape:
            raise ValueError(
                f"goal_image is {tuple(goal_image.shape[-3:])} but this env's {self.camera_uid} "
                f"renders {self._image_shape}. Re-render or resize the goal to match: the same "
                "scene at a different resolution does not encode to the same features."
            )
        # copied, not referenced: a goal captured from this env aliases the buffer ManiSkill
        # overwrites on the next step (see clone_observations)
        return goal_image.clone()

    def _require_goal(self) -> None:
        if self._goal_image is None:
            raise RuntimeError(
                f"{type(self).__name__} has no goal image. Pass goal_image= at construction, call "
                'set_goal(...), or reset(options={"goal_image": ...}).'
            )

    # ------------------------------------------------------------------ #
    # encoding and distance
    # ------------------------------------------------------------------ #

    def _image(self, obs) -> torch.Tensor:
        return obs["sensor_data"][self.camera_uid]["rgb"]

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.device is not None:
            images = images.to(self.device)
        with torch.no_grad():
            features = self.encoder(images)
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"encoder must return a torch tensor batched like its input, got {type(features)}"
            )
        if features.shape[0] != images.shape[0]:
            raise ValueError(
                f"encoder returned {features.shape[0]} encodings for {images.shape[0]} frames; it "
                "must map (N, H, W, 3) -> (N, ...)"
            )
        return features

    @staticmethod
    def _flat(features: torch.Tensor) -> torch.Tensor:
        return features.reshape(features.shape[0], -1).float()

    def _goal_reference(self, like: torch.Tensor) -> torch.Tensor:
        """The goal's flattened encoding, on `like`'s device. Encoded once, then cached.

        The device is taken from the frames rather than fixed at construction because it is
        ManiSkill's choice: `backend_kwargs` may put the renderer on a cuda index the caller never
        names, while a goal loaded from a file or captured from a `physx_cpu` probe env starts on
        the cpu. Moving a uint8 frame between devices is lossless, so aligning here changes no
        feature -- unlike `device=`, which decides where the *backbone* runs.
        """
        if self._goal_features is None or self._goal_features.device != like.device:
            target = self.device if self.device is not None else like.device
            self._goal_features = self._flat(self._encode(self._goal_image.to(target)))
        return self._goal_features

    def _distance(self, features: torch.Tensor) -> torch.Tensor:
        flat = self._flat(features)
        # broadcasts a single shared goal across the batch
        distance = torch.linalg.vector_norm(flat - self._goal_reference(flat), dim=-1)
        if self.normalize == "rms":
            distance = distance / math.sqrt(flat.shape[-1])
        return distance

    def _reward_like(self, shaped: torch.Tensor, reward):
        """`shaped` in the container the wrapped env reports rewards in."""
        if isinstance(reward, torch.Tensor):
            return shaped.reshape(reward.shape).to(dtype=reward.dtype, device=reward.device)
        if isinstance(reward, np.ndarray):
            return shaped.reshape(reward.shape).cpu().numpy().astype(reward.dtype)
        return float(shaped.reshape(-1)[0])

    def _report(self, info, distance, features) -> None:
        if not isinstance(info, dict):
            return
        info["dino_distance"] = distance
        if self.expose_features:
            info["dino_features"] = features

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #

    def reset(self, **kwargs):
        options = kwargs.get("options")
        if isinstance(options, dict) and "goal_image" in options:
            options = dict(options)
            goal_image = options.pop("goal_image")
            kwargs["options"] = options
            self.set_goal(goal_image)
        self._require_goal()

        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, None)

        features = self._encode(self._image(obs))
        self._prev_distance = self._distance(features)
        self._report(info, self._prev_distance, features)
        return result

    def step(self, action):
        self._require_goal()
        if self._prev_distance is None:
            raise RuntimeError(
                f"{type(self).__name__}.step before reset: there is no previous distance to "
                "measure progress against."
            )
        obs, reward, terminated, truncated, info = self.env.step(action)

        features = self._encode(self._image(obs))
        distance = self._distance(features)
        if self.shaping == "delta":
            shaped = self._prev_distance - distance
        else:
            shaped = -distance
        self._prev_distance = distance

        if isinstance(info, dict):
            info["env_reward"] = reward
        self._report(info, distance, features)
        shaped = self._reward_like(shaped * self.reward_scale, reward)
        return obs, shaped, terminated, truncated, info
