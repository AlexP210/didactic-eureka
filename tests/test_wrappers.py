"""Wrapper semantics, checked against a fake env rather than the simulator.

Two things here that a ManiSkill env cannot show: staggered episode ends (the tasks' time limit is
shared, so every env truncates on the same step) and cheap repetition. The fake env mimics the two
ManiSkill behaviours the wrappers have to cope with -- observation buffers reused in place, and
stepping continuing past `truncated` -- so run this as `python tests/test_wrappers.py`.
"""

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from custom_maniskill_tasks import (
    DINORewardWrapper,
    FrameSkip,
    FrameStack,
    IgnoreTerminations,
)

ACTION_DIM = 2


class FakeBatchedEnv(gym.Env):
    """A batched, torch-valued env in ManiSkill's image: `rgb` is a buffer overwritten in place,
    `info["elapsed"]` is the live step counter, and `step` keeps working after `truncated`."""

    def __init__(self, num_envs=3, limits=(2, 4, 6)):
        self.num_envs = num_envs
        self.limits = torch.tensor(limits[:num_envs])
        self.single_action_space = gym.spaces.Box(-1, 1, (ACTION_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (num_envs, ACTION_DIM), np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, (num_envs, 2, 2), np.uint8),
                "state": gym.spaces.Box(-np.inf, np.inf, (num_envs, 3), np.float32),
            }
        )
        self._rgb = torch.zeros((num_envs, 2, 2), dtype=torch.uint8)
        self.t = torch.zeros(num_envs, dtype=torch.long)
        self.actions_seen = []

    def _obs(self):
        self._rgb.copy_(self.t.reshape(-1, 1, 1).expand(-1, 2, 2).to(torch.uint8))
        return {"rgb": self._rgb, "state": self.t.reshape(-1, 1).expand(-1, 3).float()}

    def reset(self, seed=None, options=None):
        self.t.zero_()
        self.actions_seen = []
        return self._obs(), {"elapsed": self.t}

    def step(self, action):
        self.actions_seen.append(torch.as_tensor(action).clone())
        self.t += 1
        reward = torch.ones(self.num_envs)
        truncated = self.t >= self.limits
        terminated = self.t >= 1000  # never, but a real tensor
        return self._obs(), reward, terminated, truncated, {"elapsed": self.t}


class FakeSingleEnv(gym.Env):
    """Unbatched, python-scalar flags: what an env looks like under an adapter that squeezes."""

    def __init__(self, limit=5):
        self.limit = limit
        self.action_space = gym.spaces.Box(-1, 1, (ACTION_DIM,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)
        self.t = 0
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        self.steps_taken = 0
        return torch.full((3,), float(self.t)), {}

    def step(self, action):
        self.t += 1
        self.steps_taken += 1
        return torch.full((3,), float(self.t)), 1.0, False, self.t >= self.limit, {}


def test_frame_skip_action_space_and_splitting():
    env = FrameSkip(FakeBatchedEnv(), frame_skip=3)
    assert env.action_space.shape == (3, 3 * ACTION_DIM), env.action_space
    assert env.single_action_space.shape == (3 * ACTION_DIM,)
    assert env.rand_act().shape == (3, 3 * ACTION_DIM)
    assert env.rand_act().dtype == torch.float32

    env.reset()
    action = torch.arange(3 * 3 * ACTION_DIM, dtype=torch.float32).reshape(3, 3 * ACTION_DIM)
    env.step(action)
    seen = env.actions_seen
    assert len(seen) == 3, "one env step per skipped frame"
    for i, chunk in enumerate(seen):
        expected = action[:, i * ACTION_DIM : (i + 1) * ACTION_DIM]
        assert torch.equal(chunk, expected), f"sub-action {i} out of order: {chunk} != {expected}"

    try:
        env.step(torch.zeros(3, 2 * ACTION_DIM))
    except ValueError as error:
        assert "expects 3 x 2" in str(error), error
    else:
        raise AssertionError("a wrongly-sized macro action must be rejected")


def test_frame_skip_freezes_finished_envs():
    """Envs whose episode ended mid-chunk stop accumulating reward and keep their final frame."""
    env = FrameSkip(FakeBatchedEnv(num_envs=3, limits=(2, 4, 6)), frame_skip=3)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(torch.zeros(3, 3 * ACTION_DIM))

    # env 0 truncated after its 2nd sub-step, so it saw 2 of the 3 rewards
    assert torch.equal(reward, torch.tensor([2.0, 3.0, 3.0])), reward
    assert truncated.tolist() == [True, False, False], truncated
    assert not terminated.any()
    # its observation and info are frozen at the step it finished on -- this is also the check
    # that the frame was copied before the env overwrote its buffer
    assert obs["rgb"][:, 0, 0].tolist() == [2, 3, 3], obs["rgb"][:, 0, 0]
    assert obs["state"][:, 0].tolist() == [2.0, 3.0, 3.0], obs["state"]
    assert info["elapsed"].tolist() == [2, 3, 3], info["elapsed"]


def test_frame_skip_stops_at_the_episode_end():
    """No sub-step is taken once every env is done, so a chunk cannot overrun the time limit."""
    inner = FakeSingleEnv(limit=5)
    env = FrameSkip(inner, frame_skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(4 * ACTION_DIM, dtype=np.float32))
    assert (reward, truncated) == (4.0, False), (reward, truncated)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4 * ACTION_DIM, dtype=np.float32))
    assert truncated is True and reward == 1.0, (truncated, reward)
    assert inner.steps_taken == 5, f"took {inner.steps_taken} steps for a 5-step episode"
    assert torch.equal(obs, torch.full((3,), 5.0)), obs


def test_frame_skip_of_one_is_a_passthrough():
    env = FrameSkip(FakeSingleEnv(), frame_skip=1)
    assert env.action_space.shape == (ACTION_DIM,)
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
    assert reward == 1.0


def test_frame_stack_batched_history():
    env = FrameStack(FakeBatchedEnv(num_envs=3, limits=(99, 99, 99)), n_frames=3, frame_axis=1)
    assert env.observation_space["rgb"].shape == (3, 3, 2, 2), env.observation_space
    obs, _ = env.reset()
    assert obs["rgb"].shape == (3, 3, 2, 2)
    # reset fills the buffer with the initial frame
    assert obs["rgb"][0, :, 0, 0].tolist() == [0, 0, 0]
    for expected in ([0, 0, 1], [0, 1, 2], [1, 2, 3]):
        obs, *_ = env.step(np.zeros((3, ACTION_DIM), dtype=np.float32))
        assert obs["rgb"][0, :, 0, 0].tolist() == expected, obs["rgb"][0, :, 0, 0]
    assert obs["state"][0, :, 0].tolist() == [1.0, 2.0, 3.0]


def test_frame_stack_keeps_the_frame_axis_at_one():
    env = FrameStack(FakeBatchedEnv(), n_frames=1, frame_axis=1)
    obs, _ = env.reset()
    assert obs["rgb"].shape == (3, 1, 2, 2), obs["rgb"].shape


def test_frame_stack_unbatched_and_tensordict():
    class ToTensorDict(gym.Wrapper):
        """Stand-in for the S2P/DINO-WM adapters: squeeze the batch axis, hand back a TensorDict."""

        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Dict(
                {
                    "rgb": gym.spaces.Box(0, 255, (2, 2), np.uint8),
                    "state": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                }
            )

        def _obs(self, obs):
            return TensorDict({key: value[0] for key, value in obs.items()})

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._obs(obs), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return self._obs(obs), reward, terminated, truncated, info

    inner = ToTensorDict(FakeBatchedEnv(num_envs=1, limits=(99,)))
    env = FrameStack(inner, n_frames=2, frame_axis=0)
    obs, _ = env.reset()
    assert isinstance(obs, TensorDict), type(obs)
    assert obs["rgb"].shape == (2, 2, 2), obs["rgb"].shape
    obs, *_ = env.step(np.zeros((1, ACTION_DIM), dtype=np.float32))
    assert obs["rgb"][:, 0, 0].tolist() == [0, 1], obs["rgb"][:, 0, 0]


def test_frame_stack_over_frame_skip_strides_by_the_skip():
    env = FrameStack(
        FrameSkip(FakeBatchedEnv(num_envs=1, limits=(99,)), frame_skip=3),
        n_frames=3,
        frame_axis=1,
    )
    env.reset()
    for _ in range(3):
        obs, *_ = env.step(np.zeros((1, 3 * ACTION_DIM), dtype=np.float32))
    # macro steps land on sim steps 3, 6, 9 -- one stack spans 3 * frame_skip primitive steps
    assert obs["rgb"][0, :, 0, 0].tolist() == [3, 6, 9], obs["rgb"][0, :, 0, 0]


def test_ignore_terminations():
    class AlwaysTerminates(FakeBatchedEnv):
        def step(self, action):
            obs, reward, _, truncated, info = super().step(action)
            return obs, reward, torch.ones(self.num_envs, dtype=torch.bool), truncated, info

    env = IgnoreTerminations(AlwaysTerminates())
    env.reset()
    _, _, terminated, truncated, _ = env.step(np.zeros((3, ACTION_DIM), dtype=np.float32))
    assert not terminated.any(), terminated
    assert truncated.tolist() == [False, False, False]
    # and FrameSkip must not cut a chunk short because of a success flag it can no longer see
    env = FrameSkip(IgnoreTerminations(AlwaysTerminates(num_envs=1, limits=(99,))), frame_skip=4)
    env.reset()
    obs, reward, terminated, _, _ = env.step(np.zeros((1, 4 * ACTION_DIM), dtype=np.float32))
    assert float(reward[0]) == 4.0 and not terminated.any()


def test_attribute_forwarding():
    env = FrameStack(FrameSkip(IgnoreTerminations(FakeBatchedEnv()), frame_skip=2), n_frames=2)
    assert env.num_envs == 3, "gymnasium >= 1.0 needs wrappers to forward attributes themselves"
    assert env.frame_skip == 2
    try:
        env.no_such_attribute
    except AttributeError:
        pass
    else:
        raise AssertionError("a missing attribute must still raise AttributeError")


# --------------------------------------------------------------------------------------------- #
# DINORewardWrapper
# --------------------------------------------------------------------------------------------- #

IMAGE_SIZE = 2


class FakeCameraEnv(gym.Env):
    """ManiSkill's `sensor_data/<uid>/rgb` layout, with a frame that is a flat grey level.

    Every pixel of env i's frame equals its step counter, so `flat_pixel_encoder` below turns the
    frame into a constant vector and the RMS-normalized distance to a goal of level `g` is exactly
    `|level - g|` -- which makes every reward in these tests an integer computable by hand.
    """

    def __init__(self, num_envs=2, size=IMAGE_SIZE, cameras=("base_camera",)):
        self.num_envs = num_envs
        self.cameras = cameras
        self.action_space = gym.spaces.Box(-1, 1, (num_envs, ACTION_DIM), np.float32)
        rgb = gym.spaces.Box(0, 255, (num_envs, size, size, 3), np.uint8)
        single_rgb = gym.spaces.Box(0, 255, (size, size, 3), np.uint8)
        self.observation_space = self._space(rgb)
        self.single_observation_space = self._space(single_rgb)
        # one buffer per camera, overwritten in place the way ManiSkill's capture buffer is
        self._rgb = {uid: torch.zeros((num_envs, size, size, 3), dtype=torch.uint8)
                     for uid in cameras}
        self.t = torch.zeros(num_envs, dtype=torch.long)
        self.reset_options = None

    def _space(self, rgb):
        return gym.spaces.Dict(
            {
                "sensor_data": gym.spaces.Dict(
                    {uid: gym.spaces.Dict({"rgb": rgb}) for uid in self.cameras}
                )
            }
        )

    def _obs(self):
        for buffer in self._rgb.values():
            buffer.copy_(self.t.reshape(-1, 1, 1, 1).expand_as(buffer).to(torch.uint8))
        return {"sensor_data": {uid: {"rgb": buffer} for uid, buffer in self._rgb.items()}}

    def reset(self, seed=None, options=None):
        self.reset_options = options
        self.t.zero_()
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), torch.full((self.num_envs,), 7.0), torch.zeros(
            self.num_envs, dtype=torch.bool
        ), torch.zeros(self.num_envs, dtype=torch.bool), {}


def flat_pixel_encoder(counter=None):
    """A stand-in for the DINO backbone: flatten the frame, unchanged, to (N, D)."""

    def encode(images):
        if counter is not None:
            counter.append(images.clone())
        return images.reshape(images.shape[0], -1).float()

    return encode


def goal_frame(level, num_envs=None, size=IMAGE_SIZE):
    shape = (size, size, 3) if num_envs is None else (num_envs, size, size, 3)
    return torch.full(shape, level, dtype=torch.uint8)


def close(actual, expected):
    """The distances below are whole numbers in exact arithmetic, but a float32 norm over the
    flattened grid and the `/ sqrt(D)` normalization land a few ulps off, so compare loosely."""
    return torch.allclose(
        torch.as_tensor(actual, dtype=torch.float32),
        torch.as_tensor(expected, dtype=torch.float32),
        atol=1e-4,
    )


def test_dino_reward_is_the_negative_change_in_distance():
    """Frames climb 0,1,2,... towards a goal at 5: +1 per step while approaching, -1 past it."""
    env = DINORewardWrapper(
        FakeCameraEnv(num_envs=2), flat_pixel_encoder(), goal_frame(5)
    )
    _, info = env.reset()
    assert close(info["dino_distance"], [5.0, 5.0]), info

    rewards = []
    for _ in range(7):
        _, reward, _, _, info = env.step(np.zeros((2, ACTION_DIM), dtype=np.float32))
        rewards.append(reward)
    per_step = torch.stack(rewards)[:, 0]
    assert close(per_step, [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]), per_step

    # potential-based shaping telescopes: the return is d_0 - d_T and nothing else
    assert abs(float(per_step.sum()) - (5.0 - float(info["dino_distance"][0]))) < 1e-4


def test_dino_reward_keeps_the_env_reward_and_matches_its_container():
    env = DINORewardWrapper(FakeCameraEnv(num_envs=2), flat_pixel_encoder(), goal_frame(5))
    env.reset()
    _, reward, _, _, info = env.step(np.zeros((2, ACTION_DIM), dtype=np.float32))

    assert isinstance(reward, torch.Tensor) and reward.shape == (2,), reward
    assert reward.dtype == torch.float32
    assert torch.equal(info["env_reward"], torch.full((2,), 7.0)), info["env_reward"]
    assert "dino_features" not in info, "features are opt-in -- they are ~300 KB a step"

    env = DINORewardWrapper(
        FakeCameraEnv(num_envs=2), flat_pixel_encoder(), goal_frame(5), expose_features=True
    )
    env.reset()
    _, _, _, _, info = env.step(np.zeros((2, ACTION_DIM), dtype=np.float32))
    assert info["dino_features"].shape == (2, IMAGE_SIZE * IMAGE_SIZE * 3), info["dino_features"]


def test_dino_reward_negative_distance_shaping():
    """`-d_t` keeps paying at the goal, where the telescoping form pays nothing."""
    env = DINORewardWrapper(
        FakeCameraEnv(num_envs=1),
        flat_pixel_encoder(),
        goal_frame(3),
        shaping="negative_distance",
    )
    env.reset()
    rewards = [
        float(env.step(np.zeros((1, ACTION_DIM), dtype=np.float32))[1][0]) for _ in range(5)
    ]
    assert close(rewards, [-2.0, -1.0, 0.0, -1.0, -2.0]), rewards


def test_dino_reward_scale_and_unnormalized_distance():
    env = DINORewardWrapper(
        FakeCameraEnv(num_envs=1),
        flat_pixel_encoder(),
        goal_frame(5),
        normalize="none",
        reward_scale=0.5,
    )
    _, info = env.reset()
    # raw L2 over 2*2*3 pixels each 5 away from the goal
    assert close(info["dino_distance"][0], 5.0 * np.sqrt(12)), info
    _, reward, *_ = env.step(np.zeros((1, ACTION_DIM), dtype=np.float32))
    assert close(reward[0], 0.5 * np.sqrt(12)), reward


def test_dino_reward_goal_can_be_set_per_episode():
    """`options["goal_image"]` is consumed here and never reaches the wrapped env."""
    inner = FakeCameraEnv(num_envs=1)
    env = DINORewardWrapper(inner, flat_pixel_encoder(), goal_frame(5))
    _, info = env.reset(options={"goal_image": goal_frame(9), "env_idx": [0]})
    assert close(info["dino_distance"][0], 9.0), info
    assert inner.reset_options == {"env_idx": [0]}, inner.reset_options

    env.set_goal(goal_frame(2))
    assert int(env.goal_image[0, 0, 0, 0]) == 2
    _, info = env.reset()
    assert close(info["dino_distance"][0], 2.0), info


def test_dino_reward_per_env_goals():
    env = DINORewardWrapper(
        FakeCameraEnv(num_envs=2),
        flat_pixel_encoder(),
        torch.stack([goal_frame(4), goal_frame(9)]),
    )
    _, info = env.reset()
    assert close(info["dino_distance"], [4.0, 9.0]), info


def test_dino_reward_copies_the_goal_off_the_live_buffer():
    """A goal captured from the env must survive the env overwriting its capture buffer."""
    inner = FakeCameraEnv(num_envs=1)
    obs, _ = inner.reset()
    for _ in range(6):
        obs, *_ = inner.step(np.zeros((1, ACTION_DIM), dtype=np.float32))
    env = DINORewardWrapper(inner, flat_pixel_encoder(), obs["sensor_data"]["base_camera"]["rgb"])
    assert int(env.goal_image[0, 0, 0, 0]) == 6

    env.reset()  # rewinds the counter and rewrites the buffer in place
    assert int(env.goal_image[0, 0, 0, 0]) == 6, "the goal aliased the env's capture buffer"
    _, info = env.reset()
    assert close(info["dino_distance"][0], 6.0), info


def test_dino_reward_rejects_unusable_goals():
    inner = FakeCameraEnv(num_envs=2)
    encoder = flat_pixel_encoder()

    def rejects(goal, expected):
        try:
            DINORewardWrapper(inner, encoder, goal)
        except (ValueError, TypeError) as error:
            assert expected in str(error), f"{expected!r} not in {error}"
        else:
            raise AssertionError(f"an unusable goal must be rejected: {expected}")

    rejects(goal_frame(5).float() / 255.0, "must be uint8")
    rejects(torch.zeros((4, 4, 3), dtype=torch.uint8), "renders (2, 2, 3)")
    rejects(torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8), "num_envs=2")
    rejects(torch.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8), "(H, W, 3)")

    # and stepping with no goal at all fails loudly rather than silently rewarding nothing
    env = DINORewardWrapper(inner, encoder)
    try:
        env.reset()
    except RuntimeError as error:
        assert "no goal image" in str(error), error
    else:
        raise AssertionError("reset without a goal must fail")


def test_dino_reward_refuses_an_ambiguous_cuda_device():
    """`device='cuda'` names a different gpu before and after make_env -- see `_resolve_device`."""
    from custom_maniskill_tasks.wrappers import _resolve_device

    assert _resolve_device(None) is None
    assert _resolve_device("cpu") == torch.device("cpu")
    assert _resolve_device("cuda:1") == torch.device("cuda", 1)

    if torch.cuda.device_count() > 1:
        try:
            _resolve_device("cuda")
        except ValueError as error:
            assert "ambiguous" in str(error), error
        else:
            raise AssertionError("a bare 'cuda' must be refused on a multi-gpu host")
    else:
        assert _resolve_device("cuda") == torch.device("cuda", 0)


def test_dino_reward_refuses_to_guess_the_camera():
    inner = FakeCameraEnv(num_envs=1, cameras=("base_camera", "hand_camera"))
    try:
        DINORewardWrapper(inner, flat_pixel_encoder(), goal_frame(5))
    except ValueError as error:
        assert "cannot tell which camera" in str(error), error
    else:
        raise AssertionError("two rgb cameras must not be silently disambiguated")

    env = DINORewardWrapper(
        inner, flat_pixel_encoder(), goal_frame(5), camera_uid="hand_camera"
    )
    assert env.camera_uid == "hand_camera"
    _, info = env.reset()
    assert close(info["dino_distance"][0], 5.0), info


def test_dino_reward_encodes_once_per_macro_step_under_frame_skip():
    """Outside FrameSkip the backbone runs once a macro step, not once a primitive step."""
    seen = []
    env = DINORewardWrapper(
        FrameSkip(FakeCameraEnv(num_envs=1), frame_skip=3),
        flat_pixel_encoder(seen),
        goal_frame(9),
    )
    env.reset()
    assert len(seen) == 2, "the first reset encodes the goal once, then the initial frame"
    del seen[:]
    _, reward, *_ = env.step(np.zeros((1, 3 * ACTION_DIM), dtype=np.float32))
    assert len(seen) == 1, f"{len(seen)} encoder calls for one macro step"
    # d went 9 -> 6 across the chunk, so the macro reward is the whole chunk's progress
    assert close(reward[0], 3.0), reward

    # the goal's encoding is cached across steps and across episodes, not recomputed
    del seen[:]
    for _ in range(3):
        env.step(np.zeros((1, 3 * ACTION_DIM), dtype=np.float32))
    env.reset()
    assert len(seen) == 4, f"{len(seen)} encoder calls for 3 steps and a reset"


def test_dino_reward_under_an_observation_adapter():
    """It reads ManiSkill's own dict, so it has to sit underneath whatever reshapes it."""

    class ToImageTensor(gym.Wrapper):
        def _obs(self, obs):
            return obs["sensor_data"]["base_camera"]["rgb"].permute(0, 3, 1, 2)

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._obs(obs), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return self._obs(obs), reward, terminated, truncated, info

    env = ToImageTensor(
        DINORewardWrapper(FakeCameraEnv(num_envs=1), flat_pixel_encoder(), goal_frame(5))
    )
    obs, _ = env.reset()
    assert obs.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE), obs.shape
    _, reward, _, _, info = env.step(np.zeros((1, ACTION_DIM), dtype=np.float32))
    assert close(reward[0], 1.0) and close(info["dino_distance"][0], 4.0)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
