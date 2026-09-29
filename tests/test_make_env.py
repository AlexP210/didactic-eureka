"""`make_env` against the real simulator: `python tests/test_make_env.py`.

Everything here runs on physx_cpu at num_envs=1, because sapien can only enable GPU PhysX once per
process and never after CPU PhysX has been touched. The camera tests run default -> focused ->
wrist deliberately: the wrist view repoints the `panda` agent uid process-wide (see
`cameras._register_wrist_camera_panda`), so a default-view env built afterwards would also carry a
hand_camera.
"""

import numpy as np
import torch

from custom_maniskill_tasks import make_env

TASK = "PushCube-v1"
ACTION = np.array([0.4, -0.3, 0.2, 0.0], dtype=np.float32)


def _scalar(x):
    return np.asarray(x.cpu() if isinstance(x, torch.Tensor) else x).reshape(-1)[0]


def test_no_reset_of_its_own():
    """The caller owns every reset: nothing resets at construction, on success, or on truncation."""
    env = make_env(TASK, obs_mode="state")
    # ManiSkill keeps the limit on its own TimeLimitWrapper; env.spec.max_episode_steps is None
    limit = env.get_wrapper_attr("_max_episode_steps")
    env.reset(seed=0)
    for _ in range(limit):
        _, _, terminated, truncated, _ = env.step(ACTION)
    assert not bool(_scalar(terminated)), "termination should be suppressed"
    assert bool(_scalar(truncated)), f"expected truncation at {limit} steps"
    assert int(_scalar(env.unwrapped.elapsed_steps)) == limit

    # stepping past the limit keeps the episode where it is instead of silently starting a new one
    env.step(ACTION)
    assert int(_scalar(env.unwrapped.elapsed_steps)) == limit + 1

    # and resetting with a seed is reproducible, i.e. nothing consumed the RNG behind our back
    first, _ = env.reset(seed=7)
    second, _ = env.reset(seed=7)
    assert torch.equal(first, second)
    env.close()


def test_success_does_not_end_the_episode():
    """`info["success"]` still reports the goal; `terminated` no longer acts on it."""
    env = make_env(TASK, obs_mode="state")
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(ACTION)
    assert "success" in info
    assert not bool(_scalar(terminated))
    env.close()

    env = make_env(TASK, obs_mode="state", ignore_terminations=False)
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(ACTION)
    assert bool(_scalar(terminated)) == bool(_scalar(info["success"])), "opt-out must restore it"
    env.close()


def test_frame_skip_matches_stepping_one_at_a_time():
    """A macro step of k actions is the same trajectory, and the same return, as k single steps."""
    frame_skip = 3
    actions = np.stack([ACTION * (i + 1) / frame_skip for i in range(frame_skip)])

    skipped = make_env(TASK, obs_mode="state", frame_skip=frame_skip)
    skipped.reset(seed=11)
    obs_skipped, reward_skipped, _, _, _ = skipped.step(actions.reshape(-1))
    assert int(_scalar(skipped.unwrapped.elapsed_steps)) == frame_skip
    skipped.close()

    plain = make_env(TASK, obs_mode="state")
    plain.reset(seed=11)
    reward_total = 0.0
    for action in actions:
        obs_plain, reward, _, _, _ = plain.step(action)
        reward_total += float(_scalar(reward))
    plain.close()

    assert torch.allclose(obs_skipped, obs_plain), "the sub-actions were not applied in order"
    assert abs(float(_scalar(reward_skipped)) - reward_total) < 1e-5, (
        f"macro reward {float(_scalar(reward_skipped))} != summed reward {reward_total}"
    )


def test_frame_stack_shape_and_history():
    env = make_env(TASK, obs_mode="state", n_frames=3)
    obs, _ = env.reset(seed=0)
    width = env.unwrapped.single_observation_space.shape[0]
    assert obs.shape == (1, 3, width), obs.shape
    assert torch.equal(obs[0, 0], obs[0, 2]), "reset fills the buffer with the first frame"
    obs, *_ = env.step(ACTION)
    assert not torch.equal(obs[0, 1], obs[0, 2]), "the newest frame should be the current one"
    env.close()


def test_camera_views():
    for view, expected in [
        ("default", ["base_camera"]),
        ("standard", ["base_camera"]),  # the old spelling of "default"
        ("focused", ["base_camera"]),
        ("wrist", ["hand_camera"]),  # keep last: the agent override is process-wide
    ]:
        env = make_env(TASK, obs_mode="rgb", camera_view=view, camera_resolution=128)
        assert sorted(env.unwrapped._sensors) == expected, (view, sorted(env.unwrapped._sensors))
        obs, _ = env.reset(seed=0)
        rgb = obs["sensor_data"][expected[0]]["rgb"]
        assert rgb.shape == (1, 128, 128, 3), (view, rgb.shape)
        env.close()

    try:
        make_env(TASK, camera_view="birds_eye")
    except ValueError as error:
        assert "birds_eye" in str(error)
    else:
        raise AssertionError("an unknown camera_view must be rejected")


def test_image_frames_are_not_aliased():
    """Regression test: ManiSkill overwrites its rgb buffer in place, so the frame buffer copies.

    Without the copy every frame in the stack is the current one, which looks like a working env
    right up until a model is asked to infer motion from it.
    """
    # the focused view rather than the wrist one, so this test leaves the agent registry alone
    env = make_env(TASK, obs_mode="rgb", camera_view="focused", camera_resolution=64, n_frames=3)
    env.reset(seed=0)
    for _ in range(3):
        obs, *_ = env.step(ACTION)
    frames = obs["sensor_data"]["base_camera"]["rgb"][0].float()
    spreads = [float((frames[i] - frames[i + 1]).abs().mean()) for i in range(len(frames) - 1)]
    assert min(spreads) > 0.0, f"stacked frames are identical: {spreads}"
    env.close()


if __name__ == "__main__":
    tests = [
        test_no_reset_of_its_own,
        test_success_does_not_end_the_episode,
        test_frame_skip_matches_stepping_one_at_a_time,
        test_frame_stack_shape_and_history,
        test_image_frames_are_not_aliased,
        test_camera_views,  # last: leaves the `panda` uid pointing at PandaHandCam
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
