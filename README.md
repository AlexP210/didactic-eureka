# custom_maniskill_tasks

The ManiSkill tasks, cameras and lighting conditions used to train and evaluate Squeeze-to-Plan (S2P)
and the baselines. Online rollouts, planning rollouts and recorded datasets are all built through
this package, so they cannot drift apart.

```bash
pip install -e environments/custom_maniskill_tasks
```

## Tasks

Importing the package registers the three task ids used in the paper:

| id | is |
| --- | --- |
| `PushCube-v1.1` | `PushCube-v1` with no early termination |
| `PickCube-v1.1` | `PickCube-v1` with no early termination |
| `LiftPegUpright-v1.1` | `LiftPegUpright-v1` with no early termination |

"No early termination" is the convention every offline dataset was collected under: an episode is
never cut short by reaching the goal, so it runs the full horizon and holding the goal earns more
dense reward. `info["success"]`, the scene, dynamics, reward and success predicate are the stock
task's. `control_mode` defaults to `pd_ee_delta_pos`, which every recording used. These ids also take
a `lighting` kwarg (see [Lighting](#lighting)), which the stock `-v1` ids do not.

```python
import custom_maniskill_tasks  # noqa: F401  -- registers the ids
import gymnasium as gym

env = gym.make("PushCube-v1.1", num_envs=1)
```

## Usage

```python
from custom_maniskill_tasks import make_env

env = make_env(
    "PushCube-v1.1",
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    camera_view="wrist",
    lighting="default",      # see Lighting below
    camera_resolution=224,
)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.rand_act())
```

The env never resets itself, so every reset and the RNG stream that seeds it belong to the caller.
Truncation from the task's time limit is still reported, as the signal that a reset is due.

## Lighting

`lighting` picks the condition the scene is rendered under, for evaluating a policy on visual
conditions it was not trained on. It is an ordinary env kwarg, so it is recorded in
`env.spec.kwargs` and in trajectory files, and a replay rebuilds the same condition.

| preset | what it is |
| --- | --- |
| `default` | ManiSkill's own lighting (ambient `0.3`, two white directional lights). Every training dataset was recorded under it. |
| `bright-set-<A>-<B>` | ambient set to `A` and both directional lights to `B` (white) |
| `warm-set-<t>` | every light tinted `t` of the way along the blackbody curve from 6500 K (`t = 0`, i.e. `default`) to 2700 K (`t = 1`), at constant luminance; `t` runs up to 1.6 (about 2000 K) |
| `object-hue-<degrees>` | every task object's colours rotated `degrees` round the hue wheel (red → yellow → green → blue at 0 → 60 → 120 → 240); greys and white stay put, and the lights, table, ground and robot are unchanged |

`lighting="default"` reproduces ManiSkill's stock lighting exactly, so a policy evaluated at
`default` sees the same scene its training frames came from. Unknown preset names raise before the
env is built, since a shift that silently did not happen would look like a policy that is robust
to it.

### Names used in the paper

The lighting conditions in the paper map to these `lighting` values:

| Name in the paper | `lighting` value |
| --- | --- |
| Default | `default` |
| Bright | `bright-set-0.75-2.5` |
| Warm | `warm-set-1.1` |
| Colour-Shift | `object-hue-30` |
