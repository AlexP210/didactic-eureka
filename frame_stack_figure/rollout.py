import math, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/path/to/project/environments/custom_maniskill_tasks")
from environment_screenshots import build_env, load_agent, resolve_backends, as_obs_td, load_expert_config, expert_folder
from mani_skill.utils import gym_utils

OUT = Path(sys.argv[1]); seed = int(sys.argv[2])
task = "LiftPegUpright-v1.1"
device = torch.device("cuda:0")
cfg = load_expert_config(task)
env = build_env(cfg, 1, resolve_backends(device))
obs, _ = env.reset(seed=seed)
agent = load_agent(as_obs_td(obs, 1), math.prod(env.get_wrapper_attr("single_action_space").shape),
                   expert_folder(task) / "best_ckpt.pt", device)
for t in range(gym_utils.find_max_episode_steps_value(env)):
    Image.fromarray(obs["rgb"][0].clone().cpu().numpy()).save(OUT / f"wrist_{t:03d}.png")
    Image.fromarray(env.unwrapped.render_rgb_array()[0].cpu().numpy()).save(OUT / f"render_{t:03d}.png")
    with torch.no_grad():
        action = agent.get_action(as_obs_td(obs, 1))
    obs, _, _, _, info = env.step(action)
    if bool(info["success"][0]):
        print("success at step", t + 1); break
env.close()
