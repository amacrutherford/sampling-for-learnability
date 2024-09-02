# %%
import sys
import os

from sfl.train.minigrid_plr import ActorCritic
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
import pickle
import numpy as np
import jax
from typing import Sequence, NamedTuple, Any, Dict
import jax.numpy as jnp 
import wandb 
import tqdm 
import matplotlib.pyplot as plt

import pandas as pd
from jaxued.environments.maze.env import Maze
from jaxued.environments.maze.env_solved import MazeSolved
from jaxued.environments.maze.util import make_level_generator
from jaxued.wrappers.autoreplay import AutoReplayWrapper as AutoReplay
from jaxued.environments.maze.renderer import MazeRenderer


# %% [markdown]
# ### Constants

# %%
SEED = int(sys.argv[-2])
GROUP = sys.argv[-1]

NUM_ENVS_TO_SAMPLE = 10000
EVAL_ITERS = 1
TOTAL_ENVS = NUM_ENVS_TO_SAMPLE * EVAL_ITERS

N_PARRALEL = 5
ROLLOUT_LEN = 1000

N_EPISODES = 10
PATH = f'deploy/data/eval/minigrid/eval_single_agent_{TOTAL_ENVS}e.pkl'

print('Total envs sampled: ', TOTAL_ENVS)

# %% [markdown]
# ### Run ID

# %%
rng = jax.random.PRNGKey(SEED)

api = wandb.Api()


env = Maze()
env_viz = MazeRenderer(env)
env = AutoReplay(env)


with open(PATH, 'rb') as f:
    vals = pickle.load(f)
    state_set = vals['state']
    levels = vals['level']

    state_set = []
    for j, l in enumerate(levels):
        imgs = jax.vmap(env_viz.render_level, (0, None))(l, env.default_params)
        fig, axs = plt.subplots(10, 10, figsize=(20, 20))
        for i in range(100):
            axs[i // 10, i % 10].imshow(imgs[i])
            axs[i // 10, i % 10].axis('off')
            axs[i // 10, i % 10].set_title(str(i))
        plt.tight_layout()
        path = f'deploy/data/eval/minigrid/levels/{j}.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.0)