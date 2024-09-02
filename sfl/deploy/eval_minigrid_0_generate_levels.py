# %%
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', '..'))

import pickle
import numpy as np
import jax
from typing import Sequence, NamedTuple, Any, Dict
import jax.numpy as jnp 
import wandb 
import tqdm 
import matplotlib.pyplot as plt
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import functools
from flax.traverse_util import flatten_dict, unflatten_dict
from functools import partial
import distrax 

from jaxued.environments.maze.env import Maze
from jaxued.environments.maze.env_solved import MazeSolved
from jaxued.environments.maze.util import make_level_generator

# %% [markdown]
# ### Constants

# %%
SEED = 0

NUM_ENVS_TO_SAMPLE = 10
EVAL_ITERS = 100
TOTAL_ENVS = NUM_ENVS_TO_SAMPLE * EVAL_ITERS

PATH = f'sfl/data/eval/minigrid/eval_single_agent_{TOTAL_ENVS}e.pkl'
os.makedirs(os.path.dirname(PATH), exist_ok=True)
print('Total envs sampled: ', TOTAL_ENVS)

# %% [markdown]
# ### Run ID

# %%
rng = jax.random.PRNGKey(SEED)

env = Maze()
env_solved = MazeSolved()


get_random_level = make_level_generator(13, 13, 60)


state_set = []
level_set = []
# Kinda hacky, but generate twice the required levels and then filter out the bad ones
for _ in tqdm.tqdm(range(EVAL_ITERS)):
    rng, _rng = jax.random.split(rng)
    reset_rngs = jax.random.split(_rng, NUM_ENVS_TO_SAMPLE * 2)
    levels = jax.vmap(get_random_level)(reset_rngs)
    _, states = jax.vmap(env.reset_to_level)(reset_rngs, levels)
    _, states2 = jax.vmap(env_solved.reset_to_level)(reset_rngs, levels)

    steps = jax.vmap(env_solved.min_steps_to_goal)(states2)
    # print(steps.min(), steps.max(), steps.shape)
    is_bad = steps == jnp.inf
    print(is_bad.sum())

    idxs = jnp.argsort(is_bad)
    assert NUM_ENVS_TO_SAMPLE * 2 - is_bad.sum() >= NUM_ENVS_TO_SAMPLE, "Not enough valid levels"

    states = jax.tree_map(
        lambda x: x[idxs][:NUM_ENVS_TO_SAMPLE],
        states
    )
    levels = jax.tree_map(
        lambda x: x[idxs][:NUM_ENVS_TO_SAMPLE],
        levels
    )

    assert is_bad[idxs][:NUM_ENVS_TO_SAMPLE].sum() == 0
    # exit()
    state_set.append(states)
    level_set.append(levels)


with open(PATH, 'wb') as f:
    pickle.dump({
        'state': state_set,
        'level': level_set
    }, f)