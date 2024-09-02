
# %%
import sys
import os

from sfl.train.minigrid_plr import ActorCritic
# sys.path.append(os.path.join(os.getcwd(), '..', '..'))
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

from sfl.env.env import RobSimJAXMARL, State, EnvInstance, listify_reward, NUM_REWARD_COMPONENTS, REWARD_COMPONENT_DENSE, REWARD_COMPONENT_SPARSE
from sfl.runners.eval_runner import EvalSampledRunnerMinigrid
from sfl.train.utils import load_params
from sfl.util.rolling_stats import LogEpisodicStats
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from sfl.util.jaxued.jaxued_utils import compute_max_returns, max_mc, positive_value_loss, l1_value_loss, value_disagreement
import pandas as pd
from jaxued.environments.maze.env import Maze
from jaxued.environments.maze.env_solved import MazeSolved
from jaxued.environments.maze.util import make_level_generator
from jaxued.wrappers.autoreplay import AutoReplayWrapper as AutoReplay
from jaxued.environments.maze.level import Level
# %%
class Transition(NamedTuple):
    global_done: jnp.ndarray
    last_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def compute_score(config, dones, values, max_returns, advantages):
    if config['SCORE_FUNCTION'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['SCORE_FUNCTION'] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['SCORE_FUNCTION']}")

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

# %% [markdown]
# ### Constants

# %%
SEED = 0
GROUP = "minigrid_learnability_tuned_v2"

NUM_ENVS_TO_SAMPLE = 1000
EVAL_ITERS = 100
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


def do_run_for_single_group(GROUP_TO_USE):
    runs = api.runs("alex-plus/minigrid", filters={"group": GROUP_TO_USE})

    print('runs', runs)

    for run in runs:
        print(run.name, run.id, 'seed=', run.config["SEED"])


    first_config = runs[0].config
    # first_config["env"]["env_params"]['map_params']['valid_path_check'] = True

    env = Maze(normalize_obs=True)
    env = AutoReplay(env)

    # first_t_config = first_config["learning"]
    # first_t_config["LOG_DORMANCY"] = True
    network = ActorCritic(
        action_dim=env.action_space(env.default_params).n,
    )

    # with open(PATH, 'rb') as f:
    # vals = pickle.load(f)
    # state_set = vals['state']
    # levels = vals['level']

    levels = Level.load_prefabs([
    "SixteenRooms",
    "SixteenRooms2",
    "Labyrinth",
    "LabyrinthFlipped",
    "Labyrinth2",
    "StandardMaze",
    "StandardMaze2",
    "StandardMaze3",
    ])#Level.load_prefabs(levels)
    print('levels', levels)
    
    _, ss = jax.vmap(env.reset_env_to_level, (None, 0, None))(rng, levels, env.default_params)
    # state_set.append(ss)
    
    def rollout_run(run_name: str, env_states, rng):
        print('run_name', run_name)
        run = api.run("/".join(run_name))
        config = run.config

        print('run name', run_name)

        # LOAD PARAMS
        model_artificat = api.artifact(f"alex-plus/{run.project}/{run.name}-checkpoint:latest")
        name = model_artificat.download()
        network_params = load_params(name + "/model.safetensors")
            
        rng, _rng = jax.random.split(rng)
        runner = EvalSampledRunnerMinigrid(
            _rng,
            env,
            network,
            lambda x, y: ActorCritic.initialize_carry((x, )),
            hidden_size=256,
            greedy=False,
            env_init_states=env_states,
            n_episodes=N_EPISODES,
            n_envs=NUM_ENVS_TO_SAMPLE,
            is_minigrid=True
        )

        # EVALUATE
        rng, _rng = jax.random.split(rng)
        o = runner.run_and_visualise(_rng, network_params, run.name)  # TODO change rollout length. No need, minimax runs 10 eval steps.

        outcomes = jax.tree_map(lambda x: x[None], o)

        return jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), outcomes)


    # rollout runs

    for run in runs:
        # if run.config['SEED'] >= 6: continue
        ll = rollout_run(run.path, ss, rng)
        raise
        ll = jax.tree_map(lambda x: x.squeeze(), ll)
        print(run.name, ll['eval-sampled/win_rates'].shape, ll['eval-sampled/win_rates'].mean(), ll[f'eval-sampled/returns'].mean())

        data = {
            'env-id': jnp.arange(len(ll['eval-sampled/win_rates'])),
            'win-rates': ll['eval-sampled/win_rates'],
            'returns': ll['eval-sampled/returns'],
        }
        seed = run.config["SEED"]
        PATH_TO_SAVE = f'deploy/data/eval/results/minigrid/eval_seed_{SEED}/{GROUP_TO_USE}/{seed}.csv'
        os.makedirs(os.path.dirname(PATH_TO_SAVE), exist_ok=True)

        pd.DataFrame(data).to_csv(PATH_TO_SAVE, index=False)
        print("Saved", PATH_TO_SAVE)

GROUPS = [
    # "minigrid_plr_l1vl_single_v1",
    # "minigrid_dr_single_v1",
    # "minigrid_learnability_single_v1",
    # "minigrid_accel_maxmc_fixed_single_v1",
    # "minigrid_plr_pvl_single_v1",
    # "minigrid_plr_maxmc_single_v1",
    GROUP
]

for group in GROUPS:
    do_run_for_single_group(group)