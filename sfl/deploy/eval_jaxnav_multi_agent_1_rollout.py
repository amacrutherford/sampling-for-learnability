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
from flax.traverse_util import flatten_dict, unflatten_dict
from functools import partial
import pandas as pd

from jaxmarl.environments.jaxnav import JaxNav

from sfl.runners.eval_runner import EvalSampledRunner
from sfl.train.train_utils import load_params
from sfl.train.common.network import ActorCriticRNN, ScannedRNN


# %%
SEED = 0 #int(sys.argv[-2])
# GROUP = sys.argv[-1]

NUM_ENVS_TO_SAMPLE = 1000
EVAL_ITERS = 1
TOTAL_ENVS = NUM_ENVS_TO_SAMPLE * EVAL_ITERS

N_PARRALEL = 5
ROLLOUT_LEN = 1000

N_EPISODES = 10
PATH = f'sfl/data/eval/jaxnav/eval_multi_agent_{TOTAL_ENVS}e.pkl'

print('Total envs sampled: ', TOTAL_ENVS)

# %% [markdown]
# ### Run ID

# %%
rng = jax.random.PRNGKey(SEED)

api = wandb.Api()


def do_run_for_single_group(GROUP_TO_USE):
    runs = api.runs("alex-plus/multi_robot_ued", filters={"group": GROUP_TO_USE})

    print('runs', runs)

    for run in runs:
        print(run.name, run.id, 'seed=', run.config["SEED"])


    first_config = runs[0].config
    first_config["env"]["env_params"]['map_params']['valid_path_check'] = True

    env = JaxNav(num_agents=first_config["env"]["num_agents"], 
                        **first_config["env"]["env_params"])

    first_t_config = first_config["learning"]
    first_t_config["LOG_DORMANCY"] = True
    network = ActorCriticRNN(
        action_dim=env.agent_action_space().shape[0],
        config=first_t_config,
    )

    with open(PATH, 'rb') as f:
        state_set = pickle.load(f)

    def rollout_run(run_name: str, env_state_set, rng):
        print('run_name', run_name)
        run = api.run("/".join(run_name))
        config = run.config
        t_config = config["learning"]

        print('run name', run_name)

        # LOAD PARAMS
        model_artificat = api.artifact(f"alex-plus/{run.project}/{run.name}-checkpoint:latest")
        name = model_artificat.download()
        network_params = load_params(name + "/model.safetensors")
        kept_outcomes = []
        for i, env_states in enumerate(env_state_set):
            
            rng, _rng = jax.random.split(rng)
            runner = EvalSampledRunner(
                _rng,
                env,
                network,
                ScannedRNN.initialize_carry,
                hidden_size=config["learning"]["HIDDEN_SIZE"],
                greedy=False,
                env_init_states=env_states,
                n_episodes=N_EPISODES,
                n_envs=NUM_ENVS_TO_SAMPLE,
            )

            # EVALUATE
            rng, _rng = jax.random.split(rng)
            o = runner.run(_rng, network_params)  # TODO change rollout length. No need, minimax runs 10 eval steps.
            # print('shape mike', o['eval-sampled/multi-agent-win-rate'].shape)
            if i == 0:
                outcomes = jax.tree_map(lambda x: x[None], o)
            else:
                outcomes = jax.tree_map(lambda old, new: jnp.concatenate([old, new[None]], axis=0), outcomes, o)

        return jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), outcomes)          


    # rollout runs

    for run in runs:
        ll = rollout_run(run.path, state_set, rng)
        ll = jax.tree_map(lambda x: x.squeeze(), ll)
        print(run, ll['eval-sampled/multi-agent-win-rate'].shape, ll['eval-sampled/multi-agent-win-rate'].mean(), ll[f'eval-sampled/multi-agent-return'].mean())

        data = {
            'env-id': jnp.arange(len(ll['eval-sampled/multi-agent-win-rate'])),
            'win-rates': ll['eval-sampled/multi-agent-win-rate'],
            'returns': ll['eval-sampled/multi-agent-return'],
        }
        seed = run.config["SEED"]
        PATH_TO_SAVE = f'sfl/data/eval/results/jaxnav-multi/eval_{TOTAL_ENVS}_envs_seed_{SEED}/{GROUP_TO_USE}/{seed}.csv'
        os.makedirs(os.path.dirname(PATH_TO_SAVE), exist_ok=True)

        pd.DataFrame(data).to_csv(PATH_TO_SAVE, index=False)
        print("Saved", PATH_TO_SAVE)

GROUPS = [
    # GROUP
    "MA-Learnability",
    "MA-PLR-PVL",
    "MA-PLR-MaxMC",
    "MA-ACCEL-MaxMC",
    "MA-DR",
]

for group in GROUPS:
    do_run_for_single_group(group)