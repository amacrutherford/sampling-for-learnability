# %%
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', '..'))

import pickle
import jax
import wandb 
import tqdm 

from jaxmarl.environments.jaxnav import JaxNav


# %% [markdown]
# ### Constants

# %%
SEED = 0

NUM_ENVS_TO_SAMPLE = 10000
EVAL_ITERS = 1
TOTAL_ENVS = NUM_ENVS_TO_SAMPLE * EVAL_ITERS

N_PARALLEL = 5
ROLLOUT_LEN = 1000

PATH = f'sfl/data/eval/jaxnav/eval_single_agent_{TOTAL_ENVS}e.pkl'
os.makedirs(os.path.dirname(PATH), exist_ok=True)
print('Total envs sampled: ', TOTAL_ENVS)

# %% [markdown]
# ### Run ID

# %%
rng = jax.random.PRNGKey(SEED)

api = wandb.Api()

runs = api.runs("alex-plus/multi_robot_ued",
                filters={"group": "rsim_plr_pvl_single_v4"})

print('runs', runs)
for run in runs:
    print(run.name, run.id)


first_config = runs[0].config
first_config["env"]["env_params"]['map_params']['valid_path_check'] = True

env = JaxNav(num_agents=first_config["env"]["num_agents"], 
                    **first_config["env"]["env_params"])

state_set = []
for _ in tqdm.tqdm(range(EVAL_ITERS)):
    rng, _rng = jax.random.split(rng)
    reset_rngs = jax.random.split(_rng, NUM_ENVS_TO_SAMPLE)
    _, states = jax.vmap(env.reset)(reset_rngs)
    state_set.append(states)


with open(PATH, 'wb') as f:
    pickle.dump(state_set, f)
    
print('saved to ', PATH)