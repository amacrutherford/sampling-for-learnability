import os
import pandas as pd
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import wandb
import xminigrid
from flax.training.train_state import TrainState
from xminigrid.wrappers import GymAutoResetWrapper
import pickle as pkl
from flax.training.train_state import TrainState
from nn import ActorCriticRNN
from utils import load_params, rollout
from wandb.apis.public.runs import Run

env_id = "XLand-MiniGrid-R4-13x13"
benchmark_id = "high-3m"
NUM_RULESETS = 10000
eval_num_episodes = 10
SEED = 0
group = "high-13-sfl-uniform-v1"


benchmark_path = f"eval/data/rulesets/{benchmark_id}/{NUM_RULESETS}_rulesets.pkl"
with open(benchmark_path, "rb") as f:
    rulesets = pkl.load(f)

env, env_params = xminigrid.make(env_id)
env_params = env_params.replace(ruleset=rulesets)
env = GymAutoResetWrapper(env)

api = wandb.Api()

def rollout_group(group_name: str):
    
    def _rollout_run(i, run: Run):
        
        config = run.config
        rng = jax.random.key(SEED)

        network = ActorCriticRNN(
            num_actions=env.num_actions(env_params),
            action_emb_dim=config["action_emb_dim"],
            rnn_hidden_dim=config["rnn_hidden_dim"],
            rnn_num_layers=config["rnn_num_layers"],
            head_hidden_dim=config["head_hidden_dim"],
            img_obs=config["img_obs"],
        )
        
        model_artificat = api.artifact(f"alex-plus/{run.project}/{run.name}-checkpoint:v{i}")
        name = model_artificat.download()
        network_params = load_params(name + "/model.safetensors")
        print('Loaded model from:', name)
        network_params = jax.tree_map(lambda x: x[0], network_params)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optax.adamw(learning_rate=config["lr"]),  # NOTE not actually used
        )
        
        rng, eval_reset_rng = jax.random.split(rng)
        eval_reset_rng = jax.random.split(eval_reset_rng, NUM_RULESETS)
        eval_stats = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None))(
            eval_reset_rng,
            env,
            env_params,
            train_state,
            jnp.zeros((1, config["rnn_num_layers"], config["rnn_hidden_dim"])),
            eval_num_episodes,
        )
        
        PATH_TO_SAVE = f'eval/data/results/eval_{NUM_RULESETS}_envs_seed_{SEED}/{group_name}/{i}.csv'
        
        data = {
            'env-id': jnp.arange(len(eval_stats.reward)),
            'win-rates': eval_stats.success/eval_stats.episodes,
            'returns': eval_stats.reward,
        }
        os.makedirs(os.path.dirname(PATH_TO_SAVE), exist_ok=True)

        pd.DataFrame(data).to_csv(PATH_TO_SAVE, index=False)
        print("Saved", PATH_TO_SAVE)      
        
    runs = api.runs("alex-plus/xminigrid", filters={"group": group})
        
    for i, run in enumerate(runs):
        print('run id:', run.id)
        _rollout_run(i, run)              
    
rollout_group(group)
