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
import imageio

from xminigrid.rendering.text_render import print_ruleset

META_EPISODES = 10

env_id = "XLand-MiniGrid-R4-13x13"
benchmark_id = "high-3m"
NUM_RULESETS = 10000
eval_num_episodes = 10
SEED = 0
group = "high-13-sfl-uniform-v1"

env, env_params = xminigrid.make(env_id)
env = GymAutoResetWrapper(env)

api = wandb.Api()

runs = api.runs("alex-plus/xminigrid", filters={"group": group})
run = runs[0]

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

model_artificat = api.artifact(f"alex-plus/{run.project}/{run.name}-checkpoint:latest")
name = model_artificat.download()
network_params = load_params(name + "/model.safetensors")

network_params = jax.tree_map(lambda x: x[0], network_params)
train_state = TrainState.create(
    apply_fn=network.apply,
    params=network_params,
    tx=optax.adamw(learning_rate=config["lr"]),  # NOTE not actually used
)

ruleset = xminigrid.load_benchmark(benchmark_id).get_ruleset(ruleset_id=0)
env_params = env_params.replace(ruleset=ruleset)

# you can use train_state from the final state also
# we just demo here how to do it if you loaded params from the checkpoint
params = train_state.params


# jitting all functions
apply_fn, reset_fn, step_fn = jax.jit(network.apply), jax.jit(env.reset), jax.jit(env.step)

# for logging
total_reward, num_episodes = 0, 0
rendered_imgs = []

rng = jax.random.key(1)
rng, _rng = jax.random.split(rng)

# initial inputs
hidden = network.initialize_carry(1)
prev_reward = jnp.asarray(0)
prev_action = jnp.asarray(0)

timestep = reset_fn(env_params, _rng)
rendered_imgs.append(env.render(env_params, timestep))

while num_episodes < META_EPISODES:
    rng, _rng = jax.random.split(rng)
    dist, _, hidden = apply_fn(
        params,
        {
            "observation": timestep.observation[None, None, ...],
            "prev_action": prev_action[None, None, ...],
            "prev_reward": prev_reward[None, None, ...],
        },
        hidden,
    )
    action = dist.sample(seed=_rng).squeeze()

    timestep = step_fn(env_params, timestep, action)
    prev_action = action
    prev_reward = timestep.reward

    total_reward += timestep.reward.item()
    num_episodes += int(timestep.last().item())
    rendered_imgs.append(env.render(env_params, timestep))

print("Reward:", total_reward)
print("Ruleset:")
print_ruleset(ruleset)
imageio.mimsave("eval_rollout.mp4", rendered_imgs, fps=16, format="mp4")