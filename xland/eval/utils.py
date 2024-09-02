import typing
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file
import os
import jax 
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax import struct

from xminigrid.environment import Environment, EnvParams

def load_params(filename: typing.Union[str, os.PathLike]) -> typing.Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=',')

class RolloutStats(struct.PyTreeNode):
    success: jax.Array = jnp.asarray(0)
    reward: jax.Array = jnp.asarray(0.0)
    length: jax.Array = jnp.asarray(0)
    episodes: jax.Array = jnp.asarray(0)

def rollout(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                "observation": timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hstate,
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        stats = stats.replace(
            success=stats.success + jnp.where(timestep.discount == 0.0, 1, 0),
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep.last(),
        )
        carry = (rng, stats, timestep, action, timestep.reward, hstate)
        return carry

    timestep = env.reset(env_params, rng)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    init_carry = (rng, RolloutStats(), timestep, prev_action, prev_reward, init_hstate)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]