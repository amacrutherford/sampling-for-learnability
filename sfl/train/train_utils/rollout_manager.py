
import wandb
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import gymnax
import flax
import tqdm
from flax.training.train_state import TrainState
from typing import Any, Callable, Tuple
from collections import defaultdict
import optax
import chex

#from gymnax_blines.utils.ppo import policy, update, flatten_dims, update_epoch
from sfl.train.train_utils.common import save_checkpoint
from sfl.env.base_env import MultiAgentEnvironment

class RolloutManagerMultiAgent(object):
    """ Assuming each agent controlled by one policy, executed decentrally """
    def __init__(self, model, env: MultiAgentEnvironment, env_params, num_eval_envs):
        # Setup functionalities for vectorized batch rollout
        self.env = env 
        self.env_params = env_params
        self.num_agents = self.env.num_agents
        self.num_eval_envs = num_eval_envs
        self.num_actors = self.num_agents * self.num_eval_envs
        
        #self.env_params = self.env_params.replace(**env_params)   NOTE in future this is a better way, but not needed for now
        self.observation_space = self.env.agent_observation_space(self.env_params)  # by agent as PPO treated as independent
        self.action_size = self.env.agent_action_space(self.env_params).shape  # as above
        print('-- action size', self.action_size)
        self.apply_fn = model.apply

    #@partial(jax.vmap, in_axes=(None, 0, None)) 
    @partial(jax.jit, static_argnums=0)
    def select_actions(
        self,
        train_state: TrainState,
        obs: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        value, pi = train_state.apply_fn(train_state.params, obs)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value, rng
    
    def select_eval_actions(
        self,
        train_state: TrainState,
        obs: jnp.ndarray,
    ):
        _, pi = train_state.apply_fn(train_state.params, obs)
        return pi.mean()
        

    @partial(jax.jit, static_argnums=[0])
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset, in_axes=(0, None))(
            keys, self.env_params
        )

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, actions):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, actions, self.env_params
        )

    @partial(jax.jit, static_argnames=['self'])
    def batch_evaluate(self, rng_input, train_state):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.batch_reset(jax.random.split(rng_reset, self.num_eval_envs))
        obs_stack = jnp.repeat(obs[:,:,:self.env_params.sparams.num_beams], 3, axis=-1)
        
        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, obs_stack, state, train_state, rng, cum_reward, valid_mask, terms = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)

            mobs = jnp.concatenate([obs_stack, obs[:,:,-4:]], axis=-1).reshape((valid_mask.shape[0], -1))
            action_flat = self.select_eval_actions(train_state, mobs)
            action = jnp.reshape(action_flat, (obs.shape[0], obs.shape[1], -1))

            next_o, next_s, reward, dones, infos = self.batch_step(
                jax.random.split(rng_step, self.num_eval_envs),
                state,
                action,
            )  ## is there a way to stop the auto reset as it hurts us during the test phase.. 
            num_lidar_beams = int(obs.shape[-1]-4) 
            #jax.debug.print('rewards {x}', x=reward)
            next_obs_stack = jnp.concatenate([obs_stack[:,:,num_lidar_beams:], next_o[:,:,:num_lidar_beams]], axis=2)
            new_cum_reward = cum_reward + reward.reshape((valid_mask.shape[0])) * valid_mask
            new_valid_mask = valid_mask * (1 - next_s.done.reshape((valid_mask.shape[0])))
            new_terms = terms + infos["terminations"]
            #jax.debug.print('rew {x}', x=new_cum_reward)
            carry, y = [
                next_o,
                next_obs_stack,
                next_s,
                train_state,
                rng,
                new_cum_reward,
                new_valid_mask,
                new_terms
            ], [new_valid_mask]
             
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                obs_stack,
                state,
                train_state,
                rng_episode,
                jnp.array(self.num_actors * [0.0]),  # cumulative reward
                jnp.array(self.num_actors * [1.0]),  # valid mask, zero out rewards from done agents
                jnp.zeros((self.num_eval_envs, 5)),  # termination statistics tracker
            ],
            (),
            self.env_params.max_steps,
        )
        cum_return = carry_out[-3].squeeze()
        terms = jnp.sum(carry_out[-1], axis=0) 
        #jax.debug.print('final terms {x}', x=terms)
        #terms = terms + jnp.array([num_actors, 0.0, 0.0, 0.0, num_actors])  # assume the last episode does not complete, will slightly skeq results but not a major issue for now
        terms.at[1:].set(terms[1:]/terms[0])
        
        return jnp.mean(cum_return), terms