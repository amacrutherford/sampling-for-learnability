"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from operator import is_
from typing import Tuple, Optional, List
from threading import Thread
import jax.random
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav, State, listify_reward
from jaxmarl.environments.jaxnav.jaxnav_viz import JaxNavVisualizer

from jaxued.environments.maze.renderer import MazeRenderer

from sfl.util.rolling_stats import LogEpisodicStats



def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

class EvalSampledRunner:
    """ Eval over a given or randomly sampled set of maps. """
    def __init__(
        self,
        rng,
        env: JaxNav,
        network: nn.Module,
        init_carry: callable,
        hidden_size: int,
        seperate_critic=False,
        greedy=False,
        env_init_states: Optional[State]=None,
        n_envs: int=100,
        n_episodes: int=10,
        render_mode=None,
        is_minigrid=False,):

        self.is_minigrid = is_minigrid
        self.greedy = greedy
        self.network = network
        self.init_carry = init_carry
        self.hidden_size = hidden_size
        self.seperate_critic = seperate_critic

        self.env = env

        self.n_episodes = n_episodes  # NUM IN PARALLEL
        
        
        if env_init_states is not None:
            if self.is_minigrid:
                self.n_envs = env_init_states.env_state.agent_pos.shape[0]
            else:
                self.n_envs = env_init_states.pos.shape[0]
            if self.is_minigrid:
                env_init_obs = jax.vmap(env._env.get_obs)(env_init_states.env_state)    
            else:
                env_init_obs, _ = jax.vmap(env.set_state, in_axes=(0))(env_init_states)
        
        else:
            self.n_envs = n_envs
            env_init_obs, env_init_states = jax.vmap(self.env.reset, in_axes=0)(jax.random.split(rng, self.n_envs))
        
        self.n_parallel = n_episodes * self.n_envs        
        print(n_episodes, self.n_envs, self.n_parallel, 'pararasdas')
        if self.is_minigrid:
            self.num_actors = self.n_parallel
        else:
            self.num_actors = self.env.num_agents * self.n_parallel
        
        env_init_states = jax.tree_map(lambda x: jnp.repeat(x[:, None], self.n_episodes, axis=1), env_init_states)
        self.env_init_states = jax.tree_map(
            lambda x: x.reshape([self.n_parallel] + list(x.shape[2:])), env_init_states
        )
        
        env_init_obs = jax.tree_map(lambda x: jnp.repeat(x[:, None], self.n_episodes, axis=1), env_init_obs)
        self.env_init_obs = jax.tree_map(
            lambda x: x.reshape([self.n_parallel] + list(x.shape[2:])), env_init_obs
        )    
    
        self.env_has_solved_rate = []
        

        if self.is_minigrid:
            monitored_metrics = []
        else:
            monitored_metrics = self.env.get_monitored_metrics()
        self.rolling_stats = LogEpisodicStats(names=monitored_metrics)
        self._update_ep_stats = self.rolling_stats.update_stats #, in_axes=(0,0,0,None))

        self.test_return_pre = 'test_return'
        self.test_solved_rate_pre = 'test_solved_rate'
        
        self.render_mode = render_mode


    @partial(jax.jit, static_argnums=(0))
    def _get_transition(
        self,
        rng,
        params,
        state,
        init_state,
        obs,
        done,
        carry):

        if self.is_minigrid:
            obs_batch = obs
            ac_in = (
                jax.tree_map(lambda x: x[np.newaxis, np.newaxis], obs_batch),
                done[np.newaxis, np.newaxis],
            )
            carry = jax.tree_map(lambda x: x[np.newaxis], carry)
        else:
            obs_batch = batchify(obs, self.env.agents, self.env.num_agents)
            ac_in = (
                obs_batch[np.newaxis, :],
                done[np.newaxis, :],
            )
        if self.is_minigrid:
            print("shape here", done.shape, obs.image.shape)
            carry, pi, _ = self.network.apply(params, ac_in, carry)
            carry = jax.tree_map(lambda x: x.squeeze(), carry)
        else:
            if self.seperate_critic:
                carry, pi, _ = self.network.apply(params, carry, ac_in)
            else:
                carry, pi, _, _ = self.network.apply(params, carry, ac_in)
        if self.greedy:
            action = pi.mode()
        else:
            rng, rng_sample = jax.random.split(rng)
            action = pi.sample(seed=rng_sample)
            # jax.debug.print('action {}|{}', action, carry[0].sum())
        if self.is_minigrid:
            env_act = action.squeeze()
        else:
            env_act = unbatchify(
                action, self.env.agents, 1, self.env.num_agents
            )
            env_act = {k: v.squeeze() for k, v in env_act.items()}

        rng, vrngs = jax.random.split(rng)

        (next_obs, 
         next_state, 
         reward, 
         dones, 
         info) = self.env.step(vrngs, state, env_act, init_state if not self.is_minigrid else self.env.default_params)
        ep_done = dones["__all__"] if not self.is_minigrid else dones

        if self.is_minigrid:
            dones_batch = dones
        else:
            dones_batch = batchify(dones, self.env.agents, self.env.num_agents).squeeze(axis=1)
        if not self.is_minigrid:
            if self.env.do_sep_reward:
                reward_batch = listify_reward(reward, do_batchify=True)
            else:
                reward_batch = batchify(reward, self.env.agents, self.env.num_agents).squeeze(axis=1)
        else:
            reward_batch = reward
        return next_state, next_obs, carry, reward_batch, dones_batch, ep_done, info

    @partial(jax.jit, static_argnums=(0))
    def _rollout(
        self, 
        rng, 
        params,
        init_state,
        obs,
        carry,
        ep_stats):

        def _scan_rollout(scan_carry, rng):
            (state, 
             obs, 
             done,
             ep_done,
             carry,
             ep_stats) = scan_carry
            step = \
                self._get_transition(
                    rng,
                    params, 
                    state, 
                    init_state,
                    obs, 
                    done,
                    carry)

            (next_state, 
             next_obs, 
             next_carry,
             reward, 
             done, 
             ep_done,
             info) = step
            ep_stats = self._update_ep_stats(ep_stats, ep_done, jax.tree_map(lambda x: x.sum(), info), 1) 
            return (next_state, next_obs, done, ep_done, next_carry, ep_stats), (obs, state, reward, done, ep_done, info)

        if self.is_minigrid:
            n_steps = self.env.default_params.max_steps_in_episode
        else:        
            n_steps = self.env.max_steps
        rngs = jax.random.split(rng, n_steps)
        if self.is_minigrid:
            init = (init_state, obs, False, False, carry, ep_stats)
        else:
            init = (init_state, obs, jnp.full((self.env.num_agents,), False, dtype=bool), False, carry, ep_stats)
        (state, 
         obs, 
         done,
         ep_done,
         carry, 
         ep_stats), traj = jax.lax.scan(
            _scan_rollout,
            init,
            rngs,
            length=n_steps)
        
        print("MIke shape", done.shape, traj[2].shape, traj[2].sum())

        return ep_stats, traj

    @partial(jax.jit, static_argnums=(0,))
    def run(self, rng, params, rew_lambda: Optional[float] = None):
        """
        Rollout agents on each env. 

        For each env, run n_eval episodes in parallel, 
        where each is indexed to return in order.
        """        
        eval_stats = {}  # self.fake_run(rng, params)
        
        if self.network.is_recurrent:
            # rng, subrng = jax.random.split(rng)
            if self.is_minigrid:
                zero_carry = self.init_carry(self.n_parallel, self.hidden_size)
            else:
                zero_carry = self.init_carry(self.env.num_agents, self.hidden_size)
                zero_carry = jnp.repeat(zero_carry[None], self.n_parallel, axis=0)
        else:
            zero_carry = None
            
        ep_stats = self.rolling_stats.reset_stats(
            batch_shape=(self.n_parallel,)) 

        if rew_lambda is not None:
            init_states = self.env_init_states.replace(
                rew_lambda=jnp.full_like(self.env_init_states.rew_lambda, rew_lambda)
            )
            if self.is_minigrid:
                init_obs = jax.vmap(self.env._env.get_obs)(init_states.env_state)
            else:
                init_obs, _ = jax.vmap(self.env.set_state, in_axes=(0,))(init_states)
        else:
            init_states = self.env_init_states
            init_obs = self.env_init_obs
        
        ep_stats, (obs, states, reward, dones, ep_dones, info) = jax.vmap(
            self._rollout, in_axes=(0, None, 0, 0, 0, 0)
        )(
            jax.random.split(rng, self.n_parallel), 
            params,
            init_states,
            init_obs,
            zero_carry,
            ep_stats
        )
        info_by_actor = jax.tree_map(lambda x: jnp.moveaxis(x, 0, 1).reshape((-1, self.num_actors)), info)
        reward_by_actor = jnp.moveaxis(reward, 0, 1).reshape((-1, self.num_actors))
        dones_by_actor = jnp.moveaxis(dones, 0, 1).reshape((-1, self.num_actors))
        
        o = self._calc_outcomes_by_agent(self.env.max_steps if not self.is_minigrid else self.env.default_params.max_steps_in_episode, dones_by_actor, reward_by_actor, info_by_actor)

        num_agents = self.env.num_agents if not self.is_minigrid else 1        
        o_grouped_by_episodes = jax.tree_map(lambda x: x.reshape((self.n_envs, self.n_episodes, num_agents)), o)
        o_by_env_and_actor = jax.tree_map(lambda x: (x*o_grouped_by_episodes["num_episodes"]).sum(axis=1)/o_grouped_by_episodes["num_episodes"].sum(axis=1), o_grouped_by_episodes)
        o_by_actor = jax.tree_map(lambda x: x.reshape(-1), o_by_env_and_actor)
        o_by_env = jax.tree_map(lambda x: x.mean(axis=-1), o_by_env_and_actor)
        
        eval_stats[f'eval-sampled/win_rates'] = o_by_actor["success_rate"]
        eval_stats[f'eval-sampled/c_rates'] = o_by_actor["collision_rate"]
        eval_stats[f'eval-sampled/to_rates'] = o_by_actor["timeout_rate"]
        eval_stats[f'eval-sampled/returns'] = o_by_actor["ep_return"]
        eval_stats[f'eval-sampled/ep_len'] = o_by_actor["ep_len"][0]
        
        eval_stats["eval-sampled/overall_win_rate"] = o_by_env["success_rate"].mean()
        eval_stats["eval-sampled/overall_mean_return"] = o_by_env["ep_return"].mean()
        
        
        eval_stats["eval-sampled/multi-agent-win-rate"] = o_by_env["success_rate"]
        eval_stats["eval-sampled/multi-agent-return"]   = o_by_env["ep_return"]
        
        jax.debug.print('win rate {a}, returns {b}', a=o_by_env["success_rate"].mean(), b=o_by_env["ep_return"].mean())
        return eval_stats
    
    def run_and_visualise(self, rng, params, run_name: str, gif_prefix=None, plot_lidar=False, viz_only_failure=False):
        eval_stats = self.fake_run(rng, params)
        
        if self.network.is_recurrent:
            # rng, subrng = jax.random.split(rng)
            zero_carry = self.init_carry(self.env.num_agents, self.hidden_size)
            zero_carry = jnp.repeat(zero_carry[None], self.n_parallel, axis=0)
        else:
            zero_carry = None
            
        ep_stats = self.rolling_stats.reset_stats(
            batch_shape=(self.n_parallel,)) 

        ep_stats, (obs, states, reward, dones, ep_dones, info) = jax.vmap(
            self._rollout, in_axes=(0, None, 0, 0, 0, 0)
        )(
            jax.random.split(rng, self.n_parallel), 
            params,
            self.env_init_states,
            self.env_init_obs,
            zero_carry,
            ep_stats
        )
        info_by_actor = jax.tree_map(lambda x: jnp.moveaxis(x, 0, 1).reshape((-1, self.num_actors)), info)
        reward_by_actor = jnp.moveaxis(reward, 0, 1).reshape((-1, self.num_actors))
        dones_by_actor = jnp.moveaxis(dones, 0, 1).reshape((-1, self.num_actors))
        
        
        o = self._calc_outcomes_by_agent(self.env.max_steps, dones_by_actor, reward_by_actor, info_by_actor)
        o_grouped_by_episodes = jax.tree_map(lambda x: x.reshape((self.n_envs, self.n_episodes, self.env.num_agents)), o)
        o_by_env_and_actor = jax.tree_map(lambda x: (x*o_grouped_by_episodes["num_episodes"]).sum(axis=1)/o_grouped_by_episodes["num_episodes"].sum(axis=1), o_grouped_by_episodes)
        o_by_actor = jax.tree_map(lambda x: x.reshape(-1), o_by_env_and_actor)
        o_by_env = jax.tree_map(lambda x: x.mean(axis=-1), o_by_env_and_actor)
        
        # o_by_env = jax.tree_map(lambda x: x.reshape((self.n_envs, self.n_episodes, self.env.num_agents)), o)
        # o_mean = jax.tree_map(lambda x: (x*o_by_env["num_episodes"]).sum(axis=1)/o_by_env["num_episodes"].sum(axis=1), o_by_env)
        eval_stats[f'eval-sampled/win_rates'] = o_by_actor["success_rate"]
        eval_stats[f'eval-sampled/c_rates'] = o_by_actor["collision_rate"]
        eval_stats[f'eval-sampled/to_rates'] = o_by_actor["timeout_rate"]
        eval_stats[f'eval-sampled/returns'] = o_by_actor["ep_return"]
        eval_stats[f'eval-sampled/ep_len'] = o_by_actor["ep_len"][0]
        
        eval_stats["eval-sampled/overall_win_rate_by_env"] = o_by_env["success_rate"].mean()
        eval_stats["eval-sampled/overall_mean_return_by_env"] = o_by_env["ep_return"].mean()
        print('success by env', o_by_env["success_rate"])
        print(f'overall success rate {eval_stats["eval-sampled/overall_win_rate_by_env"]}, mean return: {eval_stats["eval-sampled/overall_mean_return_by_env"]}')
        
        dones_by_env = jnp.moveaxis(dones, 0, 2).reshape((-1, self.env.num_agents, self.n_envs, self.n_episodes))
        # dones = dones.reshape((-1, self.n_parallel, self.env.num_agents))
        # ep_dones_by_env = ep_dones.swapaxes(0, 1).reshape((-1, self.n_envs, self.n_episodes))
        
        threads = []
        print('obs shape', obs['agent_0'].shape)
        for i in range(self.n_envs):
            if o_by_env["success_rate"][i] == 1.0 and viz_only_failure:
                continue
            idx = i * self.n_episodes
            first_ep_done = np.argwhere(ep_dones[idx,:]).flatten()[0]
            done_frames = np.argmax(dones[idx, :first_ep_done+1], axis=0)
            
            obs_list = [jax.tree_map(lambda x: x[idx][j], obs) for j in range(first_ep_done+1)]
            state_list = [jax.tree_map(lambda x: x[idx][j], states) for j in range(first_ep_done+1)]
            
            title = f'{run_name}-{i}\n{o_by_env_and_actor["success_rate"][i]}' 
            
            viz = JaxNavVisualizer(
                    self.env,
                    obs_list,
                    state_list,
                    done_frames=done_frames,
                    title_text=title,
                    plot_lidar=plot_lidar,
                    plot_agent=True,
                    plot_path=True,
            )
            if gif_prefix is not None:
                gif_name = f'{gif_prefix}-{run_name}-{i}.gif'
            else:
                gif_name = f'{run_name}-{i}.gif'
            
            thread = Thread(target=viz.animate, args=(gif_name,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()        
    
        return eval_stats

    def fake_run(self, rng, params):
        eval_stats = {"eval/:overall_win_rate": 0.0}
        eval_stats[f'eval/win_rates'] = jnp.zeros((self.n_envs,))
        eval_stats[f'eval/returns'] = jnp.zeros((self.n_envs,))
        
        return eval_stats

    @partial(jax.vmap, in_axes=(None, None, 1, 1, 1))
    @partial(jax.jit, static_argnums=(0, 1))
    def _calc_outcomes_by_agent(self, max_steps: int, dones, returns, info):
        idxs = jnp.arange(max_steps)
        
        @partial(jax.vmap, in_axes=(0, 0))
        def __ep_outcomes(start_idx, end_idx): 
            mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
            r = jnp.sum(returns * mask)
            if self.is_minigrid:
                success   = jnp.sum((returns > 0) * mask)
                collision = 0.0 #jnp.sum((info["MapC"] + info["AgentC"]) * mask)
                timeo     = 0.0 # jnp.sum(info["TimeO"] * mask)
            else:
                success = jnp.sum(info["GoalR"] * mask)
                collision = jnp.sum((info["MapC"] + info["AgentC"]) * mask)
                timeo = jnp.sum(info["TimeO"] * mask)
            l = end_idx - start_idx
            return r, success, collision, timeo, l
        
        done_idxs = jnp.argwhere(dones, size=100, fill_value=max_steps).squeeze()
        mask_done = jnp.where(done_idxs == max_steps, 0, 1)
        ep_return, success, collision, timeo, length = __ep_outcomes(jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)        
                
        return {"ep_return": ep_return.mean(where=mask_done),
                "num_episodes": mask_done.sum(),
                "success_rate": success.mean(where=mask_done),
                "collision_rate": collision.mean(where=mask_done),
                "timeout_rate": timeo.mean(where=mask_done),
                "ep_len": length.mean(where=mask_done),
                }
        
class EvalSampledRunnerMinigrid(EvalSampledRunner):   
    
    def run_and_visualise(self, rng, params, run_name: str, gif_prefix=None, plot_lidar=False, viz_only_failure=False):
        eval_stats = self.fake_run(rng, params)
        
        if self.network.is_recurrent:
            if self.is_minigrid:
                zero_carry = self.init_carry(self.n_parallel, self.hidden_size)
            else:
                zero_carry = self.init_carry(self.env.num_agents, self.hidden_size)
                zero_carry = jnp.repeat(zero_carry[None], self.n_parallel, axis=0)
        else:
            zero_carry = None
            
        ep_stats = self.rolling_stats.reset_stats(
            batch_shape=(self.n_parallel,)) 

        ep_stats, (obs, states, reward, dones, ep_dones, info) = jax.vmap(
            self._rollout, in_axes=(0, None, 0, 0, 0, 0)
        )(
            jax.random.split(rng, self.n_parallel), 
            params,
            self.env_init_states,
            self.env_init_obs,
            zero_carry,
            ep_stats
        )
        info_by_actor = jax.tree_map(lambda x: jnp.moveaxis(x, 0, 1).reshape((-1, self.num_actors)), info)
        reward_by_actor = jnp.moveaxis(reward, 0, 1).reshape((-1, self.num_actors))
        dones_by_actor = jnp.moveaxis(dones, 0, 1).reshape((-1, self.num_actors))
        
        num_agents = self.env.num_agents if not self.is_minigrid else 1       
        o = self._calc_outcomes_by_agent(self.env.max_steps if not self.is_minigrid else self.env.default_params.max_steps_in_episode, dones_by_actor, reward_by_actor, info_by_actor)
        o_grouped_by_episodes = jax.tree_map(lambda x: x.reshape((self.n_envs, self.n_episodes, num_agents)), o)
        o_by_env_and_actor = jax.tree_map(lambda x: (x*o_grouped_by_episodes["num_episodes"]).sum(axis=1)/o_grouped_by_episodes["num_episodes"].sum(axis=1), o_grouped_by_episodes)
        o_by_actor = jax.tree_map(lambda x: x.reshape(-1), o_by_env_and_actor)
        o_by_env = jax.tree_map(lambda x: x.mean(axis=-1), o_by_env_and_actor)
        
        # o_by_env = jax.tree_map(lambda x: x.reshape((self.n_envs, self.n_episodes, self.env.num_agents)), o)
        # o_mean = jax.tree_map(lambda x: (x*o_by_env["num_episodes"]).sum(axis=1)/o_by_env["num_episodes"].sum(axis=1), o_by_env)
        eval_stats[f'eval-sampled/win_rates'] = o_by_actor["success_rate"]
        eval_stats[f'eval-sampled/c_rates'] = o_by_actor["collision_rate"]
        eval_stats[f'eval-sampled/to_rates'] = o_by_actor["timeout_rate"]
        eval_stats[f'eval-sampled/returns'] = o_by_actor["ep_return"]
        eval_stats[f'eval-sampled/ep_len'] = o_by_actor["ep_len"][0]
        
        eval_stats["eval-sampled/overall_win_rate_by_env"] = o_by_env["success_rate"].mean()
        eval_stats["eval-sampled/overall_mean_return_by_env"] = o_by_env["ep_return"].mean()
        print('success by env', o_by_env["success_rate"])
        print(f'overall success rate {eval_stats["eval-sampled/overall_win_rate_by_env"]}, mean return: {eval_stats["eval-sampled/overall_mean_return_by_env"]}')
        
        # dones = dones.reshape((-1, self.n_parallel, self.env.num_agents))
        # ep_dones_by_env = ep_dones.swapaxes(0, 1).reshape((-1, self.n_envs, self.n_episodes))
        
        threads = []
        # print('obs shape', obs['agent_0'].shape)
        for i in range(self.n_envs):
            if o_by_env["success_rate"][i] == 1.0 and viz_only_failure:
                continue
            idx = i * self.n_episodes
            first_ep_done = np.argwhere(ep_dones[idx,:]).flatten()[0]
            # done_frames = np.argmax(dones[idx, :first_ep_done+1], axis=0)
            
            # obs_list = [jax.tree_map(lambda x: x[idx][j], obs) for j in range(first_ep_done+1)]
            state_list = [jax.tree_map(lambda x: x[idx][j], states.env_state) for j in range(first_ep_done+1)]
            
            title = f'{run_name}-{i}\n{o_by_env_and_actor["success_rate"][i]}' 
            
            fig, ax = plt.subplots()
            
            viz = MazeRenderer(
                self.env._env
            )
            # im = viz.render_state(state_list[0], self.env.default_params)
            # ax.imshow(im)
            # plt.savefig(f'{run_name}-{i}.png')
            
            def render_step(state):
                # clear the previous image
                ax.clear()
                im = viz.render_state(state, self.env.default_params)
                ax.imshow(im)
            
            print('num states', len(state_list))
            ani = FuncAnimation(
                fig,
                render_step,
                frames=state_list,
                blit=False,
                # interval=self.interval,
            )
            if gif_prefix is not None:
                gif_name = f'{gif_prefix}-{run_name}-{i}.gif'
            else:
                gif_name = f'{run_name}-{i}.gif'
            
            thread = Thread(target=ani.save, args=(gif_name,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()        
        print('all viz done')
        return eval_stats