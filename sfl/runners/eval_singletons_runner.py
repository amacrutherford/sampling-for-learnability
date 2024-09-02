"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from functools import partial
from typing import Tuple, Optional, List
from threading import Thread
import numpy as np
import jax
import jax.numpy as jnp

from jaxmarl.environments.jaxnav import make_jaxnav_singleton, make_jaxnav_singleton_collection, JaxNavSingleton
from jaxmarl.environments.jaxnav.jaxnav_viz import JaxNavVisualizer

from sfl.util.rolling_stats import RollingStats, LogEpisodicStats

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

class EvalSingletonsRunner:
    def __init__(
        self,
        collection_id: str,
        network,
        init_carry: callable,
        hidden_size,
        seperate_critic=False,
        greedy=False,
        env_kwargs=None,
        n_episodes=10,
        ):
        # TODO add render mode
       
        self.greedy = greedy
        
        self.network = network
        self.init_carry = init_carry
        self.hidden_size = hidden_size
        self.seperate_critic = seperate_critic

        self.n_parallel = n_episodes  # NOTE not sure what this is for

        self.envs, self.env_ids = make_jaxnav_singleton_collection(collection_id, **env_kwargs)
        self.n_envs = len(self.envs)
        # print('envs', self.envs)
        self.env_has_solved_rate = []
        
        self.action_dtype = self.envs[0].agent_action_space().dtype

        monitored_metrics = self.envs[0].get_monitored_metrics()
        self.rolling_stats = LogEpisodicStats(names=monitored_metrics)
        self._update_ep_stats = jax.vmap(
                self.rolling_stats.update_stats, in_axes=(0,0,0,None))

        self.test_len_pre = 'test_ep_length'
        self.test_return_pre = 'test_return'
        self.test_solved_rate_pre = 'test_solved_rate'
        self.test_solved_rate_pre_by_agent = 'test_solved_rate_by_agent'
        self.test_c_rate_pre = 'test_c_rate'
        self.test_to_rate_pre = 'test_to_rate'
        
        '''
        if render_mode:
            from minimax.envs.viz.grid_viz import GridVisualizer
            self.viz = GridVisualizer()
            self.viz.show()

            if render_mode == 'ipython':
                from IPython import display
                self.ipython_display = display'''

    '''def load_checkpoint_state(self, runner_state, state):
        runner_state = list(runner_state)
        runner_state[1] = runner_state[1].load_state_dict(state[1])

        return tuple(runner_state)'''
        
    def env_list(self) -> List[JaxNavSingleton]:
        return self.envs

    @partial(jax.jit, static_argnums=(0,2))
    def _get_transition(
        self,
        rng,
        env: JaxNavSingleton,
        params,
        state,
        obs,
        done,
        carry,
        start_state):
        
        obs_batch = batchify(obs, env.agents, self.n_parallel*env.num_agents)
        ac_in = (
            obs_batch[np.newaxis, :],
            done[np.newaxis, :],
        )
        if self.seperate_critic:
            carry, pi, _ = self.network.apply(params, carry, ac_in)
        else:
            carry, pi, _, _ = self.network.apply(params, carry, ac_in)
        if self.greedy:
            action = pi.mode()
        else:
            rng, rng_sample = jax.random.split(rng)
            action = pi.sample(seed=rng_sample)

        env_act = unbatchify(
            action, env.agents, self.n_parallel, env.num_agents
        )
        env_act = {k: v.squeeze() for k, v in env_act.items()}

        rng, vrngs = jax.random.split(rng)
        vrngs = jax.random.split(vrngs, self.n_parallel)

        (next_obs, 
         next_state, 
         reward, 
         dones, 
         info) = jax.vmap(env.step)(vrngs, state, env_act, start_state)
        ep_done = dones["__all__"]

        '''# Add transition to storage
        step = (obs, action, reward, done, log_pi, value)
        if carry is not None:
            step += (carry,)'''

        '''if self.render_mode:
            self.viz.render(
                benv.env.params, 
                jax.tree_util.tree_map(lambda x: x[0][0], state))
            if self.render_mode == 'ipython':
                self.ipython_display.display(self.viz.window.fig)
                self.ipython_display.clear_output(wait=True)
        '''
        dones_batch = batchify(dones, env.agents, self.n_parallel*env.num_agents).squeeze()
        reward_batch = batchify(reward, env.agents, self.n_parallel*env.num_agents).squeeze()
        return next_state, next_obs, reward_batch, carry, dones_batch, ep_done, info

    @partial(jax.jit, static_argnums=(0, 2))
    def _rollout(
        self, 
        rng, 
        env,
        params,
        state,
        obs,
        carry,
        zero_carry,
        start_state,
        ep_stats):

        def _scan_rollout(scan_carry, rng):
            (state, 
             obs, 
             done,
             carry,
             ep_stats) = scan_carry
            
            step = \
                self._get_transition(
                    rng,
                    env,
                    params, 
                    state, 
                    obs, 
                    done,
                    carry,
                    start_state)

            (next_state, 
             next_obs, 
             reward,
             next_carry, 
             done, 
             ep_done,
             info) = step
            print('info', info)
            ep_stats = self._update_ep_stats(ep_stats, ep_done, jax.tree_map(lambda x: x.sum(axis=-1), info), 1) 
        
            return (next_state, next_obs, done, next_carry, ep_stats), (obs, state, reward, done, ep_done, info)

        n_steps = env.max_steps
        rngs = jax.random.split(rng, n_steps)
        (state, 
         obs, 
         done,
         carry, 
         ep_stats), traj = jax.lax.scan(
            _scan_rollout,
            (state, obs, jnp.full((self.n_parallel*env.num_agents), 0, dtype=bool), carry, ep_stats),
            rngs,
            length=n_steps)

        return ep_stats, traj

    @partial(jax.jit, static_argnums=(0,))
    def run(self, rng, params, rew_lambda=None):
        """
        Rollout agents on each env. 

        For each env, run n_eval episodes in parallel, 
        where each is indexed to return in order.
        """        
        eval_stats = self.fake_run(rng, params)
        win_rates = jnp.empty((self.n_envs,))
        c_rates = jnp.empty((self.n_envs,))
        rng, *rollout_rngs = jax.random.split(rng, self.n_envs+1)
        for i, env in enumerate(self.envs):
            rng, *reset_rngs = jax.random.split(rng, self.n_parallel+1)
            obs, state = jax.vmap(env.reset)(jnp.array(reset_rngs))
            if rew_lambda is not None:
                state = state.replace(
                    rew_lambda=jnp.full(state.rew_lambda.shape, rew_lambda, dtype=jnp.float32)
                )
                obs, _ = jax.vmap(env.set_state, in_axes=(0,))(state)

            if self.network.is_recurrent:
                zero_carry = self.init_carry(self.n_parallel*env.num_agents, self.hidden_size)
            else:
                zero_carry = None

            # Reset episodic stats
            ep_stats = self.rolling_stats.reset_stats(
                batch_shape=(self.n_parallel,))  # TODO add num of eps?

            ep_stats, (obs, states, reward, dones, ep_dones, info) = self._rollout(
                rollout_rngs[i],
                env,
                jax.lax.stop_gradient(params), 
                state, 
                obs,
                zero_carry,
                zero_carry,
                state,
                ep_stats)

            info_by_actor = jax.tree_map(lambda x: x.swapaxes(1, 2).reshape((-1, env.num_agents*self.n_parallel)), info)
            o = self._calc_outcomes_by_agent(env.max_steps, dones, reward, info_by_actor)
            o_by_env = jax.tree_map(lambda x: x.reshape((env.num_agents, self.n_parallel)), o)
            o_mean = jax.tree_map(lambda x: (x*o_by_env["num_episodes"]).sum(axis=1)/o_by_env["num_episodes"].sum(axis=1), o_by_env)               
            win_rates = win_rates.at[i].set(o_mean["success_rate"].mean())
            eval_stats[f'eval/:{self.test_len_pre}:{self.env_ids[i]}'] = o_mean["ep_len"].mean()
            eval_stats[f'eval/:{self.test_return_pre}:{self.env_ids[i]}'] = o_mean["ep_return"]
            eval_stats[f'eval/:{self.test_solved_rate_pre}:{self.env_ids[i]}'] = o_mean["success_rate"].mean()
            eval_stats[f'eval/:{self.test_solved_rate_pre_by_agent}:{self.env_ids[i]}'] = o_mean["success_rate"]
            eval_stats[f'eval/:{self.test_c_rate_pre}:{self.env_ids[i]}'] = o_mean["collision_rate"]
            eval_stats[f'eval/:{self.test_to_rate_pre}:{self.env_ids[i]}'] = o_mean["timeout_rate"]
        
        eval_stats["eval/:overall_win_rate"] = win_rates.mean()
        jax.debug.print('win rate {a}, overall {b}', a=win_rates, b=win_rates.mean())
        return eval_stats
    
    def run_and_visualise(self, rng, params, run_name: str, rew_lambda=None):
        """
        Rollout agents on each env. 

        For each env, run n_eval episodes in parallel, 
        where each is indexed to return in order.
        """        
        eval_stats = self.fake_run(rng, params)
        win_rates = jnp.empty((self.n_envs,))
        threads = []
        rng, *rollout_rngs = jax.random.split(rng, self.n_envs+1)
        for i, env in enumerate(self.envs):
            print('** evaluting on env ', env.name)
            rng, *reset_rngs = jax.random.split(rng, self.n_parallel+1)
            obs, state = jax.vmap(env.reset)(jnp.array(reset_rngs))
            if rew_lambda is not None:
                state = state.replace(
                    rew_lambda=jnp.full(state.rew_lambda.shape, rew_lambda, dtype=jnp.float32)
                )
                
            if self.network.is_recurrent:
                zero_carry = self.init_carry(self.n_parallel*env.num_agents, self.hidden_size)
            else:
                zero_carry = None

            # Reset episodic stats
            ep_stats = self.rolling_stats.reset_stats(
                batch_shape=(self.n_parallel,))  # TODO add num of eps?

            ep_stats, (obs, states, reward, dones, ep_dones, info) = self._rollout(
                rollout_rngs[i],
                env,
                jax.lax.stop_gradient(params), 
                state, 
                obs,
                zero_carry,
                zero_carry,
                state,
                ep_stats)            
            
            info_by_actor = jax.tree_map(lambda x: x.swapaxes(1, 2).reshape((-1, env.num_agents*self.n_parallel)), info)
            o = self._calc_outcomes_by_agent(env.max_steps, dones, reward, info_by_actor)
            o_by_env = jax.tree_map(lambda x: x.reshape((env.num_agents, self.n_parallel)), o)
            o_mean = jax.tree_map(lambda x: (x*o_by_env["num_episodes"]).sum(axis=1)/o_by_env["num_episodes"].sum(axis=1), o_by_env)
            
            o_mean["success_rate"].mean()
            dones = dones.reshape((-1, self.n_parallel, env.num_agents), order='F')
            
            title = f"{env.name}\n"# + \
                # f'mean solved rate: {o_mean["success_rate"].mean():.2f}, by agent: {np.array2string(o_mean["success_rate"], precision=2)}\n' + \
                # f'c_rate: {np.array2string(o_mean["collision_rate"], precision=2)} to_rate: {np.array2string(o_mean["timeout_rate"], precision=2)}\n' + \
                # f'mean return {o_mean["ep_return"].mean():.2f}, return {np.array2string(o_mean["ep_return"], precision=2)}, ep len {o_mean["ep_len"].mean():.2f}'
            print('title:', title)
            first_ep_done = np.argwhere(ep_dones[:,0])[0][0]
            done_frames = np.argmax(dones[:first_ep_done+1, 0], axis=0)
            
            obs_list = [jax.tree_map(lambda x: x[i][0], obs) for i in range(first_ep_done+1)]
            state_list = [jax.tree_map(lambda x: x[i][0], states) for i in range(first_ep_done+1)]
            
            viz = JaxNavVisualizer(
                    env,
                    obs_list,
                    state_list,
                    done_frames=done_frames,
                    title_text=title,
                    plot_lidar=True,
                    plot_agent=True,
                    plot_path=True,
            )
            thread = Thread(target=viz.animate, args=(f'{run_name}-{env.name}.gif',))
            thread.start()
            threads.append(thread)
        
        eval_stats["eval/:overall_win_rate"] = win_rates.mean()
        for t in threads:
            t.join()
        
        return eval_stats

    def fake_run(self, rng, params):
        eval_stats = {"eval/:overall_win_rate": 0.}
        for i, env_name in enumerate(self.env_ids):
            eval_stats.update({
                f'eval/:{self.test_len_pre}:{env_name}':0.,
                f'eval/:{self.test_return_pre}:{env_name}':0.,
                f'eval/:{self.test_solved_rate_pre}:{env_name}':0.,
                f'eval/:{self.test_c_rate_pre}:{env_name}':0.,
            })
        return eval_stats
    
    @partial(jax.vmap, in_axes=(None, None, 1, 1, 1))
    @partial(jax.jit, static_argnums=(0,1,))
    def _calc_outcomes_by_agent(self, max_steps, dones, returns, info):
        idxs = jnp.arange(max_steps)
        
        @partial(jax.vmap, in_axes=(0, 0))
        def __ep_outcomes(start_idx, end_idx): 
            mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
            r = jnp.sum(returns * mask)
            success = jnp.sum(info["GoalR"] * mask)
            collision = jnp.sum((info["MapC"] + info["AgentC"]) * mask)
            timeo = jnp.sum(info["TimeO"] * mask)
            l = end_idx - start_idx
            return r, success, collision, timeo, l
        
        done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()
        mask_done = jnp.where(done_idxs == max_steps, 0, 1)
        ep_return, success, collision, timeo, length = __ep_outcomes(jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)        
                
        return {"ep_return": ep_return.mean(where=mask_done),
                "num_episodes": mask_done.sum(),
                "success_rate": success.mean(where=mask_done),
                "collision_rate": collision.mean(where=mask_done),
                "timeout_rate": timeo.mean(where=mask_done),
                "ep_len": length.mean(where=mask_done),
                }
