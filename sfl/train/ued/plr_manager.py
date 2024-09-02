import jax
from jax import numpy as jnp
import chex
from functools import partial
from sfl.train.ued.ued_utils import *
from sfl.train.ued.map_buffer import MapBuffer
from sfl.env.maps import Map

class PLRManager:
    """ PLR Manager for multiple environments """

    def __init__(self,
                 num_agents: int,
                 num_train_envs: int,
                 map: Map, 
                 map_buffer_size=100,
                 initial_fill=1.0,
                 regret="pvl",
                 max_num_ep=100,
                 prioritisation="rank",
                 temperature=1, 
                 eps=0.05, 
                 rho=1, 
                 replay_prob=0.95, 
                 alpha=1,
                 staleness_coef=0.3):
        self.map = map
        self.map_buffer_manager = MapBuffer(
            buffer_size=map_buffer_size, 
            num_agents=num_agents,
            initial_fill=initial_fill,
            grid_size=map.map_size[0], # NOTE This assumes square maps
        )
        self.regret_type = regret
        self.max_num_ep = max_num_ep
        '''if regret == "pvl":
            self.regret_fn = positive_value_loss
        if regret == "true":
            self.regret_fn = true_regret
        else: raise NotImplementedError'''

        if prioritisation == "rank":
            self.priorisation_fn = self._rank_pri
        elif prioritisation == "proportional":
            self.priorisation_fn = self._proportional_pri
        else: raise NotImplementedError
        self.temperature = temperature
        self.eps = eps
        self.rho = rho  # percentage of buffer full before sampling replays
        self.replay_prob = replay_prob # replay prob
        self.alpha = alpha  # used in score update
        self.staleness = staleness_coef
        self.num_train_envs = num_train_envs

    def fill_initial_buffer(self, key) -> dict:
        buffer = self.map_buffer_manager.reset()
        key_initial = jax.random.split(key, self.map_buffer_manager.num_initial_gen)
        cases = self.map.sample_scenarios(key_initial)
        return self.map_buffer_manager.initial_fill(buffer, *cases)

    def generate_level(self, key):
        return self.map.sample_scenario(key)
    
    @partial(jax.jit, static_argnums=0)
    def sample_replay_decision(self, key, buffer: dict):
        """ True if sample level from replay buffer """
        
        prop_filled = (buffer["_p"] + 1)/buffer["maps"].shape[0] 
        x = jax.random.uniform(key) < self.replay_prob  # this should be min(prop_filled, self.replay_prob)
        return jnp.where(prop_filled>=self.rho, x, False)
    
    @partial(jax.jit, static_argnums=0) 
    def _compute_supports(self, c: int, map_buffer: dict):
        """ compute supports for each level in buffer
            c (int): current step
        """
        #print('scores', map_buffer["scores"])
        p_score = self.priorisation_fn(map_buffer["scores"]) ** 1/self.temperature
        p_score = p_score / jnp.sum(p_score)  # for all levels
        
        p_step = c - map_buffer["steps"]
        p_step = p_step / (jnp.sum(p_step) + 1e-8)  
        #print('p step', p_step)
        #jax.debug.print('p step {p}, p score {s}', p=p_step, s=p_score)
        p = (1-self.staleness) * p_score + self.staleness * p_step
        return p, (p_score, p_step)
            
    def _rank_pri(self, scores: chex.Array):
        """ rank priorisation function """
        temp = jnp.flip(scores.argsort()) # indicies sorted low to high based on score
        ranks = jnp.empty_like(temp)
        ranks = ranks.at[temp].set(jnp.arange(len(temp)) + 1)
        return 1/ranks
    
    def _proportional_pri(self, scores:chex.Array):
        return scores

    @partial(jax.jit, static_argnums=0)
    def sample_levels(self, key, step: int, map_buffer: dict):
        
        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0))
        def select_level(d, maps, poses, new_maps, new_poses):
            return jax.tree_map(lambda x, y: jax.lax.select(d, x, y), (maps, poses), (new_maps, new_poses))
        
        key_d, key_n, key_s = jax.random.split(key, 3)

        # Sample replay decisions
        prop_filled = (map_buffer["_p"] + 1)/map_buffer["maps"].shape[0] 
        x = jax.random.uniform(key_d, (self.num_train_envs,)) <= self.replay_prob  # if x is True, replay from the buffer
        d = jnp.where(prop_filled>=self.rho, x, False)
        d_ordered = jnp.flip(jnp.sort(d))

        # Generate new levels
        key_n = jax.random.split(key_n, self.num_train_envs)
        new_levels = jax.vmap(self.generate_level, in_axes=(0,))(key_n)
        new_maps, new_poses = new_levels

        # Sample levels from buffer
        #p = jnp.ones(map_buffer["counts"].shape) / map_buffer["counts"].shape[0] #self.sample_weights() NOTE this is not correct ???
        p, s_debug = self._compute_supports(step, map_buffer)
        
        level_idxs = jax.random.choice(key_s, jnp.arange(map_buffer["counts"].shape[0]), shape=(self.num_train_envs,), replace=False, p=p)
        
        maps, poses, _, _, _, _, _ = jax.vmap(self.map_buffer_manager.get_idx, in_axes=(None, 0))(map_buffer, level_idxs)     
        
        level_idxs = jax.vmap(jax.lax.select, in_axes=(0, 0, None))(d_ordered, level_idxs, -1)
        train_mask = jnp.full_like(d_ordered, True)
        return train_mask, select_level(d_ordered, maps, poses, new_maps, new_poses), level_idxs, (map_buffer["scores"], map_buffer["counts"], p, *s_debug)
    
    @partial(jax.jit, static_argnums=0)
    def update_with_rollout_m(self, levels: tuple, level_idxs: chex.Array, regret: chex.Array, step: int, metrics: chex.Array, max_returns: chex.Array, map_buffer: dict):
        maps, poses = levels
        p, _ = self._compute_supports(step, map_buffer)
        min_idx = jnp.argmin(p)

        for env_idx in range(self.num_train_envs):
            level_idx = level_idxs[env_idx]
            update_score = level_idx != -1  # level already in buffer
            replace = ((map_buffer["scores"][min_idx] < regret[env_idx]) & (level_idx == -1)) | update_score  # is a new level and thus can replace or replace with updated score

            # Replace
            get_idx = min_idx * (1 - update_score) + level_idx * update_score  # if update_score, replace in situ with update score, else replace at min_idx to add a new level to buffer
            old = self.map_buffer_manager.get_idx(map_buffer, get_idx)

            count = jax.lax.select(update_score, old[2] + 1, 1)  # either update score or set to 1 if new level

            new = (maps[env_idx], poses[env_idx], count, regret[env_idx], step, metrics[env_idx], max_returns[env_idx])
            case = jax.tree_map(lambda x, y: jax.lax.select(replace, x, y), new, old)

            map_buffer = self.map_buffer_manager.replace_idx(map_buffer, get_idx, *case)
            #u_p = jnp.unique(map_buffer["poses"], axis=0)
            #print('- in buffer update: num unique in buffer', u_p.shape, 'min idx', min_idx, 'env idx', env_idx, 'replace', replace)
            p, _ = self._compute_supports(step, map_buffer)
            min_idx = jnp.argmin(p)
            #jax.debug.print('min idx {i}, env idx {e}', i=min_idx, e=env_idx)
        return map_buffer
    
    def calculate_true_return(self, key, levels, env_params):
        return calculate_rrt_return(self.map, key, levels, env_params)
    
@partial(jax.jit, static_argnums=0)
def calculate_rrt_return(map_obj: Map, key: chex.PRNGKey, levels: chex.Array, env_params):
    
    def _rrt_reward(new_pos, pos, goal):
        goal_reached = jnp.linalg.norm(new_pos - goal) <= env_params.goal_radius
        rga = env_params.weight_g * (jnp.linalg.norm(pos - goal) - jnp.linalg.norm(new_pos - goal))
        return rga + goal_reached * env_params.goal_rew + env_params.dt_rew
            
    @partial(jax.vmap, in_axes=[0, 0])
    def _true_return(_g_pos, _tree):
        s_idx = jnp.argwhere(_tree[:, -1]==1, size=1, fill_value=0)
        s_idx = s_idx[0, 0]
        
        def __cond(val):
            c_idx, _ = val
            return c_idx != 0
            
        def __body(val):
            c_idx, rew = val  
            c_pos = _tree[c_idx, :2]
            p_idx = _tree[c_idx, 2].astype(int)
            p_idx = jax.lax.select(p_idx == -1, 0, p_idx)
            p_pos = _tree[p_idx, :2]
            rew += _rrt_reward(c_pos, p_pos, _g_pos)
            return (p_idx, rew)
                
        _, rew = jax.lax.while_loop(__cond, __body, (s_idx, 0.0))
        return rew
                
    map_data, poses = levels
    starts, goals = poses[:, :, 0, :2], poses[:, :, 1, :2]
    key = jax.random.split(key, map_data.shape[0])
    tree, _ = jax.vmap(map_obj.rrt_star, in_axes=(0, 0, 0, 0))(key, map_data, starts, goals)
    
    return _true_return(goals, tree)
        
            
### Scoring functions  
## Monte Carlo
@partial(jax.vmap, in_axes=[0, 0, 0, None])
@partial(jax.jit, static_argnames=['max_num_ep'])
def monte_carlo_return(value, done, max_score, max_num_ep):
    
    @partial(jax.vmap, in_axes=[0, None, None])
    def _ep_regret(s0, value, max_score):
        valid = s0 < value.shape[0]
        regret = max_score - value[s0]
        return regret, valid
    
    done_idx_list = jnp.argwhere(done == 1, size=max_num_ep, fill_value=done.shape[0]).flatten()
    done_idx_list = jnp.concatenate((jnp.array([-1]), done_idx_list))
    #s0_idx = done_idx_list + 1
    s0_idx = done_idx_list + jnp.where(done_idx_list < value.shape[0],  1, 0)
    regrets, valid = _ep_regret(s0_idx, value, max_score)

    return jnp.sum(regrets * valid) / (jnp.sum(valid) + 1e-8)

@partial(jax.vmap, in_axes=[0, 0, None])
@partial(jax.jit)
def max_return(max_r, level_idx, map_buffer):
    new_level = level_idx == -1
    legacy_r = jax.lax.select(new_level, max_r, map_buffer["max_reward"][level_idx])
    return jax.lax.select(max_r > legacy_r, max_r, legacy_r)
    

## Positive value loss
@partial(jax.vmap, in_axes=[0, 0, 0, None, None, None])
@partial(jax.jit, static_argnames=['max_num_ep'])
def positive_value_loss(reward, value, done, gamma, gae_lambda, max_num_ep):
    """ Calculate the positive value loss for each trajectory in a batch of trajectories.
    For each trajectory, will average score over """
    
    @partial(jax.vmap, in_axes=[0, None, None, None, None])
    def _calculate_td(
        t, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray, discount: float
    ):
        return reward[t] + discount * value[t + 1] * (1 - done[t]) - value[t]
    
    @partial(jax.vmap, in_axes=[0, 0, None, None, None, None])
    def _outer_ep_score(start_i, last_i, gae, idx_range, gamma, gae_lambda):
        """ vmap over the start and end pairs, which define each episode """
        ep_len = last_i - start_i + 1
        return jnp.sum(_ep_score(idx_range, start_i, last_i, gae, idx_range, gamma, gae_lambda)) * (1.0 / (ep_len + 1e-8))
    
    done_idx_list = jnp.argwhere(done == 1, size=max_num_ep, fill_value=done.shape[0]).flatten()
    done_idx_list = jnp.concatenate((jnp.array([-1]), done_idx_list))
    
    timestep_range = jnp.arange(reward.shape[0])  

    pairs = jnp.stack((done_idx_list[:-1]+1, done_idx_list[1:]), axis=1).astype(int)

    #jax.debug.print('pairs {pairs}', pairs=pairs)
    td = _calculate_td(timestep_range, value, reward, done, gamma)
    
    scores = _outer_ep_score(pairs[:, 0], pairs[:, 1], td, timestep_range, gamma, gae_lambda)
    
    score_mask = jnp.where(pairs[:, 1] < done_idx_list[-1], True, False)
    score_mask = score_mask.at[0].set(True)
    
    regret = jnp.sum(scores * score_mask) / jnp.sum(score_mask)    
    #jax.debug.print('regret {regret}, score mask {s}', regret=regret, s=score_mask)
    return regret
        
@partial(jax.vmap, in_axes=[0, None, None, None, None, None, None])
def _ep_score(t, start_i, last_i, td, idx_range, gamma, gae_lambda):
    """ loop for t """
    in_range = (t <= last_i) & (t >= start_i)
    ts = jnp.clip(jnp.sum(_timestep_score(idx_range, td, t, last_i, gamma, gae_lambda)), 0.0)  # relu
    return jax.lax.select(in_range, ts, 0.0)
    
@partial(jax.vmap, in_axes=[0, 0, None, None, None, None])
def _timestep_score(k, td, t, last_i, gamma, gae_lambda):
    in_range = (k<=last_i) & (k>=t)
    return jax.lax.select(in_range, td * ((gamma * gae_lambda) ** (k - t)), 0.0)
    
## True regret
@partial(jax.vmap, in_axes=[0, 0, 0, None])
@partial(jax.jit, static_argnames=['max_num_ep'])
def true_regret(reward, done, true_reward, max_num_ep):
    
    done_idx_list = jnp.argwhere(done == 1, size=max_num_ep, fill_value=done.shape[0]).flatten()
    done_idx_list = jnp.concatenate((jnp.array([-1]), done_idx_list))
    
    timestep_range = jnp.arange(reward.shape[0])  

    pairs = jnp.stack((done_idx_list[:-1]+1, done_idx_list[1:]), axis=1).astype(int)
    
    @partial(jax.vmap, in_axes=[0, 0, None, None, None])
    def _ep_true_regret(start_i, last_i, reward, timesteps, true_reward):
        ep_valid = last_i < done_idx_list[-1]
        mask = jnp.where((timesteps <= last_i) & (timesteps >= start_i), 1, 0)
        reward = jnp.sum(reward * mask)
        #jax.debug.print('true regret {tr}, reward {reward}, true reward {t}', tr=true_reward-reward, reward=reward, t=true_reward)
        ep_true_regret = true_reward - reward
        return ep_valid, ep_true_regret, reward 
    
    ep_valid, true_regrets, ep_rewards = _ep_true_regret(pairs[:, 0], pairs[:, 1], reward, timestep_range, true_reward)
    ep_valid = ep_valid.at[0].set(True)
    true_regrets *= ep_valid
    ep_rewards *= ep_valid
    
    #jax.debug.print('true regrets {true_regrets}, ep valid {ep_valid}', true_regrets=true_regrets, ep_valid=ep_valid)
    mean_regret = jnp.sum(true_regrets) / (jnp.sum(ep_valid) + 1e-8)
    var_regret = mean_regret ** 2 - jnp.sum(true_regrets ** 2) / (jnp.sum(ep_valid) + 1e-8)
    return mean_regret, var_regret, (true_regrets, ep_rewards, jnp.sum(ep_valid))
    
@partial(jax.vmap, in_axes=[0, 0, None])
@partial(jax.jit, static_argnames=['max_num_ep'])
def episodic_returns(reward, done, max_num_ep):
    
    done_idx_list = jnp.argwhere(done == 1, size=max_num_ep, fill_value=done.shape[0]).flatten()
    done_idx_list = jnp.concatenate((jnp.array([0]), done_idx_list))
    timestep_range = jnp.arange(reward.shape[0])  

    pairs = jnp.stack((done_idx_list[:-1], done_idx_list[1:]), axis=1).astype(int)
    
    @partial(jax.vmap, in_axes=[0, 0, None, None])
    def _ep_return(start_i, last_i, reward, timesteps):
        mask = jnp.where((timesteps < last_i) & (timesteps >= start_i), 1, 0)
        return jnp.sum(reward * mask)
    
    return _ep_return(pairs[:, 0], pairs[:, 1], reward, timestep_range)



## Metrics
@partial(jax.jit)
def compute_success_metrics(terminations: chex.Array):
    num_com = terminations[:, 0]
    succ_rates = terminations[:,1]/terminations[:,0]
    coll_rates = (terminations[:,2]+terminations[:,3])/terminations[:,0]
    return jnp.concatenate([num_com[:,None], succ_rates[:,None], coll_rates[:,None]], axis=1)

