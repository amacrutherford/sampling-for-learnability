import jax
from jax import numpy as jnp
from functools import partial
from .ued_utils import *

# TODO add steps functionality properly

class MapBuffer:
    
    def __init__(
        self,
        buffer_size,
        num_agents,
        initial_fill=1.0,
        grid_size=None,
        width=None,
        height=None,        
    ) -> None:
        
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        
        self.buffer_size = buffer_size
        self.num_initial_gen = jnp.floor(buffer_size*initial_fill).astype(int)
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.metric_count = 3
        
    @partial(jax.jit, static_argnums=0)
    def reset(self):
        return {
            "maps": jnp.empty(
                (self.buffer_size, self.height, self.width),
                dtype=jnp.int32,
            ),
            "poses": jnp.empty(
                (self.buffer_size, self.num_agents, 2, 3),
                dtype=jnp.float32,
            ),
            "counts": jnp.empty(
                (self.buffer_size,),
                dtype=jnp.int32,
            ),
            "scores": jnp.empty(
                (self.buffer_size,),
                dtype=jnp.float32,
            ),
            "steps": jnp.empty(
                (self.buffer_size,),
                dtype=jnp.int32,
            ),
            "metrics": jnp.empty(
                (self.buffer_size, self.metric_count),
                dtype=jnp.float32,
            ),
            "max_reward": jnp.empty(
                (self.buffer_size,),
                dtype=jnp.float32,
            ),
            "_p": 0,  # NOTE do we need this, not sure
        }
        
    @partial(jax.jit, static_argnums=0)
    def initial_fill(self, buffer, maps, poses, max_reward=None):
        if max_reward is None:
            max_reward = jnp.zeros((self.num_initial_gen), dtype=jnp.float32)        
        return {
            "maps": buffer["maps"].at[:self.num_initial_gen].set(maps),
            "poses": buffer["poses"].at[:self.num_initial_gen].set(poses),
            #"starts": buffer["starts"].at[:self.initial_gen].set(starts),
            "counts": buffer["counts"].at[:self.num_initial_gen].set(jnp.zeros((self.num_initial_gen), dtype=jnp.int32)),   
            "scores": buffer["scores"].at[:self.num_initial_gen].set(jnp.zeros((self.num_initial_gen), dtype=jnp.float32)),   
            "steps": buffer["counts"].at[:self.num_initial_gen].set(jnp.zeros((self.num_initial_gen), dtype=jnp.int32)),
            "metrics": buffer["metrics"].at[:self.num_initial_gen].set(jnp.zeros((self.num_initial_gen, self.metric_count), dtype=jnp.float32)),
            "max_reward": buffer["max_reward"].at[:self.num_initial_gen].set(max_reward),
            "_p": (self.num_initial_gen-1),         
        }
        
    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, map, pose, count, score, step, metric, max_reward=None):
        #assert buffer["_p"] < self.buffer_size  TODO need to change to use checkify
        #start = poses[:, 0, :]
        #goal = poses[:, 1, :]
        if max_reward is None: max_reward=0.0
        return {
            "maps": buffer["maps"].at[buffer["_p"]].set(map),
            "poses": buffer["poses"].at[buffer["_p"]].set(pose),
            #"starts": buffer["starts"].at[buffer["_p"]].set(start),
            "counts": buffer["counts"].at[buffer["_p"]].set(count),
            "scores": buffer["scores"].at[buffer["_p"]].set(score),
            "steps": buffer["steps"].at[buffer["_p"]].set(step),
            "metrics": buffer["metrics"].at[buffer["_p"]].set(metric),
            "max_reward": buffer["max_reward"].at[buffer["_p"]].set(max_reward),
            "_p": buffer["_p"] + 1 #% self.buffer_size,
        }
        
    @partial(jax.jit, static_argnums=0)
    def get_idx(self, buffer, idx):
        return (
            buffer["maps"][idx],
            buffer["poses"][idx],
            buffer["counts"][idx],
            buffer["scores"][idx],
            buffer["steps"][idx],
            buffer["metrics"][idx],
            buffer["max_reward"][idx],
        )
        
    @partial(jax.vmap, in_axes=(None, None, 0))
    @partial(jax.jit, static_argnums=0)
    def _get_idx_case(self, buffer, idx):
        """ Return a set of cases from the buffer """
        return (
            buffer["maps"][idx],
            buffer["poses"][idx],
        )
        
    @partial(jax.jit, static_argnums=0)
    def replace_idx(self, buffer, idx, map, pose, count, score, step, metric, max_reward=None):
        #buffer["scores"] = buffer["scores"].at[idx].set(score)
        #start = poses[:, 0, :]
        #goal = poses[:, 1, :]
        if max_reward is None: max_reward=0.0
        
        #jax.debug.print('replacing idx score {s} buffer {b}', s=score, b=buffer["scores"])
        #jax.debug.print('replacing idx {i}', i=idx)
        return {
            "maps": buffer["maps"].at[idx].set(map),
            "poses": buffer["poses"].at[idx].set(pose),
            "counts": buffer["counts"].at[idx].set(count),  
            "scores": buffer["scores"].at[idx].set(score), 
            "steps": buffer["steps"].at[idx].set(step), 
            "metrics": buffer["metrics"].at[idx].set(metric),
            "max_reward": buffer["max_reward"].at[idx].set(max_reward),
            "_p": buffer["_p"],         
        }
        
    @partial(jax.jit, static_argnums=[0, 3])
    def get_random_sample(self, key, buffer, num_samples):
        
        idxs = jax.random.randint(key, (num_samples,), 0, buffer["_p"])
        return self._get_idx(buffer, idxs)
        #return 
        
    @partial(jax.jit, static_argnums=[0, 2])
    def get_score_extremes(self, buffer, num_samples=5):
        """ Get the highest and lowest scoring cases in the buffer """
        # Sort array
        idxs = jnp.argsort(buffer["scores"])
        mins_idx = idxs[:num_samples]
        maxs_idx = idxs[-num_samples:]
        
        # Gather cases and scores
        mins = self._get_idx_case(buffer, mins_idx)
        maxs = self._get_idx_case(buffer, maxs_idx)
        cases = (
            jnp.concatenate((mins[0], maxs[0]), axis=0),
            jnp.concatenate((mins[1], maxs[1]), axis=0),
        )
        scores = jnp.concatenate((buffer["scores"][mins_idx], buffer["scores"][maxs_idx]), axis=0)
        counts = jnp.concatenate((buffer["counts"][mins_idx], buffer["counts"][maxs_idx]), axis=0)
        metrics = jnp.concatenate((buffer["metrics"][mins_idx], buffer["metrics"][maxs_idx]), axis=0)
        
        #jax.debug.print('\nmin i {mini}, maxi {maxi}, cases {c}, scores {s}', mini=mins_idx, maxi=maxs_idx, c=cases, s=scores)
        return cases, scores, counts, metrics
    
    ''' def update_map_buffer(self, buffer, level, score, timestep):
        if buffer["_p"] < self.buffer_size:
            buffer = self.append(buffer, *level)  # NOTE do we need to add score and timestep?
        else:
            minimal_support = None
            
            #if score(minimal_support) '''
            
    @property
    def get_map_size(self):
        """ Returns the size of the map as (height, width)"""
        return (self.height, self.width)
            
    
            
    


    
    

