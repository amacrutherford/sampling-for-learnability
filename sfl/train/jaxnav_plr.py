"""
Run PLR, ACCEL or DR on JaxNav, both single and multi-agent variants.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax import core,struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState as BaseTrainState
import chex
from enum import IntEnum
from typing import Sequence, NamedTuple, Any, Dict
import distrax
import hydra
from omegaconf import OmegaConf
import os
import wandb
import functools
import matplotlib.pyplot as plt
from PIL import Image
import time
import pickle

from jaxmarl.environments.jaxnav import JaxNav
from jaxmarl.environments.jaxnav.jaxnav_ued_utils import make_level_mutator

from sfl.util.jaxued.level_sampler import LevelSampler
from sfl.util.jaxued.jaxued_utils import compute_max_returns, l1_value_loss, max_mc, positive_value_loss

from sfl.train.train_utils import save_params
from sfl.runners import EvalSingletonsRunner, EvalSampledRunner
from sfl.util.rl.plr import PLRManager, PLRBuffer
from sfl.util.rl.ued_scores import UEDScore, compute_ued_scores

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

def _calculate_dormancy(x, layer_dim, tau):
    ins = x.squeeze()
    d = jnp.sum(jnp.abs(ins), axis=0)/ins.shape[0]  # average activation
    d = jnp.where((d / (jnp.sum(d)/layer_dim + 1e-8)) <= tau, 1, 0)
    return jnp.sum(d) / layer_dim * 100  # dormancy percentage
    

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    fc_dim_size: int = 512
    hidden_size: int = 512
    tau: float = 0.0  # dormancy threshold
    is_recurrent: bool = True
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.fc_dim_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)
        ed1 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.fc_dim_size, self.tau))


        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        hd1 = jax.lax.stop_gradient(_calculate_dormancy(hidden, self.hidden_size, self.tau))
        ed2 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.hidden_size, self.tau))
        actor_mean = nn.Dense(self.fc_dim_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        actor_mean = nn.relu(actor_mean)
        ad1 = jax.lax.stop_gradient(_calculate_dormancy(actor_mean, self.fc_dim_size, self.tau))
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        

        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(self.fc_dim_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.use_layer_norm:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        critic = nn.relu(critic)
        cd1 = jax.lax.stop_gradient(_calculate_dormancy(critic, 256, self.tau))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        dormancy = Dormancy(
            actor=ad1,
            embedding=ed1,
            hidden=hd1, 
            rnnout=ed2,
            critic=cd1
        )
        
        #jax.debug.print('dormancy {d}', d=dormancy)

        return hidden, pi, jnp.squeeze(critic, axis=-1), dormancy

class Dormancy(NamedTuple):
    actor: jnp.array
    embedding: jnp.array
    hidden: jnp.array
    rnnout: jnp.array
    critic: jnp.array

class Transition(NamedTuple):
    global_done: jnp.ndarray
    last_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray
    info: jnp.ndarray

class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    targets: jnp.ndarray
    advantages: jnp.ndarray
    # carry: jnp.ndarray
    mask: jnp.ndarray
    
@struct.dataclass
class PlrState:
    plr_buffer: PLRBuffer
    is_replay: bool
    levels: jnp.ndarray
    level_idxs: jnp.ndarray
    

def sample_trajectories_rnn(
    rng, 
    env: JaxNav,
    train_state,
    init_hstate,
    init_obs,
    init_env_state,
    config,
):
    def _env_step(runner_state, unused):
        train_state, env_state, last_obs, last_done, hstate, rng = runner_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
        ac_in = (
            obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        hstate, pi, value, dormancy = train_state.apply_fn(train_state.params, hstate, ac_in)
        action = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)
        env_act = unbatchify(
            action, env.agents, config["NUM_ENVS"], env.num_agents
        )
        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, 0)
        )(rng_step, env_state, env_act, init_env_state) 
        done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
        train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
        transition = Transition(
            jnp.tile(done["__all__"], env.num_agents),
            last_done,
            done_batch,
            action.squeeze(),
            value.squeeze(),
            batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
            log_prob.squeeze(),
            obs_batch,
            train_mask, # 0 if valid, 1 if not
            info,
        )
        runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
        return runner_state, (transition, dormancy)

    (train_state, last_env_state, last_obs, last_done, hstate, rng), traj_batch_dormancy = jax.lax.scan(
        _env_step,
        (
            train_state, 
            init_env_state,
            init_obs,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            init_hstate,
            rng,
        ),
        None,
        config["NUM_STEPS"]
    )
    
    # traj_batch, dormancy = traj_batch_dormancy
    # dormancy = jax.tree_map(lambda x: x.mean(), dormancy)            
            
    last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
    ac_in = (
        last_obs_batch[np.newaxis, :],
        last_done[np.newaxis, :],
    )
    _, _, last_val, _ = train_state.apply_fn(train_state.params, hstate, ac_in)
    last_val = last_val.squeeze()
    
    return (train_state, last_env_state, last_obs, last_done, hstate, last_val, rng), traj_batch_dormancy
    
def update_actor_critic_rnn(
    rng,
    train_state: TrainState,
    init_hstate,
    traj_batch,
    advantages,
    targets,    
    config,
    update_grad=True,
):
    
    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            init_hstate, traj_batch, advantages, targets = batch_info

            def _loss_fn_masked(params, init_hstate, traj_batch, gae, targets):
                                        
                # RERUN NETWORK
                _, pi, value, _ = train_state.apply_fn(
                    params,
                    init_hstate.transpose(),
                    (traj_batch.obs, traj_batch.last_done),
                )
                log_prob = pi.log_prob(traj_batch.action)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = 0.5 * jnp.maximum(
                    value_losses, value_losses_clipped
                ).mean(where=(1 - traj_batch.mask))
                
                # CALCULATE ACTOR LOSS
                logratio = log_prob - traj_batch.log_prob
                ratio = jnp.exp(logratio)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean(where=(1 - traj_batch.mask))
                entropy = pi.entropy().mean(where=(1 - traj_batch.mask))

                # debug
                approx_kl = jax.lax.stop_gradient(
                    ((ratio - 1) - logratio).mean()
                )
                clipfrac = jax.lax.stop_gradient(
                    (jnp.abs(ratio - 1) > config["CLIP_EPS"]).mean()
                )

                total_loss = (
                    loss_actor
                    + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
                )
                return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clipfrac)

            grad_fn = jax.value_and_grad(_loss_fn_masked, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, init_hstate, traj_batch, advantages, targets
            )
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            total_loss = jax.tree_map(lambda x: x.mean(), total_loss)
            return train_state, total_loss

        (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        ) = update_state
        rng, _rng = jax.random.split(rng)

        init_hstate = jnp.reshape(
            init_hstate, (config["HIDDEN_SIZE"], config["NUM_ACTORS"])
        )
        batch = (
            init_hstate,
            traj_batch,
            advantages.squeeze(),
            targets.squeeze(),
        )
        permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=1), batch
        )

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(
                jnp.reshape(
                    x,
                    [x.shape[0], config["NUM_MINIBATCHES"], -1]
                    + list(x.shape[2:]),
                ),
                1,
                0,
            ),
            shuffled_batch,
        )

        train_state, total_loss = jax.lax.scan(
            _update_minbatch, train_state, minibatches
        )
        # total_loss = jax.tree_map(lambda x: x.mean(), total_loss)
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        return update_state, total_loss 
    
    update_state = (
        train_state, 
        init_hstate[None, :].squeeze().transpose(),
        traj_batch,
        advantages,
        targets,
        rng,
    )
    (train_state, _, _, _, _, rng), loss_info = jax.lax.scan(
        _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
    )
    return (rng, train_state), loss_info

def compute_score(config, dones, values, max_returns, advantages):
    if config['SCORE_FUNCTION'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['SCORE_FUNCTION'] == "pvl":
        return positive_value_loss(dones, advantages)
    elif config['SCORE_FUNCTION'] == "l1vl":
        return l1_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['SCORE_FUNCTION']}")

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


@hydra.main(version_base=None, config_path="config", config_name="jaxnav-plr")
def main(config):
    config = OmegaConf.to_container(config)
    t_config = config["learning"]
    
    tags = ["RNN", "ts: "+config["env"]["test_set"], "sf: "+config["ued"]["SCORE_FUNCTION"]]
    if not config["ued"]["EXPLORATORY_GRAD_UPDATES"]:
        tags.append("robust")
    if config["ued"]["USE_ACCEL"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    
    run = wandb.init(
        group=config['GROUP_NAME'],
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    
    env = JaxNav(num_agents=config["env"]["num_agents"],
                        **config["env"]["env_params"])
        
    t_config["NUM_ACTORS"] = env.num_agents * t_config["NUM_ENVS"]
    t_config["NUM_UPDATES"] = (
        t_config["TOTAL_TIMESTEPS"] // t_config["NUM_STEPS"] // t_config["NUM_ENVS"]
    )
    t_config["MINIBATCH_SIZE"] = (
        t_config["NUM_ACTORS"] * t_config["NUM_STEPS"] // t_config["NUM_MINIBATCHES"]
    )
    t_config["CLIP_EPS"] = (
        t_config["CLIP_EPS"] / env.num_agents
        if t_config["SCALE_CLIP_EPS"]
        else t_config["CLIP_EPS"]
    )
    
    def linear_schedule(count):
        count = count // (t_config["NUM_MINIBATCHES"] * t_config["UPDATE_EPOCHS"])
        frac = (
            1.0 - count / t_config["NUM_UPDATES"]
        )
        return t_config["LR"] * frac
    
    # get ued score
    print('Using UED Score:', config["ued"]["SCORE_FUNCTION"])
        
    network = ActorCriticRNN(env.agent_action_space().shape[0],
                             fc_dim_size=t_config["FC_DIM_SIZE"],
                             hidden_size=t_config["HIDDEN_SIZE"],
                             use_layer_norm=t_config["USE_LAYER_NORM"],)

    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, t_config["NUM_ENVS"], env.lidar_num_beams+5)  # NOTE hardcoded
        ),
        jnp.zeros((1, t_config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ENVS"], t_config["HIDDEN_SIZE"])
    network_params = network.init(_rng, init_hstate, init_x)
    
    if t_config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            optax.adam(t_config["LR"], eps=1e-5),
        )

    eval_singleton_runner = EvalSingletonsRunner(
        config["env"]["test_set"],
        network,
        init_carry=ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"],
        env_kwargs=config["env"]["env_params"]
    )
    
    with open(config["EVAL_SAMPLED_SET_PATH"], "rb") as f:
      eval_env_instances = pickle.load(f)
    _, eval_init_states = jax.vmap(env.set_env_instance, in_axes=(0))(eval_env_instances)
    
    eval_sampled_runner = EvalSampledRunner(
        None,
        env,
        network,
        ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"],
        greedy=False,
        env_init_states=eval_init_states,
        n_episodes=10,
    )
        
    sample_random_level = env.sample_test_case
    mutate_level = make_level_mutator(50, env.map_obj)
    level_sampler = LevelSampler(**config["ued"]["PLR_PARAMS"])
    
    rng, _rng = jax.random.split(rng)
    pholder_level = sample_random_level(_rng) 
    sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
    pholder_level_batch = jax.tree_map(lambda x: jnp.array([x]).repeat(t_config["NUM_ENVS"], axis=0), pholder_level)
    
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
        sampler=sampler,
        update_state=0,
        num_dr_updates=0,
        num_replay_updates=0,
        num_mutation_updates=0,
        dr_last_level_batch=pholder_level_batch,
        replay_last_level_batch=pholder_level_batch,
        mutation_last_level_batch=pholder_level_batch,   
    )
    
    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition: Transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.global_done, 
                transition.value,
                transition.reward,
            )
            delta = reward + t_config["GAMMA"] * next_value * (1 - done) - value
            gae = (
                delta
                + t_config["GAMMA"] * t_config["GAE_LAMBDA"] * (1 - done) * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.value

    def log_eval(metrics):
        
        wandb.log(metrics, step=metrics["update_count"])

    # TRAIN LOOP
    @jax.jit
    def train_step(carry, unused):
        # COLLECT TRAJECTORIES
        def callback(metrics):
            wandb.log(metrics, step=metrics["update_count"])
        
        def on_new_levels(rng, train_state: TrainState):
            sampler = train_state.sampler
            
            rng, rng_levels = jax.random.split(rng)
            init_obs, new_levels = jax.vmap(env.reset)(jax.random.split(rng_levels, t_config["NUM_ENVS"]))
            init_env_state = new_levels
            
            init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
            (train_state, last_env_state, last_obs, last_done, hstate, last_val, rng), traj_batch_dormancy = sample_trajectories_rnn(
                rng,
                env,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                t_config,
            )
            traj_batch, dormancy = traj_batch_dormancy
            dormancy = jax.tree_map(lambda x: x.mean(), dormancy)
            
            advantages, targets = _calculate_gae(traj_batch, last_val)
                        
            max_returns = compute_max_returns(traj_batch.done, traj_batch.reward)
            scores = compute_score(config["ued"], traj_batch.done, traj_batch.value, max_returns, advantages)
            scores_by_level = scores.reshape((t_config["NUM_ENVS"], -1), order="F").mean(axis=1)
            max_returns_by_level = max_returns.reshape((t_config["NUM_ENVS"], -1), order="F").mean(axis=1)
            sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores_by_level, {"max_return": max_returns_by_level})

            (rng, train_state), loss_info = update_actor_critic_rnn(rng, train_state, init_hstate, traj_batch, advantages, targets, t_config, update_grad=True)
                        
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )            
            return (rng, train_state), (traj_batch.info, loss_info, dormancy)
            
        def on_replay_levels(rng, train_state: TrainState):
            sampler = train_state.sampler
            
            rng, rng_levels = jax.random.split(rng)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, t_config["NUM_ENVS"])
            init_obs, new_levels = jax.vmap(env.set_state)(levels)
            init_env_state = new_levels
            
            init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
            (train_state, last_env_state, last_obs, last_done, hstate, last_val, rng), traj_batch_dormancy = sample_trajectories_rnn(
                rng,
                env,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                t_config,
            )
            traj_batch, dormancy = traj_batch_dormancy
            dormancy = jax.tree_map(lambda x: x.mean(), dormancy)
                    
            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            max_returns = compute_max_returns(traj_batch.done, traj_batch.reward)
            max_returns_by_level = max_returns.reshape((t_config["NUM_ENVS"], -1), order="F").mean(axis=1)
            max_returns_by_level = jnp.maximum(level_sampler.get_levels_extra(sampler, level_inds)["max_return"], max_returns_by_level)
            scores = compute_score(config["ued"], traj_batch.done, traj_batch.value, max_returns, advantages)
            scores_by_level = scores.reshape((t_config["NUM_ENVS"], -1), order="F").mean(axis=1)

            sampler = level_sampler.update_batch(sampler, level_inds, scores_by_level, {"max_return": max_returns_by_level})
            (rng, train_state), loss_info = update_actor_critic_rnn(rng, train_state, init_hstate, traj_batch, advantages, targets, t_config, update_grad=True)
                        
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
                        
            return (rng, train_state), (traj_batch.info, loss_info, dormancy)
            
        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            sampler = train_state.sampler
            
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
            
            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_level, (0, 0, None))(jax.random.split(rng_mutate, t_config["NUM_ENVS"]), parent_levels, config["ued"]["NUM_EDITS"])            
            init_obs, new_levels = jax.vmap(env.set_state)(child_levels)
            init_env_state = new_levels
            
            init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
            (train_state, last_env_state, last_obs, last_done, hstate, last_val, rng), traj_batch_dormancy = sample_trajectories_rnn(
                rng,
                env,
                train_state,
                init_hstate,
                init_obs,
                init_env_state,
                t_config,
            )
            traj_batch, dormancy = traj_batch_dormancy
            dormancy = jax.tree_map(lambda x: x.mean(), dormancy)
            
            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            max_returns = compute_max_returns(traj_batch.done, traj_batch.reward)
            max_returns_by_level = max_returns.reshape((t_config["NUM_ENVS"], -1), order="F").mean(axis=1)

            scores = compute_score(config["ued"], traj_batch.done, traj_batch.value, max_returns, advantages)
            scores_by_level = scores.reshape((t_config["NUM_ENVS"], -1), order="F").mean(axis=1)
            
            sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores_by_level, {"max_return": max_returns_by_level})
                        
            (rng, train_state), loss_info = update_actor_critic_rnn(
                rng,
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                t_config,
                update_grad=config["ued"]["EXPLORATORY_GRAD_UPDATES"]
            )
                        
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
            )
                        
            return (rng, train_state), (traj_batch.info, loss_info, dormancy)
            
        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)
        
        # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
        # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`).
        if config["ued"]["USE_ACCEL"]:
            s = train_state.update_state
            branch = ((1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s).astype(int)
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)
        
        (rng, train_state), (train_info, loss_info, dormancy) = jax.lax.switch(
            branch,
            [
                on_new_levels,
                on_replay_levels,
            ] + ([] if not config["ued"]["USE_ACCEL"] else [on_mutate_levels]),
            rng, train_state
        )
        
        # LOG
        train_info = jax.tree_map(lambda x: x.sum(axis=-1).reshape((t_config["NUM_STEPS"], t_config["NUM_ENVS"])).sum(), train_info)
        ratio_0 = loss_info[1][3].at[0,0].get().mean()
        loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
        metrics = {
            "loss/": {
               "total_loss": loss_info[0],
               "value_loss": loss_info[1][0],
               "actor_loss": loss_info[1][1],
               "entropy": loss_info[1][2],
               "ratio": loss_info[1][3],
               "ratio_0": ratio_0,
               "approx_kl": loss_info[1][4],
               "clipfrac": loss_info[1][5],
            },
            "dormancy/": {
                "actor": dormancy.actor.mean(),
                "embedding": dormancy.embedding.mean(),
                "rnnout": dormancy.rnnout.mean(),
                "critic": dormancy.critic.mean(),
            },
            "terminations": {
                k: train_info[k] for k in ["NumC", "GoalR", "AgentC", "MapC", "TimeO"]
            },
        }
        
        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates  
        metrics["num_env_steps"] = metrics["update_count"] * t_config["NUM_STEPS"] * t_config["NUM_ENVS"]
        
        jax.experimental.io_callback(callback, None, metrics)
        return (rng, train_state), metrics


    @jax.jit
    def train_and_eval_step(runner_state, unused):
        
        # TRAIN
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, t_config["EVAL_FREQ"])
        
        # EVAL
        rng, eval_singleton_rng, eval_sampled_rng = jax.random.split(rng, 3)
        test_metrics = {}
        test_metrics["singleton-test-metrics"] = eval_singleton_runner.run(eval_singleton_rng, train_state.params)
        test_metrics["sampled-test-metrics"] = eval_sampled_runner.run(eval_sampled_rng, train_state.params)
        
        test_metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates  
        
        return (rng, train_state), test_metrics
    
    def log_buffer(sampler, epoch):
        
        sorted_scores = jnp.argsort(sampler["scores"])
        top = sorted_scores[-20:]
        bottom = sorted_scores[:20]
        
        num_samples = 20
        rows_per = 2 
        fig, axes = plt.subplots(2*rows_per, int(num_samples/rows_per), figsize=(20, 10))
        axes=axes.flatten()
        for i, ax in enumerate(axes[:num_samples]):
            # ax.imshow(train_state.plr_buffer.get_sample(i))
            idx = top[i]
            score = sampler["scores"][idx]
            level = jax.tree_map(lambda x: x[idx], sampler["levels"])
                        
            env.init_render(ax, level, lidar=False, ticks_off=True)
            ax.set_title(f'regret: {score:.3f}, \ntimestamp: {sampler["timestamps"][i]}')
            ax.set_aspect('equal', 'box')
            
        for i, ax in enumerate(axes[num_samples:]):
            # ax.imshow(train_state.plr_buffer.get_sample(i))
            idx = bottom[i]
            score = sampler["scores"][idx]
            level = jax.tree_map(lambda x: x[idx], sampler["levels"])
                        
            env.init_render(ax, level, lidar=False, ticks_off=True)
            ax.set_title(f'regret: {score:.3f}, \ntimestamp: {sampler["timestamps"][i]}')
            ax.set_aspect('equal', 'box')
            
        plt.tight_layout()
        fig.canvas.draw()
        im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()) 
        wandb.log({"maps": wandb.Image(im)}, step=epoch)
    
    print('eval step', t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"])
    print('num updates', t_config["NUM_UPDATES"])
    
    checkpoint_steps = t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"] // t_config["NUM_CHECKPOINTS"]
    runner_state = (rng, train_state)
    for eval_step in range(int(t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"])):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        metrics["steps_per_section"] = (t_config["EVAL_FREQ"] * t_config["NUM_STEPS"] * t_config["NUM_ENVS"]) / metrics['time_delta']
        log_eval(metrics)  #, train_state_to_log_dict(runner_state[1], level_sampler) add?
        log_buffer(runner_state[1].sampler, metrics["update_count"])
        if (eval_step % checkpoint_steps == 0) & (eval_step > 0):    
            if config["SAVE_PATH"] is not None:
                params = runner_state[1].params
                
                save_dir = os.path.join(config["SAVE_PATH"], run.name)
                os.makedirs(save_dir, exist_ok=True)
                save_params(params, f'{save_dir}/model.safetensors')
                print(f'Parameters of saved in {save_dir}/model.safetensors')
                
                # upload this to wandb as an artifact   
                artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
                artifact.add_file(f'{save_dir}/model.safetensors')
                artifact.save()
    
    if config["SAVE_PATH"] is not None:
        params = runner_state[1].params
        
        save_dir = os.path.join(config["SAVE_PATH"], wandb.run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of saved in {save_dir}/model.safetensors')
        
        # upload this to wandb as an artifact   
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()
    
    

if __name__ == "__main__":
    main()
