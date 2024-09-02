"""
Run SFL on Minigrid Maze using the JaxUED environment.
"""

import chex
import jax
import jax.experimental
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import hydra
from sfl.train.minigrid_plr import ActorCritic, evaluate_rnn
from omegaconf import OmegaConf
import os
from functools import partial
import time 
from PIL import Image
import wandb
import matplotlib.pyplot as plt

from jaxued.environments import Maze, MazeRenderer
from jaxued.wrappers import AutoReplayWrapper
from jaxued.environments.maze import Level, make_level_generator

from sfl.train.train_utils import save_params


class Transition(NamedTuple):
    global_done: jnp.ndarray
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

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
        

@hydra.main(version_base=None, config_path="config", config_name="minigrid-sfl")
def main(config):
    
    config = OmegaConf.to_container(config)
    config["NUM_ENVS"] = config["learning"]["NUM_ENVS"]
    config["EVAL_NUM_ATTEMPTS"] = config["env"]["EVAL_NUM_ATTEMPTS"]
    config["EVAL_LEVELS"] = config["env"]["EVAL_LEVELS"]
    run = wandb.init(
        group=config["GROUP_NAME"],
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "DR", f"ts: "],
        config=config,
        mode=config["WANDB_MODE"],
    )
        
    rng = jax.random.PRNGKey(config["SEED"])
    
    assert (config["learning"]["NUM_ENVS_FROM_SAMPLED"] +  config["learning"]["NUM_ENVS_TO_GENERATE"]) == config["learning"]["NUM_ENVS"]
    
    env = Maze(max_height=13, max_width=13, agent_view_size=config["env"]["AGENT_VIEW_SIZE"], normalize_obs=True)
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["env"]["N_WALLS"])
    eval_env = env
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    t_config = config["learning"]
        
    t_config["NUM_ACTORS"] = t_config["NUM_ENVS"]
    t_config["NUM_UPDATES"] = (
        t_config["TOTAL_TIMESTEPS"] // t_config["NUM_STEPS"] // t_config["NUM_ENVS"]
    )
    t_config["MINIBATCH_SIZE"] = (
        t_config["NUM_ACTORS"] * t_config["NUM_STEPS"] // t_config["NUM_MINIBATCHES"]
    )
    t_config["CLIP_EPS"] = (
        t_config["CLIP_EPS"]
        if t_config["SCALE_CLIP_EPS"]
        else t_config["CLIP_EPS"]
    )
        
    network = ActorCritic(env.action_space(env.default_params).n)


    def linear_schedule(count):
        count = count // (t_config["NUM_MINIBATCHES"] * t_config["UPDATE_EPOCHS"])
        frac = (
            1.0 - count / t_config["NUM_UPDATES"]
        )
        return t_config["LR"] * frac
    
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    obs, _ = env.reset_to_level(rng, sample_random_level(rng), env.default_params)
    obs = jax.tree_map(
    lambda x: jnp.repeat(jnp.repeat(x[None, ...], t_config["NUM_ENVS"], axis=0)[None, ...], 256, axis=0),
        obs,
    )    
    init_x = (obs, jnp.zeros((256, t_config["NUM_ENVS"])))
    init_hstate = ActorCritic.initialize_carry((t_config["NUM_ENVS"], ))
    network_params = network.init(_rng, init_x, init_hstate)
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
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    rng, _rng = jax.random.split(rng)
    # initial_singleton_test_metrics = eval_singleton_runner.run(_rng, train_state.params)  
    # initial_sampled_test_metrics = eval_sampled_runner.run(_rng, train_state.params)

    # INIT ENV
    rng, _rng, _rng2 = jax.random.split(rng, 3)
    rng_reset = jax.random.split(_rng, t_config["NUM_ENVS"])
    rng_levels = jax.random.split(_rng2, config["NUM_ENVS"])
    # obsv, env_state = jax.vmap(sample_random_level, in_axes=(0,))(reset_rng)

    new_levels = jax.vmap(sample_random_level)(rng_levels)
    obsv, env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(rng_reset, new_levels, env.default_params)
    
    
    if t_config["LAMBDA_SCHEDULE"]:
        raise NotImplementedError("Lambda schedule not implemented for finetuning")
        rng, lambda_rng = jax.random.split(rng)
        env_state = env_state.replace(
            rew_lambda = sample_lambda_set(lambda_rng, 0),
        )
    start_state = env_state
    init_hstate = ActorCritic.initialize_carry((t_config["NUM_ACTORS"],))
    
    
    
    @jax.jit
    def get_learnability_set(rng, network_params):
        
        
        BATCH_ACTORS = config["BATCH_SIZE"]
        
        
        def _batch_step(unused, rng):
            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = last_obs # batchify(last_obs, env.agents, BATCH_ACTORS)
                ac_in = (
                    jax.tree_map(lambda x: x[np.newaxis, :], obs_batch),
                    last_done[np.newaxis, :],
                )
                hstate, pi, value = network.apply(network_params, ac_in, hstate)
                action = pi.sample(seed=_rng).squeeze()
                log_prob = pi.log_prob(action)
                env_act = action
                # unbatchify(
                #     action, env.agents, config["BATCH_SIZE"], env.num_agents
                # )
                # env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["BATCH_SIZE"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, env_act, env.default_params)
                # reward = batchify(reward, env.agents, BATCH_ACTORS).squeeze()
                done_batch = done # batchify(done, env.agents, BATCH_ACTORS).squeeze()
                train_mask = (done * 0).reshape(-1)
                # info["terminated"].swapaxes(0, 1).reshape(-1)
                # train_mask = batchify(info["terminated"], env.agents, BATCH_ACTORS).squeeze()
                transition = Transition(
                    done, #(done["__all__"]),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    reward,
                    log_prob.squeeze(),
                    obs_batch,
                    train_mask,
                    info,
                )
                runner_state = (env_state, start_state, obsv, done_batch, hstate, rng)
                return runner_state, transition
            
            @partial(jax.vmap, in_axes=(None, 1, 1, 1))
            @partial(jax.jit, static_argnums=(0,))
            def _calc_outcomes_by_agent(max_steps: int, dones, returns, info):
                idxs = jnp.arange(max_steps)
                
                @partial(jax.vmap, in_axes=(0, 0))
                def __ep_outcomes(start_idx, end_idx): 
                    mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                    r = jnp.sum(returns * mask)
                    goal_r = (returns > 0) * 1.0
                    success = jnp.sum(goal_r * mask) #jnp.sum(info["GoalR"] * mask)
                    collision = 0.0 # jnp.sum((info["MapC"] + info["AgentC"]) * mask)
                    timeo = 0.0 # jnp.sum(info["TimeO"] * mask)
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
            
            # sample envs
            rng, _rng, _rng2 = jax.random.split(rng, 3)
            rng_reset = jax.random.split(_rng, config["BATCH_SIZE"])
            rng_levels = jax.random.split(_rng2, config["BATCH_SIZE"])
            # obsv, env_state = jax.vmap(sample_random_level, in_axes=(0,))(reset_rng)
            new_levels = jax.vmap(sample_random_level)(rng_levels)
            obsv, env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(rng_reset, new_levels, env.default_params)
            env_instances = new_levels
            # env_instances = EnvInstance(
            #     agent_pos=env_state.pos,
            #     agent_theta=env_state.theta,
            #     goal_pos=env_state.goal,
            #     map_data=env_state.map_data,
            #     rew_lambda=env_state.rew_lambda,
            # )
            

            init_hstate = ActorCritic.initialize_carry((BATCH_ACTORS,))
            
            runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, rng)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_STEPS"]
            )
            print('traj batch done', traj_batch.done.shape)
            # print('traj batch info', traj_batch.info["NumC"].shape)
            done_by_env = traj_batch.done.reshape((-1, config["BATCH_SIZE"]))
            reward_by_env = traj_batch.reward.reshape((-1, config["BATCH_SIZE"]))
            info_by_actor = jax.tree_map(lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            print('done_by_env', done_by_env.shape)
            print('reward_by_env', reward_by_env.shape)
            print('info_by_actor', info_by_actor)
            o = _calc_outcomes_by_agent(config["ROLLOUT_STEPS"], traj_batch.done, traj_batch.reward, info_by_actor)
            print('o', o)
            success_by_env = o["success_rate"].reshape((1, config["BATCH_SIZE"]))
            learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)
            print('learnability_by_env', learnability_by_env)
            return None, (learnability_by_env, env_instances)
            
        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (learnability, env_instances) = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"])
        
        flat_env_instances = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
        learnability = learnability.flatten()
        top_1000 = jnp.argsort(learnability)[-config["NUM_TO_SAVE"]:]
        print('top 1000', top_1000)
        
        top_1000_instances = jax.tree_map(lambda x: x.at[top_1000].get(), flat_env_instances) 
        print('top 1000 instances', top_1000_instances)
        return learnability.at[top_1000].get(), top_1000_instances
    
    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["EVAL_LEVELS"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["EVAL_LEVELS"])
        num_levels = len(config["EVAL_LEVELS"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env.default_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env.default_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env.default_params.max_steps_in_episode,
        )
        mask = jnp.arange(env.default_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
    

    
    # TRAIN LOOP
    def train_step(runner_state_instances, unused):
        # COLLECT TRAJECTORIES
        runner_state, instances = runner_state_instances
        num_env_instances = instances.agent_pos.shape[0]

        def _env_step(runner_state, unused):
            train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = last_obs # batchify(last_obs, env.agents, t_config["NUM_ACTORS"])
            ac_in = (
                jax.tree_map(lambda x: x[np.newaxis, :], obs_batch),
                last_done[np.newaxis, :],
            )
            hstate, pi, value = network.apply(train_state.params, ac_in, hstate)
            action = pi.sample(seed=_rng).squeeze()
            log_prob = pi.log_prob(action)
            env_act = action
            # unbatchify(
            #     action, env.agents, t_config["NUM_ENVS"], env.num_agents
            # )
            # env_act = {k: v.squeeze() for k, v in env_act.items()}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, t_config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(rng_step, env_state, env_act, env.default_params)
            # reward = batchify(reward, env.agents, t_config["NUM_ACTORS"]).squeeze()
            done_batch = done # batchify(done, env.agents, t_config["NUM_ACTORS"]).squeeze()
            train_mask = (done * 0).reshape(-1)
            # info["terminated"].swapaxes(0, 1).reshape(-1)
            # train_mask = batchify(info["terminated"], env.agents, t_config["NUM_ACTORS"]).squeeze()
            transition = Transition(
                done,#["__all__"],
                last_done,
                action.squeeze(),
                value.squeeze(),
                reward,
                log_prob.squeeze(),
                obs_batch,
                train_mask,
                info,
            )
            runner_state = (train_state, env_state, start_state, obsv, done_batch, hstate, update_steps, rng)
            return runner_state, (transition)

        initial_hstate = runner_state[-3]
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, t_config["NUM_STEPS"]
        )
        # traj_batch, dormancy = traj_batch_dormancy
        # dormancy = jax.tree_map(lambda x: x.mean(), dormancy)
        
        @partial(jax.vmap, in_axes=(1, 1))
        def _calc_ep_return_by_agent(dones, returns):
            idxs = jnp.arange(t_config["NUM_STEPS"])
            
            @partial(jax.vmap, in_axes=(None, 0, 0))
            def __ep_returns(rews, start_idx, end_idx): 
                mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != t_config["NUM_STEPS"])
                r = jnp.sum(rews * mask, axis=0)
                l = end_idx - start_idx
                return r, l
            
            done_idxs = jnp.argwhere(dones, size=t_config["NUM_STEPS"]//4, fill_value=t_config["NUM_STEPS"]).squeeze()
            mask_done = jnp.where(done_idxs == t_config["NUM_STEPS"], 0, 1)
            r, l = __ep_returns(returns, jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)                
            return {"episodic_return_per_agent": r.mean(where=mask_done), "episodic_length_per_agent": l.mean(where=mask_done)}
        
        reward_by_env = traj_batch.reward
        episodic_return_length = _calc_ep_return_by_agent(traj_batch.done, reward_by_env)
        episodic_return_length = jax.tree_map(lambda x: x.mean(), episodic_return_length)
        # CALCULATE ADVANTAGE
        train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state
        last_obs_batch = last_obs # batchify(last_obs, env.agents, t_config["NUM_ACTORS"])
        ac_in = (
            jax.tree_map(lambda x: x[np.newaxis, :], last_obs_batch),
            last_done[np.newaxis, :],
        )
        _, _, last_val = network.apply(train_state.params, ac_in, hstate)
        last_val = last_val.squeeze()
        print('last_val shape', last_val.shape)
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

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn_masked(params, init_hstate, traj_batch, gae, targets):
                                            
                    # RERUN NETWORK
                    _, pi, value = network.apply(
                        params,
                        (traj_batch.obs, traj_batch.done),
                        jax.tree_map(lambda x: x.transpose(), init_hstate),
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-t_config["CLIP_EPS"], t_config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(
                        value_losses, value_losses_clipped
                    )
                    critic_loss = t_config["VF_COEF"] * value_loss.mean(where=(1 - traj_batch.mask))
                    
                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    # if env.do_sep_reward: gae = gae.sum(axis=-1)
                    gae = (gae - gae.mean(where=(1-traj_batch.mask))) / (gae.std(where=(1-traj_batch.mask)) + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - t_config["CLIP_EPS"],
                            1.0 + t_config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean(where=(1 - traj_batch.mask))
                    entropy = pi.entropy().mean(where=(1 - traj_batch.mask))
                    
                    approx_kl = jax.lax.stop_gradient(
                        ((ratio - 1) - logratio).mean()
                    )
                    clipfrac = jax.lax.stop_gradient(
                        (jnp.abs(ratio - 1) > t_config["CLIP_EPS"]).mean()
                    )

                    total_loss = (
                        loss_actor
                        + critic_loss
                        - t_config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clipfrac)

                grad_fn = jax.value_and_grad(_loss_fn_masked, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
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

            init_hstate = jax.tree_map(lambda x: jnp.reshape(
                x, (256, t_config["NUM_ACTORS"])
            ), init_hstate)
            batch = (
                init_hstate,
                traj_batch,
                advantages.squeeze(),
                targets.squeeze(),
            )
            permutation = jax.random.permutation(_rng, t_config["NUM_ACTORS"])

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], t_config["NUM_MINIBATCHES"], -1]
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

        # init_hstate = initial_hstate[None, :].squeeze().transpose()
        init_hstate = jax.tree_map(lambda x: x[None, :].squeeze().transpose(), initial_hstate)
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, t_config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        metric = traj_batch.info
        metric = jax.tree_map(
            lambda x: x.sum(axis=-1).reshape(
                (t_config["NUM_STEPS"], t_config["NUM_ENVS"])  # , env.num_agents
            ),
            traj_batch.info,
        )
        rng = update_state[-1]

        def callback(metric):
            wandb.log(
                {
                    # "train-term": metric["terminations"],
                    #"reward": metric["returned_episode_returns"],
                    
                    # "eval-collision": metric["test-metrics"]["collision-by-env"].mean(),
                    # "eval-timeout": metric["test-metrics"]["timeout-by-env"].mean(),
                    "env_step": metric["update_steps"]
                        * t_config["NUM_ENVS"]
                        * t_config["NUM_STEPS"],
                    # "dormancy/": metric["dormancy"],
                    # "env-metrics/": metric["env-metrics"],
                    # "mean_ued_score": metric["mean_ued_score"],
                    **metric["episodic_return_length"],
                    **metric["loss_info"],
                    # "mean_lambda_val": metric["mean_lambda_val"],
                }
            )

        dormancy_log = {
            # "actor": dormancy.actor,
            # "embedding": dormancy.embedding,
            # "hidden": dormancy.hidden,
            # "rnnout": dormancy.rnnout,
            # "critic": dormancy.critic,
        }
        ratio0 = jnp.around(loss_info[1][3].at[0,0].get().mean(), decimals=6)
        loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
        metric["loss_info"] = {
            "total_loss": loss_info[0],
            "value_loss": loss_info[1][0],
            "actor_loss": loss_info[1][1],
            "entropy": loss_info[1][2],
            "ratio": loss_info[1][3],
            "ratio_0": ratio0,
            "approx_kl": loss_info[1][4],
            "clipfrac": loss_info[1][5],
            "mask_percentage": jnp.mean(traj_batch.mask),
        }
        metric["episodic_return_length"] = episodic_return_length
        metric["update_steps"] = update_steps
        # metric["terminations"] = {k: traj_batch.info[k] for k in ["NumC", "GoalR", "AgentC", "MapC", "TimeO"]}
        # metric["terminations"] = jax.tree_map(lambda x: x.sum(), metric["terminations"])
        metric["dormancy"] = dormancy_log
        # metric["env-metrics"] = jax.tree_map(lambda x: x.mean(), jax.vmap(env.get_env_metrics)(start_state))
        # metric["mean_lambda_val"] = env_state.rew_lambda.mean()
        jax.experimental.io_callback(callback, None, metric)
        
        # SAMPLE NEW ENVS
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        rng_reset = jax.random.split(_rng, t_config["NUM_ENVS_TO_GENERATE"])
        rng_levels = jax.random.split(_rng2, t_config["NUM_ENVS_TO_GENERATE"])
        
        # obsv_gen, env_state_gen = jax.vmap(sample_random_level, in_axes=(0,))(reset_rng)
        new_levels = jax.vmap(sample_random_level)(rng_levels)
        # obsv, env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(rng_reset, new_levels, env.default_params)
        obsv_gen, env_state_gen = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(rng_reset, new_levels, env.default_params)
    
    
        rng, _rng, _rng2 = jax.random.split(rng, 3)
        sampled_env_instances_idxs = jax.random.randint(_rng, (t_config["NUM_ENVS_FROM_SAMPLED"],), 0, num_env_instances)
        sampled_env_instances = jax.tree_map(lambda x: x.at[sampled_env_instances_idxs].get(), instances)
        myrng = jax.random.split(_rng2, t_config["NUM_ENVS_FROM_SAMPLED"])
        obsv_sampled, env_state_sampled = jax.vmap(env.reset_to_level, in_axes=(0, 0))(myrng, sampled_env_instances)
        
        obsv = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), obsv_gen, obsv_sampled)
        env_state = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), env_state_gen, env_state_sampled)
        
        start_state = env_state
        hstate = ActorCritic.initialize_carry((t_config["NUM_ACTORS"],))
        
        update_steps = update_steps + 1
        runner_state = (train_state, env_state, start_state, obsv, jnp.zeros((t_config["NUM_ACTORS"]), dtype=bool), hstate, update_steps, rng)
        return (runner_state, instances), metric
    
    def log_buffer(learnability, states, epoch):
        num_samples = states.env_state.agent_pos.shape[0]
        rows = 2 
        fig, axes = plt.subplots(rows, int(num_samples/rows), figsize=(20, 10))
        axes=axes.flatten()
        for i, ax in enumerate(axes):
            # ax.imshow(train_state.plr_buffer.get_sample(i))
            score = learnability[i]            
            state = jax.tree_map(lambda x: x[i], states)

            img = env_renderer.render_state(
                state.env_state, env.default_params
            )
            # env.init_render(ax, state, lidar=False, ticks_off=True)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'learnability: {score:.3f}')
            ax.set_aspect('equal', 'box')
                        
        plt.tight_layout()
        fig.canvas.draw()
        im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()) 
        wandb.log({"maps": wandb.Image(im)}, step=epoch)
    
    @jax.jit
    def train_and_eval_step(runner_state, eval_rng):
        
        learnability_rng, eval_singleton_rng, eval_sampled_rng, _rng = jax.random.split(eval_rng, 4)
        # TRAIN
        learnabilty_scores, instances = get_learnability_set(learnability_rng, runner_state[0].params)
        runner_state_instances = (runner_state, instances)
        runner_state_instances, metrics = jax.lax.scan(train_step, runner_state_instances, None, t_config["EVAL_FREQ"])
        # EVAL
        
        test_metrics = {
            "learnability_set_scores": learnabilty_scores,
            "learnability_set_mean_score": learnabilty_scores.mean(),
        }

        # TODO EVAL HERE
        # test_metrics["singleton-test-metrics"] = eval_singleton_runner.run(eval_singleton_rng, runner_state[0].params)
        # test_metrics["sampled-test-metrics"] = eval_sampled_runner.run(eval_sampled_rng, runner_state[0].params)
        
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(eval_singleton_rng, config["EVAL_NUM_ATTEMPTS"]), runner_state[0])
        # Collect Metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)

        log_dict = {}
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["EVAL_LEVELS"], eval_solve_rates)})
        log_dict.update({"solve_rate/mean": eval_solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["EVAL_LEVELS"], eval_returns)})
        log_dict.update({"return/mean": eval_returns.mean()})

        test_metrics.update(log_dict)

        runner_state, _ = runner_state_instances
        test_metrics["update_count"] = runner_state[-2]
        
        top_instances = jax.tree_map(lambda x: x.at[-20:].get(), instances)
        _rng = jax.random.split(_rng, 20)
        _, top_states = jax.vmap(env.reset_to_level)(_rng, top_instances)
        
        return runner_state, (learnabilty_scores.at[-20:].get(), top_states), test_metrics
    
    rng, _rng = jax.random.split(rng)
    runner_state = (
        train_state,
        env_state,
        start_state,
        obsv,
        jnp.zeros((t_config["NUM_ACTORS"]), dtype=bool),
        init_hstate,
        0,
        _rng,
    )
    checkpoint_steps = t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"] // t_config["NUM_CHECKPOINTS"]
    print('eval freq', t_config["EVAL_FREQ"])
    for eval_step in range(int(t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"])):
        start_time = time.time()
        rng, eval_rng = jax.random.split(rng)
        runner_state, instances, metrics = train_and_eval_step(runner_state, eval_rng)
        curr_time = time.time()
        log_buffer(*instances, metrics["update_count"])
        metrics['time_delta'] = curr_time - start_time
        metrics["steps_per_section"] = (t_config["EVAL_FREQ"] * t_config["NUM_STEPS"] * t_config["NUM_ENVS"]) / metrics['time_delta']
        wandb.log(metrics, step=metrics["update_count"])
        if (eval_step % checkpoint_steps == 0) & (eval_step > 0):    
            if config["SAVE_PATH"] is not None:
                params = runner_state[0].params
                
                save_dir = os.path.join(config["SAVE_PATH"], run.name)
                os.makedirs(save_dir, exist_ok=True)
                save_params(params, f'{save_dir}/model.safetensors')
                print(f'Parameters of saved in {save_dir}/model.safetensors')
                
                # upload this to wandb as an artifact   
                artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
                artifact.add_file(f'{save_dir}/model.safetensors')
                artifact.save()
                
    if config["SAVE_PATH"] is not None:
        params = runner_state[0].params
        
        save_dir = os.path.join(config["SAVE_PATH"], run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of saved in {save_dir}/model.safetensors')
        
        # upload this to wandb as an artifact   
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()
    

if __name__ == "__main__":
    main()
