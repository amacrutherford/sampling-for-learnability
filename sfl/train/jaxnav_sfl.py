"""
Run SFL on JaxNav, both single and multi-agent variations.
"""

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import os
from functools import partial
import pickle
import time 
from PIL import Image
import wandb
import matplotlib.pyplot as plt

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav, EnvInstance, NUM_REWARD_COMPONENTS, REWARD_COMPONENT_DENSE, REWARD_COMPONENT_SPARSE, listify_reward

from sfl.runners import EvalSingletonsRunner, EvalSampledRunner
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
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
        

@hydra.main(version_base=None, config_path="config", config_name="jaxnav-sfl")
def main(config):
    
    config = OmegaConf.to_container(config)
    run = wandb.init(
        group=config["GROUP_NAME"],
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "DR", f"ts: {config['env']['test_set']}"],
        config=config,
        mode=config["WANDB_MODE"],
    )
        
    rng = jax.random.PRNGKey(config["SEED"])
    
    assert (config["learning"]["NUM_ENVS_FROM_SAMPLED"] +  config["learning"]["NUM_ENVS_TO_GENERATE"]) == config["learning"]["NUM_ENVS"]
    
    env = JaxNav(num_agents=config["env"]["num_agents"],
                        **config["env"]["env_params"])  # use old config for env params to try reduce errors
    print('num agents', env.num_agents)
    t_config = config["learning"]
        
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
        
    network = ActorCriticRNN(env.agent_action_space().shape[0],
                             config=t_config)

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
    
    def linear_schedule(count):
        count = count // (t_config["NUM_MINIBATCHES"] * t_config["UPDATE_EPOCHS"])
        frac = (
            1.0 - count / t_config["NUM_UPDATES"]
        )
        return t_config["LR"] * frac
    
    # INIT NETWORK
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
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    rng, _rng = jax.random.split(rng)
    initial_singleton_test_metrics = eval_singleton_runner.run(_rng, train_state.params)  
    initial_sampled_test_metrics = eval_sampled_runner.run(_rng, train_state.params)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, t_config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    
    
    if t_config["LAMBDA_SCHEDULE"]:
        raise NotImplementedError("Lambda schedule not implemented for finetuning")
        rng, lambda_rng = jax.random.split(rng)
        env_state = env_state.replace(
            rew_lambda = sample_lambda_set(lambda_rng, 0),
        )
    start_state = env_state
    init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
    
    
    
    @jax.jit
    def get_learnability_set(rng, network_params):
        
        
        BATCH_ACTORS = config["BATCH_SIZE"] * env.num_agents
        
        
        def _batch_step(unused, rng):
            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, BATCH_ACTORS)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value, _ = network.apply(network_params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["BATCH_SIZE"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["BATCH_SIZE"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, 0)
                )(rng_step, env_state, env_act, start_state)
                if env.do_sep_reward:
                    reward = listify_reward(reward, do_batchify=True)
                else:
                    reward = batchify(reward, env.agents, BATCH_ACTORS).squeeze()
                done_batch = batchify(done, env.agents, BATCH_ACTORS).squeeze()
                train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
                # train_mask = batchify(info["terminated"], env.agents, BATCH_ACTORS).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
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
            
            # sample envs
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            env_instances = EnvInstance(
                agent_pos=env_state.pos,
                agent_theta=env_state.theta,
                goal_pos=env_state.goal,
                map_data=env_state.map_data,
                rew_lambda=env_state.rew_lambda,
            )
            
            init_hstate = ScannedRNN.initialize_carry(BATCH_ACTORS, t_config["HIDDEN_SIZE"])
            
            runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, rng)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_STEPS"]
            )
            print('traj batch done', traj_batch.done.shape)
            print('traj batch info', traj_batch.info["NumC"].shape)
            done_by_env = traj_batch.done.reshape((-1, env.num_agents, config["BATCH_SIZE"]))
            reward_by_env = traj_batch.reward.reshape((-1, env.num_agents, config["BATCH_SIZE"]))
            info_by_actor = jax.tree_map(lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            print('done_by_env', done_by_env.shape)
            print('reward_by_env', reward_by_env.shape)
            print('info_by_actor', info_by_actor)
            o = _calc_outcomes_by_agent(config["ROLLOUT_STEPS"], traj_batch.done, traj_batch.reward, info_by_actor)
            print('o', o)
            success_by_env = o["success_rate"].reshape((env.num_agents, config["BATCH_SIZE"]))
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
        
    
    # TRAIN LOOP
    def train_step(runner_state_instances, unused):
        # COLLECT TRAJECTORIES
        runner_state, instances = runner_state_instances
        num_env_instances = instances.agent_pos.shape[0]

        def _env_step(runner_state, unused):
            train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = batchify(last_obs, env.agents, t_config["NUM_ACTORS"])
            ac_in = (
                obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            hstate, pi, value, dormancy = network.apply(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(
                action, env.agents, t_config["NUM_ENVS"], env.num_agents
            )
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, t_config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, 0)
            )(rng_step, env_state, env_act, start_state)
            if env.do_sep_reward:
                reward = listify_reward(reward, do_batchify=True)
            else:
                reward = batchify(reward, env.agents, t_config["NUM_ACTORS"]).squeeze()
            done_batch = batchify(done, env.agents, t_config["NUM_ACTORS"]).squeeze()
            train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
            # train_mask = batchify(info["terminated"], env.agents, t_config["NUM_ACTORS"]).squeeze()
            transition = Transition(
                jnp.tile(done["__all__"], env.num_agents),
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
            return runner_state, (transition, dormancy)

        initial_hstate = runner_state[-3]
        runner_state, traj_batch_dormancy = jax.lax.scan(
            _env_step, runner_state, None, t_config["NUM_STEPS"]
        )
        traj_batch, dormancy = traj_batch_dormancy
        dormancy = jax.tree_map(lambda x: x.mean(), dormancy)
        
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
        
        if env.do_sep_reward:
            reward_by_env = traj_batch.reward.sum(axis=-1)
        else:
            reward_by_env = traj_batch.reward
        episodic_return_length = _calc_ep_return_by_agent(traj_batch.done, reward_by_env)
        episodic_return_length = jax.tree_map(lambda x: x.mean(), episodic_return_length)
        # CALCULATE ADVANTAGE
        train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state
        last_obs_batch = batchify(last_obs, env.agents, t_config["NUM_ACTORS"])
        ac_in = (
            last_obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
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
                    _, pi, value, _ = network.apply(
                        params,
                        init_hstate.transpose(),
                        (traj_batch.obs, traj_batch.done),
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
                    if env.do_sep_reward:
                        value_loss_sparse = value_loss[..., REWARD_COMPONENT_SPARSE].mean(where=(1 - traj_batch.mask))
                        value_loss_dense  = value_loss[..., REWARD_COMPONENT_DENSE].mean(where=(1 - traj_batch.mask))
                        
                        critic_loss = t_config["VF_COEF"] * (value_loss_sparse + value_loss_dense)
                    else:
                        critic_loss = t_config["VF_COEF"] * value_loss.mean(where=(1 - traj_batch.mask))
                    
                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    if env.do_sep_reward:
                        gae = gae.sum(axis=-1)
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
                    
                    # debug
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

            init_hstate = jnp.reshape(
                init_hstate, (t_config["HIDDEN_SIZE"], t_config["NUM_ACTORS"])
            )
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
                    "train-term": metric["terminations"],
                    #"reward": metric["returned_episode_returns"],
                    
                    # "eval-collision": metric["test-metrics"]["collision-by-env"].mean(),
                    # "eval-timeout": metric["test-metrics"]["timeout-by-env"].mean(),
                    "env_step": metric["update_steps"]
                        * t_config["NUM_ENVS"]
                        * t_config["NUM_STEPS"],
                    "dormancy/": metric["dormancy"],
                    "env-metrics/": metric["env-metrics"],
                    # "mean_ued_score": metric["mean_ued_score"],
                    **metric["episodic_return_length"],
                    **metric["loss_info"],
                    "mean_lambda_val": metric["mean_lambda_val"],
                }
            )

        dormancy_log = {
            "actor": dormancy.actor,
            "embedding": dormancy.embedding,
            "hidden": dormancy.hidden,
            "rnnout": dormancy.rnnout,
            "critic": dormancy.critic,
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
        metric["terminations"] = {k: traj_batch.info[k] for k in ["NumC", "GoalR", "AgentC", "MapC", "TimeO"]}
        metric["terminations"] = jax.tree_map(lambda x: x.sum(), metric["terminations"])
        metric["dormancy"] = dormancy_log
        metric["env-metrics"] = jax.tree_map(lambda x: x.mean(), jax.vmap(env.get_env_metrics)(start_state))
        metric["mean_lambda_val"] = env_state.rew_lambda.mean()
        jax.experimental.io_callback(callback, None, metric)
        
        # SAMPLE NEW ENVS
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, t_config["NUM_ENVS_TO_GENERATE"])
        obsv_gen, env_state_gen = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        rng, _rng = jax.random.split(rng)
        sampled_env_instances_idxs = jax.random.randint(_rng, (t_config["NUM_ENVS_FROM_SAMPLED"],), 0, num_env_instances)
        sampled_env_instances = jax.tree_map(lambda x: x.at[sampled_env_instances_idxs].get(), instances)
        obsv_sampled, env_state_sampled = jax.vmap(env.set_env_instance, in_axes=(0,))(sampled_env_instances)
        
        obsv = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), obsv_gen, obsv_sampled)
        env_state = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), env_state_gen, env_state_sampled)
        
        start_state = env_state
        hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
        
        update_steps = update_steps + 1
        runner_state = (train_state, env_state, start_state, obsv, jnp.zeros((t_config["NUM_ACTORS"]), dtype=bool), hstate, update_steps, rng)
        return (runner_state, instances), metric
    
    def log_buffer(learnability, states, epoch):
        num_samples = states.pos.shape[0]
        rows = 2 
        fig, axes = plt.subplots(rows, int(num_samples/rows), figsize=(20, 10))
        axes=axes.flatten()
        for i, ax in enumerate(axes):
            # ax.imshow(train_state.plr_buffer.get_sample(i))
            score = learnability[i]            
            state = jax.tree_map(lambda x: x[i], states)
                        
            env.init_render(ax, state, lidar=False, ticks_off=True)
            ax.set_title(f'learnability: {score:.3f}')
            ax.set_aspect('equal', 'box')
                        
        plt.tight_layout()
        fig.canvas.draw()
        im = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()) 
        wandb.log({"maps": wandb.Image(im)}, step=epoch)
    
    @jax.jit
    def train_and_eval_step(runner_state, eval_rng):
        
        learnability_rng, eval_singleton_rng, eval_sampled_rng = jax.random.split(eval_rng, 3)
        # TRAIN
        learnabilty_scores, instances = get_learnability_set(learnability_rng, runner_state[0].params)
        runner_state_instances = (runner_state, instances)
        runner_state_instances, metrics = jax.lax.scan(train_step, runner_state_instances, None, t_config["EVAL_FREQ"])
        # EVAL
        
        test_metrics = {
            "learnability_set_scores": learnabilty_scores,
            "learnability_set_mean_score": learnabilty_scores.mean(),
        }
        test_metrics["singleton-test-metrics"] = eval_singleton_runner.run(eval_singleton_rng, runner_state[0].params)
        test_metrics["sampled-test-metrics"] = eval_sampled_runner.run(eval_sampled_rng, runner_state[0].params)
        
        runner_state, _ = runner_state_instances
        test_metrics["update_count"] = runner_state[-2]
        
        top_instances = jax.tree_map(lambda x: x.at[-20:].get(), instances)
        _, top_states = jax.vmap(env.set_env_instance)(top_instances)
        
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
