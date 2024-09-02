"""
Sample a set of environments with success rates across a given number of bins.
i.e. 50 with 0-10% success, 50 with 10-20% and so on.
This set is then used when calculating regret scores. 
"""

import jax 
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pickle
import wandb
from typing import Sequence, NamedTuple, Any, Dict
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from functools import partial
import distrax

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav, EnvInstance, listify_reward
from sfl.train.train_utils import load_params
from sfl.util.rolling_stats import LogEpisodicStats
from sfl.train.common.network import ScannedRNN, _calculate_dormancy, DormancyActorCriticRNN

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
        
        # Critic predicts two things: Value Sparse and Value Dense, per agent.
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        dormancy = DormancyActorCriticRNN(
            actor=ad1,
            embedding=ed1,
            hidden=hd1, 
            rnnout=ed2,
            critic=cd1
        )
        
        #jax.debug.print('dormancy {d}', d=dormancy)

        return hidden, pi, critic, dormancy

class Transition(NamedTuple):
    global_done: jnp.ndarray
    last_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def main():
    
    NUM_ENVS_TO_SAMPLE =  1000
    N_BINS = 10
    N_ENVS_PER_BIN = 250
    bin_counts = np.zeros(N_BINS)
    success_thresholds = np.linspace(0.0, 1.0, N_BINS+1)
    success_thresholds = np.dstack([success_thresholds[:-1], success_thresholds[1:]]).squeeze()
    success_thresholds[0,0] = -np.inf
    
    
    run_name = "alex-plus/multi_robot_ued/5yvzkn8p"
    
    data_dir = "sfl/data/test_sets/"
    api = wandb.Api()
    run = api.run(run_name)
    config = run.config
    
    file_name = f'sampled_tc_uniform'
    # LOAD PARAMS
    model_artificat = api.artifact(f"alex-plus/{run.project}/{run.name}-checkpoint:latest")
    name = model_artificat.download()
    params = load_params(name + "/model.safetensors")

    
    config["env"]["env_params"]["info_by_agent"] =True,
    env = JaxNav(num_agents=config["env"]["num_agents"], **config["env"]["env_params"])
    t_config = config["learning"]
    t_config["NUM_ENVS"] = NUM_ENVS_TO_SAMPLE 
    t_config["NUM_ACTORS"] = env.num_agents * t_config["NUM_ENVS"]
    t_config["NUM_STEPS"] = env.max_steps * N_BINS 

    rng = jax.random.PRNGKey(1)
    
    t_config["LOG_DORMANCY"] = False
    t_config["USE_LAYER_NORM"] = False
    network = ActorCriticRNN(
        action_dim=env.agent_action_space().shape[0],
        fc_dim_size=t_config["FC_DIM_SIZE"],
        hidden_size=t_config["HIDDEN_SIZE"],
        use_layer_norm=t_config["USE_LAYER_NORM"]
    )
    kept_instances = []
    kept_success_by_env = []
    kept_success_by_actor = []
    kept_learn_prob_by_env = []
    
    print(f'Sampling {NUM_ENVS_TO_SAMPLE} environments with {env.num_agents} agents each')
    for i in tqdm.tqdm(range(200)):
        if np.all(bin_counts >= N_ENVS_PER_BIN):
            break
        
        rng, rngs = jax.random.split(rng)
        rngs = jax.random.split(rngs, NUM_ENVS_TO_SAMPLE)
        obs, env_states = jax.vmap(env.reset, in_axes=(0))(rngs)
        
        zero_carry = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
        rolling_stats = LogEpisodicStats(names=env.get_monitored_metrics())
        eval_stats = jax.tree_map(lambda x: x.squeeze(), rolling_stats.reset_stats(batch_shape=(t_config["NUM_ACTORS"],)))
        
        runner_state = (
            env_states,
            env_states,
            obs, 
            jnp.zeros(t_config["NUM_ACTORS"], dtype=jnp.bool_),
            zero_carry,
            rng,
            eval_stats,
        )
            
        def _env_step(runner_state, unused):
            env_state, start_state, last_obs, last_done, hstate, rng, eval_stats = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = batchify(last_obs, env.agents, t_config["NUM_ACTORS"])
            ac_in = (
                obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            hstate, pi, value, _ = network.apply(params, hstate, ac_in)
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
            done_batch = batchify(done, env.agents, t_config["NUM_ACTORS"]).squeeze()
            if env.do_sep_reward:
                reward_batch = listify_reward(reward, do_batchify=True)
            else: 
                reward_batch = batchify(reward, env.agents, t_config["NUM_ACTORS"]).squeeze()
            # rewards_together = reward_batch.sum(axis=-1)
            ep_done = jnp.tile(done["__all__"], env.num_agents)
            # print('info shape', info["NumC"].shape)
            info = jax.tree_map(lambda x: x.reshape((t_config["NUM_ACTORS"],), order="F"), info)
            transition = Transition(
                ep_done,
                last_done,
                done_batch,
                action.squeeze(),
                value.squeeze(),
                reward_batch,
                log_prob.squeeze(),
                obs_batch,
                info,
            )        
            eval_stats = rolling_stats.update_stats(eval_stats, done_batch, info, 1)
            runner_state = (env_state, start_state, obsv, done_batch, hstate, rng, eval_stats)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, t_config["NUM_STEPS"]
        )
        env_state, start_state, last_obs, last_done, hstate, rng, eval_stats = runner_state
        
        # plot over this set:
        #  - success rate by env
        #  - MaxMC score by env
        #  - pvl score by env
        #  - success rate by agent
        
        # SOLVED RATE BY AGENT
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
            
            done_idxs = jnp.argwhere(dones, size=50, fill_value=max_steps).squeeze()
            mask_done = jnp.where(done_idxs == max_steps, 0, 1)
            ep_return, success, collision, timeo, length = __ep_outcomes(jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)        
                    
            return {"ep_return": ep_return.mean(where=mask_done),
                    "num_episodes": mask_done.sum(),
                    "success_rate": success.mean(where=mask_done),
                    "collision_rate": collision.mean(where=mask_done),
                    "timeout_rate": timeo.mean(where=mask_done),
                    "ep_len": length.mean(where=mask_done),
                    }
            
        # print('num c shape', traj_batch.info["NumC"].shape)
        
        outcomes = _calc_outcomes_by_agent(t_config["NUM_STEPS"], traj_batch.done, traj_batch.reward, traj_batch.info)
        outcomes_by_env = jax.tree_map(lambda x: x.reshape((env.num_agents, t_config["NUM_ENVS"])).swapaxes(0,1), outcomes)
        
        success_by_actor = outcomes["success_rate"]
        learn_prob = (success_by_actor * (1 - success_by_actor)).reshape((env.num_agents, t_config["NUM_ENVS"])).sum(axis=0)
        mean_success_by_level = success_by_actor.reshape((env.num_agents, t_config["NUM_ENVS"])).mean(axis=0)
        kept_idx = []
        for i in range(N_BINS):
            if bin_counts[i] >= N_ENVS_PER_BIN:
                continue
            in_range = jnp.argwhere((mean_success_by_level > success_thresholds[i,0]) & (mean_success_by_level <= success_thresholds[i,1])).squeeze(axis=-1)
            if in_range.size == 0:
                continue
            # print('in range size', in_range.size)
            # print('in range shape', in_range.shape)
            # print('num in range', in_range.shape, success_thresholds[i])
            num_to_take = int(min(N_ENVS_PER_BIN - bin_counts[i], in_range.shape[0]))
            # print('num to take', num_to_take)
            in_range = in_range[:num_to_take]
            
            bin_counts[i] += in_range.shape[0]
            kept_idx.append(in_range)
            
        if len(kept_idx) == 0:
            continue
        
        kept_idx = jnp.concatenate(kept_idx)
        # print('success probs', learn_prob.shape, learn_prob)
        # print('outcomes by env', outcomes_by_env["success_rate"].shape, outcomes_by_env["success_rate"])
        # print('num learn prob over 0', jnp.sum(learn_prob > 0))
        # print('mean success by level', mean_success_by_level.shape, mean_success_by_level)
        # fig, ax = plt.subplots()
        # ax.hist(mean_success_by_level, bins=10)
        # plt.savefig("jax_multirobsim/deploy/data/" + f'{run.name}_success_hist.png')
        # plt.close()
        
        # fig, ax = plt.subplots()
        # ax.hist(learn_prob, bins=10)
        # plt.savefig("jax_multirobsim/deploy/data/" + f'{run.name}_success_hist2.png')
        # plt.close()
        
        # kept_idx = jnp.argwhere(mean_success_by_level < 1.0).squeeze(axis=-1)
        instances = EnvInstance(
            agent_pos=env_states.pos[kept_idx],
            agent_theta=env_states.theta[kept_idx],
            goal_pos=env_states.goal[kept_idx],
            map_data=env_states.map_data[kept_idx],
            rew_lambda=env_states.rew_lambda[kept_idx],
        )
        learn_prob_by_env = learn_prob[kept_idx]
        success_by_env = outcomes_by_env["success_rate"][kept_idx]
        # if kept_idx.shape[0] == 1:
        #     instances = jax.tree_map(lambda x: x[None], instances)
        #     success_by_env = success_by_env[None]
        #     learn_prob_by_env = learn_prob_by_env[None]
        kept_instances.append(instances)
        kept_success_by_env.append(success_by_env)
        kept_learn_prob_by_env.append(learn_prob_by_env)
    print('final buffer counts', bin_counts)
    
    # concat instances
    instances = EnvInstance(
        agent_pos=jnp.concatenate([i.agent_pos for i in kept_instances]),
        agent_theta=jnp.concatenate([i.agent_theta for i in kept_instances]),
        goal_pos=jnp.concatenate([i.goal_pos for i in kept_instances]),
        map_data=jnp.concatenate([i.map_data for i in kept_instances]),
        rew_lambda=jnp.concatenate([i.rew_lambda for i in kept_instances]),
    )
    print('instances', instances.agent_pos.shape, instances.agent_theta.shape, instances.goal_pos.shape, instances.map_data.shape, instances.rew_lambda.shape)
    successes_by_env = jnp.concatenate(kept_success_by_env, axis=0)
    learn_prob_by_env = jnp.concatenate(kept_learn_prob_by_env, axis=0)
    # print('successes_by_env', successes_by_env.shape)
    num_instances_saved = instances.agent_pos.shape[0]
    # print('learn_prob_by_env', learn_prob_by_env.shape, learn_prob_by_env)
    # ONLY SAVE SET NUMBER
    # num_envs_to_save = 100 
    # num_instances_saved = num_envs_to_save
    # instances = jax.tree_map(lambda x: x[:num_envs_to_save], instances)
    # successes_by_env = successes_by_env[:num_envs_to_save]
    # learn_prob_by_env = learn_prob_by_env[:num_envs_to_save]
    
    num_envs_to_plot = 100 if num_instances_saved > 100 else num_instances_saved
    idx_to_plot = jnp.argsort(learn_prob_by_env).squeeze()[-num_envs_to_plot:]
    print('idx_to_plot', idx_to_plot)
    
    fig, ax = plt.subplots(1, 2)
    ax1, ax2 = ax.flatten()
    
    ax1.hist(learn_prob_by_env, bins=10)
    ax1.set_xlabel("env learn prob")
    ax2.hist(successes_by_env.flatten(), bins=10)
    ax2.set_xlabel('individual agent success rate')
    plt.savefig(data_dir + f'{run.name}_saved_hist.png')
    
    
    file_name = file_name + f'{num_instances_saved}e_{env.num_agents}a' 
    print(f'Saving {num_instances_saved} instances, to: {file_name}')    
    
    viz_maps = False
    if viz_maps:
        
        # vizualise in batches of 40, with 4 rows, 10 columns
        num_frames = np.ceil(num_envs_to_plot / 40).astype(int)
        print('num_frames', num_frames)
        for i in tqdm.tqdm(range(num_frames)):
            i_range = range(i*40, min((i+1)*40, num_envs_to_plot))
            print('i_range', i_range)
            
            fig, ax = plt.subplots(4, 10, figsize=(40, 30))
            for j, idx in enumerate(i_range):
                env_idx = idx_to_plot[idx]
                instance = jax.tree_map(lambda x: x[env_idx], instances)
                _, state = env.set_env_instance(instance)
                env.init_render(ax[j//10, j%10], state, lidar=False, colour_agents_by_idx=True)
                title = f'env_{idx}\n' + \
                    f"learn prob: {learn_prob_by_env[env_idx]:.3f}\n" + \
                    f"s: {np.array2string(successes_by_env[env_idx], precision=3)}"
                ax[j//10, j%10].set_title(title)
                ax[j//10, j%10].axis('off')
                
            plt.savefig(data_dir + f'{file_name}_{i}.png')
    
    # use pickle
    file_name += '.pkl'
    save_path = data_dir + file_name
    with open(save_path, 'wb') as f:
        pickle.dump(instances, f)


if __name__=="__main__":
    main()
    