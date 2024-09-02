# Adapted from PureJaxRL implementation and minigrid baselines, source:
# https://github.com/lupuandr/explainable-policies/blob/50acbd777dc7c6d6b8b7255cd1249e81715bcb54/purejaxrl/ppo_rnn.py#L4
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import os
import shutil
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import jax
import jax.experimental
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import orbax
import pyrallis
import wandb
import xminigrid
from flax.jax_utils import replicate, unreplicate
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from nn import ActorCriticRNN
from utils import Transition, calculate_gae, ppo_update_networks, rollout, save_params, rollout_nsteps
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import Environment, EnvParams
from xminigrid.wrappers import GymAutoResetWrapper

# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)


@dataclass
class TrainConfig:
    project: str = "xminigrid"
    mode: str = "disabled"
    group: str = "medium-13-sfl"
    env_id: str = "XLand-MiniGrid-R4-13x13"
    benchmark_id: str = "high-3m"
    img_obs: bool = False
    # agent
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 1024
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    # training
    num_envs: int = 8192
    num_steps_per_env: int = 4096
    num_steps_per_update: int = 32
    update_epochs: int = 1
    num_minibatches: int = 16
    total_timesteps: int = 5_700_000_000
    lr: float = 0.001
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    #eval
    eval_num_envs: int = 512
    eval_num_episodes: int = 10
    eval_seed: int = 42
    train_seed: int = 42
    checkpoint_path: Optional[str] = "checkpoints"
    #sfl
    # sfl_num_episodes: int = 10
    sfl_rollout_factor: int = 10  # how many times more steps to rollout than the max_steps
    sfl_buffer_size: int = 8192
    sfl_batch_size: int = 40000
    sfl_num_batches: int = 1
    sfl_buffer_refresh_freq: int = 1
    sfl_num_envs_to_sample: int = 8192
    #logging
    log_num_images: int = 20  # number of images to log
    log_images_count: int = 16 # number of times to log images during training
    

    def __post_init__(self):
        num_devices = jax.local_device_count()
        # splitting computation across all available devices
        assert num_devices == 1, "Only single device training is supported."
        assert self.sfl_num_envs_to_sample <= self.num_envs, "SFL sample envs should be less than or equal to total envs"
        self.sfl_num_envs_to_generate = self.num_envs - self.sfl_num_envs_to_sample
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_num_envs_per_device = self.eval_num_envs // num_devices
        assert self.num_envs % num_devices == 0
        self.num_meta_updates = round(
            self.total_timesteps_per_device / (self.num_envs_per_device * self.num_steps_per_env)
        )
        print('num meta updates', self.num_meta_updates)
        self.num_outer_steps = self.num_meta_updates // self.sfl_buffer_refresh_freq
        self.num_meta_updates = self.num_outer_steps * self.sfl_buffer_refresh_freq
        self.log_images_update = self.num_meta_updates // self.log_images_count
        print('logging images every', self.log_images_update)
        print('num outer steps', self.num_outer_steps)
        self.num_inner_updates = self.num_steps_per_env // self.num_steps_per_update
        print('num inner updates', self.num_inner_updates)
        assert self.num_steps_per_env % self.num_steps_per_update == 0
        print(f"Num devices: {num_devices}, Num meta updates: {self.num_meta_updates}")


def make_states(config: TrainConfig):
    # for learning rate scheduling
    def linear_schedule(count):
        total_inner_updates = config.num_minibatches * config.update_epochs * config.num_inner_updates
        frac = 1.0 - (count // total_inner_updates) / config.num_meta_updates
        return config.lr * frac

    # setup environment
    if "XLand" not in config.env_id:
        raise ValueError("Only meta-task environments are supported.")

    env, env_params = xminigrid.make(config.env_id)
    env = GymAutoResetWrapper(env)

    # enabling image observations if needed
    if config.img_obs:
        from xminigrid.experimental.img_obs import RGBImgObservationWrapper

        env = RGBImgObservationWrapper(env)

    # loading benchmark
    benchmark = xminigrid.load_benchmark(config.benchmark_id)

    # set up training state
    rng = jax.random.key(config.train_seed)
    rng, _rng = jax.random.split(rng)

    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=config.img_obs,
    )
    # [batch_size, seq_len, ...]
    init_obs = {
        "observation": jnp.zeros((config.num_envs_per_device, 1, *env.observation_shape(env_params))),
        "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)

    network_params = network.init(_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    return rng, env, env_params, benchmark, init_hstate, train_state


def make_train(
    env: Environment,
    env_params: EnvParams,
    benchmark: Benchmark,
    config: TrainConfig,
):
    
    def log_levels(init_timestep, rulesets, env_params, step):
        
        log_dict = {}
        for i in range(rulesets.rules.shape[0]):
            r = jax.tree.map(lambda x: x[i], rulesets)
            env_params = env_params.replace(ruleset=r)
            t = jax.tree.map(lambda x: x[i], init_timestep)
            img = env.render(env_params, t)
            log_dict.update({f"images/{i}_level": wandb.Image(np.array(img))})
        print('step', step)
        wandb.log(log_dict, step=step)
    
    @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        train_state: TrainState,
        init_hstate: jax.Array,
    ):
        def _sample_learnability_buffer(rng, train_state):
            
            def _batch_step(unused, batch_rng):
                ruleset_rng, rollout_rng = jax.random.split(batch_rng)
                # sample rulesets
                ruleset_rng = jax.random.split(ruleset_rng, num=config.sfl_batch_size)
                rulesets = jax.vmap(benchmark.sample_ruleset)(ruleset_rng)
                rollout_env_params = env_params.replace(ruleset=rulesets)
                
                rollout_rng = jax.random.split(rollout_rng, num=config.sfl_batch_size)
                rollout_stats = jax.vmap(rollout_nsteps, in_axes=(0, None, 0, None, None, None))(
                    rollout_rng,
                    env,
                    rollout_env_params,
                    train_state,
                    jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                    env_params.max_steps * config.sfl_rollout_factor,
                )
                return None, (rulesets, rollout_stats)
            
            batch_rng = jax.random.split(rng, num=config.sfl_num_batches)
            _, (rulesets, rollout_stats) = jax.lax.scan(_batch_step, None, batch_rng)
            rollout_stats = jax.tree_map(lambda x: x.reshape(-1), rollout_stats)
            sucess_rate = rollout_stats.success / rollout_stats.episodes
            learnability = sucess_rate * (1 - sucess_rate)
            print('rollout stats', rollout_stats)
            print('rulesets', rulesets)
            flat_rulesets = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), rulesets)
            print('flat rulesets', flat_rulesets)
            
            top_learnability = jnp.argsort(learnability)[-config.sfl_buffer_size:]
            top_rulesets = jax.tree_map(lambda x: x.at[top_learnability].get(), flat_rulesets) 
            
            info = {
                "buffer_learnability_scores": learnability.at[top_learnability].get(),
                "buffer_success": sucess_rate.at[top_learnability].get(),
                "all_sampled_success": sucess_rate,
                "ruleset_mean_num_rules": jnp.mean(jnp.sum(jnp.where(top_rulesets.rules.at[:,0].get() > 0, 1, 0), axis=1))
            }
            
            return top_rulesets, info
                           
        
        # META TRAIN LOOP
        def _meta_step(meta_state, update_idx):
            rng, train_state, sfl_buffer = meta_state

            # INIT ENV
            rng, _rng1, _rng2, _rng3 = jax.random.split(rng, num=4)
            
            # sample rulesets for this meta update
            ruleset_gen_rng = jax.random.split(_rng1, num=config.sfl_num_envs_to_generate)
            rulesets_gen = jax.vmap(benchmark.sample_ruleset)(ruleset_gen_rng)
            
            # sample from sfl buffer
            rulesets_sampled_idxs = jax.random.randint(_rng2, (config.sfl_num_envs_to_sample,), 0, config.sfl_buffer_size)
            sampled_rules = jax.tree_map(lambda x: x.at[rulesets_sampled_idxs].get(), sfl_buffer)
            rulesets = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), rulesets_gen, sampled_rules)
            
            meta_env_params = env_params.replace(ruleset=rulesets)

            reset_rng = jax.random.split(_rng3, num=config.num_envs_per_device)
            timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_rng)
            prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
            prev_reward = jnp.zeros(config.num_envs_per_device)

            outcomes = jnp.zeros((config.num_envs_per_device, 2))
            
            # INNER TRAIN LOOP
            def _update_step(runner_state, _):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, _):
                    rng, train_state, prev_timestep, prev_action, prev_reward, outcomes, prev_hstate = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    dist, value, hstate = train_state.apply_fn(
                        train_state.params,
                        {
                            # [batch_size, seq_len=1, ...]
                            "observation": prev_timestep.observation[:, None],
                            "prev_action": prev_action[:, None],
                            "prev_reward": prev_reward[:, None],
                        },
                        prev_hstate,
                    )
                    action, log_prob = dist.sample_and_log_prob(seed=_rng)
                    # squeeze seq_len where possible
                    action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)

                    # STEP ENV
                    timestep = jax.vmap(env.step, in_axes=0)(meta_env_params, prev_timestep, action)
                    success = timestep.discount == 0.0
                    outcomes = outcomes.at[:, 0].add(jnp.where(timestep.last(), 1, 0))
                    outcomes = outcomes.at[:, 1].add(jnp.where(success, 1, 0))
                    
                    transition = Transition(
                        # ATTENTION: done is always false, as we optimize for entire meta-rollout
                        done=jnp.zeros_like(timestep.last()),
                        action=action,
                        value=value,
                        reward=timestep.reward,
                        log_prob=log_prob,
                        obs=prev_timestep.observation,
                        prev_action=prev_action,
                        prev_reward=prev_reward,
                    )
                    runner_state = (rng, train_state, timestep, action, timestep.reward, outcomes, hstate)
                    return runner_state, transition

                initial_hstate = runner_state[-1]
                # transitions: [seq_len, batch_size, ...]
                runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps_per_update)

                # CALCULATE ADVANTAGE
                rng, train_state, timestep, prev_action, prev_reward, outcomes, hstate = runner_state
                # calculate value of the last step for bootstrapping
                _, last_val, _ = train_state.apply_fn(
                    train_state.params,
                    {
                        "observation": timestep.observation[:, None],
                        "prev_action": prev_action[:, None],
                        "prev_reward": prev_reward[:, None],
                    },
                    hstate,
                )
                advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)

                # UPDATE NETWORK
                def _update_epoch(update_state, _):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, transitions, advantages, targets = batch_info
                        new_train_state, update_info = ppo_update_networks(
                            train_state=train_state,
                            transitions=transitions,
                            init_hstate=init_hstate.squeeze(1),
                            advantages=advantages,
                            targets=targets,
                            clip_eps=config.clip_eps,
                            vf_coef=config.vf_coef,
                            ent_coef=config.ent_coef,
                        )
                        return new_train_state, update_info

                    rng, train_state, init_hstate, transitions, advantages, targets = update_state

                    # MINIBATCHES PREPARATION
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                    # [seq_len, batch_size, ...]
                    batch = (init_hstate, transitions, advantages, targets)
                    # [batch_size, seq_len, ...], as our model assumes
                    batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                    shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                    # [num_minibatches, minibatch_size, ...]
                    minibatches = jtu.tree_map(
                        lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                    )
                    train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

                    update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
                    return update_state, update_info

                # hstate shape: [seq_len=None, batch_size, num_layers, hidden_dim]
                update_state = (rng, train_state, initial_hstate[None, :], transitions, advantages, targets)
                update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
                # WARN: do not forget to get updated params
                rng, train_state = update_state[:2]

                # averaging over minibatches then over epochs
                loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
                runner_state = (rng, train_state, timestep, prev_action, prev_reward, outcomes, hstate)
                return runner_state, loss_info

            # on each meta-update we reset rnn hidden to init_hstate
            runner_state = (rng, train_state, timestep, prev_action, prev_reward, outcomes, init_hstate)
            runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.num_inner_updates)
            # WARN: do not forget to get updated params
            rng, train_state = runner_state[:2]
            outcomes = runner_state[-2]
            success_rate = outcomes.at[:, 1].get() / outcomes.at[:, 0].get()
            # EVALUATE AGENT
            eval_ruleset_rng, eval_reset_rng = jax.random.split(jax.random.key(config.eval_seed))
            eval_ruleset_rng = jax.random.split(eval_ruleset_rng, num=config.eval_num_envs_per_device)
            eval_reset_rng = jax.random.split(eval_reset_rng, num=config.eval_num_envs_per_device)

            eval_ruleset = jax.vmap(benchmark.sample_ruleset)(eval_ruleset_rng)
            eval_env_params = env_params.replace(ruleset=eval_ruleset)

            eval_stats = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None))(
                eval_reset_rng,
                env,
                eval_env_params,
                train_state,
                # TODO: make this a static method?
                jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                config.eval_num_episodes,
            )

            rulesets_to_log = jax.tree.map(lambda x: x.at[:config.log_num_images].get(), rulesets)
            timesteps_to_log = jax.tree.map(lambda x: x.at[:config.log_num_images].get(), timestep)
            # jax.experimental.io_callback(log_levels, None, timesteps_to_log, rulesets_to_log, env_params, update_idx)
            
            jax.lax.cond(
                update_idx % config.log_images_update == 0,
                lambda *_: jax.experimental.io_callback(log_levels, None, timesteps_to_log, rulesets_to_log, env_params, update_idx),
                lambda *_: None,
            )

            # averaging over inner updates, adding evaluation metrics
            loss_info = jtu.tree_map(lambda x: x.mean(-1), loss_info)
            loss_info.update(
                {
                    "eval/returns_mean": eval_stats.reward.mean(0),
                    "eval/returns_median": jnp.median(eval_stats.reward),
                    "eval/lengths": eval_stats.length.mean(0),
                    "eval/success_rate_mean": jnp.mean(eval_stats.success/eval_stats.episodes),
                    "eval/lengths_20percentile": jnp.percentile(eval_stats.length, q=20),
                    "eval/returns_20percentile": jnp.percentile(eval_stats.reward, q=20),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                    "outcomes": success_rate,
                    "num_env_steps": update_idx * config.num_inner_updates * config.num_steps_per_update * config.num_envs,
                    "update_step": update_idx,
                }
            )
            
            def _callback(info):
                wandb.log(
                    info,
                    step=info["update_step"]
                )
            
            jax.experimental.io_callback(_callback, None, loss_info)
            
            meta_state = (rng, train_state, sfl_buffer)
            return meta_state, loss_info

        init_sfl_buffer, _ = _sample_learnability_buffer(rng, train_state)
        meta_state = (rng, train_state, init_sfl_buffer)
        
        def _outer_step(meta_state, outer_idx):
            rng, train_state, _ = meta_state

            rng, _rng = jax.random.split(rng)
            sfl_buffer, learnability_info = _sample_learnability_buffer(_rng, train_state)
            def __buffer_callback(x):
                info, step = x
                wandb.log(info, step=step)
            
            inner_idx = jnp.arange(config.sfl_buffer_refresh_freq) + (outer_idx)*config.sfl_buffer_refresh_freq
            jax.experimental.io_callback(__buffer_callback, None, (learnability_info, inner_idx.at[0].get()))
            meta_state, loss_info = jax.lax.scan(_meta_step, (rng, train_state, sfl_buffer), inner_idx, config.sfl_buffer_refresh_freq)
            return meta_state, (loss_info, learnability_info)
        
        meta_state, (loss_info, learnability_info) = jax.lax.scan(_outer_step, meta_state, jnp.arange(config.num_outer_steps), config.num_outer_steps)
        loss_info = jax.tree_map(lambda x: x.flatten(), loss_info)
        return {"state": meta_state[-2], "loss_info": loss_info, **learnability_info}

    return train


@pyrallis.wrap()
def train(config: TrainConfig):
    # logging to wandb
    run = wandb.init(
        project=config.project,
        group=config.group,
        config=asdict(config),
        save_code=True,
        mode=config.mode,
    )
    # removing existing checkpoints if any
    if config.checkpoint_path is not None and os.path.exists(config.checkpoint_path):
        shutil.rmtree(config.checkpoint_path)

    rng, env, env_params, benchmark, init_hstate, train_state = make_states(config)
    # replicating args across devices
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())

    print("Compiling...")
    t = time.time()
    train_fn = make_train(env, env_params, benchmark, config)
    train_fn = train_fn.lower(rng, train_state, init_hstate).compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()
    train_info = jax.block_until_ready(train_fn(rng, train_state, init_hstate))
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Logginig...")
    loss_info = unreplicate(train_info["loss_info"])

    # total_transitions = 0
    # for i in range(config.num_outer_steps):
    #     for j in range(config.sfl_buffer_refresh_freq):
    #         k = i * config.sfl_buffer_refresh_freq + j
    #         total_transitions += config.num_steps_per_env * config.num_envs_per_device * jax.local_device_count()
    #         info = jtu.tree_map(lambda x: x[k].item(), loss_info)
    #         info["transitions"] = total_transitions
    #         wandb.log(info)
            
    #     buffer_scores = loss_info["buffer_learnability_scores"][i]
    #     wandb.log({"buffer_learnability_scores": buffer_scores}, step=total_transitions)
        

    run.summary["training_time"] = elapsed_time
    run.summary["steps_per_second"] = (config.total_timesteps_per_device * jax.local_device_count()) / elapsed_time

    if config.checkpoint_path is not None:
        params = train_info["state"].params
        save_dir = os.path.join(config.checkpoint_path, run.name)
        
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of saved in {save_dir}/model.safetensors')
        
        # upload this to wandb as an artifact   
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()
        # checkpoint = {"config": asdict(config), "params": unreplicate(train_info)["state"].params}
        # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # save_args = orbax_utils.save_args_from_target(checkpoint)
        # orbax_checkpointer.save(config.checkpoint_path, checkpoint, save_args=save_args)

    print("Final return: ", float(loss_info["eval/returns_mean"][-1]))
    run.finish()


if __name__ == "__main__":
    train()
