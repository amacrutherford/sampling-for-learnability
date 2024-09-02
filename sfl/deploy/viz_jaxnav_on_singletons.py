""" 
Load a local checkpoint and run on map test set.
"""
import jax 
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
import flax.linen as nn
import numpy as np 
import functools
from typing import Sequence, NamedTuple, Any, Tuple
import distrax
import wandb 

from jaxmarl.environments.jaxnav import JaxNav

from sfl.train.train_utils import load_params
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from sfl.runners import EvalSingletonsRunner

    
def main():
    # run_id = "amacrutherford/ma_ippo_plr/om2y9k9i" # "amacrutherford/ma_ippo_plr/imcl54h7" 
    run_id = "alex-plus/multi_robot_ued/m4bcdtmb"
    api = wandb.Api()
    run = api.run(run_id)
    config = run.config
    config["test_set"] = "multi"
    
    run_name = run.name
    print(f'-- deploying {run_name} on {config["test_set"]} test set--')
    project_name = run.project
    
    rng = jax.random.PRNGKey(100)
    # config["NUM_ENVS"] = 500
    # config["NUM_ACTORS"] = config["NUM_ENVS"]
    # config["NUM_STEPS"] = 500
    # config["env_params"]["max_steps"] = 1000
    
    # config["env_params"]["evaporating"] = False  # Toggle evaporation
    init_env = JaxNav(num_agents=config["env"]["num_agents"],
                            **config["env"]["env_params"])
    t_config = config["learning"]
    t_config["LOG_DORMANCY"] = True
    t_config["USE_LAYER_NORM"] = False
    network = ActorCriticRNN(init_env.agent_action_space().shape[0],
                             config=t_config)
        
    eval_runner = EvalSingletonsRunner(
        config["test_set"],
        network,
        init_carry=ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"],
        env_kwargs=config["env"]["env_params"],
        n_episodes=2,
    )

    model_artificat = api.artifact(f"{run.entity}/{run.project}/{run.name}-checkpoint:latest") # NOTE hardcoded
    name = model_artificat.download()
    network_params = load_params(name + "/model.safetensors")
    with jax.disable_jit(False):
        stats = eval_runner.run_and_visualise(rng, network_params, run_name)
    print('stats', stats)
    
if __name__=="__main__":
    main()