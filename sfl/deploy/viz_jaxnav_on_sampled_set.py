""" 
Deploy onto a randomly generated set.

NOTE: not complete
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp 
import wandb 
import tqdm 
import matplotlib.pyplot as plt
from flax.traverse_util import flatten_dict, unflatten_dict

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav, EnvInstance

from sfl.runners.eval_runner import EvalSampledRunner
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from sfl.train.train_utils import load_params


def save_maps_to_table(env: JaxNav, env_instances: EnvInstance):
  num_envs = env_instances.map_data.shape[0]
  _, states = jax.vmap(env.set_env_instance, in_axes=(0))(env_instances)
  fig, ax = plt.subplots()
  images = []
  for i in tqdm.tqdm(range(num_envs), desc="Rendering maps"):
      state = jax.tree_map(lambda x: x[i], states)
      ax.clear()
      env.init_render(ax, state, lidar=False)
      fig.canvas.draw()
      
      image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
      image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
      images.append(wandb.Image(image))
      
  ids = np.arange(num_envs)
  zipped_data = list(zip(ids, images))
  
  table = wandb.Table(
      columns=["id", "maps"],
      data=zipped_data,
  )
  return table


def get_data_for_run(run_name: str, env_instances: EnvInstance):
  
  num_agents = env_instances.agent_pos.shape[1]
  num_envs = env_instances.agent_pos.shape[0]
  
  api = wandb.Api()
  run = api.run(run_name)
  config = run.config
  run_name = run.name
  proj = run.project
  print('run name', run_name)

  # check online for artifact

  env = JaxNav(num_agents=num_agents, do_sep_reward=False, **config["env"]["env_params"])
  obs, states = jax.vmap(env.set_env_instance, in_axes=(0))(env_instances)
  print('pos shape', states.pos.shape)
  # LOAD PARAMS
  model_artificat = api.artifact(f"{run.entity}/{run.project}/{run.name}-checkpoint:latest") # NOTE hardcoded
  name = model_artificat.download()
  network_params = load_params(name + "/model.safetensors")

  config["learning"]["LOG_DORMANCY"] = True
  config["learning"]["USE_LAYER_NORM"] = False
  network = ActorCriticRNN(
    action_dim=env.action_space().shape[1],
    config=config["learning"],
  )
  rng = jax.random.PRNGKey(10)
  

  rng, _rng = jax.random.split(rng)
  runner = EvalSampledRunner(
    _rng,
    env,
    network,
    ScannedRNN.initialize_carry,
    hidden_size=config["learning"]["HIDDEN_SIZE"],
    greedy=False,
    env_init_states=states,
    n_episodes=1,
  )

  # EVALUATE
  rng, _rng = jax.random.split(rng)
  o = runner.run_and_visualise(_rng, network_params, run.name, viz_only_failure=False, plot_lidar=True)
  print('o', o) 
  raise
  print('o', o)
  print('sba', o["eval-sampled/success_by_actor"])
  o = runner.run(_rng, network_params)
  print('o', o)
  print('sba', o["eval-sampled/success_by_actor"])
  o = runner.run(_rng, network_params)
  print('o', o)
  print('sba', o["eval-sampled/success_by_actor"])
  o = runner.run(_rng, network_params)
  print('o', o)
  print('sba', o["eval-sampled/success_by_actor"])
  
  raise
  return run_name, o

def main():
  
  dataset_path = "jax_multirobsim/data/test_sets/sampled_tc_100e_1a.pkl"

  # load
  with open(dataset_path, "rb") as f:
      env_instances = pickle.load(f)

  num_envs = env_instances.map_data.shape[0]
  num_agents = env_instances.agent_pos.shape[1]
  map_size = env_instances.map_data.shape[1:3]
  print("map_size", map_size)
  print('num_agents', num_agents)
  print("num_envs", num_envs)

  # table = save_maps_to_table(env, env_instances)

  runs = ["alex-plus/multi_robot_ued/htqyilsk"]
  for run in tqdm.tqdm(runs, desc="Evaluating runs"):
    run_name, results = get_data_for_run(run, env_instances)
    
    # performance = np.concatenate([
    #   results["eval/win_rates"][None],
    #   results["eval/c_rates"][None],
    #   results["eval/to_rates"][None],
    # ])
    # print('performance', performance.shape)
  
  #   table.add_column(f"{run_name}_w", np.array(results["eval/win_rates"]))
  #   table.add_column(f"{run_name}_c", np.array(results["eval/c_rates"]))
  #   table.add_column(f"{run_name}_t", np.array(results["eval/to_rates"]))
  
  # wandb.log({"performance": table})

if __name__=="__main__":
  main()