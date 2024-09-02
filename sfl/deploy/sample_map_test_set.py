import jax 
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pickle

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav, EnvInstance

def main():
    
    data_dir = "sfl/data/test_sets/"
    
    config = {
        "env_params": {
            "rad": 0.3,
            "map_id": "Grid-Rand-Poly",  # Grid-Rand, Grid-PreSpec, Polygon, Grid-Test2
            "map_params": {
                "map_size": [11, 11],
                "fill": 0.4,
                "start_pad": 1.5,
                "valid_path_check": True,
            }
        },  
    }
    
    num_agents = 1
    num_envs = 100
    
    env = JaxNav(num_agents=num_agents, **config["env_params"])

    rng = jax.random.PRNGKey(0)
    
    rngs = jax.random.split(rng, num_envs)
    _, states = jax.vmap(env.reset, in_axes=(0))(rngs)
    
    instances = EnvInstance(
        agent_pos=states.pos,
        agent_theta=states.theta,
        goal_pos=states.goal,
        map_data=states.map_data,
        rew_lambda=states.rew_lambda,
    )
    
    viz_maps = False 
    if viz_maps:
        # vizualise in batches of 40, with 4 rows, 10 columns
        num_frames = np.ceil(num_envs / 40).astype(int)
        print('num_frames', num_frames)
        for i in tqdm.tqdm(range(num_frames)):
            i_range = range(i*40, min((i+1)*40, num_envs))
            print('i_range', i_range)
            
            fig, ax = plt.subplots(4, 10, figsize=(40, 30))
            for j, idx in enumerate(i_range):
                
                state = jax.tree_map(lambda x: x[idx], states)
                env.init_render(ax[j//10, j%10], state, lidar=False)
                # ax[j//10, j%10].imshow(states.map_data[idx])
                ax[j//10, j%10].set_title(f'env_{idx}')
                ax[j//10, j%10].axis('off')
                
            plt.savefig(data_dir + f'env_maps_{i}.png')
    
    # use pickle
    file_name = f'sampled_tc_{num_envs}e_{num_agents}a.pkl'
    save_path = data_dir + file_name
    with open(save_path, 'wb') as f:
        pickle.dump(instances, f)


if __name__=="__main__":
    main()
    