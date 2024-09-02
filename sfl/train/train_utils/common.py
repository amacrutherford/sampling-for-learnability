from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)

import os
import wandb
import numpy as np
from typing import Callable
from tqdm.auto import tqdm
import pickle

import chex 
import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Tuple

import typing 
import os 
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file

def save_params(params: typing.Dict, filename: typing.Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)
    
def load_params(filename: typing.Union[str, os.PathLike]) -> typing.Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=',')

def load_config(config_fname, seed_id=None, lrate=None):
    """Load training configuration and random seed of experiment."""
    import yaml
    import re
    from dotmap import DotMap

    def load_yaml(config_fname: str) -> dict:
        """Load in YAML config file."""
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        with open(config_fname) as file:
            yaml_config = yaml.load(file, Loader=loader)
        return yaml_config

    config = load_yaml(config_fname)
    if seed_id is not None:
        config["train_config"]["seed_id"] = seed_id
    if lrate is not None:
        if "lr_begin" in config["train_config"].keys():
            config["train_config"]["lr_begin"] = lrate
            config["train_config"]["lr_end"] = lrate
        else:
            try:
                config["train_config"]["opt_params"]["lrate_init"] = lrate
            except Exception:
                pass
    return DotMap(config)

def save_checkpoint(state, epoch):
    import pickle 
    
    #with open(ckpt_path, "wb") as outfile:
    #    outfile.write(msgpack_serialize(to_state_dict(state)))
    print(f'Saving checkpoint at epoch {epoch}')
    artifact = wandb.Artifact(
        f'{wandb.run.name}-checkpoint', type='model'
    )
    #artifact.add_file(ckpt_path)
    with artifact.new_file(f'{epoch}-checkpoint', mode='wb') as file:
        pickle.dump(state, file, pickle.HIGHEST_PROTOCOL)
        
    wandb.log_artifact(artifact, aliases=["latest", f"epoch_{epoch}"])


'''def load_checkpoint(ckpt_file, state):
    artifact = wandb.use_artifact(
        f'{wandb.run.name}-checkpoint:latest'
    )
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, ckpt_file)
    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()
    return from_bytes(state, byte_data)'''


def load_checkpoint(dir_path, epoch=None):
    print('loading from: ', dir_path)
    if epoch is None:
        prefixed = [filename for filename in os.listdir(dir_path)]
        print('prefixed', prefixed)
        epoch = max([p.split("-")[0] for p in prefixed])
    
    filename = f"{epoch}-checkpoint"
    with open(dir_path + "/" + filename, "rb") as input:
        params = pickle.load(input)
    return params

def load_artifact(dir_path, artifact="checkpoint", epoch=None):
    if epoch is None:
        prefixed = [filename for filename in os.listdir(dir_path)]
        print('prefixed', prefixed)
        epoch = max([p.split("-")[0] for p in prefixed])
    
    filename = f"{epoch}-{artifact}"
    with open(dir_path + "/" + filename, "rb") as input:
        data = pickle.load(input)
    return data
    
    
### Obs normalising with stats aggregated through training, not currently used

    
@struct.dataclass
class RunningMuStd:
    mu: chex.Array 
    var: chex.Array 
    count: int 
  
def _batchMuStd(obs: Tuple, num_lidar_beams: int):
    batch_means = jnp.concatenate([
        jnp.array([jnp.mean(obs[:,:num_lidar_beams])]), 
        jnp.mean(obs[:,num_lidar_beams:], axis=0)])
    
    batch_var = jnp.concatenate([
        jnp.var(obs[:,:num_lidar_beams])[jnp.newaxis], 
        jnp.var(obs[:,num_lidar_beams:], axis=0)])
    batch_count = jnp.shape(obs)[0]
    return batch_means, batch_var, batch_count
  
@partial(jax.jit, static_argnames=['num_lidar_beams'])
def initRunningMuStd(obs: Tuple, num_lidar_beams: int) -> RunningMuStd:
    batch_means, batch_var, batch_count = _batchMuStd(obs, num_lidar_beams)
    #batch_var = batch_var.at[1:3].set(1.0)
    return RunningMuStd(
        mu=batch_means,
        var=batch_var,
        count=batch_count
    )

@partial(jax.jit, static_argnames=['num_lidar_beams'])
def updateRunningMuStd(current, obs: Tuple, num_lidar_beams: int):
        # [n, 15]
        
        def update_from_moments(current, batch_mean, batch_var, batch_count):
            delta = batch_mean - current.mu
            tot_count = current.count + batch_count

            new_mu = current.mu + delta * batch_count / tot_count
            m_a = current.var * (current.count)
            m_b = batch_var * (batch_count)
            M2 = m_a + m_b + jnp.square(delta) * current.count * batch_count / (current.count + batch_count)
            new_var = M2 / (current.count + batch_count)

            new_count = batch_count + current.count

            return RunningMuStd(
                mu=new_mu,
                var=new_var,
                count=new_count
            )
        
        batch_means, batch_var, batch_count = _batchMuStd(obs, num_lidar_beams)
        
        return update_from_moments(current, batch_means, batch_var, batch_count)

@partial(jax.jit, static_argnames=['num_lidar_beams', 'obs_clip'])
def norm_obs(factors, obs, num_lidar_beams, obs_clip):
    
    lobs = (obs[:,:num_lidar_beams] - factors.mu[0]) / jnp.sqrt(factors.var[0])
    oobs = (obs[:,num_lidar_beams:] - factors.mu[1:]) / jnp.sqrt(factors.var[1:])
    return jnp.clip(jnp.concatenate([lobs, oobs], axis=-1), -obs_clip, obs_clip)
