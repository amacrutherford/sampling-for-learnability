import jax
from jax import numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=["h", "w"])
def flatten_coordinate_grid(h, w):
    x, y = jnp.meshgrid(jnp.arange(h), jnp.arange(w))
    x_flat = x.flatten()
    y_flat = y.flatten()
    return x_flat, y_flat

@partial(jax.jit, static_argnames=["num_idx", "length"])
def sample_idx(key, map_flat, num_idx, length):
    """ Sample a number of index's from the free space of a map """

    free_i = jnp.arange(0, length)

    # Select all indices corresponding to free space
    free_space_indices_p = jnp.where(map_flat == 0, 1, 0)
    free_space_indices_p = free_space_indices_p / jnp.sum(free_space_indices_p)

    # Sample a random index from the indices corresponding to free space
    return jax.random.choice(key, free_i, shape=(num_idx,), replace=False, p=free_space_indices_p)

@jax.jit  # from gymnax
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])