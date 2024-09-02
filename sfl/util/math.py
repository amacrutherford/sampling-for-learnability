import jax.numpy as jnp

def rotation_matrix(theta: float) -> jnp.ndarray:
    """ Rotate about the z axis. Assume theta in radians """
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta)]
    ])
    
