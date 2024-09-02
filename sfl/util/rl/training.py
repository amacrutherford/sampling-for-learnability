"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import core
from flax import struct
import optax
import chex

from .plr import PLRManager


class VmapTrainState(struct.PyTreeNode):
  n_iters: chex.Array
  n_updates: chex.Array # per agent
  n_grad_updates: chex.Array # per agent
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any]
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState
  plr_buffer: PLRManager = None

  def apply_gradients(self, *, grads, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)

    return self.replace(
        n_grad_updates=self.n_updates + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, 
  	apply_fn, 
  	params, 
  	tx,
  	**kwargs
  ):
    opt_state = jax.vmap(tx.init)(params)
    return cls(
        n_iters=jnp.array(jax.vmap(lambda x: 0)(params), dtype=jnp.uint32),
        n_updates=jnp.array(jax.vmap(lambda x: 0)(params), dtype=jnp.uint32),
        n_grad_updates=jnp.array(jax.vmap(lambda x: 0)(params), dtype=jnp.uint32),
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )

  def increment(self):
    return self.replace(
      n_iters=self.n_iters + 1,
    )

  def increment_updates(self):
    return self.replace(
      n_updates=self.n_updates + 1,
    ) 

  @property
  def state_dict(self):
    return dict(
      n_iters=self.n_iters,
      n_updates=self.n_updates,
      n_grad_updates=self.n_grad_updates,
      params=self.params,
      opt_state=self.opt_state
    )

  def load_state_dict(self, state):
    return self.replace(
      n_iters=state['n_iters'],
      n_updates=state['n_updates'],
      n_grad_updates=state['n_grad_updates'],
      params=state['params'],
      opt_state=state['opt_state']
    )
    

from typing import Any, Callable

from flax import core
from flax import struct
import optax


class TrainState(struct.PyTreeNode):
  """Simple train state for the common case with a single Optax optimizer.

  Synopsis::

      state = TrainState.create(
          apply_fn=model.apply,
          params=variables['params'],
          tx=tx)
      grad_fn = jax.grad(make_loss_fn(state.apply_fn))
      for batch in data:
        grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)

  Note that you can easily extend this dataclass by subclassing it for storing
  additional data (e.g. additional variable collections).

  For more exotic usecases (e.g. multiple optimizers) it's probably best to
  fork the class and modify it.

  Args:
    step: Counter starts at 0 and is incremented by every call to
      `.apply_gradients()`.
    apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
      convenience to have a shorter params list for the `train_step()` function
      in your training loop.
    params: The parameters to be updated by `tx` and used by `apply_fn`.
    tx: An Optax gradient transformation.
    opt_state: The state for `tx`.
  """
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  plr_buffer: PLRManager = None

  def apply_gradients(self, *, grads, **kwargs):
    """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

    Note that internally this function calls `.tx.update()` followed by a call
    to `optax.apply_updates()` to update `params` and `opt_state`.

    Args:
      grads: Gradients that have the same pytree structure as `.params`.
      **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

    Returns:
      An updated instance of `self` with `step` incremented by one, `params`
      and `opt_state` updated by applying `grads`, and additional attributes
      replaced as specified by `kwargs`.
    """
    updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )

