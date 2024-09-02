import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import functools

def _calculate_dormancy(x, layer_dim, tau):
    ins = x.squeeze()
    d = jnp.sum(jnp.abs(ins), axis=0)/ins.shape[0]  # average activation
    d = jnp.where((d / (jnp.sum(d)/layer_dim + 1e-8)) <= tau, 1, 0)
    return jnp.sum(d) / layer_dim * 100  # dormancy percentage

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    tau: float = 0.0  # dormancy threshold
    is_recurrent: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)
        ed1 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["FC_DIM_SIZE"], self.tau))


        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        hd1 = jax.lax.stop_gradient(_calculate_dormancy(hidden, self.config["HIDDEN_SIZE"], self.tau))
        ed2 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["HIDDEN_SIZE"], self.tau))
        actor_mean = nn.Dense(self.config["HIDDEN_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        actor_mean = nn.relu(actor_mean)
        ad1 = jax.lax.stop_gradient(_calculate_dormancy(actor_mean, self.config["FC_DIM_SIZE"], self.tau))
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        

        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        critic = nn.relu(critic)
        cd1 = jax.lax.stop_gradient(_calculate_dormancy(critic, 256, self.tau))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        if self.config["LOG_DORMANCY"]:
            dormancy = DormancyActorCriticRNN(
                actor=ad1,
                embedding=ed1,
                hidden=hd1, 
                rnnout=ed2,
                critic=cd1
            )
            
            return hidden, pi, jnp.squeeze(critic, axis=-1), dormancy
        else:
            return hidden, pi, jnp.squeeze(critic, axis=-1)

class DormancyActorCriticRNN(NamedTuple):
    actor: jnp.array
    embedding: jnp.array
    hidden: jnp.array
    rnnout: jnp.array
    critic: jnp.array

class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    tau: float = 0.0  # dormancy threshold
    is_recurrent: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)
        ed1 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["FC_DIM_SIZE"], self.tau))


        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        hd1 = jax.lax.stop_gradient(_calculate_dormancy(hidden, self.config["HIDDEN_SIZE"], self.tau))
        ed2 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["HIDDEN_SIZE"], self.tau))
        actor_mean = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        actor_mean = nn.relu(actor_mean)
        ad1 = jax.lax.stop_gradient(_calculate_dormancy(actor_mean, self.config["FC_DIM_SIZE"], self.tau))
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        

        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))


        dormancy = DormancyActorRNN(
            actor=ad1,
            embedding=ed1,
            hidden=hd1, 
            rnnout=ed2,
        )
        
        #jax.debug.print('dormancy {d}', d=dormancy)

        return hidden, pi, dormancy

class DormancyActorRNN(NamedTuple):
    actor: jnp.array
    embedding: jnp.array
    hidden: jnp.array
    rnnout: jnp.array

class CriticRNN(nn.Module):
    config: Dict
    tau: float = 0.0  # dormancy threshold
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)
        ed1 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["FC_DIM_SIZE"], self.tau))
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        hd1 = jax.lax.stop_gradient(_calculate_dormancy(hidden, self.config["HIDDEN_SIZE"], self.tau))
        ed2 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["HIDDEN_SIZE"], self.tau))

        
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        critic = nn.relu(critic)
        cd1 = jax.lax.stop_gradient(_calculate_dormancy(critic, self.config["FC_DIM_SIZE"], self.tau))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        dormancy = DormancyCriticRNN(
            critic=cd1,
            embedding=ed1,
            hidden=hd1,
            rnnout=ed2,
        )
        
        return hidden, jnp.squeeze(critic, axis=-1), dormancy

class ActorCriticRNNMultiHead(nn.Module):
    config: Dict
    action_dim: Sequence[int]
    num_reward_components: int = 1
    tau: float = 0.0  # dormancy threshold
    is_recurrent: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)
        ed1 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["FC_DIM_SIZE"], self.tau))


        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        hd1 = jax.lax.stop_gradient(_calculate_dormancy(hidden, self.config["HIDDEN_SIZE"], self.tau))
        ed2 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["HIDDEN_SIZE"], self.tau))
        actor_mean = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        actor_mean = nn.relu(actor_mean)
        ad1 = jax.lax.stop_gradient(_calculate_dormancy(actor_mean, self.config["FC_DIM_SIZE"], self.tau))
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        

        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        critic = nn.relu(critic)
        cd1 = jax.lax.stop_gradient(_calculate_dormancy(critic, 256, self.tau))
        
        # Critic predicts two things: Value Sparse and Value Dense, per agent.
        critic = nn.Dense(self.num_reward_components, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
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

class CriticRNNMultiHead(nn.Module):
    config: Dict
    num_reward_components: int = 1
    tau: float = 0.0  # dormancy threshold
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)
        ed1 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["FC_DIM_SIZE"], self.tau))
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        hd1 = jax.lax.stop_gradient(_calculate_dormancy(hidden, self.config["HIDDEN_SIZE"], self.tau))
        ed2 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["HIDDEN_SIZE"], self.tau))

        
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        critic = nn.relu(critic)
        cd1 = jax.lax.stop_gradient(_calculate_dormancy(critic, self.config["FC_DIM_SIZE"], self.tau))
        critic = nn.Dense(self.num_reward_components, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        dormancy = DormancyCriticRNN(
            critic=cd1,
            embedding=ed1,
            hidden=hd1,
            rnnout=ed2,
        )
        
        return hidden, critic, dormancy

class DormancyCriticRNN(NamedTuple):
    critic: jnp.array
    embedding: jnp.array
    hidden: jnp.array
    rnnout: jnp.array
    
    
#### DISCRETE


class DiscreteActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    tau: float = 0.0  # dormancy threshold
    is_recurrent: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        embedding = nn.relu(embedding)
        ed1 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["FC_DIM_SIZE"], self.tau))


        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        hd1 = jax.lax.stop_gradient(_calculate_dormancy(hidden, self.config["HIDDEN_SIZE"], self.tau))
        ed2 = jax.lax.stop_gradient(_calculate_dormancy(embedding, self.config["HIDDEN_SIZE"], self.tau))
        actor_mean = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        actor_mean = nn.relu(actor_mean)
        ad1 = jax.lax.stop_gradient(_calculate_dormancy(actor_mean, self.config["FC_DIM_SIZE"], self.tau))
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        if self.config["USE_LAYER_NORM"]:
            embedding = nn.LayerNorm(use_scale=False)(embedding)
        critic = nn.relu(critic)
        cd1 = jax.lax.stop_gradient(_calculate_dormancy(critic, 256, self.tau))
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

        return hidden, pi, jnp.squeeze(critic, axis=-1), dormancy