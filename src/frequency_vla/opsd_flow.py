"""Native π0.5 flow operations, with an observation-only trace for training.

The traced sampler retains the pinned Pi0.sample_actions mask, positions, KV
behavior, Euler updates and 10-step stopping rule. It adds latent outputs only.
Formal evaluation always calls the original model.sample_actions implementation.
"""
import einops
import jax
import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask


def velocity(model, observation, latent, time):
    """The native compute_loss forward pass, exposing its velocity prediction."""
    observation = model_lib.preprocess_observation(None, observation, train=False)
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, condition = model.embed_suffix(observation, latent, time)
    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
    (_, output), _ = model.PaliGemma.llm(
        [prefix, suffix], mask=make_attn_mask(mask, ar),
        positions=jnp.cumsum(mask, axis=1) - 1, adarms_cond=[None, condition])
    return model.action_out_proj(output[:, -model.action_horizon:])


def sample_with_trace(model, observation, noise, num_steps=10):
    observation = model_lib.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    dt = -1.0 / num_steps
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    _, cache = model.PaliGemma.llm(
        [prefix, None], mask=make_attn_mask(prefix_mask, prefix_ar),
        positions=jnp.cumsum(prefix_mask, axis=1) - 1)

    def step(carry):
        latent, time, index, trace, times = carry
        trace = trace.at[index].set(latent)
        times = times.at[index].set(time)
        suffix, suffix_mask, suffix_ar, condition = model.embed_suffix(
            observation, latent, jnp.broadcast_to(time, batch_size))
        mask = jnp.concatenate([
            einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1]),
            make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix], mask=mask, positions=positions, kv_cache=cache,
            adarms_cond=[None, condition])
        assert prefix_out is None
        v = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        return latent + dt * v, time + dt, index + 1, trace, times

    def condition(carry):
        return carry[1] >= -dt / 2

    initial = (noise, jnp.array(1., dtype=noise.dtype), jnp.array(0),
               jnp.zeros((num_steps,) + noise.shape, dtype=noise.dtype),
               jnp.zeros(num_steps, dtype=noise.dtype))
    actions, _, count, trace, times = jax.lax.while_loop(condition, step, initial)
    return actions, trace, times, count
