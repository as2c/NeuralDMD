"""
NeuralDMD   ––  core model definition.

This file holds:
* Spatial MLP
* Temporal Omega / b MLPs
* Forward call returning (W0, W_half, W, Ω, b)
"""

import jax, jax.numpy as jnp, equinox as eqx
from .activations import Silu, ReLU
from .encoding    import SinusoidalEncoding, LearnableFourierEncoding

__all__ = ["NeuralDMD"]


class _TemporalOmegaMLP(eqx.Module):
    latent: jax.Array
    mlp: eqx.nn.Sequential
    r_half: int = eqx.static_field()

    def __init__(self, *, r_half: int, latent_dim: int = 16,
                 hidden_size: int = 64, layers: int = 2, key):
        self.r_half = r_half
        k = jax.random.split(key, layers + 1)
        self.latent = jax.random.normal(k[0], (latent_dim,))
        seq = []
        in_dim = latent_dim
        for i in range(layers):
            seq += [eqx.nn.Linear(in_dim, hidden_size, key=k[i+1]), ReLU()]
            in_dim = hidden_size
        seq += [eqx.nn.Linear(in_dim, 2 * r_half + 1, key=k[-1])]
        self.mlp = eqx.nn.Sequential(seq)

    def __call__(self):
        out = self.mlp(self.latent)  # Shape: (2 * r_half,)
        raw_alphas = out[0:self.r_half]
        raw_thetas = out[self.r_half:2 * self.r_half]
        alphas = -2 * jax.nn.sigmoid(raw_alphas)              # ensures alphas in [-2, 0]
        thetas = jax.nn.sigmoid(raw_thetas)         # ensures thetas in (0, 1)
        return alphas, thetas


class _TemporalBMLP(eqx.Module):
    latent: jax.Array
    mlp: eqx.nn.Sequential
    r_half: int = eqx.static_field()

    def __init__(self, *, r_half: int, latent_dim: int = 16,
                 hidden_size: int = 64, layers: int = 2, key):
        self.r_half = r_half
        k = jax.random.split(key, layers + 1)
        self.latent = jax.random.normal(k[0], (latent_dim,))
        seq = []
        in_dim = latent_dim
        for i in range(layers):
            seq += [eqx.nn.Linear(in_dim, hidden_size, key=k[i+1]), ReLU()]
            in_dim = hidden_size
        seq += [eqx.nn.Linear(in_dim, 1 + 2 * r_half, key=k[-1])]
        self.mlp = eqx.nn.Sequential(seq)

    def __call__(self):
        out = self.mlp(self.latent)          # (1 + 2 r_half,)
        b0 = out[0:1]
        raw = out[1:].reshape(self.r_half, 2)
        b_half = raw[:, 0] + 1j * raw[:, 1]
        return b0, b_half


class NeuralDMD(eqx.Module):
    """Full Neural-DMD model (spatial network + temporal networks)."""

    mlp: eqx.nn.Sequential
    encoding: eqx.Module = eqx.static_field()
    temporal_omega: _TemporalOmegaMLP
    temporal_b: _TemporalBMLP
    scale: jax.Array
    bias: jax.Array
    r_half: int = eqx.static_field()

    def __init__(self, *, r: int, key, hidden_size: int = 256,
                 layers: int = 4, num_frequencies: int = 10,
                 use_learnable_encoding: bool = False, 
                 temporal_latent_dim=32, temporal_hidden=64, temporal_layers=2):
        assert r % 2 == 0, "`r` must be even so modes pair with conjugates"
        self.r_half = r // 2
        k = jax.random.split(key, layers + 3)
        self.scale = jnp.array(1.0)
        self.bias = jnp.array(1e-3)

        # Positional encoding
        if use_learnable_encoding:
            self.encoding = LearnableFourierEncoding(
                input_dim=2, num_frequencies=num_frequencies, key=k[0]
            )
        else:
            self.encoding = SinusoidalEncoding(num_frequencies=num_frequencies)

        # Spatial MLP
        in_dim = 2 * (2 * num_frequencies + 1)
        seq = []
        for i in range(layers):
            seq += [eqx.nn.Linear(in_dim, hidden_size, key=k[i+1]), Silu()]
            in_dim = hidden_size
        seq += [eqx.nn.Linear(in_dim, 2 * self.r_half + 1, key=k[layers + 1])]
        self.mlp = eqx.nn.Sequential(seq)

        # Temporal networks
        self.temporal_omega = _TemporalOmegaMLP(r_half=self.r_half,
                                                latent_dim=temporal_latent_dim,
                                               hidden_size=temporal_hidden,
                                               layers=temporal_layers,
                                                 key=k[-2])
        self.temporal_b = _TemporalBMLP(r_half=self.r_half, 
                                        latent_dim=temporal_latent_dim,
                                       hidden_size=temporal_hidden,
                                       layers=temporal_layers,
                                       key=k[-1])

    # ------------------------------------------------------------------
    #                        forward utilities
    # ------------------------------------------------------------------
    def spatial_forward(self, xy):
        enc = self.encoding(xy)
        out = self.mlp(enc)
        W0 = out[0:1]
        real, imag = jnp.split(out[1:], 2)
        W_half = real + 1j * imag
        W = jnp.concatenate([W_half, W0, jnp.conj(W_half)])
        return W0, W_half, W

    def __call__(self, xy: jnp.ndarray):
        W0, W_half, W = jax.vmap(self.spatial_forward)(xy)
        alpha, theta = self.temporal_omega()
        Omega = alpha + 1j * theta
        b0, b_half = self.temporal_b()
        b = jnp.concatenate([b_half, b0, jnp.conj(b_half)])
        return W0, W_half, W, Omega, b
