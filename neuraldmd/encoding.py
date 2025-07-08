import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

__all__ = ["SinusoidalEncoding", "LearnableFourierEncoding"]


class SinusoidalEncoding(eqx.Module):
    """Non-learned 2-D Fourier feature encoding."""

    frequencies: jax.Array = eqx.static_field()

    def __init__(self, *, num_frequencies: int = 10):
        self.frequencies = 2 ** jnp.arange(num_frequencies)

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        x, y = xy[0], xy[1]
        enc_x, enc_y = [x], [y]
        for f in self.frequencies:
            enc_x += [jnp.sin(f * x), jnp.cos(f * x)]
            enc_y += [jnp.sin(f * y), jnp.cos(f * y)]
        
        return jnp.array(enc_x + enc_y)


class LearnableFourierEncoding(eqx.Module):
    """Learnable per-dimension frequencies (cf. Tancik et al.)."""

    frequencies: jax.Array
    input_dim: int = eqx.static_field()
    num_frequencies: int = eqx.static_field()

    def __init__(self, *, input_dim: int = 2, num_frequencies: int = 10, key):
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.frequencies = eqx.nn.Parameter(
            jnp.abs(jax.random.normal(key, (input_dim, num_frequencies)))
        )

    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        parts = [xy]
        for d in range(self.input_dim):
            scaled = self.frequencies[d] * xy[d]
            parts += [jnp.sin(scaled), jnp.cos(scaled)]
        return jnp.concatenate(parts)
