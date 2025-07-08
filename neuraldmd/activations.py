import jax
import equinox as eqx

__all__ = ["Silu", "Tanh", "ReLU"]


class Silu(eqx.Module):
    """SiLU / swish activation."""

    def __call__(self, x, *, key=None):
        return jax.nn.silu(x)


class Tanh(eqx.Module):
    def __call__(self, x, *, key=None):
        return jax.nn.tanh(x)


class ReLU(eqx.Module):
    def __call__(self, x, *, key=None):
        return jax.nn.relu(x)
