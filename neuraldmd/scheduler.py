import jax.numpy as jnp


class PlateauScheduler:
    """
    Manual reduce-on-plateau scheduler compatible with Optax.

    Usage:
        sched = PlateauScheduler(initial_lr=1e-3)
        lr = sched.step(val_loss)
    """

    def __init__(self, *, initial_lr: float, factor: float = 0.5,
                 patience: int = 2000, min_lr: float = 1e-8):
        self.lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = jnp.inf
        self.tick = 0

    def step(self, current_loss: float) -> float:
        if current_loss < self.best:
            self.best = current_loss
            self.tick = 0
        else:
            self.tick += 1
        if self.tick >= self.patience:
            new_lr = max(self.lr * self.factor, self.min_lr)
            if new_lr < self.lr:
                print(f"[Plateau] lr: {self.lr:.2e} → {new_lr:.2e}", flush=True)
                self.lr = new_lr
            self.tick = 0
        return self.lr
