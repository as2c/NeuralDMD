import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import os
from tqdm import tqdm
import optax
import matplotlib.pyplot as plt

# -------------------------
# Activation Modules
# -------------------------
class SiluActivation(eqx.Module):
    def __call__(self, x, key=None):
        return jax.nn.silu(x)

class TanhActivation(eqx.Module):
    def __call__(self, x, key=None):
        return jax.nn.tanh(x)

class ReLUActivation(eqx.Module):
    def __call__(self, x, key=None):
        return jax.nn.relu(x)

# -------------------------
# Plateau Scheduler
# -------------------------
class PlateauScheduler:
    def __init__(self, initial_lr, factor=1.0, patience=1000, min_lr=1e-8):
        self.lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = jnp.inf
        self.epochs_since_improvement = 0
    def step(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        if self.epochs_since_improvement >= self.patience:
            new_lr = max(self.lr * self.factor, self.min_lr)
            if new_lr < self.lr:
                print(f"Reducing learning rate from {self.lr:.6f} to {new_lr:.6f}", flush=True)
                self.lr = new_lr
                self.epochs_since_improvement = 0
        return self.lr

# -------------------------
# Sinusoidal Encoding
# -------------------------
class LearnableFourierEncoding(eqx.Module):
    # frequencies is a learnable parameter of shape (input_dim, num_frequencies)
    frequencies: jnp.ndarray  
    input_dim: int = eqx.static_field()
    num_frequencies: int = eqx.static_field()

    def __init__(self, input_dim: int = 2, num_frequencies: int = 10, key=None):
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        # Initialize frequencies with a moderate scale (for example, log-uniform around 1)
        # Here we use a normal initializer scaled by 1.0
        self.frequencies = eqx.nn.Parameter(jnp.abs(jax.random.normal(key, (input_dim, num_frequencies))))

    def __call__(self, xy: jnp.ndarray):
        # xy should be a 1D array of shape (input_dim,)
        # We include the original coordinates and then for each coordinate, sin and cos of (frequency * coordinate)
        encoded_parts = [xy]
        for d in range(self.input_dim):
            # shape of self.frequencies[d] is (num_frequencies,)
            scaled = self.frequencies[d] * xy[d]  # shape: (num_frequencies,)
            encoded_parts.append(jnp.sin(scaled))
            encoded_parts.append(jnp.cos(scaled))
        return jnp.concatenate(encoded_parts)


class SinusoidalEncoding(eqx.Module):
    frequencies: np.ndarray = eqx.static_field()
    def __init__(self, num_frequencies=10):
        self.frequencies = 2**(jnp.arange(num_frequencies))
    def __call__(self, xy):
        x, y = xy[0], xy[1]
        encoding_x = [x]
        encoding_y = [y]
        for freq in self.frequencies:
            encoding_x.append(jnp.sin(freq * x))
            encoding_x.append(jnp.cos(freq * x))
            encoding_y.append(jnp.sin(freq * y))
            encoding_y.append(jnp.cos(freq * y))
        return jnp.array(encoding_x + encoding_y)

# -------------------------
# TemporalOmegaMLP: Outputs raw parameters for Omega
# -------------------------
class TemporalOmegaMLP(eqx.Module):
    latent: jax.Array         # Learned latent vector.
    mlp: eqx.nn.Sequential    # MLP mapping latent to 2 * r_half outputs.
    r_half: int = eqx.static_field()
    def __init__(self, r_half, latent_dim=16, hidden_size=64, num_layers=2, key=None):
        self.r_half = r_half
        self.latent = jax.random.normal(key, (latent_dim,))
        layers = []
        in_size = latent_dim
        keys = jax.random.split(key, num_layers + 1)
        for i in range(num_layers):
            layers.append(eqx.nn.Linear(in_size, hidden_size, key=keys[i]))
            layers.append(SiluActivation())
            in_size = hidden_size
        layers.append(eqx.nn.Linear(in_size, 2 * r_half + 1, key=keys[-1]))
        self.mlp = eqx.nn.Sequential(layers)
    def __call__(self):
        out = self.mlp(self.latent)  # Shape: (2 * r_half,)
        raw_alphas = out[0:self.r_half]
        raw_thetas = out[self.r_half:2 * self.r_half]
        alphas = -2 * jax.nn.sigmoid(raw_alphas)              #-2 ≤ ensures alphas ≤ 0
        thetas = jax.nn.sigmoid(raw_thetas)         # ensures thetas ∈ (0, 1)
        # thetas = raw_thetas
        return alphas, thetas

# -------------------------
# TemporalBMLP: Outputs learned b parameters, including b0 and b_half
# -------------------------
class TemporalBMLP(eqx.Module):
    latent: jax.Array         # Learned latent vector.
    mlp: eqx.nn.Sequential    # MLP mapping latent to (1 + 2 * r_half) outputs.
    r_half: int = eqx.static_field()
    def __init__(self, r_half, latent_dim=16, hidden_size=64, num_layers=2, key=None):
        self.r_half = r_half
        self.latent = jax.random.normal(key, (latent_dim,))
        layers = []
        in_size = latent_dim
        keys = jax.random.split(key, num_layers + 1)
        for i in range(num_layers):
            layers.append(eqx.nn.Linear(in_size, hidden_size, key=keys[i]))
            layers.append(SiluActivation())
            in_size = hidden_size
        layers.append(eqx.nn.Linear(in_size, 1 + 2 * r_half, key=keys[-1]))
        self.mlp = eqx.nn.Sequential(layers)
    def __call__(self):
        out = self.mlp(self.latent)  # Shape: (1 + 2 * r_half,)
        b0 = out[0:1]  # Learned scalar b0.
        b_half_raw = out[1: 2 * self.r_half + 1]
        b_half_raw = b_half_raw.reshape((self.r_half, 2))
        b_half = b_half_raw[:, 0] + 1j * b_half_raw[:, 1]
        return b0, b_half

# -------------------------
# NeuralDMD: Uses a spatial MLP and the two temporal networks.
# -------------------------
class NeuralDMD(eqx.Module):
    mlp: eqx.nn.Sequential       # Spatial network (for W).
    encoding: SinusoidalEncoding # Positional encoding.
    output_size: int = eqx.static_field()
    temporal_omega: TemporalOmegaMLP
    temporal_b: TemporalBMLP
    scale: jax.Array
    bias: jax.Array
    r_half: int = eqx.static_field()
    def __init__(self, r, hidden_size=512, num_layers=10, key=None, num_frequencies=10,
                 temporal_latent_dim=32, temporal_hidden=64, temporal_layers=2):
        assert r % 2 == 0, "r must be even to ensure complex conjugate symmetry"
        self.r_half = r // 2
        self.scale = jnp.array(1.0)
        self.bias = jnp.array(0.0)
        keys = jax.random.split(key, num_layers + 3)
        self.output_size = 2 * self.r_half + 1
        self.encoding = SinusoidalEncoding(num_frequencies=num_frequencies)
        layers = []
        in_size = 2 * (2 * num_frequencies + 1)
        for i in range(num_layers):
            layers.append(eqx.nn.Linear(in_size, hidden_size, key=keys[i]))
            layers.append(SiluActivation())
            in_size = hidden_size
        layers.append(eqx.nn.Linear(hidden_size, self.output_size, key=keys[num_layers]))
        # layers.append(TanhActivation())
        self.mlp = eqx.nn.Sequential(layers)
        self.temporal_omega = TemporalOmegaMLP(r_half=self.r_half,
                                               latent_dim=temporal_latent_dim,
                                               hidden_size=temporal_hidden,
                                               num_layers=temporal_layers,
                                               key=keys[num_layers+1])
        self.temporal_b = TemporalBMLP(r_half=self.r_half,
                                       latent_dim=temporal_latent_dim,
                                       hidden_size=temporal_hidden,
                                       num_layers=temporal_layers,
                                       key=keys[num_layers+2])
    def spatial_forward(self, xy):
        encoded = self.encoding(xy)
        output = self.mlp(encoded)
        W0 = jnp.expand_dims(output[0], axis=0)
        W_real, W_imag = jnp.split(output[1:], 2, axis=-1)
        W_half = W_real + 1j * W_imag
        W = jnp.concatenate([W_half, W0, jnp.conj(W_half)], axis=-1)
        return W0, W_half, W
    def __call__(self, xy):
        # Compute spatial outputs via vmap.
        W0, W_half, W = jax.vmap(self.spatial_forward)(xy)
        # Compute temporal parameters once (global to the batch).
        alphas, thetas = self.temporal_omega()
        Omega = alphas + 1j * thetas
        b0, b_half = self.temporal_b()
        # Form the full b: here we concatenate b_half, b0, and the conjugate of b_half.
        b = jnp.concatenate([b_half, b0, jnp.conj(b_half)], axis=0)
        return W0, W_half, W, Omega, b

# -------------------------
# Loss Function
# -------------------------
def tv_loss(image):
    """Compute Total Variation loss for a 2D image."""
    diff_vert = jnp.abs(image[:-1, :] - image[1:, :])
    diff_horiz = jnp.abs(image[:, :-1] - image[:, 1:])
    return jnp.sum(diff_vert) + jnp.sum(diff_horiz)

# ---------------------------------------------------------------------
# Sparsity–promoting loss helper
# ---------------------------------------------------------------------
def sparsity_loss(W0, W_half):
    """ℓ₁ penalty encouraging sparse spatial modes."""
    return jnp.mean(jnp.abs(W0)) + jnp.mean(jnp.abs(W_half))

def loss_fn(model, xy, target_values, time_indices, beta, frame_max, frame_min):
    """
    Args:
      model: NeuralDMD instance.
      xy: Coordinates for each pixel, shape (B, 2)
      target_values: Ground truth signal for each pixel over time, shape (B, T)
      time_indices: Array of time points, shape (T,)
      alpha, beta, delta: Hyperparameters.
    
    Returns:
      total_loss, (reconstruction_loss, orthogonality_loss)
    """
    # Compute spatial outputs via vmap.
    W0, W_half, _ = jax.vmap(model.spatial_forward)(xy)
    # Compute global temporal parameters.
    alphas, thetas = model.temporal_omega()  # each shape: (r_half,)
    Omega = alphas + 1j * thetas              # shape: (r_half,)
    b0, b_half = model.temporal_b()           # b0: (1,), b_half: (r_half,)
    
    # Compute temporal evolution.
    lambda_exp = jnp.exp(Omega[:, None] * time_indices[None, :] * 160.0)  # shape: (r_half, T)
    
    # Reconstruction: use spatial mode W_half and combine with temporal evolution and b_half.
    X_reconstructed = 2 * jnp.real(jnp.einsum('br, rt, r -> bt', W_half, lambda_exp, b_half)) + W0[:, 0:1] * b0[0]
    X_reconstructed = X_reconstructed * (frame_max - frame_min) + frame_min
    reconstruction_loss = jnp.sum((target_values - X_reconstructed)**2)
    total_loss = reconstruction_loss
    
    # Orthogonality loss on spatial modes.
    # W_half_normed = W_half / (jnp.linalg.norm(W_half, axis=0, keepdims=True) + 1e-10)
    # W_half_t = jnp.conjugate(W_half_normed).T
    # W_gram = jnp.matmul(W_half_t, W_half_normed)
    # I = jnp.eye(W_gram.shape[0])
    # orthogonality_loss = jnp.linalg.norm(W_gram - I, ord='fro')
    
    # total_loss += beta * orthogonality_loss

    negative_penalty = jnp.sum(jax.nn.relu(-X_reconstructed)**2)
    beta = 1e-2
    total_loss += beta * negative_penalty

    sparse_penalty = sparsity_loss(W0, W_half)
    gamma = 1e-2
    total_loss += gamma * sparse_penalty

    b_sparse_penalty = sparsity_loss(b0, b_half)
    b_weight = 1.0
    total_loss += b_weight * b_sparse_penalty

    return total_loss, (reconstruction_loss, reconstruction_loss)

# -------------------------
# Training Step and Loop (unchanged structure)
# -------------------------
@eqx.filter_jit
def train_step(model, opt_state, xy, target_values, time_indices, optimizer, beta, frame_max, frame_min):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, xy, target_values, time_indices, beta, frame_max, frame_min
    )
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    reconstruction_loss, orthogonality_loss = aux
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, reconstruction_loss, orthogonality_loss, grads

@eqx.filter_jit
def train_epoch_jit(model, opt_state, xy_array, pix_array, time_array, optimizer, beta, key, frame_max, frame_min):
    """
    xy_array: (num_batches, sensor_batch_size, 2) = (B, S, 2)
    pix_array: (num_batches, sensor_batch_size, time_batch_size) = (B, S, T_b)
    time_array: (num_batches, time_batch_size) = (B, T_b)
    key: PRNG key.
    """
    def scan_fn(carry, batch_idx):
        model, opt_state, key = carry
        key, subkey = jax.random.split(key)
        xy = xy_array[batch_idx]        # (S, 2)
        pixels = pix_array[batch_idx]     # (S, T_b)
        times = time_array[batch_idx]     # (T_b, )
        noise = jax.random.normal(subkey, shape=xy.shape) * 0.01
        xy_noisy = xy + noise
        new_model, new_opt_state, loss, rec_loss, ortho_loss, grads = train_step(
            model, opt_state, xy_noisy, pixels, times, optimizer, beta, frame_max, frame_min
        )
        return (new_model, new_opt_state, key), (loss, rec_loss, ortho_loss, grads)
    num_batches = xy_array.shape[0]
    init_carry = (model, opt_state, key)
    (final_model, final_opt_state, _), (losses, rec_losses, ortho_losses, grads) = jax.lax.scan(
        scan_fn, init_carry, jnp.arange(num_batches)
    )
    avg_loss = jnp.sum(losses)
    rec_avg = jnp.sum(rec_losses)
    ortho_avg = jnp.sum(ortho_losses)
    return final_model, final_opt_state, avg_loss, rec_avg, ortho_avg, grads

def plot_losses(rec_losses, ortho_losses, total_losses, output_dir):
    epochs = range(1, len(rec_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rec_losses, label="Reconstruction Loss")
    plt.plot(epochs, ortho_losses, label="Orthogonality Loss")
    plt.plot(epochs, total_losses, label="Total Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses Over Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "losses.png"))
    plt.close()

def print_grad_stats(name, grad):
    norm = np.linalg.norm(grad)
    grad_min = np.min(grad)
    grad_max = np.max(grad)
    print(f"{name}: norm = {norm:.4e}, min = {grad_min:.4e}, max = {grad_max:.4e}")

def print_param_norms(model):
    params = eqx.filter(model, eqx.is_array)
    def norm_fn(x):
        return jnp.linalg.norm(x)
    norms_tree = jax.tree_util.tree_map(norm_fn, params)
    leaves, _ = jax.tree_util.tree_flatten(norms_tree)
    for i, leaf in enumerate(leaves):
        print(f"Parameter {i} norm: {np.array(leaf):.4e}")

def schedule_fn(step):
    return scheduler.lr

def print_all_gradients(grads, model):
    # Print spatial MLP gradients.
    print("Spatial MLP gradients:")
    # grads.mlp is assumed to have the same structure as model.mlp.
    for i, module in enumerate(model.mlp):
        if isinstance(module, eqx.nn.Linear):
            grad_module = grads.mlp[i]
            print_grad_stats(f"Spatial layer {i} weight", grad_module.weight)
            print_grad_stats(f"Spatial layer {i} bias", grad_module.bias)

    # Print Temporal Omega MLP gradients.
    print("Temporal Omega MLP gradients:")
    for i, module in enumerate(model.temporal_omega.mlp):
        if isinstance(module, eqx.nn.Linear):
            grad_module = grads.temporal_omega.mlp[i]
            print_grad_stats(f"Temporal Omega layer {i} weight", grad_module.weight)
            print_grad_stats(f"Temporal Omega layer {i} bias", grad_module.bias)

    # Print Temporal B MLP gradients.
    print("Temporal B MLP gradients:")
    for i, module in enumerate(model.temporal_b.mlp):
        if isinstance(module, eqx.nn.Linear):
            grad_module = grads.temporal_b.mlp[i]
            print_grad_stats(f"Temporal B layer {i} weight", grad_module.weight)
            print_grad_stats(f"Temporal B layer {i} bias", grad_module.bias)

def train_model(model, train_loader, num_epochs, key, beta, data_dir, initial_lr, plots_dir, frame_max, frame_min):
    global scheduler
    scheduler = PlateauScheduler(initial_lr=initial_lr)
    optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=schedule_fn, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    rec_losses = []
    ortho_losses = []
    total_losses = []
    previous_loss = jnp.inf
    checkpoints_dir = "./checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            coords_list, pix_list, time_indices = train_loader.get_epoch_data()
            xy_array = coords_list
            pix_array = pix_list

            model, opt_state, avg_loss, rec_loss, ortho_loss, grads = train_epoch_jit(
                model, opt_state, xy_array, pix_array, time_indices, optimizer, beta, key, frame_max, frame_min
            )
            rec_losses.append(float(rec_loss))
            ortho_losses.append(float(ortho_loss))
            total_losses.append(float(avg_loss))
            if epoch % 10 == 0:
                print_all_gradients(grads, model)
                print_param_norms(model)
            current_lr = scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss={float(avg_loss):.6f}, Rec={float(rec_loss):.6f} LR={current_lr:.2e}", flush=True)
            pbar.update(1)
            if epoch % 100 == 0:
                eqx.tree_serialise_leaves(os.path.join(checkpoints_dir, "trained_model.eqx"), model)

            if avg_loss < previous_loss:
                avg_loss = previous_loss
                eqx.tree_serialise_leaves(os.path.join(data_dir, "trained_model.eqx"), model)
            if epoch > 4 and epoch % 2 == 0:
                from_epoch = 2
                plot_losses(rec_losses[from_epoch:], ortho_losses[from_epoch:], total_losses[from_epoch:], plots_dir)
                print(f"Plotted losses up to epoch {epoch+1}.")
    return model, total_losses, rec_losses, ortho_losses