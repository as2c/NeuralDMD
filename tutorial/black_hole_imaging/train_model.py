import sys
sys.path.append("../../.")
import numpy as np
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
from dmd_data_loader import DMDDataLoader
import jax
from neuraldmd import NeuralDMD
from neuraldmd.scheduler import PlateauScheduler
from neuraldmd.training import train_step_visibilities as train_step
from neuraldmd.losses import fourier_loss_fn as loss_fn
import optax
import equinox as eqx
from util_funcs import load_hdf5
from tqdm import tqdm

@eqx.filter_jit
def train_epoch_jit(model, opt_state, batch_list, optimizer, alpha, beta, key, frame_max, frame_min):
    """
    xy_array: (num_batches, batch_size, 2)
    pix_array: (num_batches, batch_size, T)
    time_idx_array: (T,) - same for all batches.
    key: PRNG key.
    """
    frame_batches, pixel_coords, As_batches, targets_batches, sigmas_batches, mask_batches, time_batches, num_vis_batches = batch_list
    def scan_fn(carry, batch_idx):
        model, opt_state, key = carry
        key, subkey = jax.random.split(key)
        frame_batch = frame_batches[batch_idx]         # shape: (batch_size, H*W)
        target_vis_batch = targets_batches[batch_idx]    # shape: (batch_size, max_vis)
        time_indices = time_batches[batch_idx]           # shape: (batch_size,)
        A = As_batches[batch_idx]                        # shape: (batch_size, max_vis, H*W)
        sigma = sigmas_batches[batch_idx]                # shape: (batch_size, max_vis)
        mask = mask_batches[batch_idx]                   # shape: (batch_size, max_vis)
        num_vis_batch = num_vis_batches[batch_idx]
        
        noise = jax.random.normal(subkey, shape=pixel_coords.shape) * 0.01
        xy_noisy = pixel_coords + noise
        new_model, new_opt_state, loss, rec_loss, ortho_loss, grads = train_step(
            model, opt_state, xy_noisy, frame_batch, target_vis_batch, num_vis_batch,  A, sigma, 
            time_indices, mask, optimizer, alpha, beta, frame_max, frame_min, loss_fn=loss_fn
        )
        return (new_model, new_opt_state, key), (loss, rec_loss, ortho_loss, grads)
    
    num_batches = frame_batches.shape[0]
    init_carry = (model, opt_state, key)
    (final_model, final_opt_state, _), (losses, rec_losses, ortho_losses, grads) = jax.lax.scan(
        scan_fn, init_carry, jnp.arange(num_batches)
    )
    avg_loss = jnp.mean(losses)
    rec_avg = jnp.sum(rec_losses)
    ortho_avg = jnp.sum(ortho_losses)
    return final_model, final_opt_state, avg_loss, rec_avg, ortho_avg, grads

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

def train_model(model, train_loader, num_epochs, key, alpha, beta, data_dir, initial_lr, plots_dir, frame_max, frame_min):
    os.makedirs(plots_dir, exist_ok=True)
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
            epoch_data = train_loader.get_epoch_data(epoch)
            if len(epoch_data) == 0:
                print(f"No full batches for epoch {epoch}—skipping.", flush=True)
                continue
            model, opt_state, avg_loss, rec_loss, ortho_loss, grads = train_epoch_jit(
                model, opt_state, epoch_data, optimizer, alpha, beta, key, frame_max, frame_min
            )
            rec_losses.append(float(rec_loss))
            ortho_losses.append(float(ortho_loss))
            total_losses.append(float(avg_loss))
            if epoch % 10 == 0:
                print_all_gradients(grads, model)
                print_param_norms(model)
            current_lr = scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, r-chi2={float(avg_loss):.6f}, Rec={float(rec_loss):.6f} LR={current_lr:.2e}", flush=True)
            pbar.update(1)
            if avg_loss < previous_loss:
                avg_loss = previous_loss
                eqx.tree_serialise_leaves(os.path.join(data_dir, "trained_model.eqx"), model)

            if epoch % 10 == 0:
                eqx.tree_serialise_leaves(os.path.join(checkpoints_dir, f"e{epoch}.eqx"), model)
                np.savetxt(os.path.join(checkpoints_dir, f"loss{epoch}.txt"), np.array([avg_loss, rec_loss]))
            
            if epoch > 4 and epoch % 2 == 0:
                from_epoch = 2
                plot_losses(rec_losses[from_epoch:], ortho_losses[from_epoch:], total_losses[from_epoch:], plots_dir)
                print(f"Plotted losses up to epoch {epoch+1}.")
                
    return model, total_losses, rec_losses, ortho_losses

models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)
plots_dir = "../../../plots"
os.makedirs(plots_dir, exist_ok=True)
fov_x, fov_y = jnp.pi, jnp.pi

# Taken from generate_data notebook:
hs_data_dir = "../../../hs_data"
array_name = 'ngEHT'
movie_name = "orbiting_hs"
fractional_noise = 0.05
#####################################
obs_path = os.path.join(hs_data_dir, f"{array_name}/{movie_name}_f{fractional_noise}")

frames, times = load_hdf5(obs_path, "gt_video.hdf5") # Load the ground truth video
frame_max, frame_min = frames.max(), frames.min()
n = frames.shape[0]
train_frames = frames[:n,]
time_fraction = 0.6 # Fraction of the total time to use for training
batch_size = 32     # Batch size for training
num_epochs = 20000 # Number of epochs to train the model for
times = (times - times.min()) / (times.max() - times.min()) # normalize times to [0, 1]
train_loader = DMDDataLoader(train_frames, batch_size=batch_size, data_dir=obs_path, 
                                epochs=num_epochs, times=times, fov_x=fov_x, fov_y=fov_y, time_fraction=time_fraction)
r = 24 # number of modes to learn: 12 modes + 12 conjugates + 1 static mode
key = jax.random.PRNGKey(42) # define random key
num_frequencies = 2 # degree of frequencies to use for the positional encoding
model = NeuralDMD(r=r, key=key, hidden_size=256, layers=4, num_frequencies=2,
                 temporal_latent_dim=32, temporal_hidden=64, temporal_layers=2) # initialize the model
continue_training = True # set to True to continue training from a saved model
if continue_training:
    model = eqx.tree_deserialise_leaves(os.path.join(models_dir, "trained_model.eqx"), model)

beta = 0
alpha = 0
lr = 2.5e-4 # learning rate

trained_model, total_losses, reconstruction_losses, orthogonality_losses = train_model(
    model, train_loader, num_epochs, key, alpha, beta, models_dir, lr, plots_dir, frame_max, frame_min
)