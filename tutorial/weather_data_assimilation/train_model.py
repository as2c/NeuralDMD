# tutorial/pixel/train_model.py

import os, numpy as np, xarray as xr
import jax, jax.numpy as jnp
import optax, equinox as eqx
from tqdm import tqdm

from dmd_data_loader import WeatherDMDDataLoader
from neuraldmd import NeuralDMD
from neuraldmd.losses import pixel_loss_fn
from neuraldmd.training import train_step_pixels as train_step
from neuraldmd.scheduler import PlateauScheduler
import matplotlib.pyplot as plt

def plot_losses(rec_losses, ortho_losses, total_losses, output_dir):
    epochs = range(1, len(rec_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rec_losses, label="Reconstruction Loss")
    # plt.plot(epochs, ortho_losses, label="Orthogonality Loss")
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
            model, opt_state, xy_noisy, pixels, times, optimizer, beta, frame_max, frame_min, pixel_loss_fn
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


def train_model(
    model,
    train_loader,
    num_epochs,
    key,
    beta,
    data_dir,
    initial_lr,
    plots_dir,
    frame_max,
    frame_min
):
    # Initialize learning rate scheduler and optimizer
    global scheduler
    scheduler = PlateauScheduler(initial_lr=initial_lr)
    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=schedule_fn, weight_decay=1e-4
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Tracking losses
    total_losses = []
    rec_losses = []
    ortho_losses = []

    # Setup checkpointing
    previous_loss = jnp.inf
    checkpoints_dir = "./checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Training loop
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            coords_b, pix_b, time_b = train_loader.get_epoch_data()

            model, opt_state, avg_loss, rec_loss, ortho_loss, grads = train_epoch_jit(
                model,
                opt_state,
                coords_b,
                pix_b,
                time_b,
                optimizer,
                beta,
                key,
                frame_max,
                frame_min
            )

            total_losses.append(float(avg_loss))
            rec_losses.append(float(rec_loss))
            ortho_losses.append(float(ortho_loss))

            # Print gradient statistics every 10 epochs
            if epoch % 10 == 0:
                print_all_gradients(grads, model)
                print_param_norms(model)

            # Step learning rate scheduler
            current_lr = scheduler.step(avg_loss)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss={float(avg_loss):.6f}, "
                f"Rec={float(rec_loss):.6f}, LR={current_lr:.2e}",
                flush=True
            )
            pbar.update(1)

            # Save intermediate checkpoint
            if epoch % 100 == 0:
                eqx.tree_serialise_leaves(
                    os.path.join(checkpoints_dir, "trained_model.eqx"), model
                )

            # Save best model
            if avg_loss < previous_loss:
                previous_loss = avg_loss
                eqx.tree_serialise_leaves(
                    os.path.join(data_dir, "trained_model.eqx"), model
                )

            # Plot and save loss curves
            if epoch > 4 and epoch % 2 == 0:
                from_epoch = 2
                plot_losses(
                    rec_losses[from_epoch:],
                    ortho_losses[from_epoch:],
                    total_losses[from_epoch:],
                    plots_dir
                )
                print(f"Plotted losses up to epoch {epoch+1}.")

    return model, total_losses, rec_losses, ortho_losses



def main():
    # Config
    DATA_NC     = "./data/data_stream-oper_stepType-instant.nc"
    MODEL_DIR   = "./models"
    SEED        = 42
    RANK_R      = 40
    NUM_FREQ    = 4
    SENSOR_FRAC = 0.10
    TIME_FRAC   = 0.30
    BATCH_SIZE  = 128
    NUM_EPOCHS  = 1000
    LEARNING_RATE = 1e-4
    CKPT_EVERY  = 100
    CONTINUE_TRAINING = True
    PLOTS_DIR  = "./plots"
    BETA = 0

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load Dataset
    ds = xr.open_dataset(DATA_NC)
    N_sensors = int(SENSOR_FRAC * ds.latitude.size * ds.longitude.size)

    loader = WeatherDMDDataLoader(
        ds,
        N_sensors         = N_sensors,
        sensor_batch_size = BATCH_SIZE,
        time_fraction     = TIME_FRAC,
        fov_lon           = np.pi,
        fov_lat           = np.pi,
        seed              = SEED,
    )
    frame_max, frame_min = loader.orig_max, loader.orig_min

    # Model
    key = jax.random.PRNGKey(SEED)
    model = NeuralDMD(r=RANK_R, key=key, num_frequencies=NUM_FREQ, hidden_size=512, layers=10)

    ckpt_path = os.path.join(MODEL_DIR, "trained_model.eqx")
    if CONTINUE_TRAINING and os.path.exists(ckpt_path):
        model = eqx.tree_deserialise_leaves(ckpt_path, model)
        print(f"Resumed from {ckpt_path}")

    trained_model, total_losses, rec_losses, ortho_losses = train_model(
        model,
        loader,
        NUM_EPOCHS,
        key,
        BETA,
        MODEL_DIR,
        LEARNING_RATE,
        PLOTS_DIR,
        frame_max,
        frame_min
    )

    print("Training complete.")


if __name__ == "__main__":
    main()
