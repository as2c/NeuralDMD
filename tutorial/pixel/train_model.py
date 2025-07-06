"""
train_model.py

Train a NeuralDMD model on near-surface wind data (u10, v10) saved in a
GRIB-to-NetCDF file.

Pipeline
--------
1. Create ./data and ./plots directories.
2. Load the dataset with xarray.
3. Build a WeatherDMDDataLoader that picks N_sensors random stations and
   streams a fraction of the time steps.
4. Get global min and max for normalizing frames.
5. Instantiate a NeuralDMD model with rank r and Fourier features.
6. Optionally resume from an Equinox checkpoint.
7. Train for num_epochs, logging checkpoints and loss plots.
"""


import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import os
import xarray as xr
from dmd_data_loader import WeatherDMDDataLoader
from neural_dmd import NeuralDMD, train_model

if __name__ == "__main__":
    # Create output directories
    data_dir = "./data"
    plots_dir = "./plots"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Path to your GRIB-derived NetCDF of u10/v10
    data_path = "./data/data_stream-oper_stepType-instant.nc"

    # 1) Load the dataset
    ds = xr.open_dataset(data_path)

    # 2) Build a data loader that simulates sparse sensors
    total_N =  146401
    sensor_fraction = 0.1
    N_sensors     = int(sensor_fraction * total_N)        # number of random stations
    time_fraction = 0.3        # use all time steps
    batch_size    = 128
    fov_lon = np.pi
    fov_lat = np.pi
    num_epochs    = 5000
    seed = 42
    loader = WeatherDMDDataLoader(
        ds,
        N_sensors=N_sensors,
        time_fraction=time_fraction,
        sensor_batch_size=batch_size,
        fov_lon=fov_lon,
        fov_lat=fov_lat,
        seed=seed
    )

    # 3) Compute normalization range for wind field
    frame_max = float(loader.w_max)
    frame_min = float(loader.w_min)
    # 4) Instantiate the NeuralDMD model
    r   = 40
    key = jax.random.PRNGKey(seed)
    model = NeuralDMD(r, key=key, num_frequencies=4)

    # 5) Optionally continue training from checkpoint
    continue_training = False
    if continue_training:
        ckpt_path = os.path.join(data_dir, "trained_model.eqx")
        if os.path.exists(ckpt_path):
            model = eqx.tree_deserialise_leaves(ckpt_path, model)

    # 6) Train!
    beta       = 0.0      # weight on orthogonality loss
    learning_rate = 1e-4

    trained_model, total_losses, rec_losses, ortho_losses = train_model(
        model,
        loader,
        num_epochs,
        key,
        beta,
        data_dir,
        learning_rate,
        plots_dir,
        frame_max,
        frame_min
    )
    print("Training complete.")