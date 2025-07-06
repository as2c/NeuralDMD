import sys
sys.path.append("../../.")
import numpy as np
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
from dmd_data_loader import DMDDataLoader
import jax
from neural_dmd import NeuralDMD, train_model
import equinox as eqx
from util_funcs import load_hdf5


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
model = NeuralDMD(r, key=key, num_frequencies=num_frequencies) # initialize the model
continue_training = True # set to True to continue training from a saved model
if continue_training:
    model = eqx.tree_deserialise_leaves(os.path.join(models_dir, "trained_model.eqx"), model)

beta = 0
alpha = 0
lr = 2.5e-4 # learning rate

trained_model, total_losses, reconstruction_losses, orthogonality_losses = train_model(
    model, train_loader, num_epochs, key, alpha, beta, models_dir, lr, plots_dir, frame_max, frame_min
)