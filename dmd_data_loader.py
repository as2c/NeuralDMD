import os
import numpy as np
import jax.numpy as jnp
from jax import random
import jax

class DMDDataLoader:
    def __init__(self, 
                 data,         # Image cube: shape (num_frames, H, W) [can be numpy or JAX array]
                 batch_size,   # Number of time frames per batch
                 epochs,       # Total number of epochs (for precomputing time indices)
                 data_dir,     # Directory where the npy files ("As.npy", "targets.npy", etc.) are stored
                 times=None,
                 fov_x=1.,    # Field-of-view in microarcseconds (x direction)
                 fov_y=1.,    # Field-of-view in microarcseconds (y direction)
                 time_fraction=1.,  # Fraction (0 to 1) of time frames to use each epoch
                 shuffle=True,
                 seed=42):
        """
        Initialize the data loader.
        
        Parameters:
          data: Numpy array of shape (num_frames, H, W) representing the image cube.
          batch_size: Number of time frames per batch.
          epochs: Number of epochs (for precomputing time indices).
          data_dir: Directory containing the pre-generated npy files:
                    "As.npy", "targets.npy", "sigmas.npy", "masks.npy".
          times: Array of time values corresponding to each frame.
          fov_x, fov_y: Field of view along x and y.
          time_fraction: Fraction of the total time frames to sample in each epoch.
          shuffle: Whether to shuffle the time indices.
          seed: Random seed.
        """
        self.data_dir = data_dir
        self.times = times
        
        # Convert the image cube to a JAX array.
        self.data = jnp.array(data)  # shape: (num_frames, H, W)
        self.num_frames, self.height, self.width = self.data.shape
        self.num_samples = self.height * self.width
        
        # Save FOV details for converting pixel indices.
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.pixel_size_x = fov_x / self.width
        self.pixel_size_y = fov_y / self.height
        
        self.shuffle = shuffle
        self.rng_key = random.PRNGKey(seed)
        
        # Precompute pixel coordinates (using all pixels).
        self.indices = jnp.arange(self.num_samples)
        self.pixel_coords = self._pixel_to_physical(self.indices)
        
        # Time parameters.
        self.time_fraction = time_fraction
        self.num_time_samples = int(self.time_fraction * self.num_frames)
        self.batch_size = batch_size  # number of time frames per batch
        self.epochs = epochs
        

        # Precompute time indices for each epoch.
        self._precompute_time_indices()
        
        # Load the precomputed npy files (these are assumed to have been generated earlier)
        self.As_full = jnp.array(np.load(os.path.join(data_dir, "As.npy")))         # shape: (T_full, max_vis, num_samples)
        print("As shape:", self.As_full.shape)
        self.targets_full = jnp.array(np.load(os.path.join(data_dir, "targets.npy")))   # shape: (T_full, max_vis)
        self.sigmas_full = jnp.array(np.load(os.path.join(data_dir, "sigmas.npy")))     # shape: (T_full, max_vis)
        # max_sigma = jnp.max(self.sigmas_full)
        # min_sigma = jnp.min(self.sigmas_full)
        # self.sigmas_full = self.sigmas_full / min_sigma
        self.masks_full = jnp.array(np.load(os.path.join(data_dir, "masks.npy")))       # shape: (T_full, max_vis)
        self.num_vis_list = jnp.array(np.load(os.path.join(data_dir, "num_vis_list.npy")))
        # We assume T_full = number of time frames in obs_frames.
    
    def _pixel_to_physical(self, pixel_indices):
        """
        Convert pixel indices (0 to num_samples-1) to physical coordinates in microarcseconds.
        Returns a jnp.array of shape (num_samples, 2) where each row is [theta_x, theta_y].
        """
        x_coords = pixel_indices % self.width
        y_coords = pixel_indices // self.width
        theta_x = (x_coords - self.width / 2) * self.pixel_size_x
        theta_y = (y_coords - self.height / 2) * self.pixel_size_y
        return jnp.stack([theta_x, theta_y], axis=-1)
    
    def _precompute_time_indices(self):
        """Precompute a random set of time indices (of length num_time_samples) for each epoch."""
        self.precomputed_time_indices = []
        frame_indices = jnp.arange(self.num_frames)
        for _ in range(self.epochs):
            self.rng_key, subkey = random.split(self.rng_key)
            t_indices = random.choice(subkey, frame_indices, shape=(self.num_time_samples,), replace=False)
            self.precomputed_time_indices.append(t_indices)
    
    def get_epoch_data(self, epoch):
        """
        For the given epoch, select a random subset of time frames and then batch the
        corresponding A matrices, targets, sigma values, and masks over time.
        
        Returns a tuple:
          (As_batches, targets_batches, sigmas_batches, mask_batches, time_batches)
        with shapes:
          - As_batches: (num_batches, batch_size, max_vis, num_samples)
          - targets_batches: (num_batches, batch_size, max_vis)
          - sigmas_batches: (num_batches, batch_size, max_vis)
          - mask_batches: (num_batches, batch_size, max_vis)
          - time_batches: (num_batches, batch_size)
        """
        # Get the precomputed time indices for this epoch.
        time_indices = self.precomputed_time_indices[epoch]
        if self.shuffle:
            self.rng_key, subkey = random.split(self.rng_key)
            time_indices = random.permutation(subkey, time_indices)
        
        # Trim to a multiple of batch_size.
        total_time = time_indices.shape[0]
        total_trim = total_time - (total_time % self.batch_size)
        time_indices = time_indices[:total_trim]
        times = self.times[time_indices]
        time_batches = jnp.reshape(times, (-1, self.batch_size))
        
        # Get ground-truth frame data from the image cube.
        frame_selected = self.data.reshape(self.num_frames, -1)[time_indices, :]  # (total_trim, H*W)
        num_batches = time_batches.shape[0]
        frame_batches = jnp.reshape(frame_selected, (num_batches, self.batch_size, -1))
        
        # Now, select the corresponding entries from the loaded npy arrays.
        # We assume that these arrays have the first dimension corresponding to time.
        As_selected = self.As_full[time_indices, ...]         # shape: (T_trim, max_vis, num_samples)
        targets_selected = self.targets_full[time_indices, ...]   # shape: (T_trim, max_vis)
        sigmas_selected = self.sigmas_full[time_indices, ...]     # shape: (T_trim, max_vis)
        masks_selected = self.masks_full[time_indices, ...]       # shape: (T_trim, max_vis)
        num_vis_selected = self.num_vis_list[time_indices, ...]
        
        num_batches = time_batches.shape[0]
        As_batches = jnp.reshape(As_selected, (num_batches, self.batch_size, *As_selected.shape[1:]))
        targets_batches = jnp.reshape(targets_selected, (num_batches, self.batch_size, *targets_selected.shape[1:]))
        sigmas_batches = jnp.reshape(sigmas_selected, (num_batches, self.batch_size, *sigmas_selected.shape[1:]))
        mask_batches = jnp.reshape(masks_selected, (num_batches, self.batch_size, *masks_selected.shape[1:]))
        num_vis_batches = jnp.reshape(num_vis_selected, (num_batches, self.batch_size, *num_vis_selected.shape[1:]))
        
        return frame_batches, self.pixel_coords, As_batches, targets_batches, sigmas_batches, mask_batches, time_batches, num_vis_batches