import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import imageio.v3 as iio
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_modes(W, height, width, file_dir, title):
    r = W.shape[1]
    cols = 6
    rows = r // cols + (r % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 2 * rows))
    axes = axes.flatten()
    
    first_mode = np.real(W[:, 0].reshape(height, width))
    vmin, vmax = first_mode.min(), first_mode.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    for i in range(r):
        mode_i = np.real(W[:, i].reshape(height, width))
        im = axes[i].imshow(mode_i, cmap='inferno', norm=norm)
        axes[i].set_title(f"{title} Mode {i+1}")
        axes[i].axis("off")
        cbar = fig.colorbar(im, ax=axes[i])
        cbar.set_ticks([vmin, (vmin+vmax)/2, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{(vmin+vmax)/2:.2f}", f"{vmax:.2f}"])
    for j in range(r, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(file_dir)

def plot_modes_normalized(W, height, width, file_dir, title):
    r = W.shape[1]
    cols = r  # Ensure we don't request more columns than modes available
    rows = 1  # We are only displaying one row

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))  # 1 row, up to 5 columns

    axes = np.atleast_1d(axes)  # Ensure axes is always an iterable array

    for i in range(cols):  # Loop over available modes
        W_i = W[:, i] / jnp.linalg.norm(W[:, i])
        mode_i = np.real(W_i.reshape(height, width))
        if i == 0:
            mode_i = np.abs(mode_i)
        # Set min/max for each mode
        vmin, vmax = mode_i.min(), mode_i.max()
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot mode
        im = axes[i].imshow(mode_i, cmap='inferno', norm=norm)
        axes[i].set_title(f"Mode {i+1}")
        axes[i].axis("off")

        # Create smaller colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        # Set custom tick labels for each colorbar
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{(vmin+vmax)/2:.2f}", f"{vmax:.2f}"])

    fig.subplots_adjust(wspace=0.5, right=0.85)
    fig.suptitle("Neural DMD Modes/ngEHT Coverage", fontsize=20, y=0.77)
    plt.savefig(file_dir, dpi=1200, bbox_inches='tight', pad_inches=0)

def plot_circle_Lambda(Lambda_tilde, title, plot_dir):
    plt.figure(figsize=(8,8))
    plt.scatter(Lambda_tilde.real, Lambda_tilde.imag, c=range(len(Lambda_tilde)))
    plt.colorbar(label="index")
    plt.title(title)
    theta = np.linspace(0, 2*np.pi, 500)
    x = np.cos(theta)
    y = np.sin(theta)
    plt.plot(x, y, label="Unit Circle")
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir)

def make_gif(frames, num_frames, plots_dir, gif_name):
    gif_frames = []
    temp_file = os.path.join(plots_dir, "temp.png")
    for i in range(num_frames):
        fig, ax = plt.subplots()
        ax.imshow(frames[i], cmap='afmhot', vmin=0, vmax=1)
        ax.axis("off")
        plt.savefig(temp_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        gif_frames.append(iio.imread(temp_file))
    gif_path = os.path.join(plots_dir, gif_name)
    iio.imwrite(gif_path, gif_frames, duration=20, loop=0)
    os.remove(temp_file)

def load_hdf5(dir, file):
    with h5py.File(f"{dir}/{file}", "r") as f:
        frames = f["I"][:]
        times = f["times"][:]
    return frames, times

def gram_schmidt(W):
    N, r = W.shape
    W_orth = jnp.zeros_like(W, dtype=W.dtype)
    for i in range(r):
        w_i = W[:, i]
        for j in range(i):
            w_j = W_orth[:, j]
            proj = jnp.dot(jnp.conj(w_j), w_i) / jnp.dot(jnp.conj(w_j), w_j)
            w_i -= proj * w_j
        W_orth = W_orth.at[:, i].set(w_i / jnp.linalg.norm(w_i))
    return jnp.array(W_orth)

def calc_psnr(frame1, frame2, max_pixel_value=1.0):
    mse = jnp.mean((frame1-frame2)**2)
    if mse==0:
        return float('inf')
    psnr = 10 * jnp.log10((max_pixel_value**2)/mse)
    return psnr

def pixel_to_physical(x_grid, y_grid, width, height, pixel_size_x, pixel_size_y):
    theta_x = (x_grid - width/2) * pixel_size_x
    theta_y = (y_grid - height/2) * pixel_size_y
    return np.stack([theta_x.flatten(), theta_y.flatten()], axis=-1)