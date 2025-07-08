"""
test_model.py

-------
Visualize and evaluate a trained NeuralDMD model on 10 m wind speed data.

----------
- Load a NetCDF file with u10 and v10 winds.
- Convert to wind speed magnitude and build spatial / temporal coordinates
   exactly as used during training.
- Restore a trained NeuralDMD checkpoint.
- Plot:
   - Spatial modes (real and imaginary parts)
   - DMD eigenvalues on the unit circle
- Reconstruct the full wind field and write:
   - Ground-truth and reconstruction GIF animations
   - Per-frame PNG snapshots every 10 steps
   - NetCDF file with the reconstructed data

Outputs
-------
plots/
    spatial_modes_real.png
    spatial_modes_imag.png
    unit_circle.png
    ground_truth.gif
    reconstruction.gif
    W_<n>.gif
rec_analyses.nc   (reconstructed wind dataset)
"""


import os
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import imageio.v3 as iio
from neuraldmd import NeuralDMD
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -----------------------------------------------------------------------------
# plotting utilities
# -----------------------------------------------------------------------------
def plot_modes(W, lons, lats, file_path, title, real_part=True):
    """
    W : np.ndarray, shape (nlat*nlon, r)
    lons, lats : 1D arrays of the native grid (degrees east, degrees north)
    """
    nlat, nlon = len(lats), len(lons)
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    # print(lat_min, lat_max, lon_min, lon_max)
    # breakpoint()

    r = W.shape[1]
    cols = 6
    rows = (r + cols - 1) // cols

    # bump up the figsize so each map is ~4"×4"
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4, rows * 4),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    axes = axes.flatten()

    # pick first mode for vmin/vmax
    first = np.real(W[:,0]) if real_part else np.imag(W[:,0])
    vmin, vmax = first.min(), first.max()
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i in range(r):
        ax = axes[i]
        data = (np.real(W[:,i]) if real_part else np.imag(W[:,i]))
        data = data.reshape(nlat, nlon)

        im = ax.imshow(
            data,
            origin='lower',
            extent=[lon_min, lon_max, lat_min, lat_max],
            transform=ccrs.PlateCarree(),
            cmap='inferno',
            norm=norm
        )
        ax.coastlines(resolution='50m')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
        ax.set_title(f"{'Re' if real_part else 'Im'} mode {i+1}", fontsize=10)
        ax.axis('off')

        # small colorbar inside each subplot
        fig.colorbar(im, ax=ax, orientation='vertical',
                     fraction=0.04, pad=0.02)

    # remove empty subplots
    for j in range(r, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(pad=1.0, rect=[0,0,1,0.95])
    fig.savefig(file_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# def plot_modes(W, height, width, file_path, title, real_part=True):
#     """
#     Plot either real or imaginary part of each column in W.
#     W: (height*width, r)
#     """
#     r = W.shape[1]
#     cols = 6
#     rows = (r + cols - 1) // cols
#     fig, axes = plt.subplots(
#         rows, cols,
#         figsize=(cols * 3.5, rows * 3.5),
#         subplot_kw={'projection': ccrs.PlateCarree()},
#     )
#     axes = axes.flatten()

#     # pick first mode to set vmin/vmax
#     base = (np.real(W[:,0]) if real_part else np.imag(W[:,0]))
#     base = base.reshape(height, width)
#     vmin, vmax = base.min(), base.max()
#     norm = Normalize(vmin=vmin, vmax=vmax)

#     for i in range(r):
#         data = (np.real(W[:,i]) if real_part else np.imag(W[:,i])).reshape(height, width)
#         ax = axes[i]
#         im = ax.imshow(
#             data, origin='lower',
#             transform=ccrs.PlateCarree(),
#             cmap='inferno', norm=norm
#         )
#         ax.coastlines(resolution='50m')
#         ax.set_title(f"{'Re' if real_part else 'Im'} mode {i+1}", fontsize=8)
#         ax.axis('off')
#         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     # remove unused axes
#     for j in range(r, len(axes)):
#         fig.delaxes(axes[j])

#     plt.suptitle(title, fontsize=14)
#     plt.tight_layout(pad=1.0)
#     plt.savefig(file_path, dpi=200, bbox_inches="tight")
    # plt.close()

def plot_modes_normalized(W, file_dir, real_part=True):
    """
    W : np.ndarray, shape (nlat*nlon, r)
    lons, lats : 1D arrays of the native grid (degrees east, degrees north)
    """
    nlat, nlon = len(lats), len(lons)
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()

    r = W.shape[1]
    cols = 6
    rows = (r + cols - 1) // cols

    # bump up the figsize so each map is ~4"×4"
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4, rows * 4),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    axes = axes.flatten()
    # pick first mode for vmin/vmax
    # W_0 = np.abs(W[:, 0] / jnp.linalg.norm(W[:, 0]))
    # first = np.real(W_0) if real_part else np.imag(W_0)
    # vmin, vmax = first.min(), first.max()
    # norm = Normalize(vmin=vmin, vmax=vmax)

    for i in range(r):
        # W_i = W[:, i] / jnp.linalg.norm(W[:, i]) * -1
        W_i = W[:, i]
        if i == 0:
            W_i = np.abs(W_i)

        ax = axes[i]
        data = (np.real(W_i) if real_part else np.imag(W_i))
        data = data.reshape(nlat, nlon)

        im = ax.imshow(
            data,
            origin='lower',
            extent=[lon_min, lon_max, lat_min, lat_max],
            transform=ccrs.PlateCarree(),
            cmap='inferno'
        )
        ax.coastlines(resolution='50m')
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
        # ax.set_title(f"{'Re' if real_part else 'Im'} mode {i+1}", fontsize=10)
        ax.axis('off')

        # small colorbar inside each subplot
        fig.colorbar(im, ax=ax, orientation='vertical',
                     fraction=0.04, pad=0.02)

    # remove empty subplots
    for j in range(r, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=1.0, rect=[0,0,1,0.95])
    fig.savefig(file_dir, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_unit_circle(Omega_full, file_path):
    """
    Plot all e^{Ω} on the complex plane with unit‐circle.
    """
    Λ = np.exp(Omega_full)
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter([1, *Λ.real], [0, *Λ.imag], s=20)
    θ = np.linspace(0,2*np.pi,400)
    ax.plot(np.cos(θ), np.sin(θ), '--', color='gray')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    # ax.set_title("DMD Spectrum (all modes)")
    # cbar = plt.colorbar(sc, ax=ax, pad=0.1, fraction=0.046)
    # cbar.set_label("mode index")
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    nc_file    = "/home/as2c/gfs_data/20250401_07/uv_weather/data_stream-oper_stepType-instant.nc"
    model_file = "/home/as2c/iccv_paper/weather/neural_dmd/data/trained_model.eqx"
    out_dir    = "./plots"
    os.makedirs(out_dir, exist_ok=True)

    # 1) load & wrap & slice to native lon/lat box
    ds = xr.open_dataset(nc_file)
    ds = ds.assign_coords(
        longitude=(((ds.longitude + 180) % 360) - 180)
    ).sortby('longitude')
    lats = ds.latitude.values
    lons = ds.longitude.values
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    print(lat_min, lat_max, lon_min, lon_max)
    # 2) extract wind [T, nlat, nlon]
    wind = np.hypot(ds.u10, ds.v10).values
    T, nlat, nlon = wind.shape

    # 3) build normalized spatial coords exactly as in loader
    fov = np.pi
    # a) true lon/lat grid in radians
    lon2d, lat2d = np.meshgrid(lons, lats, indexing='xy')
    lon2d = np.deg2rad(lon2d)
    lat2d = np.deg2rad(lat2d)
    # b) center & min–max → [–0.5, +0.5] → scale by fov
    lon_center = 0.5 * (lon2d.min() + lon2d.max())
    lat_center = 0.5 * (lat2d.min() + lat2d.max())
    lon_flat = (lon2d - lon_center).ravel()
    lat_flat = (lat2d - lat_center).ravel()
    lon_frac = lon_flat / (lon_flat.max() - lon_flat.min())
    lat_frac = lat_flat / (lat_flat.max() - lat_flat.min())
    lon_scaled = lon_frac * fov
    lat_scaled = lat_frac * fov
    xy = np.stack([lon_scaled, lat_scaled], axis=-1)   # (nlat*nlon, 2)

    # 4) normalize times to [0,1] using actual valid_time
    times = ds.valid_time.values
    t0, t1 = times.min(), times.max()
    dt = (times - t0).astype('timedelta64[ns]').astype(float)
    tot = (t1 - t0).astype('timedelta64[ns]').astype(float)
    t_norm = (dt / tot).astype(float)                 # shape (T,)

    # 5) load model & compute modes
    key = jax.random.PRNGKey(0)
    r = 40
    model = NeuralDMD(r=r, key=key, num_frequencies=4, hidden_size=512, layers=10)
    model = eqx.tree_deserialise_leaves(model_file, model)

    # spatial
    W0, W_half, W = jax.vmap(model.spatial_forward)(jnp.array(xy))
    W0      = np.array(W0)        # (nlat*nlon, 1)
    W_half  = np.array(W_half)    # (nlat*nlon, r_half)
    W       = np.array(W)         # (nlat*nlon, r)
    # temporal
    alphas, thetas = model.temporal_omega()
    Omega  = np.array(alphas + 1j*thetas)  # (r_half,)
    b0, b_half     = model.temporal_b()
    b0      = float(np.array(b0))
    b_half  = np.array(b_half)            # (r_half,)
    # full spectrum + coefficients
    Omega_full = np.concatenate([Omega, np.conj(Omega)])
    b_full     = np.concatenate([b_half, [b0], np.conj(b_half)])

    Lambda = np.exp(Omega)
    sort_idx = np.argsort(np.real(Lambda))[::-1]
    W_sorted = np.concatenate([W0, W_half[:, sort_idx]], axis=1)
    Omega_sorted = np.concatenate([np.array([0]), Omega[sort_idx]])
    # 6) plot spatial modes
    # plot_modes(W, nlat, nlon, os.path.join(out_dir,'spatial_modes_real.png'),
    #            "Spatial modes (real part)", real_part=True)
    # plot_modes(W, nlat, nlon, os.path.join(out_dir,'spatial_modes_imag.png'),
    #            "Spatial modes (imag part)", real_part=False)
    # plot_modes(W_sorted, lons, lats,
    #        "./plots/spatial_modes_real.png",
    #        "Spatial modes (real part)",
    #        real_part=True)
    
    plot_modes_normalized(W_sorted[:, :4], os.path.join("./plots", "W_half_sorted_normalized.pdf"), real_part=True)
    plot_modes_normalized(W_sorted[:, :4], os.path.join("./plots", "W_im_half_sorted_normalized.pdf"), real_part=False)
    
    plot_modes(W_sorted, lons, lats,
            "./plots/spatial_modes_imag.png",
            "Spatial modes (imag part)",
            real_part=False)
    
    # 7) plot full spectrum
    plot_unit_circle(Omega_full, os.path.join(out_dir,'unit_circle.png'))

    # 8) ground-truth GIF
    vmin, vmax = wind.min(), wind.max()
    gt_frames = []
    for t in tqdm(range(T)):
        fig, ax = plt.subplots(1,1,figsize=(8,5),
                               subplot_kw={'projection':ccrs.PlateCarree()})
        pcm = ax.pcolormesh(
            lons, lats, wind[t],
            transform=ccrs.PlateCarree(),
            cmap='viridis', vmin=vmin, vmax=vmax,
            shading='auto'
        )
        ax.set_extent([lon_min,lon_max,lat_min,lat_max], ccrs.PlateCarree())
        ax.coastlines(resolution='50m')
        # ax.set_title(f"True wind (frame {t})", fontsize=10)
        fig.tight_layout()

        # grab RGB buffer
        canvas = fig.canvas; canvas.draw()
        w,h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(h,w,4)
        gt_frames.append(buf[:,:,1:4])
        if t % 10 == 0:
            plt.savefig(os.path.join(out_dir,f"gt_{t}.png"),
                        dpi=200, bbox_inches='tight')
        plt.close(fig)

    iio.imwrite(os.path.join(out_dir,'ground_truth.gif'),
                gt_frames, duration=0.2)

    # 9) reconstruct using the same codepath as training
    lambda_exp = np.exp(Omega[:,None] * t_norm[None,:] * 160.0)   # (r_half, T)
    Xrec0 = 2*np.real(np.einsum('br,rt,r->bt', W_half, lambda_exp, b_half)) \
            + (W0[:,0,None] * b0)
    # un‐normalize back to physical m/s
    Xrec = Xrec0 * (vmax - vmin) + vmin
    rec = Xrec.T.reshape(T,nlat,nlon)

    # 10) reconstruction GIF
    rec_frames = []
    for t in tqdm(range(T)):
        fig, ax = plt.subplots(1,1,figsize=(8,5),
                               subplot_kw={'projection':ccrs.PlateCarree()})
        pcm = ax.pcolormesh(
            lons, lats, rec[t],
            transform=ccrs.PlateCarree(),
            cmap='viridis', vmin=vmin, vmax=vmax,
            shading='auto'
        )
        ax.set_extent([lon_min,lon_max,lat_min,lat_max], ccrs.PlateCarree())
        ax.coastlines(resolution='50m')
        # ax.set_title(f"Reconstruction (frame {t})", fontsize=10)
        fig.tight_layout()

        canvas = fig.canvas; canvas.draw()
        w,h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(h,w,4)
        rec_frames.append(buf[:,:,1:4])
        if t % 10 == 0:
            plt.savefig(os.path.join(out_dir,f"reconstruction_{t}.png"),
                        dpi=200, bbox_inches='tight')
            
        plt.close(fig)

    iio.imwrite(os.path.join(out_dir,'reconstruction.gif'),
                rec_frames, duration=0.2)

    da_rec = xr.DataArray(
        data=rec,
        dims=["valid_time","latitude","longitude"],
        coords={
            "valid_time": ds.valid_time,    # original timestamp coordinate
            "latitude": ds.latitude,
            "longitude": ds.longitude,
        },
        name="wind2m_rec"
    )
    ds_rec = da_rec.to_dataset()

    rec_nc = "./rec_analyses.nc"
    if os.path.exists(rec_nc):
        os.remove(rec_nc)
    ds_rec.to_netcdf(rec_nc)
    print(f"Saved NeuralDMD reconstructions to {rec_nc}")

    n = 1
    lambda_exp = np.exp(Omega_sorted[n] * t_norm * 160.0)
    W_rec = 2*np.real(np.einsum('b,t->bt', W_sorted[:, n], lambda_exp) * b_half[n - 1])
    vmin, vmax = W_rec.min(), W_rec.max()
    rec = W_rec.T.reshape(T, nlat, nlon)
    rec_frames = []
    for t in tqdm(range(T)):
        fig, ax = plt.subplots(1,1,figsize=(8,5),
                               subplot_kw={'projection':ccrs.PlateCarree()})
        pcm = ax.pcolormesh(
            lons, lats, rec[t],
            transform=ccrs.PlateCarree(),
            cmap='viridis', vmin=vmin, vmax=vmax,
            shading='auto'
        )
        ax.set_extent([lon_min,lon_max,lat_min,lat_max], ccrs.PlateCarree())
        ax.coastlines(resolution='50m')
        # ax.set_title(f"Reconstruction (frame {t})", fontsize=10)
        fig.tight_layout()

        canvas = fig.canvas; canvas.draw()
        w,h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(h,w,4)
        rec_frames.append(buf[:,:,1:4])
        if t % 10 == 0:
            plt.savefig(os.path.join(out_dir,f"reconstruction_{t}.png"),
                        dpi=200, bbox_inches='tight')
            
        plt.close(fig)

    iio.imwrite(os.path.join(out_dir, f'W_{n}.gif'),
                rec_frames, duration=0.2)