import numpy as np
import xarray as xr

class WeatherDMDDataLoader:
    """
    DataLoader for NeuralDMD on weather data using fully spatiotemporal minibatches:
      - A fixed set of N_sensors is chosen once (seeded) and never changes.
      - Each epoch (call to get_epoch_data) samples num_batches batches.
      - For each batch, randomly sample a small set of time indices (time_batch_size)
        and a small set of sensors (sensor_batch_size) from the N_sensors pool.
    """
    def __init__(
        self,
        ds: xr.Dataset,
        N_sensors: int = 100,
        sensor_batch_size: int = 32,
        time_fraction: float = 1.0,
        num_batches: int = None,
        fov_lon: float = np.pi,
        fov_lat: float = np.pi,
        seed: int = 0,
    ):
        # Wrap longitudes into [-180, +180] and sort
        ds2 = ds.assign_coords(
            longitude=(((ds.longitude + 180) % 360) - 180)
        ).sortby("longitude")

        # Convert grid coords to radians
        lons = np.deg2rad(ds2.longitude.values)  # (nlon,)
        lats = np.deg2rad(ds2.latitude .values)  # (nlat,)
        nlat, nlon = lats.size, lons.size

        # Build flattened meshgrid in lat-major (matching wind data layout)
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='xy')  # (nlat,nlon)
        lon_flat = lon_grid.ravel()  # (nlat*nlon,)
        lat_flat = lat_grid.ravel()

        # Normalize coords into [-fov/2, +fov/2]
        lon_min, lon_max = lon_flat.min(), lon_flat.max()
        lat_min, lat_max = lat_flat.min(), lat_flat.max()
        lon_frac = (lon_flat - lon_min) / (lon_max - lon_min)  # [0,1]
        lat_frac = (lat_flat - lat_min) / (lat_max - lat_min)
        lon_c = (lon_frac - 0.5) * fov_lon
        lat_c = (lat_frac - 0.5) * fov_lat
        coords_flat = np.stack([lon_c, lat_c], axis=-1)  # (npix,2)

        # Load and min–max normalize wind speed over full grid
        wind3d = np.hypot(ds2.u10, ds2.v10).values  # (T,nlat,nlon)
        self.orig_min, self.orig_max = wind3d.min(), wind3d.max()
        wind3d = (wind3d - self.orig_min) / (self.orig_max - self.orig_min)
        T = wind3d.shape[0]

        # Flatten only the chosen N_sensors once
        rng = np.random.default_rng(seed)
        npix = nlat * nlon
        sensor_idx = rng.choice(npix, size=N_sensors, replace=False)
        self.coords_sensors = coords_flat[sensor_idx]        # (N_sensors, 2)
        wind2d = wind3d.reshape(T, -1)                       # (T, npix)
        self.wind2d_sensors = wind2d[:, sensor_idx]         # (T, N_sensors)
        self.w_min, self.w_max = wind2d.min(), wind2d.max()

        # Normalize time stamps to [0,1]
        times = ds2.valid_time.values.astype('datetime64[ns]')  # (T,)
        t0, t1 = times.min(), times.max()
        dt = (times - t0).astype('timedelta64[ns]').astype(np.float64)
        tot = (t1 - t0).astype('timedelta64[ns]').astype(np.float64)
        self.time_norm = dt / tot  # (T,)

        # Store parameters
        self.T = T
        self.N_sensors = N_sensors
        self.sensor_batch_size = sensor_batch_size
        self.time_batch_size = int(time_fraction * self.T)
        # default num_batches: cover each sensor once per epoch
        if num_batches is None:
            self.num_batches = N_sensors // sensor_batch_size
        else:
            self.num_batches = num_batches
        self.rng = rng

    def get_epoch_data(self):
        """
        Returns arrays for all batches in this epoch:
          coords_batches: shape (num_batches, sensor_batch_size, 2)
          data_batches:   shape (num_batches, sensor_batch_size, time_batch_size)
          time_batches:   shape (num_batches, time_batch_size)
        """
        B = self.num_batches
        S = self.sensor_batch_size
        T_b = self.time_batch_size

        # Sample sensors and times for each batch in a vectorized way
        pix_idx = self.rng.choice(
            self.N_sensors, size=(B, S), replace=True
        )  # (B, S)
        t_idx = self.rng.choice(
            self.T, size=(B, T_b), replace=True
        )  # (B, T_b)

        # coords_batches[b] = coords of sensors in batch b
        coords_batches = self.coords_sensors[pix_idx]  # (B, S, 2)

        # data_temp[b] = wind2d_sensors[t_idx[b]] → shape (T_b, N_sensors)
        data_temp = self.wind2d_sensors[t_idx]         # (B, T_b, N_sensors)
        # pick only the sensors in each batch and transpose to (S, T_b)
        data_sel = np.take_along_axis(
            data_temp,
            pix_idx[:, None, :],  # broadcast to (B,1,S)
            axis=2
        )  # (B, T_b, S)
        data_batches = data_sel.transpose(0, 2, 1)     # (B, S, T_b)

        # time_norm_batches[b] = normalized times for this batch
        time_batches = self.time_norm[t_idx]           # (B, T_b)

        return coords_batches, data_batches, time_batches
