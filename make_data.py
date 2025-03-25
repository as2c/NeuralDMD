import ehtim as eh
from astropy import units as u
from ehtim.imaging.imager_utils import chisqdata, chisqdata_vis
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
import h5py
import ngehtsim.obs.obs_generator as og
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

def save_movie_to_hdf5(im_list, movie, filename):
    """
    Save an ehtim Movie object to an HDF5 file.
    
    Parameters:
      movie: an ehtim Movie object (e.g., obtained via eh.movie.load_hdf5 or created otherwise)
      filename: the output filename, e.g. "my_movie.hdf5"
    """
    # Get the list of image frames
    n_frames = len(im_list)
    
    # Assume each image stores its pixel data as a flattened array in _imdict['I']
    # and that each image has dimensions xdim and ydim.
    xdim = im_list[0].xdim
    ydim = im_list[0].ydim

    # Create an array to hold all frames.
    frames = np.empty((n_frames, ydim, xdim), dtype=np.float32)
    for i, im in enumerate(im_list):
        # Reshape the flattened image data into a 2D array.
        frames[i, :, :] = im._imdict['I'].reshape(ydim, xdim)
    
    # Save times and frames to HDF5.
    with h5py.File(filename, "w") as f:
        f.create_dataset("times", data=np.array(movie.times))
        f.create_dataset("I", data=frames)
        # Optionally, you can store additional metadata such as:
        f.attrs['xdim'] = xdim
        f.attrs['ydim'] = ydim
        f.attrs['psize'] = im_list[0].psize
        # You might also save other metadata like ra, dec, etc.
        # For example:
        f.attrs['ra'] = im_list[0].ra
        f.attrs['dec'] = im_list[0].dec
        f.attrs['rf'] = im_list[0].rf
        f.attrs['mjd'] = im_list[0].mjd
        
def resize_images(grmhd_ims, npix, npix_old, new_psize):
    """
    Resize a list of ehtim Image objects to a new size, and adjust the flux scaling accordingly.
    """
    new_images = []
    for i, im in enumerate(grmhd_ims):
        # Reshape the original flattened image into a 2D array
        im_orig = im._imdict['I'].reshape(npix_old, npix_old)
        
        # Resize the image to (npix_new, npix_new)
        im_resized = resize(im_orig, (npix, npix), anti_aliasing=True)
        im_resized = np.clip(im_resized, 0, None)
        # Adjust the flux scaling to account for the change in pixel area
        flux_scaling = (npix_old**2) / (npix**2)  # (400^2)/(100^2) = 16
        im_resized *= flux_scaling
        ##########
        new_im = eh.image.Image(im_resized, 
                                ra=im.ra,          # copy right ascension
                                dec=im.dec,        # copy declination
                                rf=im.rf,          # observing frequency
                                mjd=im.mjd,        # observation time
                                psize=new_psize)
        new_images.append(new_im)
    return new_images

def create_obs(image_true, grmhd_movie, npix, array, is_array):
    t_frames = grmhd_movie.times * u.h
    # image_true = grmhd_movie.get_frame(0)
    # image_true = grmhd_movie.im_list()[0]
    tstart_hr = grmhd_movie.times[0]
    tstop_hr = grmhd_movie.times[-1]
    t_gather = (t_frames[-1] - t_frames[0]).to('s').value / (len(t_frames) - 1) # 105.68

    tint = 5
    tadv = np.floor(t_gather - tint)
    # Generate synthetic observations
    if is_array:
        settings = {'source': 'SgrA',
                'frequency': image_true.rf/1e9,
                'bandwidth': 2.0,
                'month': 'Apr',
                'day': '21',
                'year': '2022',
                't_start': tstart_hr,         # start time of observation, in hours
                'dt': tstop_hr - tstart_hr,             # total duration of observation, in hours
                't_int': tint,          # integration time, in seconds
                't_rest': tadv,        # time interval between consecutive integrations, in seconds
                'array': array,
                'ttype': 'direct',     # type of Fourier transform ('direct', 'nfft', or 'fast')
                'fft_pad_factor': 2,   # zero pad the image to fft_pad_factor * image size in the FFT
                'random_seed': 42,  # random number seed; if blank, will be auto-generated
                'weather': 'exact'
            }
    else:
        settings = {'source': 'SgrA',
            'frequency': image_true.rf/1e9,
            'bandwidth': 2.0,
            'month': 'Apr',
            'day': '21',
            'year': '2022',
            't_start': tstart_hr,         # start time of observation, in hours
            'dt': tstop_hr - tstart_hr,             # total duration of observation, in hours
            't_int': tint,          # integration time, in seconds
            't_rest': tadv,        # time interval between consecutive integrations, in seconds
            'sites': array,
            'ttype': 'direct',     # type of Fourier transform ('direct', 'nfft', or 'fast')
            'fft_pad_factor': 2,   # zero pad the image to fft_pad_factor * image size in the FFT
            'random_seed': 42,  # random number seed; if blank, will be auto-generated
            'weather': 'exact'
           }
    
    
    obsgen = og.obs_generator(settings)
    obs = obsgen.make_obs(image_true)
    # obs_old = array.obsdata(tint=tint, tadv=tadv, tstart=tstart_hr, tstop=tstop_hr, ra=image_true.ra, 
    #                     dec=image_true.dec, rf=image_true.rf, mjd=image_true.mjd,
    #                     bw=image_true.rf, timetype='UTC', polrep='stokes')

    fov = image_true.fovx()
    obs_frames = obs.split_obs(t_gather=t_gather) # NOTE: What does this (len(t_frames) + 1) mean?
    prior = eh.image.make_square(obs, npix, fov)
    return obs, obs_frames, prior, fov
    
def diagnostics(grmhd_ims, obs_frames, fractional_noise, prior, fov, npix_old, noisy):
    max_vis = 0
    all_target_values, all_sigma_values = [], []
    r_list = []
    for i in range(len(grmhd_ims)):
        grmhd_ims[i].ra = grmhd_ims[1].ra
        grmhd_ims[i].dec = grmhd_ims[1].dec
        
    for i, obs_frame in tqdm(enumerate(obs_frames)):
        
        """ NOTE: next two lines are very important, wihtout them we have:
        Image (grmhd_ims[i]) RA, DEC: 17.761121055814954 -29.0078430557251
        Observation (obs_frame) RA, DEC: 17.76112 -29.007797
        Indicating different numerical precision in the GRMHD frames and obs_frames.
        """
        obs_frame.ra = grmhd_ims[i].ra
        obs_frame.dec = grmhd_ims[i].dec
        # NOTE: the following line is just to make sure mjd of the grmhd frames matches the observation time, but it's not necessary
        grmhd_ims[i].mjd = obs_frame.mjd
        
        obs = grmhd_ims[i].observe_same(obs_frame, ttype='direct', ampcal=not noisy, phasecal=not noisy, rlgaincal=not noisy) # if keyword arguments=True, then 
        obs = obs.add_fractional_noise(fractional_noise)
        target, sigma, A = chisqdata(obs, prior, mask=[], pol='I', dtype='vis')
        max_vis = max(max_vis, len(target))
        all_target_values.append(target)
        all_sigma_values.append(sigma)
        r_list.append(np.max(np.sqrt(obs.data['u']**2 + obs.data['v']**2)))
        # NOTE: uncomment to plot uv-coverage
        plt.scatter(obs.data['u'], obs.data['v'])
    plt.savefig("uv.png")
    # TODO: figure this out
    r_list = np.array(r_list)
    max_freq_lambda = r_list.max()
    max_freq_natural = max_freq_lambda / fov
    delta_x = fov / npix_old
    r_max = 1 / max_freq_natural / fov / delta_x
    
    print("number of pixels: ", 2 * r_max)
    all_target_values = np.concatenate(all_target_values)
    all_sigma_values = np.concatenate(all_sigma_values)
    print("max: ", np.max(all_sigma_values))
    print("min: ", np.min(all_sigma_values))
    print("max vis: ", np.max(np.abs(all_target_values)))
    print("min vis: ", np.min(np.abs(all_target_values)))
    
    print(f"Max number of visibilities: {max_vis}")
    return max_vis

def generate_data(obs_path, obs_frames, fractional_noise, grmhd_ims, prior, npix, max_vis, noisy):
    As, targets, sigmas, masks, num_vis_list = [], [], [], [], []
    chisq = 0
    all_obs = []
    for i, obs_frame in tqdm(enumerate(obs_frames)):
        # NOTE: alternative
        # obs = grmhd_ims[i].observe_same(obs_frame, ttype='direct')
    
        obs = grmhd_ims[i].observe_same(obs_frame, ttype='direct', ampcal=not noisy, phasecal=not noisy, rlgaincal=not noisy) # if keyword arguments=True, then 
        # target1, sigma1, A1 = chisqdata(obs, prior, mask=[], pol='I', dtype='vis')
        
        obs = obs.add_fractional_noise(fractional_noise)
        target, sigma, A = chisqdata(obs, prior, mask=[], pol='I', dtype='vis')
        # plt.scatter(range(len(sigma1)), sigma1, label="sigma1")
        # plt.scatter(range(len(sigma)), sigma, label="sigma")
        # plt.savefig("sigmas.png")
        # plt.close()
        # plt.scatter(range(len(sigma)), sigma - sigma1, label="delta sigma")
        # plt.savefig("delta_sigma.png")
        # print(sigma - sigma1)
        num_visibilities = len(target)
        sigma_padded = np.ones((max_vis,), dtype=np.float32) * 1e6
        sigma_padded[:num_visibilities] = sigma
        
        # **Pad A matrix & target with zeros**
        A_padded = np.zeros((max_vis, npix * npix), dtype=np.complex64)
        A_padded[:num_visibilities, :] = A # Only fill real data

        target_padded = np.zeros((max_vis,), dtype=np.complex64)
        target_padded[:num_visibilities] = target          
        # **Generate Mask (1 for real data, 0 for padded values)**
        mask = np.zeros((max_vis,), dtype=np.float32)
        mask[:num_visibilities] = 1.0  # Valid entries
        print("num_vis: ", num_visibilities, np.sum(np.abs((target_padded) - A_padded @ grmhd_ims[i]._imdict['I'])))
        print("num_vis: ", num_visibilities, 1 / num_visibilities * np.sum((np.abs((target) - A @ grmhd_ims[i]._imdict['I']))**2 / sigma**2))
        print("num_vis: ", num_visibilities, 1 / num_visibilities * np.sum((np.abs((target_padded) - A_padded @ grmhd_ims[i]._imdict['I']))**2 / sigma_padded**2))
        chisq += np.sum((np.abs((target) - A @ grmhd_ims[i]._imdict['I']))**2 / sigma**2)

        As.append(A_padded)
        targets.append(target_padded)
        sigmas.append(sigma_padded)
        masks.append(mask)
        num_vis_list.append(num_visibilities)
        all_obs.append(obs)

    print(chisq / np.sum(num_vis_list))
    As = np.array(As)
    targets = np.array(targets)
    sigmas = np.array(sigmas)
    masks = np.array(masks)
    num_vis_list = np.array(num_vis_list)
    np.save(os.path.join(obs_path, "As.npy"), As)
    np.save(os.path.join(obs_path, "targets.npy"), targets)
    np.save(os.path.join(obs_path, "sigmas.npy"), sigmas)
    np.save(os.path.join(obs_path, "masks.npy"), masks)
    np.save(os.path.join(obs_path, "num_vis_list.npy"), num_vis_list)
    
    final_obs = eh.obsdata.merge_obs(all_obs)

    # Save as UVFITS file
    final_obs.save_uvfits(os.path.join(obs_path, "obs.uvfits"))

def plot_uv_coverage(obs, plot_dir, ax=None, fontsize=14, s=None, cmap='rainbow', add_conjugate=True, xlim=(-9.5, 9.5), ylim=(-9.5, 9.5),
                     shift_inital_time=True, cbar=True, cmap_ticks=[0, 4, 8, 12], time_units='Hrs'):
    """
    Plot the uv coverage as a function of observation time.
    x axis: East-West frequency
    y axis: North-South frequency

    Parameters
    ----------
    obs: ehtim.Obsdata
        ehtim Observation object
    ax: matplotlib axis,
        A matplotlib axis object for the visualization.
    fontsize: float, default=14,
        x/y-axis label fontsize.
    s: float,
        Marker size of the scatter points
    cmap : str, default='rainbow'
        A registered colormap name used to map scalar data to colors.
    add_conjugate: bool, default=True,
        Plot the conjugate points on the uv plane.
    xlim, ylim: (xmin/ymin, xmax/ymax), default=(-9.5, 9.5)
        x-axis range in [Giga lambda] units
    shift_inital_time: bool,
        If True, observation time starts at t=0.0
    cmap_ticks: list,
        List of the temporal ticks on the colorbar
    time_units: str,
        Units for the colorbar
    """

    giga = 10**9

    # u = np.array([obsdata[0][6] for obsdata in obs.tlist()]) / giga
    # v = np.array([obsdata[0][7] for obsdata in obs.tlist()]) / giga
    # t = np.array([obsdata[0][0] for obsdata in obs.tlist()])
    
    # Extract u, v, and time using fixed indices
    u_list = []
    v_list = []
    t_list = []

    for obsdata in obs.tlist():
        u_list.append(np.array([entry[6] for entry in obsdata]))  # u-coordinates
        v_list.append(np.array([entry[7] for entry in obsdata]))  # v-coordinates
        t_list.append(np.array([entry[0] for entry in obsdata]))  # time

    # Concatenate lists into single arrays
    u = np.concatenate(u_list) / giga
    v = np.concatenate(v_list) / giga
    t = np.concatenate(t_list)
    
    if shift_inital_time:
        t -= t.min()

    if add_conjugate:
        u = np.concatenate([u, -u])
        v = np.concatenate([v, -v])
        t = np.concatenate([t, t])

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    if time_units == 'mins':
        t *= 60.0
    sc = ax.scatter(u, v, c=t, cmap=plt.cm.get_cmap(cmap), s=s)
    ax.set_xlabel(r'East-West Freq $[G \lambda]$', fontsize=fontsize)
    ax.set_ylabel(r'North-South Freq $[G \lambda]$', fontsize=fontsize)
    ax.invert_xaxis()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    
    # Determine max time for better ticks
    t_max = t.max()
    num_ticks = 5  # Number of ticks we want on the color bar

    # Create evenly spaced time ticks
    cmap_ticks = np.linspace(0, t_max, num_ticks)

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3.5%', pad=0.2)
        cbar = fig.colorbar(sc, cax=cax, ticks=cmap_ticks)

        # Set color bar labels in Hrs or Mins
        if time_units == 'mins':
            cbar_labels = [f"{tick * 60:.1f} mins" for tick in cmap_ticks]
        else:
            cbar_labels = [f"{tick:.1f} Hrs" for tick in cmap_ticks]

        cbar.set_ticklabels(cbar_labels)
    
    # if cbar:
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes('right', size='3.5%', pad=0.2)
    #     cbar = fig.colorbar(sc, cax=cax, ticks=cmap_ticks)
    #     cbar.set_ticklabels(['{} {}'.format(tick, time_units) for tick in cbar.get_ticks()])
    plt.tight_layout()
    plt.savefig(plot_dir)



def plot_uv_coverage_dual(obs1, obs2, plot_dir, fontsize=20, s=None, cmap='rainbow', 
                          add_conjugate=True, xlim=(-9.5, 9.5), ylim=(-9.5, 9.5), 
                          shift_initial_time=True, time_units='Hrs'):
    """
    Plot the uv coverage of two eht-imaging Obsdata objects side by side, sharing the y-axis and colorbar.
    """
    giga = 10**9

    def extract_uvt(obs):
        u_list, v_list, t_list = [], [], []
        for obsdata in obs.tlist():
            u_list.append(np.array([entry[6] for entry in obsdata]))  # u-coordinates
            v_list.append(np.array([entry[7] for entry in obsdata]))  # v-coordinates
            t_list.append(np.array([entry[0] for entry in obsdata]))  # time
        
        u = np.concatenate(u_list) / giga
        v = np.concatenate(v_list) / giga
        t = np.concatenate(t_list)
        
        if shift_initial_time:
            t -= t.min()

        if add_conjugate:
            u = np.concatenate([u, -u])
            v = np.concatenate([v, -v])
            t = np.concatenate([t, t])

        return u, v, t

    # Extract data for both observations
    u1, v1, t1 = extract_uvt(obs1)
    u2, v2, t2 = extract_uvt(obs2)

    # Find global time range for colorbar
    t_min = min(t1.min(), t2.min())
    t_max = max(t1.max(), t2.max())
    num_ticks = 5  # Number of ticks we want on the color bar
    cmap_ticks = np.linspace(t_min, t_max, num_ticks)

    # Use the modern colormap method
    cmap = plt.colormaps.get_cmap(cmap)

    # Create figure with GridSpec for better spacing control
    fig = plt.figure(figsize=(10, 5))  # Adjust figure size
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.15)  # 2 plots + 1 narrow colorbar

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)  # Share y-axis
    cbar_ax = fig.add_subplot(gs[2])  # Colorbar in the last column

    # Scatter plots
    sc1 = ax1.scatter(u1, v1, c=t1, cmap=cmap, s=s)
    sc2 = ax2.scatter(u2, v2, c=t2, cmap=cmap, s=s)

    # Labels and formatting
    ax1.set_xlabel(r'East-West Freq $[G \lambda]$', fontsize=fontsize)
    ax2.set_xlabel(r'East-West Freq $[G \lambda]$', fontsize=fontsize)
    ax1.set_ylabel(r'North-South Freq $[G \lambda]$', fontsize=fontsize)  # Shared y-axis
    ax1.invert_xaxis()
    ax2.invert_xaxis()
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    # Set titles
    ax1.set_title("ngEHT+ Coverage", fontsize=fontsize)
    ax2.set_title("ngEHT Coverage", fontsize=fontsize)
    ax2.tick_params(labelleft=False)  # Hide y-axis labels on the second plot
    # Create single shared colorbar in the third column
    cbar = fig.colorbar(sc1, cax=cbar_ax, ticks=cmap_ticks)

    # Set color bar labels in Hrs or Mins
    if time_units == 'mins':
        cbar_labels = [f"{tick * 60:.1f} mins" for tick in cmap_ticks]
    else:
        cbar_labels = [f"{tick:.1f} Hrs" for tick in cmap_ticks]

    cbar.set_ticklabels(cbar_labels)
    # cbar.set_label("Observation Time", fontsize=fontsize)

    plt.savefig(plot_dir, bbox_inches='tight', dpi=1200)
    plt.close(fig)