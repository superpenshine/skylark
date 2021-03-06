# Simulation data loader

import os
import cv2
import pdb
import h5py
import platform
import matplotlib
import numpy as np

from pathlib import Path
from functools import partial
# from scipy.ndimage import geometric_transform, map_coordinatesl
from util.transform import LogPolartoPolar, Resize

# Change backend for Mac users
if platform.system() == "Darwin":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def test(data, log_grid):
    toPolar = LogPolartoPolar()
    toPolar(log_grid, data)


def get_ny(data, log_grid, nvar):
    '''
    Calculate ny from data shapes
    '''
    nx = log_grid.shape[0]
    div, mod = divmod(data.shape[0] / nvar,  nx)

    if mod != 0:
        raise ValueError(
            "Invalid data format, data:{}, log_grid:{}, {}".format(data.shape, log_grid.shape, nvar))

    return int(div)


def save_to_h5(config, polar = False, size = None, trva=None):
    '''
    Save data to one huge .h5 file of structure:
        dataset: [n_disk, n_steps, nx, ny, nvar]
    trva: split to train/valid data or not
    polar: transfer data to polar coords
    nvar: [nx, ny, 0] -- gas
          [nx, ny, 1] -- dust for size 0.01cm
          [nx, ny, 2] -- dust for size 0.1cm
          [nx, ny, 3] -- dust for size 0.001cm
    '''
    nvar = config.nvar
    file_pattern = config.pattern
    h5_dir = config.h5_dir_linux
    data_dir = config.data_dir
    # For debug on Windows
    if os.name == 'nt':
        h5_dir = str(config.h5_dir_win)
    if polar:
        toPolar = LogPolartoPolar()
    if size: 
        resize = Resize(size)
    if trva:
        out_tr = h5py.File(Path(str(h5_dir) + "_tr.h5"))
        out_va = h5py.File(Path(str(h5_dir) + "_va.h5"))
    else:
        h5_fout = h5py.File(h5_dir.with_suffix('.h5'))

    # Use dataset with shape if disks' frame numbers are the same
    # d_set = h5_fout.create_dataset(
    #     'data', 
    #     dtype=np.float32,
    #     shape=(1, 150, 1024, 1024, 4),
    #     maxshape=(None, 150, 1024, 1024, 4), # hard code it for now
    #     compression='gzip', 
    #     compression_opts=4) # default ratio is 4

    # Iterate through disk data folders
    for i, fo in enumerate([data_dir]):
        fo_name = fo.stem
        grid_dir = data_dir / config.f_gird
        log_grid = np.loadtxt(grid_dir)
        f_names = sorted(fo.glob(file_pattern))

        if trva:
            disk_grp_tr = out_tr.create_group(fo_name)
            disk_grp_va = out_va.create_group(fo_name)
            disk_grp_tr["log_grid"] = log_grid
            disk_grp_va["log_grid"] = log_grid
        else:
            disk_grp = h5_fout.create_group(fo_name)
            disk_grp["log_grid"] = log_grid

        # Split data(Should avoid code duplication)
        if trva:
            # Assume even for train, odd for validation
            f_len = len(f_names)
            
            # f_names_tr = list(map(f_names.__getitem__, list(range(0, f_len, 2))))
            # f_names_va = list(map(f_names.__getitem__, list(range(1, f_len, 2))))
            
            # Assume split data with given validation percentage
            split = int(f_len * (1 - config.valid_size))
            f_names_tr = f_names[:split]
            f_names_va = f_names[split:]
            
            # Write to tr/va files
            for j, f in enumerate(f_names_tr):
                print("tr", j)
                data = load_from_binary(f, log_grid, nvar, polar)
                if polar:
                    data = toPolar(log_grid, data)
                if size:
                    data = resize(data)
                disk_grp_tr[str(j)] = data

            for j, f in enumerate(f_names_va):
                print("va", j)
                data = load_from_binary(f, log_grid, nvar, polar)
                if polar:
                    data = toPolar(log_grid, data)
                if size:
                    data = resize(data)
                disk_grp_va[str(j)] = data
        else:
            for j, f in enumerate(f_names):
                print(j)
                data = load_from_binary(f, log_grid, nvar, polar)
                if polar:
                    data = toPolar(log_grid, data)
                if size:
                    data = resize(data)
                disk_grp[str(j)] = data


def load_from_binary(f, log_grid, nvar, polar=False):
    '''
    Load data in correct shape
    '''
    data = np.fromfile(f.open('rb'), dtype=np.float32)
    nx = log_grid.shape[0]
    ny = get_ny(data, log_grid, nvar)
    data = np.reshape(data, (nx, ny, nvar))

    return data


def load(dir, disk, step, return_gird = True):
    '''
    Load data from .h5 file [n_discs, n_steps, nx, ny, nvar]
    '''
    step = str(step)
    h5_fout = h5py.File(dir)
    if return_gird:
        return (np.asarray(h5_fout[disk][step]), np.asarray(h5_fout[disk]["log_grid"]))

    return np.asarray(h5_fout[disk][step])


def expand(log_grid, n):
    '''
    Turn log_grid numbers to actual pixel index in img
    '''
    return (log_grid - log_grid[0]) / (log_grid[-1] - log_grid[0]) * n


def logpolar_to_polar(log_grid, output_coords):
    '''
    Transfer x from log to linear space
    log_grid: || |  |   |    |, bin boundaries in linear space
    output_coords: HWC where H == x, W == y (all x and y looped)
    '''
    x = np.digitize([output_coords[0]], log_grid)[0] - 1

    return (x, output_coords[1])


def preview(data, log_grid, polar = False, var = 1):
    '''
    Preview the image from ndarray data
    data: ndarray, data should be in log-polar space
    log_grid: ndarray of shape [nx,]
    polar: whether to show image with polar or log_polar
    var: channel to visualize
    var:  0 -- gas
          1 -- dust for size 0.01cm
          2 -- dust for size 0.1cm
          3 -- dust for size 0.001cm

    '''
    nx, ny = data.shape[:2]
    theta = (np.arange(0, ny) * 1.0 / ny + 0.5 / ny) * 2 * np.pi

    # Show img in log-polar

    # Show img in polar
    if polar:
        phi = (np.arange(0, ny) * 1.0 / ny + 0.5 / ny) * 2 * np.pi
        plt.subplot(1, 1, 1, projection='polar')
        # vertical orbit
        # rp1, rp2 = np.meshgrid(log_grid, phi)
        # plt.pcolormesh(rp1, rp2, data[:,:,var].T)
        # horizontal orbit
        rp1, rp2 = np.meshgrid(phi, log_grid)
        plt.pcolormesh(rp1, rp2, data[:,:,var])
        plt.axis([0, 2*np.pi, 0, 2])
    else:
        plt.imshow(data[:,:,var])

    plt.colorbar()
    plt.show()

    # plt.imshow(data[:,:,var])
    # print(np.sum(np.square(data[:,:,var])))
    # print(np.histogram(data[:,:,0], bins=10, range=(0, 7)))
    # plt.colorbar()
    # # Show in polar coords with mapping
    # rp1, rp2 = np.meshgrid(np.linspace(0, nx, nx), np.digitize(np.linspace(0, ny, ny), expand(log_grid, ny))-1)
    # polar_data = map_coordinates(data[:,:,var], (rp2, rp1))
    # # Looping over pixels, SLOW!
    # # polar_data = geometric_transform(data[:,:,var], partial(logpolar_to_polar, expand(log_grid, nx)))
    # crop = Crop((0, 0), (439, 1024))
    # plt.imshow(crop(data[:,:,var]))

    # Show in polar coords using pcolormesh
    # rp1, rp2 = np.meshgrid(theta, log_grid)
    # plt.pcolormesh(rp1, rp2, data[:,:,var])
    # plt.colorbar()

    # Code Experiment1
    # linear1, linear2 = np.meshgrid(theta, np.linspace(0, 1024, nx))
    # plt.pcolormesh(linear1, linear2, geometric_transform(data[:,:,var], partial(logpolar_to_polar, log_grid_expanded)))
    # plt.imshow(data[:,:,var])

    # Code Experiment2
    # rp1, rp2 = np.meshgrid(theta, log_grid)
    # fig, ax = plt.subplots(nrows=1)
    # log_data = ax.pcolormesh(rp1, rp2, data[:,:,var], cmap='gray').get_array()
    # plt.imshow(log_data.reshape(1023, 1023), cmap='gray')
    # print(data[:1023, :1023, 1] - log_data.reshape(1023, 1023))

    # plt.axis('off')
    plt.show()


def preview_raw(config, f_dir, grid_dir, polar = False, var = 1):
    '''
    Preview the image from raw data
    f_dir: directory to raw data
    polar: whether to show image in polar cords
    var: integer from nvar:=[0, 3], indicating the simulation obj to view
    var:  0 -- gas
          1 -- dust for size 0.01cm
          2 -- dust for size 0.1cm
          3 -- dust for size 0.001cm
    '''
    data = np.fromfile(Path(f_dir).open('rb'),dtype=np.float32)
    log_grid = np.loadtxt(Path(grid_dir))
    # Reshape data
    nvar = config.nvar
    nx = log_grid.shape[0]
    ny = get_ny(data, log_grid, nvar)
    data = np.reshape(data, (nx, ny, nvar))

    preview(data, log_grid, polar = polar, var = var)


def get_stats(h5_dir_tr, h5_dir_va, n_chan, verbose=False):
    '''
    Get dataset stats
    '''
    n = 0
    _sum = np.zeros(n_chan)
    sum_minus_mean_square = np.zeros(n_chan)

    for _dir in [h5_dir_tr, h5_dir_va]:
        data = h5py.File(_dir, "r")
        d_names = list(data.keys())
        for d_name in d_names:
            disk_data = data[d_name]
            d_step = len(disk_data.keys()) - 1
            for d_step in range(d_step):
                img = np.array(disk_data[str(d_step)])
                _sum += np.sum(np.sum(img, axis=0), axis=0)
                h, w = img.shape[:-1]
                n += h * w

    mean = _sum * 1.0 / n

    for _dir in [h5_dir_tr, h5_dir_va]:
        data = h5py.File(_dir, "r")
        d_names = list(data.keys())
        for d_name in d_names:
            disk_data = data[d_name]
            d_step = len(disk_data.keys()) - 1
            for d_step in range(d_step):
                img = np.array(disk_data[str(d_step)])
                sum_minus_mean_square += np.sum(np.sum(np.square(img - mean), axis=0), axis=0)

    std = np.sqrt(sum_minus_mean_square / n)
    std = np.float32(std)
    mean = np.float32(mean)
    if verbose:
        print("Total number of pixels: {}".format(n))
        for chan in range(n_chan):
            print("Channel{} mean: {}, std: {}".format(chan, mean[chan], std[chan]))

    return mean, std


def make_video():
    '''
    Make video out of frames
    ''' 
    image_folder = './frames'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 8, (width,height))
    for i in range(len(images)):
        video.write(cv2.imread(os.path.join(image_folder, str(i) + '.png')))

    cv2.destroyAllWindows()
    video.release()


def get_frames(solver, data_dir, d_name='sigma_data', frames_path='./frames/', var=1, polar=True, mode='inter'):
    '''
    Turn image data to frames
    solver: solver gives the interpolation or extrapolation result
    data_dir: directory to h5py data
    d_name: dataset name to use
    frames_path: path to frame output folder
    polar: whether the output frames is in polar projection
    mode: inter for interpolation, extra for extrapolation, None for 
    ground truth frames only.
    '''
    data = h5py.File(data_dir, 'r')
    data_d = data[d_name]
    frame_ids = list(data_d.keys())
    frame_ids.remove('log_grid')
    log_grid = data_d['log_grid']

    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    
    # Make gt frame list
    gt_frames, frames = [], []
    for i in range(len(frame_ids)):
        gt_frames.append(np.array(data_d[str(i)]))

    # Interpolation
    if mode == 'inter':
        frames.append(gt_frames[0][:,:,var])
        for i in range(len(gt_frames) - 1):
            mid_frame = solver(gt_frames[i], gt_frames[i + 1], mode=0)
            frames.extend([mid_frame[:,:,var], gt_frames[i + 1][:,:,var]])

        frames = np.array(frames)
    # Extrpolation
    elif mode == 'extra':
        raise NotImplementedError()
    else: 
        frames = np.array(gt_frames)[:,:,:,var]

    #Prepare frames
    _, ny = np.array(frames[0]).shape
    phi = (np.arange(0, ny) * 1.0 / ny + 0.5 / ny) * 2 * np.pi

    for frame_id in range(len(frames)):
        img = frames[frame_id]
        path = frames_path + str(frame_id) + '.png'
        if polar:
            plt.subplot(1, 1, 1, projection='polar')
            rp1, rp2 = np.meshgrid(phi, log_grid)
            plt.pcolormesh(rp1, rp2, img)
            plt.axis([0, 2 * np.pi, 0, 2])
            plt.savefig(path)
            plt.close()
        else:
            plt.imsave(path, img, vmin=np.amin(img), vmax=np.amax(img))
