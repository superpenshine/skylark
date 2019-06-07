# Simulation data loader

import pdb
import h5py
import platform
import matplotlib
import numpy as np

from pathlib import Path
from functools import partial
from scipy.ndimage import geometric_transform, map_coordinates
from util.transform import LogPolartoPolar

# Change backend for Mac users
if platform.system() == "Darwin":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def get_ny(data, log_grid, nvar):
    '''
    Calculate ny from data shapes
    '''
    nx = log_grid.shape[0]
    div, mod = divmod(data.shape[0] / nvar,  nx)

    if mod != 0:
        raise ValueError("Invalid data format, data:{}, log_grid:{}, {}".format(data.shape, log_grid.shape, nvar))

    return int(div)


def save_to_h5(config, polar = False, trva=None):
    '''
    Save data to one huge .h5 file of structure:
        dataset: [n_disk, n_steps, nx, ny, nvar]
    trva: split to train/valid data or not

    nvar: [nx, ny, 0] -- gas
          [nx, ny, 1] -- dust for size 0.01cm
          [nx, ny, 2] -- dust for size 0.1cm
          [nx, ny, 3] -- dust for size 0.001cm
    '''
    file_pattern = config.pattern
    if polar:
        toPolar = LogPolartoPolar()
    if trva:
        out_tr = h5py.File(Path(str(config.h5_dir) + "_tr.h5"))
        out_va = h5py.File(Path(str(config.h5_dir) + "_va.h5"))
    else:
        h5_fout = h5py.File(config.h5_dir.with_suffix('.h5'))

    # Use dataset with shape if disks' frame numbers are the same
    # d_set = h5_fout.create_dataset(
    #     'data', 
    #     dtype=np.float32,
    #     shape=(1, 150, 1024, 1024, 4),
    #     maxshape=(None, 150, 1024, 1024, 4), # hard code it for now
    #     compression='gzip', 
    #     compression_opts=4) # default ratio is 4

    # Iterate through disk data folders
    for i, fo in enumerate([config.data_dir]):
        fo_name = fo.stem
        grid_dir = config.data_dir / config.f_gird
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
            f_names_tr = list(map(f_names.__getitem__, list(range(0, f_len, 2))))
            f_names_va = list(map(f_names.__getitem__, list(range(1, f_len, 2))))
            
            # Assume split data with given validation percentage
            # split = int(f_len * (1 - config.valid_size))
            # f_names_tr = f_names[:split]
            # f_names_va = f_names[split:]
            
            # Write to tr/va files
            for j, f in enumerate(f_names_tr):
                print("tr", j)
                data = load_from_binary(f, log_grid, config.nvar, polar)
                if polar:
                    data = toPolar(log_grid, data)
                disk_grp_tr[str(j)] = data

            for j, f in enumerate(f_names_va):
                print("va", j)
                data = load_from_binary(f, log_grid, config.nvar, polar)
                if polar:
                    data = toPolar(log_grid, data)
                disk_grp_va[str(j)] = data
        else:
            for j, f in enumerate(f_names):
                print(j)
                data = load_from_binary(f, log_grid, config.nvar, polar)
                if polar:
                    data = toPolar(log_grid, data)
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
          3 -- dust for size 0.001cm.

    '''
    # fig = plt.figure()
    # plt.subplot(111, polar=polar)
    nx, ny = data.shape[:2]
    theta = (np.arange(0, ny) * 1.0 / ny + 0.5 / ny) * 2 * np.pi

    # Show in polar coords with mapping
    rp1, rp2 = np.meshgrid(np.linspace(0, nx, nx), np.digitize(np.linspace(0, ny, ny), expand(log_grid, ny))-1)
    polar_data = map_coordinates(data[:,:,var], (rp2, rp1))
    # Looping over pixels, SLOW!
    # polar_data = geometric_transform(data[:,:,var], partial(logpolar_to_polar, expand(log_grid, nx)))
    plt.imshow(polar_data)

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

