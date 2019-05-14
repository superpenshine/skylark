# Simulation data loader

import pdb
import h5py
import platform
import matplotlib
import numpy as np
from pathlib import Path

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


def save_to_h5(config, raw_data = True, polar = False):
    '''
    Save data to one huge .h5 file of structure:
        dataset: [n_disk, n_steps, nx, ny, nvar]

    nvar: [nx, ny, 0] -- gas
          [nx, ny, 1] -- dust for size 0.01cm
          [nx, ny, 2] -- dust for size 0.1cm
          [nx, ny, 3] -- dust for size 0.001cm
    '''
    nvar = config.nvar
    ndisc = 1 #config.ndisc
    file_pattern = config.pattern
    h5_fout = h5py.File(config.h5_dir)

    # d_set = h5_fout.create_dataset(
    #     'data', 
    #     dtype=np.float32,
    #     shape=(1, 150, 1024, 1024, 4),
    #     maxshape=(None, 150, 1024, 1024, 4), # hard code it for now
    #     compression='gzip', 
    #     compression_opts=4) # default ratio is 4

    for i, fo in enumerate([config.data_dir]):
        fo_name = fo.stem
        grid_dir = config.data_dir / config.f_gird

        log_grid = np.loadtxt(grid_dir)
        f_names = sorted(fo.glob(file_pattern))

        nx = log_grid.shape[0]

        disk_grp = h5_fout.create_group(fo_name)
        disk_grp["log_grid"] = log_grid

        for j, f in enumerate(f_names):
            print(j)
            data = np.fromfile(f.open('rb'), dtype=np.float32)
            ny = get_ny(data, log_grid, nvar)
            data = np.reshape(data, (nx, ny, nvar))

            if raw_data:
                disk_grp[str(j)] = data
                continue


def load(dir, disk, step, retrun_gird = True):
    '''
    Load data from .h5 file [n_discs, n_steps, nx, ny, nvar]
    '''
    step = str(step)
    h5_fout = h5py.File(dir)
    if retrun_gird:
        return (np.asarray(h5_fout[disk][step]), np.asarray(h5_fout[disk]["log_grid"]))

    return np.asarray(h5_fout[disk][step])


def to_array(buf):
    pass


def preview(data, log_grid, polar = False, var = 1):
    '''
    Preview the image from ndarray data
    data: ndarray, data should be in log-polar space
    log_grid: ndarray of shape [nx,]
    polar: whether to show image with polar projection
    var: integer from nvar:=[0, 3], indicating the simulation obj to view
    var:  0 -- gas
          1 -- dust for size 0.01cm
          2 -- dust for size 0.1cm
          3 -- dust for size 0.001cm.

    '''
    fig = plt.figure()
    plt.subplot(111, polar=polar)

    if not polar:
        plt.imshow(data[:,:,var].T, origin='lower')
        # plt.axis([0.4, 2, 0, 2 * np.pi])
        plt.show()
        return

    # Transfer to polar system first then bend it
    ny = data.shape[1]
    theta = (np.arange(0, ny) * 1.0 / ny + 0.5 / ny) * 2 * np.pi;
    nx = data.shape[0]  
    rp1, rp2 = np.meshgrid(log_grid, theta)
    # rp1, rp2 = np.meshgrid(np.linspace(0, 16, nx), theta)
    plt.pcolormesh(rp1, rp2, data[:,:,var])
    plt.colorbar()
    plt.show()

    # fig.savefig("a.png", frameon=False, bbox_inces='tight', pad_inches=0)

    # fig.canvas.draw()

    # a = np.array(fig.canvas.renderer._renderer)
    # print(a.shape)
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, frameon=False)
    # ax2.imshow(a)
    # plt.show()


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

