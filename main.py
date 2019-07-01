#!/usr/bin/env python3
# main file
import sys
from config import get_config, print_usage
# from network_tf import nn
from network_pt import network
from util.data_util import *
from dataset.Astrodata import Astrodata

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


config, unparsed = get_config()

if len(unparsed) > 0:
    print_usage()
    exit(1)


def main():
    # preview_raw(config, 'D:/sigma_data/sigma_data0150.bin', config.data_dir / config.f_gird, polar=False, var=1)
    # save_to_h5(config, trva=True, polar=True)
    
    # data, grid = load(str(config.h5_dir_win) + "_tr.h5", "sigma_data", 71)
    # preview(data, grid, polar=False)
    # test(data, grid)

    # Test for TF network
    # network = nn(config)
    # network.load_img_on_tb(data, grid)

    solver = network(config)
    if config.m == 'v':
        solver.test_single(triplet_id=10, step_diff=(40, None), dataset='va')
    elif config.m == 'save':
        save_to_h5(config, trva=True, polar=False, size=config.input_size)
    elif config.m == 'stats':
        get_stats(str(config.h5_dir_win) + "_tr.h5", str(config.h5_dir_win) + "_va.h5", config.nvar, verbose=True)
    elif config.m == 'p':
        data, grid = load(str(config.h5_dir_win) + "_tr.h5", "sigma_data", 60)
        preview(data, grid, polar=False)
    elif config.m == 'sanity':
        solver.sanity_check_regular_loss()
        # solver.sanity_check_randcrop_interpo_loss()
        # solver.sanity_visualize()
    elif config.m == 'c':
        solver.clean_up()
    else:
        solver.run()

if __name__ == "__main__":
    main()
