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

    solver = network(config, arch='resnet', mode='extra')
    if config.m == 'v':
        # visualization
        solver.test_single(triplet_id=0, step_diff=(None, 2), dataset='te', audience='normal', var=1) # set min step diff to make the input fixed
    elif config.m == 'psnr':
        # calc avg. psnr over dataset
        solver.mean_psnr(var=1)
    elif config.m == 'save':
        # create datasets
        save_to_h5(config, trva=True, polar=False, size=config.input_size)
    elif config.m == 'stats':
        # get stats about data
        get_stats(str(config.h5_dir_win) + "_tr.h5", str(config.h5_dir_win) + "_va.h5", config.nvar, verbose=True)
    elif config.m == 'p':
        # Uncomment the line below
        # data, grid = load(str(config.h5_dir_win) + "_tr.h5", "sigma_data", 60)
        data, grid = load(str("D:/sigma_data/data_logpolar") + "_tr.h5", "sigma_data", 60)
        preview(data, grid, polar=False)
    elif config.m == 'a':
        # animate
        make_video(fps=2)
    elif config.m == 'a2':
        # animate dual video
        make_dual_video(fps=2, image_folder=('./frames_tegt_round', './frames_te_round'))
    elif config.m == 'frame':
        # make frames
        solver.setup()
        # get_frames(solver.solve, str(config.h5_dir_win) + "_tr.h5", d_name='sigma_data', var=1, mode='inter')
        get_frames(solver.solve, 'D:/sigma_data/data_logpolar_resized32_te.h5', d_name='sigma_data', var=1, mode=None, polar=True, start_frame=0, size=(1024, 1024))
    elif config.m == 'sanity':
        # sanity check
        solver.sanity_check_regular_loss()
        # solver.sanity_check_randcrop_interpo_loss()
        # solver.sanity_visualize()
    elif config.m == 'c':
        # clean up run results
        solver.clean_up()
    else:
        solver.run()

if __name__ == "__main__":
    main()
