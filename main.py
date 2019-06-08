# main file

from config import get_config, print_usage
# from network_tf import nn
from network_pt import network
from util.data_util import *
from dataset.Astrodata import Astrodata

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


config, unparsed = get_config()

if len(unparsed) > 0:
    print_usage()
    exit(1)


def main():
    # preview_raw(config, 'D:/sigma_data/sigma_data0150.bin', config.data_dir / config.f_gird, polar=False, var=1)
    # save_to_h5(config, trva=True, polar=True)
    
    # data, grid = load(str(config.h5_dir) + "_tr.h5", "sigma_data", 10)
    # preview(data, grid, polar=False)
    # test(data, grid)

    # network = nn(config)
    # network.load_img_on_tb(data, grid)

    solver = network(config)
    solver.run()
    # solver.sanity_check_randcrop()
    # solver.sanity_check_no_randcrop()
    # solver.test_single(triplet_id = 6)
    # solver.test_full()


if __name__ == "__main__":
    main()
