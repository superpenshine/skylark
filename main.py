# main file

import io
from config import get_config, print_usage
from network_tf import nn
from util.data_util import *
from dataset.Astrodata import Astrodata

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


config, unparsed = get_config()

if len(unparsed) > 0:
    print_usage()
    exit(1)


def main():
    # preview_raw(config, 'D:/sigma_data/sigma_data0150.bin', config.data_dir / config.f_gird, polar=False, var=1)
    # save_to_h5(config)
    
    # data, grid = load(config.h5_dir, "sigma_data", 150)
    # preview(data, grid, polar=False)

    # network = nn(config)
    # network.load_img_on_tb(data, grid)

    # import pdb
    # pdb.set_trace()
    
    data = Astrodata(config.h5_dir, 
                     min_step_diff = 10, 
                     max_step_diff = 1500, 
                     rtn_log_grid = False, 
                     transform = transforms.Compose([
                        transforms.ToPILImage(), 
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()
                        ])
                     )
    dataloader = DataLoader(data, batch_size=4, shuffle=True)

    writer = SummaryWriter(log_dir = str(config.log_dir))

    for b_id, (i0, i1, label) in enumerate(dataloader):
        i1 = i1[0,1,:,:]
        writer.add_image('test_figure0', i1, 0, dataformats='HW')
        writer.close()
        break


if __name__ == "__main__":
    main()
