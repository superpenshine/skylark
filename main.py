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
    # save_to_h5(config, trva=True)
    
    # data, grid = load(config.h5_dir, "sigma_data", 150)
    # preview(data, grid, polar=False)

    # network = nn(config)
    # network.load_img_on_tb(data, grid)

    # import pdb
    # pdb.set_trace()
    
    # dataset, loader test
    # data = Astrodata(config.h5_dir, 
    #                  min_step_diff = 10, 
    #                  max_step_diff = 1500, 
    #                  rtn_log_grid = False, 
    #                  transform = transforms.Compose([
    #                     transforms.ToPILImage(), 
    #                     # transforms.Resize((32, 32)),
    #                     transforms.ToTensor()
    #                     ])
    #                  )

    # n_data = len(data)
    # np.random.seed(1234)
    # indices = list(range(len(data)))
    # np.random.shuffle(indices)
    # split = int(n_data * config.valid_size)

    # train, valid = indices[:split], indices[split:]
    # train_sampler = SubsetRandomSampler(train)
    # valid_sampler = SubsetRandomSampler(valid)

    # train_loader = DataLoader(data, 
    #                           batch_size = 4, 
    #                           sampler = train_sampler)
    # valid_loader = DataLoader(data, 
    #                           batch_size = 4, 
    #                           sampler = valid_sampler)

    # writer = SummaryWriter(log_dir = str(config.log_dir))

    # for b_id, (i0, i1, label) in enumerate(train_loader):
    #     i1 = i1[0,1,:,:]
    #     writer.add_image('test_figure0', i1, 0, dataformats='HW')
    #     writer.close()
    #     break

    solver = network(config)
    # solver.run()
    solver.test_single()


if __name__ == "__main__":
    main()
