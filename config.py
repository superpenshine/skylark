# configuration file

import os
import argparse
from pathlib import Path
from torch.cuda import is_available


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


sys_arg = add_argument_group("Data")

# Params you want to change for local system uses
sys_arg.add_argument("-m", type=str,
                    default=None,
                    help='''Running mode, 
                    v: for visulization
                    s: for save to h5py
                    p: for preview''')

sys_arg.add_argument("--data_dir", type=Path,
                    # default='D:/sigma_data',
                    default='/home/sht/data/sigma_data',
                    help="Directory to data folder")

sys_arg.add_argument("--h5_dir_win", type=Path,
                    # default='D:/sigma_data/data_polar',
                    # default='D:/sigma_data/data_logpolar',
                    default='C:/sigma_data/data_logpolar_resized',
                    help="Win data file without the .h5 suffix")

sys_arg.add_argument("--h5_dir_linux", type=Path,
                    # default='/home/sht/data/sigma_data/data_polar', 
                    # default='/home/sht/data/sigma_data/data_logpolar', 
                    default='/home/sht/data/sigma_data/data_logpolar_resized', 
                    help="Linux data file without the .h5 suffix")

sys_arg.add_argument("--min_step_diff", type=int,
                    default=None,
                    help="None or an integer indicating min step difference")

sys_arg.add_argument("--max_step_diff", type=int,
                    default=None,
                    help="None or an integer indicating max step difference")

sys_arg.add_argument("--batch_size", type=int,
                    default=100,
                    help="Batch size")

sys_arg.add_argument("--lr", type=float,
                    default=0.00001,
                    help="Learning rate")

sys_arg.add_argument("--epochs", type=int,
                    default=20000,
                    help="Number of epochs")

sys_arg.add_argument("--num_workers", type=int,
                    default=8,
                    help="Number of dataloader workers")

sys_arg.add_argument("--f_gird", type=Path,
                    default='log_grid.dat',
                    help="log grid file name")

sys_arg.add_argument("--log_dir", type=Path,
                    default='./tmp',
                    help="Directory to where TF store logs")

sys_arg.add_argument("--pattern", type=str,
                    default='sigma_data*.bin',
                    help="Data file pattern")

sys_arg.add_argument("--nvar", type=int,
                    default=4,
                    help="nvar number")

sys_arg.add_argument("--valid_size", type=float,
                    default=0.2,
                    help="Validation sample percentage (<=1.0)")

sys_arg.add_argument("--input_size", type=tuple, 
                    # default=(64, 64), 
                    default=(32, 32), 
                    help="Input image size")

sys_arg.add_argument("--crop_size", type=tuple,
                    default=(48, 48),  
                    # default=(32, 32), 
                    # default=(24, 24),
                    help="Ramdom crop image size, tiling size")

sys_arg.add_argument("--label_size", type=tuple,
                    default=(32, 32),  
                    # default=(16, 16),
                    # default=(8, 8), 
                    help="Label image size, this is related to network model")

sys_arg.add_argument("--checkpoint_freq", type=int, 
                    default=20, 
                    help="Number of epochs between each checkpoint")

sys_arg.add_argument("--report_freq", type=int, 
                    default=20, 
                    help="Number of epochs between each summary write")

sys_arg.add_argument("--cuda", type=bool, 
                    default=is_available(), 
                    help="Cuda enabled or use cpu")

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()
