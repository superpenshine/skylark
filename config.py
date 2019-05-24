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
sys_arg.add_argument("--data_dir", type=Path,
                    default='D:/sigma_data',
                    # default='/home/sht/data/sigma_data',
                    help="Directory to data folder")

sys_arg.add_argument("--h5_dir", type=Path,
                    default='D:/sigma_data/data',
                    # default='/home/sht/data/sigma_data/data', 
                    help="Data file name without the .5 extension")

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

sys_arg.add_argument("--min_step_diff", type=int,
                    default=20,
                    help="None or an integer indicating min step difference")

sys_arg.add_argument("--max_step_diff", type=int,
                    default=None,
                    help="None or an integer indicating max step difference")

sys_arg.add_argument("--input_size", type=tuple, 
                    default=(130, 130), 
                    help="Input image size")

sys_arg.add_argument("--label_size", type=tuple, 
                    default=(114, 114), 
                    help="Label image size")

sys_arg.add_argument("--batch_size", type=int,
                    default=5,
                    help="Batch size")

sys_arg.add_argument("--lr", type=float,
                    default=0.0001,
                    help="Learning rate")

sys_arg.add_argument("--epochs", type=int,
                    default=3,
                    help="Number of epochs")

sys_arg.add_argument("--checkpoint_freq", type=int, 
                    default=1, 
                    help="Number of epochs between each checkpoint")

sys_arg.add_argument("--cuda", type=bool, 
                    default=is_available(), 
                    help="Cuda enabled or use cpu")

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()
