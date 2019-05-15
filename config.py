# configuration file

import os
import argparse
from pathlib import Path


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
                    help="Directory to data folder")

sys_arg.add_argument("--f_gird", type=Path,
                    default='log_grid.dat',
                    help="log grid file name")

sys_arg.add_argument("--h5_dir", type=Path,
                    default='D:/sigma_data/data.h5',
                    help="Suffix of data file name")

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
                    default=0.8,
                    help="Validation sample percentage (<=1.0)")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()