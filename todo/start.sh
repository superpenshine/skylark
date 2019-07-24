#!/bin/bash
source /home/sht/.bashrc
source /home/sht/scratch/venv/bin/activate
module load cuda cudnn
/home/sht/scratch/skylark/main.py
