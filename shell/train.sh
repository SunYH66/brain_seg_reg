#!/bin/bash
#SBATCH -J train_seg_06_with_GT
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:1
#SBATCH -o ./out/train_seg_06_with_GT.out
#SBATCH -e ./out/train_seg_06_with_GT.error

/public/home/liujm1/miniconda3/envs/test/bin/python ../train.py

