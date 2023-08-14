#!/bin/bash
#SBATCH -J syh
#SBATCH -p bme_gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH -o ./shell/slurm-%j.out
#SBATCH -e ./shell/slurm-%j.out

python test_whole_reg.py

