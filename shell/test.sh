#!/bin/bash
#SBATCH -J test_seg
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:1
#SBATCH -o ./out/test_seg.out
#SBATCH -e ./out/test_seg.error

python /hpc/data/home/bme/v-sunyh2/programs/brain_seg_reg/test.py

