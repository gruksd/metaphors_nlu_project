#!/bin/bash

#SBATCH -A project_2001403
#SBATCH -J train_t5
#SBATCH -e err/%x_%j.err
#SBATCH -o out/%x_%j.out
#SBATCH -p gpu
#SBATCH -t 0-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL

ml purge && ml pytorch
source .venv/bin/activate

srun python3 code/train.py --model_name google/flan-t5-base
