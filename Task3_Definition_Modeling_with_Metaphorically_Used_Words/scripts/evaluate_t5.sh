#!/bin/bash

#SBATCH -A project_2001403
#SBATCH -J evaluate_t5
#SBATCH -e err/%x_%j.err
#SBATCH -o out/%x_%j.out
#SBATCH -p gputest
#SBATCH -t 15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL

ml purge && ml pytorch
source .venv/bin/activate

srun python3 code/predict_and_score.py --data_path data/oxford_test.json \
	--model_name_or_path models/flan-t5-base/checkpoint-82200
srun python3 code/predict_and_score.py --data_path data/wordnet_test.json \
	--model_name_or_path models/flan-t5-base/checkpoint-82200
srun python3 code/predict_and_score.py --data_path data/metaphor_paraphrase.json \
	--model_name_or_path models/flan-t5-base/checkpoint-82200
