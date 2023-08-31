#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=transbathy
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuin0084
#SBATCH --output=transbathy.%j.out
#SBATCH --error=transbathy.%j.err

module load cuda/11.2
module load miniconda/3
conda activate transbath
cd ~/code/Transbathy
python main.py
