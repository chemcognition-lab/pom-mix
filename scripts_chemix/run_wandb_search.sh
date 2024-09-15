#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=16384
#SBATCH --job-name=chemix_saug
#SBATCH --output=./slurm_outputs_wandb/log_%a.txt
#SBATCH --array=1-200

source ~/.bashrc
conda init bash
conda activate dream

wandb agent --count 1 rajaonsonella/iclr_chemix_sweep/su0bxvw8