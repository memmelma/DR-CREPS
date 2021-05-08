#!/usr/bin/env bash

###############################################################################
# SLURM Configurations
#SBATCH -J segway_2021-03-17_10-40-07
#SBATCH -a 0-0
#SBATCH -t 0-05:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=5000
#SBATCH -o ./logs/segway/%A_%a-out.txt
#SBATCH -e ./logs/segway/%A_%a-err.txt
###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
python3 segway_mi_el.py \
		${@:2} \
		--seed $SLURM_ARRAY_TASK_ID \
		--results-dir $1
