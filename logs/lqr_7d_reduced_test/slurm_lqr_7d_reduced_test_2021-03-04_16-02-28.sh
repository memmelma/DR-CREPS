#!/usr/bin/env bash

###############################################################################
# SLURM Configurations
#SBATCH -J lqr_7d_reduced_test_2021-03-04_16-02-28
#SBATCH -a 0-0
#SBATCH -t 0-05:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=5000
#SBATCH -o ./logs/lqr_7d_reduced_test/%A_%a-out.txt
#SBATCH -e ./logs/lqr_7d_reduced_test/%A_%a-err.txt
###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
python3 lqr_mi_el.py \
		${@:2} \
		--seed $SLURM_ARRAY_TASK_ID \
		--results-dir $1
