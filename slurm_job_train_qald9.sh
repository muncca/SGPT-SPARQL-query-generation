#!/bin/bash -l
#SBATCH --job-name="SGPT: TRAIN QALD-9"
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=staff
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=staff

srun python ~/dev/SGPT-SPARQL-query-generation/train.py --dataset qald9 --epochs 70 --output_dir "/scratch/capolcorsin/SGPT-SPARQL-query-generation/runs" --exp_name "sgpt_ep70_lr6e-4"
