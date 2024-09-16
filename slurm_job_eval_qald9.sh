#!/bin/bash -l
#SBATCH --job-name="SGPT-SPARQL-query-generation: EVAL QALD-9"
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=staff
#SBATCH --gpus-per-node=a100:1

srun python -u eval.py --generate /scratch/capolcorsin/SGPT-SPARQL-query-generation/runs/sgpt/qald9/ --dataset qald9 --generation_params_file config/gpt-2-base/generation_params.json --eval_dataset test  --output_file outputs/predictions_gpt2-base.json 
