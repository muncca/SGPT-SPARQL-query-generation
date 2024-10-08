#!/bin/bash -l
#SBATCH --job-name="SGPT: TRAIN QALD-9"
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=staff
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition=staff

epochs=75
knowledge_max_tokens=50
temp=0.4
maxlength=160
topk=10
exp_name="sgpt_ep${epochs}_lr6e-4_modelgpt2_knowledgemaxtokens${knowledge_max_tokens}_temperature${temp}_maxlength${maxlength}_topk${topk}"

echo "$exp_name"

source activate sgpt
srun python ~/dev/SGPT-SPARQL-query-generation/train.py --dataset qald9 --epochs $epochs --knowledge_max_tokens $knowledge_max_tokens --output_dir "/scratch/capolcorsin/SGPT-SPARQL-query-generation/runs" --exp_name "${exp_name}"
srun python -u eval.py --generate /scratch/capolcorsin/SGPT-SPARQL-query-generation/runs/${exp_name}/qald9/ --dataset qald9 --generation_params_file config/gpt-2-base/generation_params.json --eval_dataset test  --output_file outputs/predictions_gpt2-${exp_name}.json
conda deactivate
