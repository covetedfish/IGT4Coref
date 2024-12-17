#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4   # Number of requested cores
#SBATCH --mem=32G
#SBATCH --qos=preemptable
#SBATCH --out=logs/predict_es_coref.%j.out    # Output file name
#SBATCH --error=logs/predict_es_coref.%j.err
module purge
module load gcc/11.2.0
source /curc/sw/anaconda3/latest

# Run Python Script
conda activate glossy
cd "/projects/enri8153/CorefUD/src"

torchrun pred_coref.py