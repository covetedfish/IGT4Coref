#!/bin/bash
#SBATCH --nodes=1           # Number of requested nodes
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4   # Number of requested cores
#SBATCH --mem=32G
#SBATCH --qos=preemptable
#SBATCH --out=logs/train_coref.%j.out    # Output file name
#SBATCH --error=logs/train_coref.%j.err
module purge
module load gcc/11.2.0
source /curc/sw/anaconda3/latest

# Run Python Script
conda activate glossy
cd "/projects/enri8153/CorefUD/"

torchrun --nproc_per_node=4 src/corefud.py -c configs/small_turkish.cfg