#!/bin/bash
#SBATCH --job-name=classify_eval
#SBATCH --output=logs/classify_%j.out
#SBATCH --error=logs/classify_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:L40:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sujankhadka@u.boisestate.edu

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate classify_env

# Go to working directory
cd /bsuhome/sujankhadka/claude

# Create logs directory if not exists
mkdir -p logs

# Show GPU info
nvidia-smi

# Run the script
python finetune_longformer.py

echo "End: $(date)"
