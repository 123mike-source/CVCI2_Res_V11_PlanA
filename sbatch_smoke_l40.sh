#!/bin/bash
#SBATCH -J cvci2-smoke
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:l40:1
#SBATCH --chdir=/share/home/u21012/nixiran/CVCI2_Res
#SBATCH -o /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.out
#SBATCH -e /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/11.6
source /share/home/u21012/anaconda3/etc/profile.d/conda.sh
conda activate wmmpc
export PYTHONUNBUFFERED=1
python /share/home/u21012/nixiran/CVCI2_Res/smoke_forward.py
