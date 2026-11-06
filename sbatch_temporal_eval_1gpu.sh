#!/bin/bash
#SBATCH -J cvci2res-eval
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:l40:1
#SBATCH --exclude=gpu4015,gpu4016,gpu4020
#SBATCH --mem=128G
#SBATCH --chdir=/share/home/u21012/nixiran/CVCI2_Res
#SBATCH -o /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.out
#SBATCH -e /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.err
set -euo pipefail
ROOT=/share/home/u21012/nixiran/CVCI2_Res
mkdir -p "$ROOT/logs" "$ROOT/results"
bash "$ROOT/tools/run_in_job.sh" tools/eval_temporal.py \
  --output-dir "$ROOT/results/eval_${SLURM_JOB_ID}" --workers 4 "$@"
