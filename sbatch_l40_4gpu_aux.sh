#!/bin/bash
# Clean from-scratch comparison against the original CVCI2 run.
# Global batch remains 4 GPUs x 12 samples = 48.
#SBATCH -J cvci2-aux
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:l40:4
#SBATCH --exclude=gpu4020
#SBATCH --chdir=/share/home/u21012/nixiran/CVCI2_Res
#SBATCH -o /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.out
#SBATCH -e /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.err
set -euo pipefail

source /etc/profile.d/modules.sh
module load cuda/11.6
source /share/home/u21012/anaconda3/etc/profile.d/conda.sh
conda activate wmmpc
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=6
export TMPDIR=/share/home/u21012/nixiran/CVCI2_Res/cache/tmp
mkdir -p "$TMPDIR" /share/home/u21012/nixiran/CVCI2_Res/logs /share/home/u21012/nixiran/CVCI2_Res/experiments
RUN_DIR=/share/home/u21012/nixiran/CVCI2_Res/experiments/transfuser_dit_aux_$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN_DIR"
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc_per_node=4 --master_port="$MASTER_PORT" /share/home/u21012/nixiran/CVCI2_Res/train_nusc.py \
  --data-root "${NUSCENES_DATA_ROOT:-/mnt/workspace/dataSet/nuscene}" \
  --work-dir "$RUN_DIR" \
  --image-checkpoint /share/home/u21012/nixiran/CVCI2_Res/ckpts/resnet34-b627a593.pth \
  --batch-size 12 --workers 6 --epochs 10 --lr 1e-4 "$@"
