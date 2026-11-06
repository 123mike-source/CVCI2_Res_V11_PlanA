#!/bin/bash
#SBATCH -J cvci2res-matched95bf16
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:l40:8
#SBATCH --exclude=gpu4015,gpu4016,gpu4020
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/share/home/u21012/nixiran/CVCI2_Res
#SBATCH -o /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.out
#SBATCH -e /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.err
set -euo pipefail
ROOT=/share/home/u21012/nixiran/CVCI2_Res
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$ROOT/logs" "$ROOT/experiments"
RUN_DIR=${RUN_DIR:-"$ROOT/experiments/matched95_bf16_gate12_8gpu_${SLURM_JOB_ID}"}
echo "[$(date -Is)] job=$SLURM_JOB_ID host=$(hostname) work=$RUN_DIR"
echo 'V6 gate12 x0 same-time direct samples=95 (36/25/16/9/9); qformer=false; CVCI2 checkpoint initialization; joint perception/planner; IRWM frozen'
echo 'BF16 trunk; FP32 head/loss/DDIM/IRWM; batch_per_gpu=6 grad_accum=1 workers=4'
echo 'epochs=10 peak_lr=1e-4 warmup=500 optimizer updates; cosine budget computed from data'
echo "TensorBoard=$RUN_DIR/tensorboard monitoring=$RUN_DIR/monitoring.jsonl"
bash "$ROOT/tools/run_in_job.sh" -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="$SLURM_JOB_ID" \
  "$ROOT/tools/train_temporal_matched_deform" --work-dir "$RUN_DIR" \
  --batch-size 6 --grad-accum 1 --workers 4 --epochs 10 --lr 1e-4 --precision bf16 \
  --points-per-map 36 25 16 9 9 --window-m 6 3 --offset-mode learned \
  --future-dropout 0.1 \
  --monitor-every 100 "$@"
