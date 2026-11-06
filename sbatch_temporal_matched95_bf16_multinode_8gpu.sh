#!/bin/bash
#SBATCH -J cvci2res-m95bf16-2n4g
#SBATCH -p L40
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:l40:4
#SBATCH --exclude=gpu4015,gpu4016,gpu4020,gpu4041
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/share/home/u21012/nixiran/CVCI2_Res
#SBATCH -o /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.out
#SBATCH -e /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.err
set -euo pipefail
export ROOT=/share/home/u21012/nixiran/CVCI2_Res
export TRAIN_GPUS_PER_NODE=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 NCCL_ASYNC_ERROR_HANDLING=1 NCCL_IB_DISABLE=1
if [[ "${SLURM_NNODES}" != 2 || "${SLURM_NTASKS}" != 2 ]]; then
  echo 'This recipe requires 2 nodes, one launcher task per node and four L40 GPUs per task.' >&2
  exit 2
fi
mkdir -p "$ROOT/logs" "$ROOT/experiments"
export RUN_DIR=${RUN_DIR:-"$ROOT/experiments/matched95_bf16_random_2n4g_${SLURM_JOB_ID}"}
readarray -t TRAIN_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=${TRAIN_HOSTS[0]}
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "[$(date -Is)] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST work=$RUN_DIR"
echo "layout=2_nodes_x_4_L40 total_gpu=8 cpu_per_node=28 mem_per_node=128G master=$MASTER_ADDR:$MASTER_PORT NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo 'V6 gate12 x0 same-time direct samples=95 (36/25/16/9/9); qformer=false; CVCI2 checkpoint initialization; joint perception/planner; IRWM frozen'
echo 'BF16 trunk; FP32 head/loss/DDIM/IRWM; batch_per_gpu=6 grad_accum=1 workers=4 global_batch=48'
echo 'epochs=10 peak_lr=1e-4 warmup=500 optimizer updates; cosine budget computed from data; fresh training unless --resume is passed'
echo "TensorBoard=$RUN_DIR/tensorboard monitoring=$RUN_DIR/monitoring.jsonl"
# Match the reference launcher: one torchrun agent per node, four local workers.
# NCCL_IB_DISABLE=1 matches the reference TCP transport; local GPU P2P stays enabled.
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NTASKS" --ntasks-per-node=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" --kill-on-bad-exit=1 --mpi=none --export=ALL \
  bash -c '
    echo "[$(date -Is)] node_rank=$SLURM_PROCID host=$(hostname) visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
    exec bash "$ROOT/tools/run_in_job.sh" -m torch.distributed.run \
      --nnodes="$SLURM_NNODES" --nproc_per_node="$TRAIN_GPUS_PER_NODE" --node_rank="$SLURM_PROCID" \
      --rdzv_backend=static --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --max_restarts=0 "$@"
  ' launch_matched95 \
  "$ROOT/tools/train_temporal_matched_deform" --work-dir "$RUN_DIR" \
  --batch-size 6 --grad-accum 1 --workers 4 --epochs 10 --lr 1e-4 --precision bf16 \
  --weight-decay 1e-4 --seed 2026 --log-every 25 \
  --points-per-map 36 25 16 9 9 --window-m 6 3 --offset-mode learned \
  --future-dropout 0.1 \
  --monitor-every 100 "$@"
