#!/bin/bash
#SBATCH -J cvci2res-eval4
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:l40:4
#SBATCH --exclude=gpu4015,gpu4016,gpu4020
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --chdir=/share/home/u21012/nixiran/CVCI2_Res
#SBATCH -o /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.out
#SBATCH -e /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.err
set -euo pipefail
ROOT=/share/home/u21012/nixiran/CVCI2_Res
export OMP_NUM_THREADS=1
mkdir -p "$ROOT/logs" "$ROOT/results"
if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch sbatch_temporal_eval_4gpu.sh /absolute/path/to/current_epoch.pth [evaluation options]" >&2
  exit 2
fi
ADAPTER_ARGS=()
if [[ "$1" != --* ]]; then
  ADAPTER_ARGS=(--adapter-checkpoint "$1")
  shift
fi
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/results/eval_cvci3_protocol_4gpu_${SLURM_JOB_ID}"}
bash "$ROOT/tools/run_in_job.sh" -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="${SLURM_JOB_ID:-local}" --nnodes=1 --nproc_per_node=4 \
  "$ROOT/tools/eval_cvci3_protocol.py" "${ADAPTER_ARGS[@]}" --output-dir "$OUTPUT_DIR" \
  --mode future --seed 2026 --ddim-steps 1 --ddim-schedule pretrained --workers 6 --prefetch-factor 2 \
  --log-every 20 --log-seconds 10 "$@"
