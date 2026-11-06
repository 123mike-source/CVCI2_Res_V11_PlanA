#!/bin/bash
#SBATCH -J cvci2res-val99
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:l40:4
#SBATCH --exclude=gpu4020
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --chdir=/share/home/u21012/nixiran/CVCI2_Res
#SBATCH -o /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.out
#SBATCH -e /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.err
set -euo pipefail
ROOT=/share/home/u21012/nixiran/CVCI2_Res
export OMP_NUM_THREADS=1
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT/results/val_l2_pretrained_ddim_4gpu_${SLURM_JOB_ID}"}
mkdir -p "$OUTPUT_ROOT"
echo "[$(date -Is)] START job=$SLURM_JOB_ID host=$(hostname) output=$OUTPUT_ROOT"
echo 'DDIM=[99,74,50,25,0] eta=0 seed=2026 cold_start=3 workers_per_gpu=6'
echo 'Selection: validation ego endpoint L2 mean at 1/2/3s with individual masks after cold start.'
trap 'code=$?; echo "[$(date -Is)] EXIT status=$code output=$OUTPUT_ROOT"' EXIT
mkdir -p "$OUTPUT_ROOT/run_receipts/$SLURM_JOB_ID"
sha256sum tools/eval_cvci3_protocol.py tools/summarize_pretrained_ddim_sweep.py \
  tools/validate_eval_cvci3.py sbatch_val_l2_pretrained_ddim_4gpu.sh \
  > "$OUTPUT_ROOT/run_receipts/$SLURM_JOB_ID/source.sha256"
if [[ ! -f "$OUTPUT_ROOT/source.sha256" ]]; then
  cp "$OUTPUT_ROOT/run_receipts/$SLURM_JOB_ID/source.sha256" "$OUTPUT_ROOT/source.sha256"
fi
bash tools/run_in_job.sh tools/validate_progress_io.py
if [[ ! -f "$OUTPUT_ROOT/unit.json" ]]; then
  echo "[$(date -Is)] Checking sampler, RNG and cold-start invariants"
  bash tools/run_in_job.sh tools/validate_eval_cvci3.py --output "$OUTPUT_ROOT/unit.json"
fi
if [[ ! -f "$OUTPUT_ROOT/smoke/metrics.json" ]]; then
  echo "[$(date -Is)] Four-scene native-DDIM smoke test"
  bash tools/run_in_job.sh -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="${SLURM_JOB_ID:-local}" --nnodes=1 --nproc_per_node=4 \
    tools/eval_cvci3_protocol.py --output-dir "$OUTPUT_ROOT/smoke" \
    --mode future --seed 2026 --ddim-steps 1 --ddim-schedule pretrained \
    --workers 6 --prefetch-factor 2 --max-scenes 4 --log-every 20 --log-seconds 10
fi
# Prioritize epoch 10 and 8 for early comparison; still evaluate every epoch.
for LABEL in baseline epoch_10 epoch_08 epoch_01 epoch_02 epoch_03 epoch_04 epoch_05 epoch_06 epoch_07 epoch_09; do
  if [[ -f "$OUTPUT_ROOT/$LABEL/metrics.json" ]]; then
    echo "[$(date -Is)] Existing result $LABEL; validating before reuse"
  else
    echo "[$(date -Is)] BEGIN $LABEL"
    if [[ -e "$OUTPUT_ROOT/$LABEL" ]]; then
      ARCHIVE="$OUTPUT_ROOT/interrupted/$LABEL-before-job$SLURM_JOB_ID"
      mkdir -p "$OUTPUT_ROOT/interrupted"
      test ! -e "$ARCHIVE"
      mv -- "$OUTPUT_ROOT/$LABEL" "$ARCHIVE"
      echo "Preserved interrupted attempt: $ARCHIVE"
    fi
    MODE=baseline
    ADAPTER_ARGS=()
    if [[ "$LABEL" != baseline ]]; then
      MODE=future
      ADAPTER_ARGS=(--adapter-checkpoint "$ROOT/experiments/future_adapter_8gpu_3043141/$LABEL.pth")
    fi
    bash tools/run_in_job.sh -m torch.distributed.run --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="${SLURM_JOB_ID:-local}" --nnodes=1 --nproc_per_node=4 \
      tools/eval_cvci3_protocol.py --output-dir "$OUTPUT_ROOT/$LABEL" \
      --mode "$MODE" "${ADAPTER_ARGS[@]}" --seed 2026 --ddim-steps 1 --ddim-schedule pretrained \
      --workers 6 --prefetch-factor 2 --log-every 20 --log-seconds 10
  fi
  bash tools/run_in_job.sh tools/summarize_pretrained_ddim_sweep.py --root "$OUTPUT_ROOT"
  echo "[$(date -Is)] END $LABEL"
done
bash tools/run_in_job.sh tools/summarize_pretrained_ddim_sweep.py --root "$OUTPUT_ROOT" --require-complete
echo "[$(date -Is)] COMPLETE baseline+10 epochs; see selection.json and leaderboard.csv"
