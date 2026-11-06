#!/bin/bash
#SBATCH -J cvci2res-v5check
#SBATCH -p L40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:l40:4
#SBATCH --exclude=gpu4015,gpu4016,gpu4020
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --chdir=/share/home/u21012/nixiran/CVCI2_Res
#SBATCH -o /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.out
#SBATCH -e /share/home/u21012/nixiran/CVCI2_Res/logs/slurm-%x-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2
GPUS=${SLURM_GPUS_ON_NODE:-4}
if [[ "$GPUS" != 1 && "$GPUS" != 2 && "$GPUS" != 4 && "$GPUS" != 8 ]]; then echo 'Expected 1/2/4/8 GPUs'; exit 1; fi
EVAL_GPUS=$((GPUS > 1 ? GPUS / 2 : 1))
T="experiments/matched_deform_v5_validation_${SLURM_JOB_ID}"
V="results/matched_deform_v5_validation_${SLURM_JOB_ID}"
mkdir -p "$T" "$V"
run() { bash tools/run_in_job.sh "$@"; }
echo "[$(date -Is)] job=$SLURM_JOB_ID host=$(hostname) GPUs=$GPUS validation=$V"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
run -c 'import torch; print("CUDA",torch.cuda.device_count(),flush=True); [print(i,torch.cuda.get_device_name(i),float(torch.ones(1,device="cuda:"+str(i)).sum()),flush=True) for i in range(torch.cuda.device_count())]' > "$V/cuda_preflight.log" 2>&1
run tools/validate_matched_deform --output "$V/unit.json" > "$V/unit.log" 2>&1
echo 'PASS unit geometry/masks/task gradients/compatibility'
train() { run -m torch.distributed.run --nnodes=1 --nproc_per_node="$GPUS" \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="${SLURM_JOB_ID}-train" \
  tools/train_temporal_matched_deform --work-dir "$T" --batch-size 2 --grad-accum 3 --workers 1 \
  --epochs 10 --steps-per-epoch 6 --lr 3e-5 --log-every 1 --monitor-every 1 --verify-ddp "$@"; }
train --stop-after-epoch 1 > "$T/train_phase1.log" 2>&1
echo "PASS $GPUS-GPU batch=2 first epoch"
train --resume "$T/last.pth" --stop-after-epoch 2 > "$T/train_resume.log" 2>&1
echo "PASS $GPUS-GPU exact resume and monitoring"
IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm must expose devices}"
if [[ ${#DEVICES[@]} -ne $GPUS ]]; then echo 'Unexpected visible GPU count'; exit 1; fi
FIRST=$(IFS=,; echo "${DEVICES[*]:0:EVAL_GPUS}")
SECOND=$(IFS=,; echo "${DEVICES[*]:EVAL_GPUS:EVAL_GPUS}")
evaluate() { run -m torch.distributed.run --nnodes=1 --nproc_per_node="$EVAL_GPUS" \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="${SLURM_JOB_ID}-$1" \
  tools/eval_cvci3_protocol.py --adapter-checkpoint "$T/epoch_02.pth" --output-dir "$V/$1" \
  --mode "${3:-future}" --ddim-schedule pretrained --max-scenes 1 --max-frames "$2" \
  --workers 1 --log-every 1 --log-seconds 2; }
if [[ "$GPUS" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="$FIRST" evaluate eval_a 6 > "$V/eval_a.log" 2>&1
  CUDA_VISIBLE_DEVICES="$FIRST" evaluate eval_b 12 > "$V/eval_b.log" 2>&1
else
  CUDA_VISIBLE_DEVICES="$FIRST" evaluate eval_a 6 > "$V/eval_a.log" 2>&1 &
  PID_A=$!
  CUDA_VISIBLE_DEVICES="$SECOND" evaluate eval_b 12 > "$V/eval_b.log" 2>&1 &
  PID_B=$!
  wait "$PID_A"
  wait "$PID_B"
fi
CUDA_VISIBLE_DEVICES="$FIRST" evaluate disabled 6 disabled > "$V/disabled.log" 2>&1
echo "PASS evaluations and disabled-future mode; train_world=$GPUS eval_world=$EVAL_GPUS"
run tools/validate_matched_deform --experiments "$T" --results "$V" --expected-train-world "$GPUS" \
  --output "$V/summary.json" > "$V/comparison.log" 2>&1
cat "$V/summary.json"
