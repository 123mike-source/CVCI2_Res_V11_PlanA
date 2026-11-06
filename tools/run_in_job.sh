#!/bin/bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/11.6
source /share/home/u21012/anaconda3/etc/profile.d/conda.sh
conda activate wmmpc
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."
# Multiprocessing UNIX sockets belong on node-local storage, not shared NFS.
TASK_TMP=$(mktemp -d "/tmp/cvci2res-${UID}.XXXXXX")
export TMPDIR="$TASK_TMP"
cleanup() {
  case "$TASK_TMP" in /tmp/cvci2res-"${UID}".*) rm -rf -- "$TASK_TMP" ;; esac
}
trap cleanup EXIT
python -u "$@"
