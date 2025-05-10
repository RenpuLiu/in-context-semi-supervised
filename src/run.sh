#!/usr/bin/env bash
set -euo pipefail                               # safer bash


source ~/miniconda3/etc/profile.d/conda.sh      # adjust path if different
conda activate in-context-learning


LOGDIR="$(dirname "$0")/logs"
mkdir -p "$LOGDIR"

UNLAB_LIST=(0 1 2 3 4 5 6 7 8 9 10)

for M in "${UNLAB_LIST[@]}"; do
  echo "[`date '+%F %T'`]  starting unlab_id=$M" | tee -a "$LOGDIR/runner.log"
  python train.py --config conf/semi_supervised.yaml \
        > "$LOGDIR/job_${M}_$(date +%F_%H-%M-%S).log" 2>&1
done
