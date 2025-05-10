#!/usr/bin/env bash
# Run IC‑SSL training ten times

set -e  # Abort on first error

for i in {1..10}; do
    echo "===== Run ${i}/10: $(date '+%F %T') ====="
    python train.py --config conf/semi_supervised.yaml
done
