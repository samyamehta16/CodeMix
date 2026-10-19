#!/usr/bin/env bash
# Multi-seed transformer grid. Does NOT overwrite seed-42 model directories.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

MODELS=(indicbert-v2 muril mbert xlm-roberta)
SPLITS=(random hard)
SEEDS=(1 2 3 4 5)

mkdir -p eval_results_1400/logs
LOG=eval_results_1400/logs/multiseed.log
: > "$LOG"

for split in "${SPLITS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      tag="${model}_${split}_seed${seed}"
      metrics="eval_results_1400/metrics/${tag}_test.json"
      if [[ -f "$metrics" ]]; then
        echo "SKIP existing $tag" | tee -a "$LOG"
        continue
      fi
      echo "=== TRAIN $tag ===" | tee -a "$LOG"
      python scripts/eval_1400/train_one.py \
        --model "$model" --split "$split" --seed "$seed" \
        --epochs 4 --lr 2e-5 --warmup-ratio 0 --batch-size 16 --max-length 64 \
        2>&1 | tee -a "eval_results_1400/logs/${tag}.log" | tee -a "$LOG" | tail -20
    done
  done
done

echo "Multiseed grid done." | tee -a "$LOG"
