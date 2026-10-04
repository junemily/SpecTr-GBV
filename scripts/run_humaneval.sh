#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRAFT_MODEL="${DRAFT_MODEL:-JackFram/llama-160m}"
TARGET_MODEL="${TARGET_MODEL:-meta-llama/Llama-2-7b-hf}"
SEEDS="${SEEDS:-1001 1002 1003}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"

mkdir -p "$RESULTS_DIR/logs"
cd "$ROOT"

for seed in $SEEDS; do
  python benchmark/eval_humaneval.py \
    --draft_model "$DRAFT_MODEL" \
    --target_model "$TARGET_MODEL" \
    --num_candidates 7 --gamma 6 \
    --max_tokens 64 --max_prompts 32 \
    --temp 0.7 --top_p 1.0 --dtype float16 \
    --seed "$seed" --output_dir "$RESULTS_DIR" -e "spectrgbv_seed${seed}" \
    | tee "$RESULTS_DIR/logs/spectrgbv_seed${seed}.log"

  python benchmark/eval_humaneval.py \
    --draft_model "$DRAFT_MODEL" \
    --target_model "$TARGET_MODEL" \
    --num_candidates 1 --gamma 6 \
    --max_tokens 64 --max_prompts 32 \
    --temp 0.7 --top_p 1.0 --dtype float16 \
    --seed "$seed" --output_dir "$RESULTS_DIR" -e "gbv_seed${seed}" \
    | tee "$RESULTS_DIR/logs/gbv_seed${seed}.log"
done

python scripts/summarize_results.py "$RESULTS_DIR"/logs/*.log
