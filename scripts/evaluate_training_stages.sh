#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${1:?usage: bash scripts/evaluate_training_stages.sh BASE_MODEL SFT_MODEL GRPO_MODEL [BENCHMARK] [OUTPUT_ROOT]}"
SFT_MODEL="${2:?missing SFT model}"
GRPO_MODEL="${3:?missing exported GRPO model}"
BENCHMARK="${4:-$ROOT/data/evaluation/tasks.jsonl}"
OUTPUT_ROOT="${5:-$ROOT/outputs/evaluation/training-stage-comparison}"
ENV_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_PORT="${LLM_PORT:-8000}"
LLM_URL="http://127.0.0.1:${LLM_PORT}"
MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
MODEL_PID=""
ENV_PID=""

for path in "$BASE_MODEL" "$SFT_MODEL" "$GRPO_MODEL" "$BENCHMARK"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done
cleanup_model() {
  if [[ -n "$MODEL_PID" ]] && kill -0 "$MODEL_PID" 2>/dev/null; then
    kill "$MODEL_PID"
    wait "$MODEL_PID" 2>/dev/null || true
  fi
  MODEL_PID=""
}

cleanup_all() {
  cleanup_model
  if [[ -n "$ENV_PID" ]] && kill -0 "$ENV_PID" 2>/dev/null; then
    kill "$ENV_PID"
    wait "$ENV_PID" 2>/dev/null || true
  fi
  ENV_PID=""
}
trap cleanup_all EXIT INT TERM

mkdir -p "$OUTPUT_ROOT"
if ! curl -sS --max-time 5 --output /dev/null "$ENV_URL"; then
  echo "starting ShopSimulator at $ENV_URL"
  bash "$ROOT/scripts/start_environment.sh" >"$OUTPUT_ROOT/environment.log" 2>&1 &
  ENV_PID=$!
  env_ready=0
  for _ in $(seq 1 60); do
    if curl -sS --max-time 5 --output /dev/null "$ENV_URL"; then
      env_ready=1
      break
    fi
    if ! kill -0 "$ENV_PID" 2>/dev/null; then
      echo "ShopSimulator exited during startup" >&2
      tail -n 80 "$OUTPUT_ROOT/environment.log" >&2
      exit 1
    fi
    sleep 5
  done
  if [[ "$env_ready" != 1 ]]; then
    echo "ShopSimulator did not become reachable within 5 minutes" >&2
    exit 1
  fi
fi
if curl -fsS "$LLM_URL/health" >/dev/null 2>&1; then
  echo "port $LLM_PORT already has a live model server; stop it first." >&2
  exit 1
fi

labels=(base sft grpo-step50)
models=("$BASE_MODEL" "$SFT_MODEL" "$GRPO_MODEL")
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  model="${models[$index]}"
  run_dir="$OUTPUT_ROOT/$label"
  mkdir -p "$run_dir"
  echo "[$label] starting vLLM from $model"
  VLLM_USE_FLASHINFER_SAMPLER=0 \
    LLM_PORT="$LLM_PORT" SERVED_MODEL_NAME="$MODEL_NAME" \
    bash "$ROOT/scripts/serve_model.sh" "$model" >"$run_dir/vllm.log" 2>&1 &
  MODEL_PID=$!

  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "$LLM_URL/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$MODEL_PID" 2>/dev/null; then
      echo "[$label] vLLM exited during startup" >&2
      tail -n 80 "$run_dir/vllm.log" >&2
      exit 1
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    echo "[$label] vLLM did not become ready within 15 minutes" >&2
    exit 1
  fi

  echo "[$label] evaluating $(wc -l < "$BENCHMARK") frozen tasks"
  PYTHONPATH="$ROOT/src" "$ROOT/.venv/bin/python" \
    "$ROOT/scripts/evaluate_shop_benchmark.py" \
    --benchmark "$BENCHMARK" \
    --output "$run_dir/trajectories.jsonl" \
    --summary "$run_dir/summary.json" \
    --base-url "$ENV_URL" \
    --model "$MODEL_NAME" \
    --llm-base-url "$LLM_URL/v1" \
    --api-key EMPTY \
    --max-steps 35 \
    --temperature 0 \
    --top-p 1 \
    --max-tokens 512 \
    --context-window 24576
  "$ROOT/.venv/bin/python" "$ROOT/scripts/build_eval_report.py" --run-dir "$run_dir"
  cleanup_model
  sleep 5
done

"$ROOT/.venv/bin/python" "$ROOT/scripts/summarize_stage_comparison.py" \
  --run "Base=$OUTPUT_ROOT/base" \
  --run "SFT=$OUTPUT_ROOT/sft" \
  --run "GRPO-50=$OUTPUT_ROOT/grpo-step50" \
  --json "$OUTPUT_ROOT/comparison.json" \
  --markdown "$OUTPUT_ROOT/comparison.md"
