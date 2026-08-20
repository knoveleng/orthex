#!/usr/bin/env bash
# Runs the orthex v1 pipeline sequentially (one GPU, one model at a time)
# against all 6 confirmed-compatible target models, logging progress lines
# to stdout that a Monitor can watch for start/finish/failure per model.
set -uo pipefail
cd "$(dirname "$0")/.."

# name:model-id:architecture-adapter (model-id/adapter blank -> use
# configs/default.yaml's own values, i.e. Llama)
MODELS=(
  "llama-3.2-3b-instruct::"
  "qwen2.5-3b-instruct:Qwen/Qwen2.5-3B-Instruct:qwen2"
  "qwen3-4b:Qwen/Qwen3-4B:qwen3"
  "qwen3.5-4b:Qwen/Qwen3.5-4B:qwen3_5"
  "gemma-2-2b-it:google/gemma-2-2b-it:gemma2"
  "gemma-3-4b-it:google/gemma-3-4b-it:gemma3"
)

for entry in "${MODELS[@]}"; do
  IFS=':' read -r name model_id adapter <<< "${entry}"
  echo "MODEL_START ${name}"
  args=(--config configs/default.yaml)
  [ -n "${model_id}" ] && args+=(--set "model.id=${model_id}")
  [ -n "${adapter}" ] && args+=(--set "model.architecture_adapter=${adapter}")
  python -m orthex.cli "${args[@]}" 2> "out_logs/${name}.err.log" 1> "out_logs/${name}.out.log"
  rc=$?
  if [ "${rc}" -eq 0 ]; then
    echo "MODEL_DONE ${name}"
  else
    echo "MODEL_FAILED ${name} rc=${rc}"
  fi
done

echo "RUN_ALL_COMPLETE"
