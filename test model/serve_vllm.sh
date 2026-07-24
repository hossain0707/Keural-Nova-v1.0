#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server for Keural MoE Pro v1.0 on both H200s.
# Run this in one tmux pane; run the chatbot (run_chatbot.sh) in another.
#
# Model source: defaults to the LOCAL merged weights, which are byte-identical
# to mkd-hossain/Keural-MoE-Pro-v1.0 on Hugging Face (same merge fingerprint) —
# avoids a 72 GB re-download. To serve straight from the Hub instead:
#   KEURAL_MODEL=mkd-hossain/Keural-MoE-Pro-v1.0 ./serve_vllm.sh
set -euo pipefail

PROJECT="/home/work/keural MoE Pro"
# space-free python (flashinfer JIT breaks on spaces in the interpreter path)
PY="/home/work/.keural_venv/bin/python"

MODEL="${KEURAL_MODEL:-$PROJECT/outputs/merged/Keural-35B-A3B-ckpt7500}"
PORT="${KEURAL_PORT:-8001}"

export PYTHONPATH=
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Serving: $MODEL"
echo "Port:    $PORT   (served model name: keural-v1.0)"

exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name keural-v1.0 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port "$PORT"
