#!/usr/bin/env bash
# Start both Keural Nova model backends, each on its own GPU, bound to LOCALHOST
# only (not reachable from outside — the gateway is the only public entry point).
#   v1.0 -> GPU 0, 127.0.0.1:8001, served name "keural-nova-v1.0"
#   v1.1 -> GPU 1, 127.0.0.1:8002, served name "keural-nova-v1.1"
# Run in a tmux pane; Ctrl-C stops both. Wait for BOTH "Application startup complete".
set -uo pipefail

PY="/home/work/.keural_venv/bin/python"
PROJECT="/home/work/keural MoE Pro"
V10="$PROJECT/outputs/merged/Keural-35B-A3B-ckpt7500"   # == mkd-ai/Keural-Nova-v1.0
V11="$PROJECT/outputs/merged/Keural-35B-A3B"            # == mkd-ai/Keural-Nova-v1.1
LOGDIR="$PROJECT/gateway/logs"; mkdir -p "$LOGDIR"
export PYTHONPATH= HF_HUB_ENABLE_HF_TRANSFER=0

echo "Starting Keural Nova v1.0 on GPU0 -> 127.0.0.1:8001 ..."
CUDA_VISIBLE_DEVICES=0 "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$V10" --served-model-name keural-nova-v1.0 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 8192 --dtype bfloat16 --host 127.0.0.1 --port 8001 \
  > "$LOGDIR/v1.0.log" 2>&1 &
P1=$!

echo "Starting Keural Nova v1.1 on GPU1 -> 127.0.0.1:8002 ..."
CUDA_VISIBLE_DEVICES=1 "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$V11" --served-model-name keural-nova-v1.1 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 8192 --dtype bfloat16 --host 127.0.0.1 --port 8002 \
  > "$LOGDIR/v1.1.log" 2>&1 &
P2=$!

trap 'echo stopping; kill $P1 $P2 2>/dev/null' INT TERM
echo "v1.0 PID $P1 (gateway/logs/v1.0.log) | v1.1 PID $P2 (gateway/logs/v1.1.log)"
echo "Both loading (~3-4 min each, parallel)..."
wait
