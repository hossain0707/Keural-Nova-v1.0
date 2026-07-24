#!/usr/bin/env bash
# Serve BOTH models at once for live side-by-side comparison:
#   Keural Nova v1.0  -> GPU 0, port 8001, name "keural-v1.0"
#   Qwen3.6-35B-A3B base -> GPU 1, port 8002, name "qwen-base"
# One model per GPU (no tensor-parallel split) so both fit simultaneously
# (~70 GB weights each on a 143 GB H200). Run in a tmux pane; Ctrl-C stops both.
set -uo pipefail

PY="/home/work/.keural_venv/bin/python"
PROJECT="/home/work/keural MoE Pro"
KEURAL="$PROJECT/outputs/merged/Keural-35B-A3B-ckpt7500"
QWEN="/home/work/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
LOGDIR="$PROJECT/test model/logs"; mkdir -p "$LOGDIR"

export PYTHONPATH=
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Starting Keural v1.0 on GPU0:8001 ..."
CUDA_VISIBLE_DEVICES=0 "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$KEURAL" --served-model-name keural-v1.0 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 8192 --dtype bfloat16 --host 0.0.0.0 --port 8001 \
  > "$LOGDIR/keural.log" 2>&1 &
KPID=$!

echo "Starting Qwen3.6 base on GPU1:8002 ..."
CUDA_VISIBLE_DEVICES=1 "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$QWEN" --served-model-name qwen-base \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
  --max-model-len 8192 --dtype bfloat16 --host 0.0.0.0 --port 8002 \
  > "$LOGDIR/qwen.log" 2>&1 &
QPID=$!

# stop both if this script is interrupted
trap 'echo "stopping..."; kill $KPID $QPID 2>/dev/null' INT TERM
echo "Keural PID $KPID (log: test model/logs/keural.log)"
echo "Qwen   PID $QPID (log: test model/logs/qwen.log)"
echo "Both loading (~3-4 min each, in parallel). Waiting..."
wait
