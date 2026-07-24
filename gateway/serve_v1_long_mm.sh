#!/usr/bin/env bash
# Keural Nova v1.0 ONLY — 1M context (YaRN) + Image-Text-to-Text, TP=2 (both GPUs).
# Serves on 127.0.0.1:8001 so the existing gateway/tunnel keeps working.
# NOTE: YaRN on this mrope/VL model is not guaranteed — verify output coherence
# after launch. 1M as a permanent default may reduce short-context quality.
set -uo pipefail
PY="/home/work/.keural_venv/bin/python"
MODEL="/home/work/keural MoE Pro/outputs/merged/Keural-35B-A3B-ckpt7500"
export PYTHONPATH= PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0

exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name keural-nova-v1.0 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 \
  --max-model-len 1010000 \
  --hf-overrides '{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}' \
  --limit-mm-per-prompt '{"image":2}' \
  --dtype bfloat16 --host 127.0.0.1 --port 8001
