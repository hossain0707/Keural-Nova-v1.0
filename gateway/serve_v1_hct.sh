#!/usr/bin/env bash
# Keural Nova v1.0 — matched to HCT's production Qwen serving config.
# Differences from HCT reference, forced by our hardware:
#   TP=4 -> TP=2 (we have 2x H200, not 4). Everything else matches.
# 1M context via the CORRECT rope override (full mrope block preserved, factor 4.0).
set -uo pipefail
PY="/home/work/.keural_venv/bin/python"
MODEL="/home/work/keural MoE Pro/outputs/merged/Keural-35B-A3B-ckpt7500"

export PYTHONPATH= PYTHONNOUSERSITE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_DEEP_GEMM=0
export HF_HUB_ENABLE_HF_TRANSFER=0

exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name keural-nova-v1.0 \
  --port 8001 --host 127.0.0.1 \
  --tensor-parallel-size 2 \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --dtype auto \
  --attention-backend flashinfer \
  --moe-backend triton \
  --gpu-memory-utilization 0.85 \
  --max-model-len 1000000 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":2}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
