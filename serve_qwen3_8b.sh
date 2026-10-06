#!/bin/bash
# Serve ONE Qwen3-8B embedding server as the encoder behind `clm-serve`.
# LAST-token pooling, same as the precompute, so the embeddings match what the
# head was trained on.
#
# ENGINE picks the backend: "vllm" (Linux + NVIDIA GPU) or "llama.cpp" (Apple
# Silicon, Metal-accelerated, via llama-server). Defaults to "llama.cpp" on
# Darwin and "vllm" everywhere else; see "Running on Apple Silicon" in the
# README for why vLLM doesn't apply there and how quantization affects the
# projection heads.
#
# Usage: GPU=0 PORT=8090 UTIL=0.35 ./serve_qwen3_8b.sh                  # vllm
#        ENGINE=llama.cpp QUANT=BF16 PORT=8090 ./serve_qwen3_8b.sh      # llama.cpp
set -u
ENGINE="${ENGINE:-$([ "$(uname -s)" = "Darwin" ] && echo llama.cpp || echo vllm)}"
PORT="${PORT:-8090}"
MAXLEN="${MAXLEN:-2048}"
LOGDIR="${LOGDIR:-$(cd "$(dirname "$0")/.." && pwd)/logs}"
mkdir -p "$LOGDIR"

if [ "$ENGINE" = "llama.cpp" ]; then
    QUANT="${QUANT:-BF16}"
    echo "serving Qwen3-8B ($QUANT) embeddings on port $PORT (context $MAXLEN) via llama-server"
    exec llama-server -hf "unsloth/Qwen3-8B-GGUF:$QUANT" \
        --alias qwen3-8b \
        --embedding \
        --pooling last \
        -ngl 1024 \
        -fa on \
        -c "$MAXLEN" \
        --port "$PORT" \
        >> "$LOGDIR/llama_server_demo_8b.log" 2>&1
else
    GPU="${GPU:-0}"
    UTIL="${UTIL:-0.35}"
    echo "serving Qwen3-8B pooling on GPU $GPU port $PORT (util $UTIL) via vllm serve"
    CUDA_VISIBLE_DEVICES=$GPU exec vllm serve Qwen/Qwen3-8B \
        --served-model-name qwen3-8b \
        --runner pooling \
        --enforce-eager \
        --enable-prefix-caching \
        --max-model-len "$MAXLEN" \
        --gpu-memory-utilization "$UTIL" \
        --max-num-seqs 32 \
        --port "$PORT" \
        >> "$LOGDIR/vllm_demo_8b.log" 2>&1
fi
