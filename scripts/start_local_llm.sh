#!/usr/bin/env bash
set -euo pipefail

# Local overrides live in the untracked .env at the project root.
ENV_FILE="$(dirname "$0")/../.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${LOCAL_LLM_MODEL_PATH:-$HOME/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
HOST="${LOCAL_LLM_HOST:-127.0.0.1}"
PORT="${LOCAL_LLM_PORT:-8080}"
THREADS="${LOCAL_LLM_THREADS:-4}"
CONTEXT="${LOCAL_LLM_CONTEXT:-2048}"

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "llama-server is not built: $LLAMA_SERVER" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model file not found: $MODEL_PATH" >&2
    exit 1
fi

exec "$LLAMA_SERVER" \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --threads "$THREADS" \
    --ctx-size "$CONTEXT" \
    --batch-size 128 \
    --parallel 1 \
    --cont-batching
