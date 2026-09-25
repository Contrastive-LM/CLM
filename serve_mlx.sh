#!/usr/bin/env bash
# Launch CLM on Apple Silicon: MLX embedding server + clm-serve --device mlx.
# Preflights unified-memory pressure so we do not jetsam mid-download next to a
# resident 30B+ Metal server.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${CLM_MLX_MODEL:-mlx-community/Qwen3-8B-4bit}"
EMB_PORT="${CLM_MLX_EMB_PORT:-8090}"
CLM_PORT="${CLM_PORT:-8700}"
LOGDIR="${CLM_LOGDIR:-/tmp/clm-mlx}"
WAIT_S="${CLM_MLX_WAIT_S:-0}"
mkdir -p "$LOGDIR"

export CLM_DEVICE="${CLM_DEVICE:-mlx}"

# shellcheck disable=SC1091
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

echo "[serve_mlx] memory preflight"
python - <<'PY'
from clm.memory import estimate_model_gib, format_snapshot, require_free_memory, MemoryPressureError
import os, sys
model = os.environ.get("CLM_MLX_MODEL", "mlx-community/Qwen3-8B-4bit")
need = float(os.environ.get("CLM_MLX_MIN_FREE_GIB") or estimate_model_gib(model))
wait = float(os.environ.get("CLM_MLX_WAIT_S", "0"))
try:
    require_free_memory(need, wait_s=wait)
except MemoryPressureError as e:
    print(e, file=sys.stderr)
    print("[serve_mlx] tip: stop the big mlx-serve (or anything holding >10 GiB Metal RSS), then retry.",
          file=sys.stderr)
    sys.exit(2)
PY

echo "[serve_mlx] embedder $MODEL on :$EMB_PORT"
clm-embed-mlx --model "$MODEL" --port "$EMB_PORT" --host 127.0.0.1 \
  --wait-for-memory "$WAIT_S" \
  >>"$LOGDIR/embed.log" 2>&1 &
EMB_PID=$!
echo "$EMB_PID" >"$LOGDIR/embed.pid"

cleanup() {
  kill "$EMB_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until /v1/models answers (or embedder dies)
for _ in $(seq 1 180); do
  if ! kill -0 "$EMB_PID" 2>/dev/null; then
    echo "[serve_mlx] embedder exited; see $LOGDIR/embed.log" >&2
    tail -40 "$LOGDIR/embed.log" >&2 || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${EMB_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:${EMB_PORT}/v1/models" >/dev/null \
  || { echo "[serve_mlx] embedder failed to start; see $LOGDIR/embed.log"; tail -40 "$LOGDIR/embed.log"; exit 1; }

echo "[serve_mlx] clm-serve on :$CLM_PORT"
exec clm-serve --port "$CLM_PORT" --device mlx \
  --emb-url "http://127.0.0.1:${EMB_PORT}/v1/embeddings" \
  --emb-model qwen3-8b "$@"
