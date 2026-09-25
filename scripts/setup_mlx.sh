#!/usr/bin/env bash
# Create a local Python venv and install CLM for Apple Silicon (Metal / MLX).
# Does not ship a virtualenv — run this after unpacking the archive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
if [[ -z "${PY}" ]]; then
  echo "Need Python 3.10+ (python3.12 recommended)." >&2
  exit 1
fi
ver="$("$PY" -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' \
  || { echo "Python $ver is too old; need 3.10+."; exit 1; }

echo "[setup] using $PY ($ver)"
if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -e '.[mlx]'

echo "[setup] downloading reference projection head (~75 MB)..."
clm-download

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
echo "[setup] done."
echo
echo "  source .venv/bin/activate"
echo "  bash serve_mlx.sh          # embeddings :8090 + API :8700"
echo "  # or: clm-embed-mlx --port 8090   &&   clm-serve --device mlx --emb-url http://127.0.0.1:8090/v1/embeddings"
echo
echo "Optional: export CLM_MLX_MODEL=mlx-community/Qwen3-8B-4bit"
echo "Memory: stop other large Metal servers first (need ~8–12 GiB reclaimable)."
