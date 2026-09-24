#!/bin/bash
# Download the trained Qwen3-8B projection head from Hugging Face:
#   https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
# into checkpoints/CLM_v0.1-8B.pt (~75 MB).  `clm-download` does the same thing.
#
# Usage: ./download_head.sh
set -eu

REPO="Contrastive-LM/CLM-v0.1-8B"
FILE="CLM_v0.1-8B.pt"
DEST_DIR="$(cd "$(dirname "$0")" && pwd)/checkpoints"
DEST="$DEST_DIR/$FILE"

if [ -f "$DEST" ]; then
    echo "Already downloaded: $DEST"
    exit 0
fi
mkdir -p "$DEST_DIR"

if command -v hf >/dev/null 2>&1; then
    hf download "$REPO" "$FILE" --local-dir "$DEST_DIR"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" "$FILE" --local-dir "$DEST_DIR"
else
    echo "hf CLI not found, falling back to curl..."
    curl -fL "https://huggingface.co/$REPO/resolve/main/$FILE" -o "$DEST"
fi

ls -lh "$DEST"
echo "Done: $DEST"
