# CLM on Apple Silicon

The CLM API and projection head run on the CPU. The Qwen3-8B encoder runs on
the Mac GPU through MLX. `tools/mlx_embed_server.py` exposes its final-token
hidden state through the OpenAI-compatible `/v1/embeddings` endpoint expected
by CLM. It does not use an MTP head or generate text.

This path was exercised on an Apple Silicon Mac with the reference CLM head
and an MXFP8 Qwen3-8B checkpoint. It is separate from the upstream Linux/vLLM
path. The reference head was trained against Qwen3-8B embeddings, so compare
classification results with the reference encoder before depending on a new
quantization or checkpoint.

## Install

Use Python 3.12 on an Apple Silicon Mac. In a fresh environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[serve,hf]'
python -m pip install mlx==0.32.2 safetensors==0.8.0
python -m pip install 'mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@47e43a4526d42a6dcee758f07eb04b78a8439ecb'
```

Those MLX versions are the versions used for the working local setup. The FP8
conversion also uses PyTorch, supplied by CLM's `serve` extra.

## Prepare the FP8 model

Download the official Qwen3-8B FP8 checkpoint with the Hugging Face CLI. The
download remains in the standard Hugging Face cache; the converter writes a
separate MLX checkpoint to a directory you choose:

```bash
SOURCE=$(hf download Qwen/Qwen3-8B-FP8)
python tools/convert_qwen_fp8_to_mlx.py "$SOURCE" ./models/qwen3-8b-mxfp8
```

The converter restores and requantizes one linear layer at a time. It does not
download the BF16 checkpoint. Allow space for both the downloaded FP8 files
and the converted model while preparing it.

## Serve

In one terminal:

```bash
python tools/mlx_embed_server.py --model ./models/qwen3-8b-mxfp8 --port 8091
```

In another:

```bash
clm-serve --host 127.0.0.1 --port 8701 --device cpu \
  --emb-url http://127.0.0.1:8091/v1/embeddings --emb-model qwen3-8b
```

The playground is at <http://127.0.0.1:8701/>. Test the encoder before using
the playground:

```bash
curl -fsS http://127.0.0.1:8091/v1/models
curl -fsS http://127.0.0.1:8091/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b","input":["What causes tides?","The Moon’s gravity"]}'
```

`clm-serve` downloads the reference projection head if it is not already
cached. The encoder and API are local to the Mac by default; neither command
configures them to start after a reboot.

## Limits

- The model uses unified memory. Heavy memory pressure and swap can turn
  short classifications into multi-second requests.
- The encoder processes texts sequentially within one embeddings request.
- Conversion preserves FP8-class storage, but has not been shown to preserve
  every classification decision from the original BF16 model. Validate on
  representative traces before relying on it for production decisions.
- The endpoint implements the fields CLM's `Embedder` uses. It is not a
  general-purpose OpenAI embeddings server.
