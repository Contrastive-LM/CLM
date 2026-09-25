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

If the checkout is in an iCloud-managed `Documents` folder, put the virtual
environment in a local directory such as `~/.local/share/clm-mac/venv` instead
of `.venv`. macOS can evict dependency files from `Documents`; this makes the
first PyTorch and Transformers imports unexpectedly slow.

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

## Local latency benchmark

On a MacBook Pro `Mac16,8` (Apple M4 Pro, 14 CPU cores, 48 GB unified memory;
macOS 26.5.2), the local API returned these timings on 2026-09-25:

| Cache condition | n | Server median | Server p95 | Local wall median | Local wall p95 | Median encoder tokens |
|---|---:|---:|---:|---:|---:|---:|
| First request after API cache reset | 1 | 424.8 ms | — | 425.6 ms | — | 46 |
| New state and new choices | 30 | 418.0 ms | 425.2 ms | 418.5 ms | 425.7 ms | 103 |
| New state, fixed choices | 30 | 161.6 ms | 168.8 ms | 162.1 ms | 169.3 ms | 37 |
| Identical cached request | 30 | 0.1 ms | 0.1 ms | 0.3 ms | 0.4 ms | 0 |

The encoder was already loaded on `127.0.0.1:8091`. A separate API process on
`127.0.0.1:8702` started with an empty vector cache and the reference
`CLM_v0.1-8B.pt` projection head on CPU. Requests were sequential over
localhost, using `clm-latest`, one choice question, three short department
criteria, and the default 2,048-token embedding limit. The benchmark gives
each new state and choice a unique case ID, then reuses the fixed choices, then
repeats an identical request. The API's `X-CLM-Latency-Ms` header is server
time; local wall time includes the HTTP round trip. Encoder token counts
confirm that the first two 30-request phases were misses and the identical
repeats were cache hits.

The encoder used `Qwen/Qwen3-8B-FP8` at Hugging Face revision
`220b46e3b2180893580a4454f21f22d3ebb187d3`, converted to MLX MXFP8 E4M3
with 32-element groups by `tools/convert_qwen_fp8_to_mlx.py`. Versions were
MLX 0.32.2, MLX-LM `0.31.4.dev129+g47e43a452`, Transformers 5.17.0, and
CLM 0.1.0. The local head file's SHA-256 was
`b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5`.
These are short classification requests, not long agent traces or a comparison
with another GPU. See the [raw samples](benchmarks/m4-pro-macos-2026-09-25.json).

To repeat the measurement, keep the encoder running, start a fresh API on
port 8702 with the same head, then run:

```bash
clm-serve --host 127.0.0.1 --port 8702 --device cpu \
  --ckpt /path/to/CLM_v0.1-8B.pt \
  --emb-url http://127.0.0.1:8091/v1/embeddings --emb-model qwen3-8b
python tools/benchmark_clm_api.py http://127.0.0.1:8702 \
  --samples 30 --json-out benchmark.json
```

Run the two commands in separate terminals. The benchmark refuses an API
whose vector cache is already used.

## Limits

- The model uses unified memory. Heavy memory pressure and swap can turn
  short classifications into multi-second requests.
- The encoder processes texts sequentially within one embeddings request.
- Conversion preserves FP8-class storage, but has not been shown to preserve
  every classification decision from the original BF16 model. Validate on
  representative traces before relying on it for production decisions.
- The endpoint implements the fields CLM's `Embedder` uses. It is not a
  general-purpose OpenAI embeddings server.
