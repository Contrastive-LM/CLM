"""MLX Qwen3 last-token embedding server (OpenAI-compatible ``/v1/embeddings``).

Replaces ``vllm serve ... --runner pooling`` on Apple Silicon.  The reference
CLM head expects Qwen3-8B last-token pooling + L2-normalised vectors.

Refuses to load under memory pressure (see ``clm.memory``) so jetsam does not
kill a half-finished Hugging Face fetch with an empty log.
"""
from __future__ import annotations

import argparse
import base64
import os
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
from starlette.requests import Request

from .memory import MemoryPressureError, MemoryWatchdog, estimate_model_gib, require_free_memory, snapshot

DEFAULT_MODEL = os.environ.get("CLM_MLX_MODEL", "mlx-community/Qwen3-8B-4bit")
DEFAULT_PORT = int(os.environ.get("CLM_MLX_EMB_PORT", "8090"))
DEFAULT_MAX_TOKENS = int(os.environ.get("CLM_EMB_MAX_TOKENS", "2048"))
DEFAULT_MIN_FREE = float(os.environ.get("CLM_MLX_MIN_FREE_GIB", "0"))  # 0 => estimate from model id


def l2_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


class MlxEmbedder:
    """In-process last-token pooler over an mlx-lm Qwen3 checkpoint."""

    def __init__(self, model_id: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS,
                 cache_size: int = 200_000, batch: int = 8,
                 min_free_gib: float | None = None, wait_for_memory_s: float = 0.0):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.batch = batch
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self._lock = threading.Lock()

        need = min_free_gib if min_free_gib is not None else DEFAULT_MIN_FREE
        if need <= 0:
            need = estimate_model_gib(model_id)
        print(f"[clm-mlx-embed] preflight memory for {model_id} (need ≥ {need:.1f} GiB)", flush=True)
        require_free_memory(need, wait_s=wait_for_memory_s)
        dog = MemoryWatchdog(need * 0.5).start()
        t0 = time.perf_counter()
        try:
            print(f"[clm-mlx-embed] loading {model_id} ...", flush=True)
            self.model, self.tokenizer = load(model_id)
            # Hidden states live on ``model.model`` (before lm_head).
            self.backbone = self.model.model
            self.hidden = int(getattr(self.model.args, "hidden_size", 4096))
            mx.eval(self.backbone.parameters())
        finally:
            dog.stop()
            if dog.trips:
                print(f"[clm-mlx-embed] memory watchdog tripped {dog.trips} time(s) during load", flush=True)
        print(f"[clm-mlx-embed] ready in {time.perf_counter() - t0:.1f}s "
              f"(hidden={self.hidden}); {snapshot().free_gib:.1f} GiB reclaimable left", flush=True)

    def _tokenize(self, texts: list[str]) -> tuple[Any, Any]:
        """Pad to a rectangular batch; returns (input_ids [B,T], attention_mask [B,T])."""
        mx = self.mx
        encoded = [
            self.tokenizer.encode(t, add_special_tokens=True)[: self.max_tokens]
            for t in texts
        ]
        if not encoded:
            return mx.zeros((0, 0), dtype=mx.int32), mx.zeros((0, 0), dtype=mx.int32)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        tmax = max(len(e) for e in encoded)
        ids = np.full((len(encoded), tmax), pad_id, dtype=np.int32)
        mask = np.zeros((len(encoded), tmax), dtype=np.int32)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = e
            mask[i, : len(e)] = 1
        return mx.array(ids), mx.array(mask)

    def _forward(self, input_ids, attention_mask) -> np.ndarray:
        """Last non-pad token hidden state, L2-normalised, float32 numpy [B, H]."""
        mx = self.mx
        hidden = self.backbone(input_ids)  # [B, T, H]
        # last real token index per row
        lengths = mx.sum(attention_mask, axis=1) - 1
        lengths = mx.maximum(lengths, 0)
        batch = mx.arange(hidden.shape[0])
        pooled = hidden[batch, lengths]
        pooled = pooled.astype(mx.float32)
        norms = mx.linalg.norm(pooled, ord=2, axis=-1, keepdims=True)
        pooled = pooled / mx.maximum(norms, 1e-9)
        mx.eval(pooled)
        return np.array(pooled, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        """Embed ``texts`` (preserving order); returns vectors + approximate token count."""
        if not texts:
            return [], 0
        tokens = 0
        out: list[np.ndarray | None] = [None] * len(texts)
        # cache hits
        todo_idx: list[int] = []
        todo_text: list[str] = []
        with self._lock:
            for i, t in enumerate(texts):
                v = self.cache.get(t)
                if v is None:
                    todo_idx.append(i)
                    todo_text.append(t)
                else:
                    self.cache.move_to_end(t)
                    out[i] = v
        # unique miss texts, then expand
        unique: list[str] = list(dict.fromkeys(todo_text))
        got: dict[str, np.ndarray] = {}
        for i in range(0, len(unique), self.batch):
            chunk = unique[i:i + self.batch]
            ids, mask = self._tokenize(chunk)
            tokens += int(np.array(mask).sum())
            vecs = self._forward(ids, mask)
            for t, v in zip(chunk, vecs):
                got[t] = v
        with self._lock:
            for t, v in got.items():
                self.cache[t] = v
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        for i, t in zip(todo_idx, todo_text):
            out[i] = got[t]
        return [v for v in out], tokens  # type: ignore[misc]


def create_app(embedder: MlxEmbedder, served_name: str = "qwen3-8b"):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="CLM MLX Embeddings", version="0.1.0")
    app.state.embedder = embedder

    @app.get("/health")
    def health():
        return {"ok": True, "model": embedder.model_id, "hidden": embedder.hidden}

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": served_name, "object": "model", "owned_by": "clm-mlx"}]}

    @app.post("/v1/embeddings")
    async def embeddings(req: Request):
        body = await req.json()
        raw = body.get("input", [])
        if isinstance(raw, str):
            texts = [raw]
        elif isinstance(raw, list):
            texts = [str(t) for t in raw]
        else:
            return JSONResponse({"error": "input must be a string or list of strings"}, status_code=422)
        encoding = body.get("encoding_format", "float")
        vecs, tokens = embedder.embed_texts(texts)
        data = []
        for i, v in enumerate(vecs):
            if encoding == "base64":
                emb: Any = base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode("ascii")
            else:
                emb = v.tolist()
            data.append({"object": "embedding", "index": i, "embedding": emb})
        return {
            "object": "list",
            "data": data,
            "model": body.get("model") or served_name,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve Qwen3 last-token embeddings over Metal/MLX.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="mlx-lm model id or local path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--served-name", default=os.environ.get("CLM_EMB_MODEL", "qwen3-8b"))
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cache-size", type=int, default=200_000)
    ap.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE or None,
                    help="refuse to load unless this much reclaimable RAM is free "
                         "(default: estimate from model size; env CLM_MLX_MIN_FREE_GIB)")
    ap.add_argument("--wait-for-memory", type=float, default=float(os.environ.get("CLM_MLX_WAIT_S", "0")),
                    help="seconds to wait for free memory before failing (0 = fail immediately)")
    args = ap.parse_args()

    try:
        embedder = MlxEmbedder(args.model, max_tokens=args.max_tokens,
                               cache_size=args.cache_size, batch=args.batch,
                               min_free_gib=args.min_free_gib,
                               wait_for_memory_s=args.wait_for_memory)
    except MemoryPressureError as e:
        raise SystemExit(str(e)) from e
    app = create_app(embedder, served_name=args.served_name)
    print(f"[clm-mlx-embed] POST http://{args.host}:{args.port}/v1/embeddings", flush=True)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
