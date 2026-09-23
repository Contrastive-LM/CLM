"""Encoder side: an OpenAI-compatible ``/v1/embeddings`` endpoint (vLLM pooling
server) plus an LRU cache of L2-normalised embeddings.

The reference head expects Qwen3-8B with last-token pooling, e.g.

    vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling \\
         --enable-prefix-caching --max-model-len 2048 --port 8090
"""
from __future__ import annotations

import base64
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
import requests


class EmbedderError(RuntimeError):
    pass


def l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


class Embedder:
    def __init__(self, url: str = "http://127.0.0.1:8090/v1/embeddings", model: str = "qwen3-8b",
                 max_tokens: int | None = 2048, cache_size: int = 200_000, batch: int = 32,
                 timeout: float = 300.0, api_key: str | None = None):
        self.url, self.model, self.max_tokens, self.batch, self.timeout = url, model, max_tokens, batch, timeout
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self._lock = threading.Lock()
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _fetch(self, texts: list[str]) -> tuple[list[np.ndarray], int]:
        body: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "base64"}
        if self.max_tokens:
            body["truncate_prompt_tokens"] = self.max_tokens
        try:
            r = self.session.post(self.url, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise EmbedderError(f"embedder unreachable at {self.url}: {e}") from e
        if r.status_code != 200:
            raise EmbedderError(f"embedder error {r.status_code}: {r.text[:300]}")
        j = r.json()
        out: list[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
        for d in j["data"]:
            e = d["embedding"]
            v = (np.frombuffer(base64.b64decode(e), dtype=np.float32) if isinstance(e, str)
                 else np.asarray(e, dtype=np.float32))
            out[d["index"]] = l2(v.astype(np.float32))
        return out, int((j.get("usage") or {}).get("prompt_tokens", 0) or 0)

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        """-> ([n, hidden] L2-normalised embeddings, encoder tokens spent on cache misses)."""
        vecs: dict[str, np.ndarray] = {}
        todo: list[str] = []
        with self._lock:
            for t in dict.fromkeys(texts):
                v = self.cache.get(t)
                if v is None:
                    todo.append(t)
                else:
                    self.cache.move_to_end(t); vecs[t] = v
        tokens = 0
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            got, tk = self._fetch(chunk)
            tokens += tk
            with self._lock:
                for t, v in zip(chunk, got):
                    vecs[t] = v; self.cache[t] = v
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return np.stack([vecs[t] for t in texts]), tokens

    def healthy(self) -> bool:
        try:
            return self.session.get(self.url.rsplit("/v1/", 1)[0] + "/v1/models", timeout=5).status_code == 200
        except requests.RequestException:
            return False
