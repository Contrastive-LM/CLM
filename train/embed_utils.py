"""Encoder embeddings (Qwen3-8B, last-token pooling) with the token recipe of the
DeepSWE precompute in the research repo's ``main`` branch.

* ``Recipe.state_ids``: chat template, last ``max_len - 1`` tokens; string states are
  tokenized as text, tail kept.
* ``Recipe.text_ids``: text without special tokens; ``keep="head"`` keeps the first
  ``max_len - 1`` tokens, ``keep="tail"`` the last.

Backends take token ids: ``OfflineBackend`` (in-process vLLM) and ``ServerBackend``
(``vllm serve <model> --runner pooling``).
"""
from __future__ import annotations

import base64
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.embedder import EmbedderError, l2  # noqa: E402


class Recipe:
    def __init__(self, model: str = "Qwen/Qwen3-8B", max_len: int = 8192):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.cap = max_len - 1

    @staticmethod
    def _flatten(out) -> list[int]:
        # transformers 4.x returns List[int]; 5.x may return a BatchEncoding / nested list
        if hasattr(out, "keys"):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return list(out)

    def text_ids(self, text: str, keep: str = "head") -> list[int]:
        ids = self._flatten(self.tok(text, add_special_tokens=False)["input_ids"])
        if not ids:
            ids = self._flatten(self.tok(" ", add_special_tokens=False)["input_ids"])
        return ids[:self.cap] if keep == "head" else ids[-self.cap:]

    def state_ids(self, state) -> list[int]:
        if isinstance(state, str):
            return self.text_ids(state, keep="tail")
        ids = self._flatten(self.tok.apply_chat_template(state, tokenize=True, add_generation_prompt=False))
        return ids[-self.cap:]


class OfflineBackend:
    """In-process vLLM pooling model, loaded on first use."""

    def __init__(self, model: str, max_len: int, gpu_mem: float = 0.85):
        self.model, self.max_len, self.gpu_mem = model, max_len, gpu_mem
        self._llm = None

    def embed(self, id_lists: list[list[int]]) -> np.ndarray:
        from vllm.inputs import TokensPrompt
        if self._llm is None:
            from vllm import LLM
            self._llm = LLM(model=self.model, runner="pooling", enforce_eager=False,
                            max_model_len=self.max_len, trust_remote_code=True,
                            gpu_memory_utilization=self.gpu_mem)
        outs = self._llm.embed([TokensPrompt(prompt_token_ids=ids) for ids in id_lists], use_tqdm=True)
        return l2(np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outs]))


class ServerBackend:
    """OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(self, url: str, model: str, batch: int = 64, timeout: float = 600.0,
                 api_key: str | None = None):
        import requests
        self.url, self.model, self.batch, self.timeout = url, model, batch, timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def embed(self, id_lists: list[list[int]]) -> np.ndarray:
        import requests
        out: list[np.ndarray] = []
        for i in range(0, len(id_lists), self.batch):
            chunk = id_lists[i:i + self.batch]
            body = {"model": self.model, "input": chunk, "encoding_format": "base64"}
            try:
                r = self.session.post(self.url, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                raise EmbedderError(f"embedder unreachable at {self.url}: {e}") from e
            if r.status_code != 200:
                raise EmbedderError(f"embedder error {r.status_code}: {r.text[:300]}")
            got: list = [None] * len(chunk)
            for d in r.json()["data"]:
                e = d["embedding"]
                got[d["index"]] = (np.frombuffer(base64.b64decode(e), dtype=np.float32) if isinstance(e, str)
                                   else np.asarray(e, dtype=np.float32))
            out.extend(got)
            if (i // self.batch) % 50 == 0:
                print(f"[embed] server {min(i + self.batch, len(id_lists))}/{len(id_lists)}", flush=True)
        return l2(np.stack(out).astype(np.float32))


def make_backend(url: str | None, model: str, max_len: int, gpu_mem: float = 0.85, served_name: str | None = None):
    """Server backend when ``url`` is given, else offline vLLM."""
    return ServerBackend(url, served_name or model) if url else OfflineBackend(model, max_len, gpu_mem)
