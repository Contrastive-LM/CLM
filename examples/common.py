"""One caller for both System One endpoints, used by the T-Rex example.

CLM's ``POST /v1/systemone`` follows the TypeSafe wire schema, so the same
``clm.CLMClient`` talks to a local ``clm-serve`` and to ``api.typesafe.ai``:

    judge = Judge("clm")   # CLM_BASE_URL (default http://127.0.0.1:8700), model clm-latest
    judge = Judge("jev")   # TYPESAFE_API_KEY (or JEV_API_KEY, or <repo>/.env), model jev-latest
    reply = judge.ask(state, {"pick": {"type": "choice", "instructions": ..., "criteria": {...}}})
    reply.answers["pick"]["choice"], reply.latency_ms

What the wrapper adds on top of the client: retries with exponential back-off
(429 / 529 / 5xx / transport errors), per-request latency and token accounting
(``judge.stats()``), an optional sqlite answer cache so an interrupted run resumes
for free (cached replies never count toward latency), and ``choose`` for a Choice
over more options than the endpoint accepts (Jev allows 255 per question: bigger
sets run as a chunk tournament, CLM has no limit).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import statistics
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import requests

from clm import CLMClient, CLMError

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RETRY_STATUS = {429, 500, 502, 503, 504, 529}

ENDPOINTS = {
    "clm": {"base_url_env": "CLM_BASE_URL", "base_url": "http://127.0.0.1:8700",
            "model_env": "CLM_MODEL", "model": "clm-latest", "key_envs": ("CLM_API_KEY",), "max_options": None},
    "jev": {"base_url_env": "TYPESAFE_BASE_URL", "base_url": "https://api.typesafe.ai",
            "model_env": "TYPESAFE_MODEL", "model": "jev-latest", "key_envs": ("TYPESAFE_API_KEY", "JEV_API_KEY"),
            "max_options": 255},
}


def load_env(path: str = os.path.join(REPO, ".env")) -> None:
    """Export ``KEY=VALUE`` lines of ``<repo>/.env`` into the environment (existing values win)."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _key(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def percentile(values: Sequence[float], q: float) -> float:
    """``q``-th percentile of a non-empty sequence (nearest rank)."""
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


@dataclass
class Reply:
    answers: dict[str, dict]
    latency_ms: float          # client-side wall-clock of the successful request (0 when cached)
    input_tokens: int
    output_tokens: int
    model: str
    cached: bool = False
    retries: int = 0


@dataclass
class Stats:
    requests: int = 0
    cached: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        lat = self.latencies_ms
        return {"requests": self.requests, "cached": self.cached, "retries": self.retries,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
                "latency_ms_p50": round(percentile(lat, 0.5), 1) if lat else None,
                "latency_ms_p95": round(percentile(lat, 0.95), 1) if lat else None}


class Judge:
    """``ask(state, questions)`` against one endpoint, with retries, timing and a cache."""

    def __init__(self, name: str, cache: str | None = None, timeout: float = 120.0, max_retries: int = 6,
                 model: str | None = None):
        if name not in ENDPOINTS:
            raise ValueError(f"model must be one of {list(ENDPOINTS)}")
        load_env()
        spec = ENDPOINTS[name]
        self.name = name
        key = next((os.environ[k] for k in spec["key_envs"] if os.environ.get(k)), None)
        if name == "jev" and not key:
            raise RuntimeError("set TYPESAFE_API_KEY (or JEV_API_KEY), e.g. in <repo>/.env")
        self.model: str = model or os.environ.get(spec["model_env"]) or spec["model"]
        self.base_url = os.environ.get(spec["base_url_env"], spec["base_url"])
        self.client = CLMClient(base_url=self.base_url, api_key=key, timeout=timeout, model=self.model)
        self.max_options = spec["max_options"]
        self.max_retries = max_retries
        self.stats = Stats()
        self._lock = threading.Lock()
        self.db = None
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            self.db = sqlite3.connect(cache, check_same_thread=False)
            self.db.execute("CREATE TABLE IF NOT EXISTS ans (k TEXT PRIMARY KEY, v TEXT)")

    # ------------------------------------------------------------------ one request
    def _post(self, state: Any, questions: dict[str, dict]) -> dict:
        """Raw response body (answers as plain dicts), retried on rate limits and transport errors."""
        body = {"state": state, "model": self.model, "questions": questions}
        url = f"{self.client.base_url}/v1/systemone"
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client._s.post(url, json=body, timeout=self.client.timeout)
                if r.status_code == 200:
                    return r.json() | {"_retries": attempt}
                if r.status_code not in RETRY_STATUS:
                    raise CLMError(r.status_code, r.text[:300])
                last = CLMError(r.status_code, r.text[:300])
                wait = float(r.headers.get("Retry-After") or 0) or min(30.0, 0.5 * 2 ** attempt)
            except requests.RequestException as e:
                last = e
                wait = min(30.0, 0.5 * 2 ** attempt)
            time.sleep(wait + random.uniform(0, 0.25))
        raise RuntimeError(f"{self.name}: request failed after {self.max_retries} retries: {last}")

    def ask(self, state: Any, questions: dict[str, dict]) -> Reply:
        k = _key(self.name, self.model, state, questions)
        if self.db is not None:
            with self._lock:
                row = self.db.execute("SELECT v FROM ans WHERE k=?", (k,)).fetchone()
            if row:
                j = json.loads(row[0])
                with self._lock:
                    self.stats.cached += 1
                return Reply(j["answers"], 0.0, 0, 0, j.get("model", self.model), cached=True)
        t0 = time.perf_counter()
        j = self._post(state, questions)
        ms = (time.perf_counter() - t0) * 1000
        usage = j.get("usage") or {}
        reply = Reply(j["answers"], ms, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0),
                      j.get("model", self.model), retries=int(j.get("_retries", 0)))
        with self._lock:
            s = self.stats
            s.requests += 1; s.retries += reply.retries
            s.input_tokens += reply.input_tokens; s.output_tokens += reply.output_tokens
            s.latencies_ms.append(ms)
            if self.db is not None:
                self.db.execute("INSERT OR REPLACE INTO ans VALUES (?,?)",
                                (k, json.dumps({"answers": reply.answers, "model": reply.model}, ensure_ascii=False)))
                self.db.commit()
        return reply

    # ------------------------------------------------------------------ Choice over many options
    def choose(self, state: Any, options: Mapping[str, Any], instructions: Any) -> tuple[dict[str, float], list[Reply]]:
        """Probability per option key.  Above the endpoint's option limit the set is split into
        chunks, each chunk is asked once, and a final Choice over the top of every chunk decides;
        options that miss the final keep their chunk probability scaled by the chunk's share."""
        keys = list(options)
        cap = self.max_options or len(keys)
        if len(keys) <= cap:
            r = self.ask(state, {"pick": {"type": "choice", "instructions": instructions, "criteria": dict(options)}})
            return {k: float(v) for k, v in r.answers["pick"]["probabilities"].items()}, [r]
        chunks = [keys[i:i + cap] for i in range(0, len(keys), cap)]
        replies, chunk_probs = [], []
        for ck in chunks:
            p, rs = self.choose(state, {k: options[k] for k in ck}, instructions)
            chunk_probs.append(p); replies += rs
        per_chunk = max(2, cap // len(chunks))
        finalists = [k for p in chunk_probs for k, _ in sorted(p.items(), key=lambda kv: -kv[1])[:per_chunk]]
        final, rs = self.choose(state, {k: options[k] for k in finalists}, instructions)
        replies += rs
        probs: dict[str, float] = {}
        for ck, cp in zip(chunks, chunk_probs):
            share = sum(final.get(k, 0.0) for k in ck)
            mass = sum(cp.get(k, 0.0) for k in ck if k in final) or 1e-9
            for k in ck:
                probs[k] = final[k] if k in final else cp.get(k, 0.0) * share / mass * 0.5
        return probs, replies

    def close(self) -> None:
        if self.db is not None:
            self.db.close()


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print(f"wrote {path}")
