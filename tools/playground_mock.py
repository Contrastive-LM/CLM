#!/usr/bin/env python3
"""Run the playground without a GPU — for working on the UI, not for measuring anything.

This starts the real ``clm.server`` app (so the routes, the static mount and the
answer schema are exactly what ``clm-serve`` exposes) against a **fake encoder**:
character n-gram feature hashing instead of Qwen3-8B, and no projection head.
The numbers it returns are lexical-overlap noise, not CLM predictions.  The
server reports ``{"mock": true}`` on ``/health`` and the page shows a warning
banner so nobody mistakes a screenshot of this for a result.

    pip install fastapi uvicorn numpy
    python tools/playground_mock.py --port 8700        # then open http://localhost:8700/
    python tools/playground_mock.py --broken           # pretend the encoder is down (502s)

For real answers, serve the encoder and run ``clm-serve`` — see the README.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from clm.embedder import EmbedderError              # noqa: E402
from clm.schema import answer_from_logits, build_pairs  # noqa: E402

DIM = 512
SCALE = 28.0
RELEASE = "2026-09-19"


def _embed(text: str) -> list[float]:
    """Hashed character 3/4-grams, L2 normalised — deterministic across runs."""
    v = [0.0] * DIM
    t = " " + " ".join(text.lower().split()) + " "
    for n in (3, 4):
        for i in range(len(t) - n + 1):
            h = int.from_bytes(hashlib.blake2b(t[i:i + n].encode(), digest_size=4).digest(), "big")
            v[h % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class MockEmbedder:
    def __init__(self, broken: bool = False):
        self.broken = broken
        self.url = "mock://n-gram"
        self.tokens = 0

    def embed(self, texts: list[str]):
        if self.broken:
            raise EmbedderError("embedder unreachable at http://127.0.0.1:8090/v1/embeddings: "
                                "[Errno 61] Connection refused (this is the mock's --broken mode)")
        self.tokens += sum(max(1, len(t) // 4) for t in texts)
        return [_embed(t) for t in texts], sum(max(1, len(t) // 4) for t in texts)

    def healthy(self) -> bool:
        return not self.broken


class MockEngine:
    """Same surface as ``clm.Engine`` for the three things the server calls."""

    mock = True
    arena = None          # no vector cache; /health reports it as off

    def __init__(self, broken: bool = False):
        self.embedder = MockEmbedder(broken)
        self.heads = {"clm-latest": None}

    def models(self) -> list[dict[str, str]]:
        return [
            {"name": "clm-latest", "release_date": RELEASE,
             "description": "MOCK — character n-grams, not a contrastive language model"},
            {"name": "clm-raw", "release_date": RELEASE,
             "description": "MOCK — the same n-grams without the (absent) projection head"},
        ]

    def answer(self, state, questions, model="clm-latest", temperature=1.0) -> dict:
        if not questions:
            raise ValueError("questions must not be empty")
        if not (0 < temperature <= 100):
            raise ValueError("temperature must be in (0, 100]")
        if model not in ("clm-latest", "clm-raw"):
            from clm.engine import ModelNotFound
            raise ModelNotFound(f"unknown model {model!r}; available: ['clm-latest', 'clm-raw']")

        pairs = build_pairs(state, questions)                      # the real schema
        scale = SCALE if model == "clm-latest" else SCALE * 0.45   # flatter, like the raw-space ablation
        answers, tokens = {}, 0
        for qid, (s_text, keys, texts) in pairs.items():
            (s_vec,), tk = self.embedder.embed([s_text])
            tokens += tk
            cand, tk = self.embedder.embed(texts)
            tokens += tk
            logits = [scale * sum(a * b for a, b in zip(s_vec, c)) / temperature for c in cand]
            answers[qid] = answer_from_logits(questions[qid], keys, logits)
        return {"model": model, "answers": answers,
                "usage": {"billing_units": len(questions), "input_tokens": tokens, "output_tokens": 0}}


    def rank(self, state, candidates, instructions=None, model="clm-latest", temperature=1.0):
        """Mirrors ``Engine.rank``: a choice question over the candidates, best first."""
        q = {"type": "choice", "instructions": instructions,
             "criteria": {str(i): c for i, c in enumerate(candidates)}}
        a = self.answer(state, {"rank": q}, model, temperature)["answers"]["rank"]
        order = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])
        return [{"rank": r + 1, "candidate": candidates[int(i)], "prob": p} for r, (i, p) in enumerate(order)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--broken", action="store_true", help="simulate an unreachable encoder (502s)")
    ap.add_argument("--cors", action="store_true")
    args = ap.parse_args()

    from clm.server import create_app
    app = create_app(MockEngine(args.broken), os.environ.get("CLM_API_KEY"), cors=args.cors)
    print(f"[mock] FAKE ENCODER — numbers are meaningless; playground at http://{args.host}:{args.port}/",
          flush=True)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
