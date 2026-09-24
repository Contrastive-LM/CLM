"""The inference engine: state + typed questions -> distributions, no HTTP needed.

    from clm import Engine
    engine = Engine()                                      # embedder at :8090, reference head
    out = engine.answer(state, {"ok": {"type": "noul", "instructions": "Is this fine?"}})
    out["answers"]["ok"]["noul"]

For each question the state (with the question's instructions appended) goes
through the state head and every option's description through the action head;
the softmax over ``scale * cosine`` is the answer.  ``clm-raw`` skips the heads
(cosine in the encoder's own space) as an ablation.
"""
from __future__ import annotations

import glob
import os
from typing import Any

from .cache import CacheDisabled, VectorArena
from .embedder import Embedder
from .heads import HIDDEN, HeadPair, default_checkpoint, default_device
from .schema import answer_from_logits, build_pairs

DEFAULT_MODEL = "clm-latest"
RAW_MODEL = "clm-raw"
RAW_SCALE = 100.0
RAW_SHARE = 0.125      # of the arena, for the raw ablation's wider vectors
RELEASE = "2026-09-19"


class ModelNotFound(KeyError):
    pass


class Engine:
    def __init__(self, embedder: Embedder | None = None, checkpoint: str | None = None,
                 models: dict[str, str] | None = None, checkpoint_dir: str | None = None, device: str | None = None,
                 emb_url: str | None = None, emb_model: str | None = None, action_cache: Any = None):
        self.embedder = embedder or Embedder(url=emb_url or os.environ.get("CLM_EMB_URL", "http://127.0.0.1:8090/v1/embeddings"),
                                             model=emb_model or os.environ.get("CLM_EMB_MODEL", "qwen3-8b"))
        device = device or default_device()
        self.heads: dict[str, HeadPair] = {}
        ck = checkpoint or default_checkpoint()
        if ck:
            self.heads[DEFAULT_MODEL] = HeadPair(DEFAULT_MODEL, ck, device)
        for p in sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt"))) if checkpoint_dir else []:
            if not ck or os.path.abspath(p) != os.path.abspath(ck):
                self.heads.setdefault(os.path.splitext(os.path.basename(p))[0], HeadPair(os.path.basename(p)[:-3], p, device))
        for name, path in (models or {}).items():
            self.heads[name] = HeadPair(name, path, device)
        for h in self.heads.values():
            h.ensure()
        self.device = device
        self.arena = self._reserve(action_cache)

    def _reserve(self, budget: Any) -> VectorArena | None:
        """Claim the arena up front, so its cost is paid at start-up or not at all.

        Projections are 512-d and encoder embeddings 4096-d, so the raw ablation's
        pool gets an eighth of the arena and still holds a useful number of rows.
        """
        if budget is None:
            budget = os.environ.get("CLM_ACTION_CACHE")
        if budget is None and self.device.startswith("mps"):
            budget = 0
        try:
            arena = VectorArena(self.device, budget)
        except CacheDisabled:
            return None
        for dim in sorted({h.proj_dim for h in self.heads.values()}):
            arena.reserve(dim, RAW_SHARE if dim == HIDDEN else (1.0 - RAW_SHARE))
        arena.reserve(HIDDEN, RAW_SHARE)       # clm-raw works in the encoder's own space
        return arena

    # ------------------------------------------------------------------ cached vectors
    def _cached(self, namespace: str, dim: int, texts: list[str], tokens: list[int], project=None):
        """Vectors for ``texts``, from the arena where possible; ``tokens`` collects misses."""
        def compute(missing: list[str]):
            emb, spent = self.embedder.embed(missing)
            tokens.append(spent)
            return project(emb) if project else self._to_device(emb)

        if self.arena is None:
            return compute(texts)
        return self.arena.get(namespace, dim, texts, compute)

    def _to_device(self, x):
        import torch
        return torch.from_numpy(x).to(self.device)

    # ------------------------------------------------------------------ models
    def models(self) -> list[dict[str, str]]:
        desc = {DEFAULT_MODEL: "Contrastive language model: Qwen3-8B encoder + trained projection heads",
                RAW_MODEL: "Ablation: cosine in the raw encoder embedding space, no projection head"}
        names = ([DEFAULT_MODEL] if DEFAULT_MODEL in self.heads else []) + \
            sorted(n for n in self.heads if n != DEFAULT_MODEL) + [RAW_MODEL]
        return [{"name": n, "release_date": RELEASE,
                 "description": desc.get(n) or f"Projection-head checkpoint {os.path.basename(self.heads[n].path)}"}
                for n in names]

    def has(self, model: str) -> bool:
        return model == RAW_MODEL or model in self.heads

    # ------------------------------------------------------------------ inference
    def answer(self, state: Any, questions: dict[str, dict], model: str = DEFAULT_MODEL,
               temperature: float = 1.0) -> dict:
        """-> {"model", "answers": {id: Answer}, "usage"}; raises ValueError on bad questions."""
        if not questions:
            raise ValueError("questions must not be empty")
        if not (0 < temperature <= 100):
            raise ValueError("temperature must be in (0, 100]")
        if model == RAW_MODEL:
            head = None
        elif model in self.heads:
            head = self.heads[model].ensure()
        else:
            raise ModelNotFound(f"unknown model {model!r}; available: {[m['name'] for m in self.models()]}")
        pairs = build_pairs(state, questions)          # ValueError on malformed questions
        states = [p[0] for p in pairs.values()]
        cands = [t for p in pairs.values() for t in p[2]]
        tokens: list[int] = []
        if head is None:
            # The raw ablation reuses the encoder's own vectors, in the encoder's own space.
            zq = self._cached("raw/state", HIDDEN, states, tokens)
            za = self._cached("raw/action", HIDDEN, cands, tokens)
            scale = RAW_SCALE
        else:
            ns, dim = head.namespace, head.proj_dim
            zq = self._cached(f"{ns}/state", dim, states, tokens, head.project_states)
            za = self._cached(f"{ns}/action", dim, cands, tokens, head.project_actions)
            scale = head.scale
        answers, k = {}, 0
        for i, (qid, (_, keys, texts)) in enumerate(pairs.items()):
            cos = za[k:k + len(texts)] @ zq[i]
            k += len(texts)
            answers[qid] = answer_from_logits(questions[qid], keys, (scale * cos / temperature).tolist())
        return {"model": model, "answers": answers,
                "usage": {"billing_units": len(questions), "input_tokens": sum(tokens), "output_tokens": 0}}

    def rank(self, state: Any, candidates: list[str], instructions: str | None = None,
             model: str = DEFAULT_MODEL, temperature: float = 1.0) -> list[dict]:
        """Rank free-form candidate strings against a state (best first).

        ``state`` is the context and ``instructions`` the question; the state head sees
        ``context + question`` and the action head sees each candidate verbatim.
        """
        q = {"type": "choice", "instructions": instructions,
             "criteria": {str(i): c for i, c in enumerate(candidates)}}
        a = self.answer(state, {"rank": q}, model, temperature)["answers"]["rank"]
        order = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])
        return [{"rank": r + 1, "candidate": candidates[int(i)], "prob": p} for r, (i, p) in enumerate(order)]
