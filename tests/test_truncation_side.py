"""Which end of an over-long text the encoder keeps: a state its tail, a candidate its head.

No GPU and no network: the embedder's HTTP session is replaced by a stub that records
every ``/v1/embeddings`` request body and answers with a vector that depends on the
text and on the truncation side asked for.

    python -m unittest discover tests
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.embedder import Embedder  # noqa: E402
from clm.heads import HIDDEN  # noqa: E402


class FakeResponse:
    status_code = 200

    def __init__(self, payload: dict):
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class FakeEncoder:
    """Stands in for ``requests.Session.post`` on the vLLM pooling server."""

    def __init__(self):
        self.bodies: list[dict] = []

    def __call__(self, url, json=None, timeout=None):
        self.bodies.append(json)
        side = json.get("truncation_side")
        data = [{"index": i, "embedding": base64.b64encode(self.vector(side, t).tobytes()).decode()}
                for i, t in enumerate(json["input"])]
        return FakeResponse({"data": data, "usage": {"prompt_tokens": len(json["input"])}})

    @staticmethod
    def vector(side: str | None, text: str) -> np.ndarray:
        rng = np.random.default_rng(zlib.crc32(f"{side}\x00{text}".encode()))
        return rng.standard_normal(HIDDEN).astype(np.float32)


def make_embedder(max_tokens: int | None = 2048) -> tuple[Embedder, FakeEncoder]:
    emb = Embedder(max_tokens=max_tokens)
    emb.session.post = FakeEncoder()
    return emb, emb.session.post


class EmbedderTest(unittest.TestCase):
    def test_side_is_sent_with_truncation(self):
        emb, enc = make_embedder()
        emb.embed(["a"], truncation_side="left")
        emb.embed(["b"], truncation_side="right")
        emb.embed(["c"])                                    # no side: the request is unchanged
        self.assertEqual([b.get("truncation_side") for b in enc.bodies], ["left", "right", None])
        self.assertEqual([b["truncate_prompt_tokens"] for b in enc.bodies], [2048] * 3)

    def test_no_side_without_truncation(self):
        emb, enc = make_embedder(max_tokens=None)
        emb.embed(["a"], truncation_side="left")
        self.assertNotIn("truncate_prompt_tokens", enc.bodies[0])
        self.assertNotIn("truncation_side", enc.bodies[0])

    def test_cache_keeps_sides_apart(self):
        emb, enc = make_embedder()
        tail, _ = emb.embed(["s"], truncation_side="left")
        head, _ = emb.embed(["s"], truncation_side="right")
        self.assertEqual(len(enc.bodies), 2)                # the other side is not a cache hit
        self.assertFalse(np.allclose(tail, head))
        tail2, spent_tail = emb.embed(["s"], truncation_side="left")
        head2, spent_head = emb.embed(["s"], truncation_side="right")
        self.assertEqual(len(enc.bodies), 2)                # both now come from the cache
        np.testing.assert_array_equal(tail2, tail)
        np.testing.assert_array_equal(head2, head)
        self.assertEqual(spent_tail + spent_head, 0)


class EngineTest(unittest.TestCase):
    """Engine sends states with "left" and candidates with "right", through the vector arena."""

    @classmethod
    def setUpClass(cls):
        import torch
        from clm.heads import make_head
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ckpt = os.path.join(cls.tmp.name, "tiny.pt")
        torch.save({"state_head": make_head(8, 2, 4).state_dict(), "action_head": make_head(8, 2, 4).state_dict(),
                    "logit_scale": torch.tensor(0.0), "cfg": {"width": 8, "depth": 2, "projection_dim": 4}},
                   cls.ckpt)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_states_keep_tail_candidates_keep_head(self):
        from clm.engine import Engine
        for model in ("clm-latest", "clm-raw"):
            with self.subTest(model=model):
                emb, enc = make_embedder()
                engine = Engine(emb, checkpoint=self.ckpt, device="cpu", action_cache="4MiB")
                self.assertIsNotNone(engine.arena)
                # The question is also an option, so one text is embedded as a state and as a candidate.
                engine.rank("", ["Is it raining?", "No."], instructions="Is it raining?", model=model)
                sent = {(b["truncation_side"], t) for b in enc.bodies for t in b["input"]}
                self.assertEqual(sent, {("left", "Is it raining?"), ("right", "Is it raining?"), ("right", "No.")})


if __name__ == "__main__":
    unittest.main()
