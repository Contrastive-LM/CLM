"""Offline tests for ``clm.finetune``: a hashing tokenizer and a local embedding server
stand in for Qwen3-8B, so no GPU, model download or network is needed.

    pip install pytest && pytest tests
"""
import json
import os
import re
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import clm  # noqa: E402
from clm import adapters, embed_utils, trainer  # noqa: E402

DIM = 512
ANIMALS = ["otter", "heron", "lynx", "bison", "gecko", "ibis", "koala", "lemur", "moose", "newt", "okapi",
           "panda", "quail", "raven", "skunk", "tapir"]
CODES = {a: f"token {a[::-1]} {i * 7 % 97}" for i, a in enumerate(ANIMALS)}
#: small heads trained from scratch: enough to learn the toy task in a few seconds on a CPU
FAST = dict(model="fake/enc", init=None, width=256, depth=3, proj=64, batch_size=32, epochs=15, seed=0,
            verbose=False)


def token_ids(text: str) -> list[int]:
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    return [zlib.crc32(f.encode()) for f in words + [f"{a} {b}" for a, b in zip(words, words[1:])]] or [0]


class FakeRecipe:
    def __init__(self, model="fake/enc", max_len=8192):
        self.cap = max_len - 1

    def text_ids(self, text, keep="head"):
        ids = token_ids(text)
        return ids[:self.cap] if keep == "head" else ids[-self.cap:]

    def state_ids(self, state):
        return self.text_ids(state if isinstance(state, str) else " ".join(m["content"] for m in state), "tail")


class Handler(BaseHTTPRequestHandler):
    requests = 0

    def log_message(self, *args):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"data": [{"id": "fake-enc", "root": "fake/enc", "max_model_len": 4096}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] == "fake-enc", body["model"]
        Handler.requests += len(body["input"])
        out = []
        for i, item in enumerate(body["input"]):
            v = np.zeros(DIM, dtype=np.float32)
            for t in (token_ids(item) if isinstance(item, str) else item):      # signed feature hashing
                v[t % DIM] += 1.0 if (t >> 16) & 1 else -1.0
            out.append({"index": i, "embedding": (v / (np.linalg.norm(v) + 1e-12)).tolist()})
        self._send({"data": out, "usage": {"prompt_tokens": 0}})


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(embed_utils, "Recipe", FakeRecipe)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CLM_DEVICE", "cpu")
    monkeypatch.setenv("CLM_ACTION_CACHE", "0")
    monkeypatch.delenv("CLM_EMB_URL", raising=False)


def pairs(templates):
    return [{"input": t.format(a), "answer": CODES[a]} for a in ANIMALS for t in templates]


def choices(templates):
    out = []
    for i, a in enumerate(ANIMALS):
        wrong = [CODES[ANIMALS[(i + k) % len(ANIMALS)]] for k in (3, 7)]
        for t in templates:
            out.append({"state": t.format(a), "candidates": [wrong[0], CODES[a], wrong[1]], "rewards": [0, 1, 0]})
    return out


TRAIN = ["What is the code for the {}?", "Give me the {} code.", "Code of {}, please.", "{}: which code?"]
VAL = ["Tell me the code that belongs to the {}."]


# ------------------------------------------------------------------ adapters
def test_step_transitions_pass_through_unchanged():
    rec = {"state": [{"role": "user", "content": "fix it"}], "action": "pytest -q", "task_id": "t", "step_idx": 3}
    assert adapters.as_transitions([rec])[0] is rec
    assert adapters.task_of([rec]) == "clm"


def test_plain_pairs_become_one_task_each_and_share_a_group_per_answer():
    a, b, c = adapters.as_transitions([{"input": "q1", "answer": "x"}, {"question": "q2", "output": "x"},
                                       {"prompt": "q3", "response": "y"}])
    assert (a["state"], a["action"], a["step_idx"]) == ("q1", "x", 0)
    assert len({a["task_id"], b["task_id"], c["task_id"]}) == 3
    assert a["group"] == b["group"] != c["group"]


def test_choice_records_take_label_rewards_target_or_negatives():
    ex = list(adapters.choice_examples([
        {"state": "s", "candidates": ["a", "b", "c"], "rewards": [0, 2, 2]},
        {"state": "s", "candidates": ["a", "b"], "label": "b"},
        {"state": "s", "candidates": {"x": "first", "y": ""}, "target": {"x": 0.25, "y": 0.75}},
        {"question": "capital?", "answer": "Paris", "negatives": ["Lyon"], "instructions": "Pick one."},
        {"state": "s", "candidates": ["a", "b"], "rewards": [0, 0]},      # no correct candidate: skipped
    ]))
    assert [e.label for e in ex] == [1, 1, 1, 0]
    assert ex[0].target == [0.0, 0.5, 0.5]
    assert (ex[2].keys, ex[2].candidates) == (["x", "y"], ["first", "y"])
    assert (ex[3].state_text, ex[3].candidates) == ("capital?\n\nPick one.", ["Paris", "Lyon"])
    assert adapters.task_of([{"state": "s", "candidates": ["a", "b"], "label": 0}]) == "choice"
    with pytest.raises(ValueError, match="gold answer"):
        list(adapters.choice_examples([{"state": "s", "candidates": ["a", "b"]}]))
    with pytest.raises(ValueError, match="mixes"):
        adapters.task_of([{"input": "q", "answer": "a"}, {"state": "s", "candidates": ["a", "b"], "label": 0}])


# ------------------------------------------------------------------ finetune
def test_candidates_with_outcomes_train_the_choice_task(server, tmp_path):
    r = clm.finetune(choices(TRAIN), choices(VAL), base_url=server, out_dir=str(tmp_path), **FAST)
    assert (r.task, r.metric) == ("choice", "val_acc")
    assert r.best_score >= 0.8 and r.baseline_score <= 0.4 and r.improvement > 0.3   # chance is 1/3
    assert os.path.exists(r.checkpoint) and r.history[0]["epoch"] == 0
    ranked = r.engine().rank("Which code goes with the lynx?", [CODES["bison"], CODES["lynx"], CODES["newt"]])
    assert ranked[0]["candidate"] == CODES["lynx"]


def test_plain_pairs_train_the_clm_task(server, tmp_path):
    r = clm.finetune(pairs(TRAIN), pairs(VAL), base_url=server, out_dir=str(tmp_path), **FAST)
    assert (r.task, r.metric) == ("clm", "val_loss")           # one step per task: nothing to rank within
    assert r.best_score < r.baseline_score and r.improvement > 0
    assert r.history[r.best_epoch]["val_top1"] >= 0.9
    ranked = r.engine().rank("What code does the okapi have?", list(CODES.values()))
    assert ranked[0]["candidate"] == CODES["okapi"]


def test_step_traces_from_a_file_split_by_task(server, tmp_path):
    path = tmp_path / "steps.jsonl"
    with open(path, "w") as f:
        for a in ANIMALS:
            for s, t in enumerate(TRAIN):
                f.write(json.dumps({"state": t.format(a), "action": f"{CODES[a]} step {s}", "task_id": a,
                                    "step_idx": s, "trajectory_id": a, "reward": 1}) + "\n")
    r = clm.finetune(str(path), base_url=server, out_dir=str(tmp_path / "run"), val_frac=0.25, **FAST)
    assert (r.task, r.metric) == ("clm", "within_task_top1")
    split = json.load(open(tmp_path / "run" / "task_split.json"))
    assert len(split["val_tasks"]) == 4 and not set(split["val_tasks"]) & set(split["train_tasks"])


def test_the_command_line_trains_the_same_head(server, tmp_path):
    api = clm.finetune(choices(TRAIN), choices(VAL), base_url=server, out_dir=str(tmp_path / "api"), **FAST)
    trainer.main(["--task", "choice", "--data", str(tmp_path / "api" / "data" / "train.jsonl"),
                  "--val-data", str(tmp_path / "api" / "data" / "val.jsonl"), "--embed-model", "fake/enc",
                  "--embed-url", server, "--width", "256", "--depth", "3", "--proj", "64", "--batch", "32",
                  "--epochs", "15", "--seed", "0", "--out-dir", str(tmp_path / "cli")])
    a, b = (torch.load(p, map_location="cpu") for p in (api.checkpoint, tmp_path / "cli" / "best_head.pt"))
    assert all(torch.equal(a[h][k], b[h][k]) for h in ("state_head", "action_head") for k in a[h])


def test_embeddings_are_reused_only_for_the_same_file(server, tmp_path):
    kw = dict(base_url=server, out_dir=str(tmp_path), **{**FAST, "epochs": 2})
    clm.finetune(pairs(TRAIN), **kw)
    before = Handler.requests
    clm.finetune(pairs(TRAIN), **kw)                           # same records: nothing to embed
    assert Handler.requests == before
    clm.finetune(pairs(TRAIN[:2] + VAL), **kw)                 # same path, new records: embedded again
    assert Handler.requests > before


def test_bad_calls_raise_instead_of_exiting(server, tmp_path):
    kw = dict(base_url=server, out_dir=str(tmp_path), **FAST)
    with pytest.raises(TypeError, match="bogus"):
        clm.finetune(pairs(TRAIN), bogus=1, **kw)
    with pytest.raises(ValueError, match="exceeds the embedding server's limit"):
        clm.finetune(pairs(TRAIN), max_len=100_000, **kw)
    with pytest.raises(ValueError, match="holds 'choice' records"):
        clm.finetune(choices(TRAIN), task="clm", **kw)
    with pytest.raises(ValueError, match="needs a state"):
        clm.finetune([{"text": "no state or action here"}], **kw)
    with pytest.raises(FileNotFoundError):
        clm.finetune(str(tmp_path / "missing.jsonl"), **kw)
