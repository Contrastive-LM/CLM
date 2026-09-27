#!/usr/bin/env python3
"""Logistic-regression baseline on frozen encoder state embeddings for a ``choice`` dataset.

For each question id, fits StandardScaler + LogisticRegression on the state embeddings
of every train row (gold option key as the class) and reports test accuracy. It never
sees candidate texts, so it can only predict options that occur in train; on unseen-label
test sets it scores 0 by construction.

Embeddings are read from the cache ``train/finetune.py --task choice`` writes
(``<embed-cache>/choice_<model>_<max-len>.npz``, keyed by sha1 of the state text), so run
finetune.py on the same ``--data`` / ``--workflow`` first; nothing is re-embedded.

  python evaluation/intent_routing/logreg_baseline.py --data data/intent_routing/banking77 \\
      --workflow seen --embed-cache runs/banking77_seen/embeddings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "train"))
sys.path.insert(0, os.path.join(REPO, "src"))
import adapters  # noqa: E402
from finetune import TextCache, _slug, load_typed_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="local dir, parquet file or HF id (as for finetune.py)")
    ap.add_argument("--workflow", default="all")
    ap.add_argument("--embed-cache", required=True, help="finetune.py --embed-cache (default there: OUT/embeddings)")
    ap.add_argument("--embed-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    path = os.path.join(args.embed_cache, f"choice_{_slug(args.embed_model)}_{args.max_len}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"no embedding cache at {path}; run train/finetune.py --task choice first")
    cache = TextCache(path)
    ex = {s: list(adapters.typed_decision_examples(load_typed_rows(args.data, s, args.workflow, args.hf_cache)))
          for s in ("train", "test")}
    missing = cache.missing(e.state_text for v in ex.values() for e in v)
    if missing:
        raise SystemExit(f"{len(missing)} state texts are not in {path}; was it built from the same data?")

    def xy(es):
        return (np.stack([cache[e.state_text] for e in es]).astype(np.float32), [e.keys[e.label] for e in es])

    by = {s: defaultdict(list) for s in ex}
    for s, es in ex.items():
        for e in es:
            by[s][e.qid].append(e)
    hit = n = unseen = 0
    per = {}
    for qid, test in sorted(by["test"].items()):
        train = by["train"].get(qid, [])
        xt, yt = xy(test)
        classes = {e.keys[e.label] for e in train}
        unseen += sum(y not in classes for y in yt)
        if not train:
            pred = [None] * len(yt)
        elif len(classes) < 2:
            pred = [next(iter(classes))] * len(yt)
        else:
            x, y = xy(train)
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=args.max_iter)).fit(x, y)
            pred = model.predict(xt).tolist()
        h = sum(p == g for p, g in zip(pred, yt))
        per[qid] = round(h / len(yt), 4)
        hit += h
        n += len(yt)
    res = {"data": args.data, "workflow": args.workflow, "train_questions": len(ex["train"]),
           "test_questions": n, "test_acc": hit / max(1, n), "test_gold_not_in_train": unseen,
           "per_question": per}
    print(f"[logreg] {args.data} [{args.workflow}] TEST acc {res['test_acc']:.4f} over {n} questions "
          f"({unseen} with a gold label absent from train) per_question {per}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
