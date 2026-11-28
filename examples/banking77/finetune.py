#!/usr/bin/env python3
"""Fine-tune CLM on BANKING77: which of 77 intents is this customer message about?

    python examples/banking77/finetune.py          # with the encoder from the quickstart running

BANKING77 (``PolyAI/banking77``, CC-BY-4.0) holds 10,003 training and 3,080 test messages
sent to a bank's support chat, each labelled with one of 77 intents. A message becomes one
``choice`` record: the message is the state, the 77 intent names are the candidates and the
labelled intent is the gold answer.

    {"state": "I am still waiting on my card?",
     "candidates": ["activate my card", "age limit", ..., "wrong exchange rate for cash withdrawal"],
     "label": "card arrival"}

``clm.finetune`` holds out 10% of the training messages to pick the best epoch. The test
split is scored afterwards with ``Engine.rank``, by the reference head (zero-shot) and by
the head trained here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))
import clm  # noqa: E402

DATASET = "PolyAI/banking77"
LR = 5e-3       # 10x the trainer's default for choice: see the README for what it buys on the validation split


def load(split: str) -> list[dict]:
    """The ``train`` or ``test`` split as ``choice`` records."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    table = pq.read_table(hf_hub_download(DATASET, f"data/{split}-00000-of-00001.parquet", repo_type="dataset"))
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    intents = [n.replace("_", " ").strip(" ?").lower() for n in names]      # "card_arrival" -> "card arrival"
    return [{"state": row["text"].strip(), "candidates": intents, "label": intents[row["label"]]}
            for row in table.to_pylist()]


def accuracy(engine, records: list[dict]) -> float:
    """Share of records whose top-ranked candidate is the labelled intent."""
    hits = 0
    for r in records:
        hits += engine.rank(r["state"], r["candidates"])[0]["candidate"] == r["label"]
    return hits / len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="runs/banking77", help="checkpoints, history and cached embeddings")
    ap.add_argument("--base-url", default="auto",
                    help="encoder server; default: CLM_EMB_URL, else the one on :8090, else in-process vLLM")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--results", default=os.path.join(HERE, "results", "finetune.json"))
    args = ap.parse_args()

    train, test = load("train"), load("test")
    result = clm.finetune(train, out_dir=args.out_dir, base_url=args.base_url, epochs=args.epochs, lr=LR)

    tuned = result.engine()
    reference = clm.Engine(embedder=tuned.embedder)      # the released head on the same encoder
    report = {
        "dataset": DATASET, "train_messages": len(train), "test_messages": len(test), "intents": 77,
        "validation": {"metric": result.metric, "reference_head": round(result.baseline_score, 4),
                       "fine_tuned": round(result.best_score, 4), "best_epoch": result.best_epoch},
        "test_accuracy": {"reference_head": round(accuracy(reference, test), 4),
                          "fine_tuned": round(accuracy(tuned, test), 4)},
        "checkpoint": result.checkpoint,
    }
    print(json.dumps(report, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
