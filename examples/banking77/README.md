# BANKING77: fine-tuning CLM as an intent classifier

`clm.finetune` on a closed-set classification task: which of 77 banking intents is a customer
message about? The encoder stays frozen and the two projection heads train for under a minute.

**Task.** [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) (CC-BY-4.0) holds
10,003 training and 3,080 test messages sent to a bank's support chat, each labelled with one of
77 intents (40 test messages per intent, so chance is 1.3%). No message appears in both splits.

**Records.** One `choice` record per message. The message is the state, the 77 intent names
(`card_arrival` written as `card arrival`) are the candidates, and the labelled intent is the gold
answer:

```json
{"state": "I am still waiting on my card?",
 "candidates": ["activate my card", "age limit", "...", "wrong exchange rate for cash withdrawal"],
 "label": "card arrival"}
```

[`finetune.py`](finetune.py) builds the records, calls
`clm.finetune(train, lr=5e-3, out_dir="runs/banking77")`, and scores the test split with
`Engine.rank`: the top-ranked candidate is the prediction. `clm.finetune` holds out 1,000 of the
training messages to pick the best epoch, so the test split plays no part in training.

**Results.** Accuracy, reference head (`CLM_v0.1-8B.pt`, zero-shot) against the head trained here:

| | Reference head | Fine-tuned |
| --- | --- | --- |
| Validation (1,000 held-out training messages) | 13.1% | 88.8% |
| Test (3,080 messages) | 13.4% | **88.0%** |

Run on 2026-11-27 with the Qwen3-8B encoder from the quickstart (`--max-model-len 2048`) on one
H100 and the heads on another. The whole script took 67 seconds: 26 for `clm.finetune` (embedding
10,076 texts, then 18 epochs before early stopping; epoch 13 was best) and the rest to rank the
test split twice. [`results/finetune.json`](results/finetune.json) holds the report. Numbers vary
slightly between runs, because the encoder's embeddings are not bit-identical from one pass to
the next.

**Settings.** Chosen on the validation split, not on the test split:

- Learning rate. The trainer's default for `choice` (5e-4) reaches 83.3% validation accuracy and
  `lr=5e-3` reaches 88.8%, so the script sets it.
- Warm start. The same settings from a fresh head (`init=None`) reach 72.3% in 20 epochs.
- No `instructions`. Adding the question "Which banking intent is this customer message about?" to
  every record changed little after fine-tuning (89.7% against 88.8%) and lowered the reference
  head's zero-shot accuracy (5.0% against 13.1%), so the records carry the message alone.

**Reproduce.**

```bash
pip install -r requirements.txt
# from the repo root, with the encoder running (see Quickstart in the main README)
python examples/banking77/finetune.py
```

`--out-dir` (default `runs/banking77`) receives the checkpoints, the metric history and the cached
embeddings, so a second run skips the embedding pass. `--base-url` points at another encoder
server, and `--epochs` changes the schedule.

**Use the head.** `clm-serve --ckpt runs/banking77/best_head.pt` serves it over HTTP. In Python:

```python
import clm

engine = clm.Engine(checkpoint="runs/banking77/best_head.pt")
engine.rank("why was I charged twice for my coffee this morning", intents)[:3]
# -> transaction charged twice (0.972), extra charge on statement (0.010), lost or stolen card (0.003)
engine.rank("The cash machine kept my card and I'm travelling tomorrow.", intents)[:3]
# -> lost or stolen card (0.553), card swallowed (0.212), lost or stolen phone (0.177)
```

`intents` is the list of 77 candidate strings the head was trained with.
