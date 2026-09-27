# Intent routing benchmark

A reproducible benchmark for routing customer messages to intents with CLM
v0.1-8B, zero-shot and with fine-tuned heads, using only the existing
`train/finetune.py --task choice` pipeline.

---

## Motivation

Routing a message to one of many queues or intents is a common System One
decision, and the `choice` question type fits it directly. The released
evaluations cover agentic tasks (computer use, games, tool calling, verification);
there is no routing or classification benchmark yet. This directory adds one,
built from public intent datasets, and reports our results on it.

It measures:

- **Zero-shot:** the released head choosing among many hand-described intents.
- **Fine-tuned, seen intents:** heads trained on the same intents they are tested on.
- **Fine-tuned, unseen intents:** heads tested on intents held out of training,
  which is the case where a contrastive head should beat a closed-set classifier.
- **Baseline:** a logistic regression on the same frozen state embeddings.

---

## Data

| Dataset | Source | Licence | Labels (unseen) | Train | Test, seen | Test, unseen |
|---|---|---|---|---|---|---|
| Banking77 | `PolyAI/banking77` (the CSVs its loader reads, `PolyAI-LDN/task-specific-datasets` @ `57ec275`) | CC-BY-4.0 | 77 (15) | 5000 | 1500 | 1000 |
| Bitext insurance | `bitext/Bitext-insurance-llm-chatbot-training-dataset` @ `3ce42aa` | CDLA-Sharing-1.0 | 39 (8) | 5000 | 1500 | 1000 |
| typed-decisions | `LocalLLaMA/typed-decisions` @ `f7a2487`, config `all` | Apache-2.0 | mixed question types | 1200 rows | 400 rows (2000 questions) | n/a |

`prepare_data.py` downloads the pinned sources and builds the Banking77 and
Bitext splits; the data itself is not committed. For each dataset, with seed 0
throughout:

- Texts are stripped. Exact duplicates are dropped, and a text that appears
  under two labels is dropped entirely (Banking77: 13072 rows left, upstream
  train and test pooled; Bitext: 38983).
- `round(0.2 × n_labels)` labels are chosen as **unseen**. None of their
  messages appear in training.
- Every label is split 50/50 into train and test.
- Each part is subsampled to the caps above, stratified by label.

typed-decisions is used as released, with its own train/test split.

### Formatting

Each message becomes one row in the typed-decisions schema, so
`train/adapters.py:typed_decision_examples` and `clm.schema.build_pairs` handle
it unchanged:

- `state`: the message text.
- `questions`: one `choice` question, `queue`, with instructions
  "Which support queue should handle this customer message?" and one
  hand-written, one-sentence description per label as `criteria`, e.g.
  `card_arrival`: "Customer says an ordered card has not arrived yet."
- `gold`: `{"queue": {"label": <intent>}}`.

The state text is therefore the message, a blank line, then the question; the
candidates are the label descriptions verbatim.

- **Train rows list the seen labels only**, so unseen descriptions never reach
  training, not even as negatives.
- **Test rows list every label**, seen and unseen, including in the unseen-intent
  test set.

Layout written by `prepare_data.py`:

```
data/intent_routing/<dataset>/
├── labels.json                          # question, label descriptions, unseen labels, source
├── seen/{train,test}-00000-of-00001.parquet    # train; test on seen intents
└── unseen/{train,test}-00000-of-00001.parquet  # same train; test on unseen intents
```

`--workflow seen` and `--workflow unseen` therefore train the same head on the
same rows (same validation split too) and differ only in the test set.

---

## Reproduce

`finetune.py` embeds with offline vLLM unless `--embed-url` points at a pooling
server (see [Serve](../../README.md#serve)). Sharing `--embed-cache` across runs
embeds each text once.

```bash
pip install pandas scikit-learn   # on top of requirements.txt
python evaluation/intent_routing/prepare_data.py --out data/intent_routing

CKPT="$(clm-download)"            # CLM_v0.1-8B.pt
EMB=runs/intent_routing/embeddings
for ds in banking77 bitext; do
  for wf in seen unseen; do
    for loss in infonce softce; do
      python train/finetune.py --task choice --data data/intent_routing/$ds --workflow $wf \
          --init-ckpt "$CKPT" --loss $loss --embed-cache $EMB \
          --out-dir runs/intent_routing/${ds}_${wf}_${loss}
    done
  done
  python evaluation/intent_routing/logreg_baseline.py --data data/intent_routing/$ds \
      --workflow seen --embed-cache $EMB
done

# typed-decisions
for loss in infonce softce; do
  python train/finetune.py --task choice --data LocalLLaMA/typed-decisions --workflow all \
      --init-ckpt "$CKPT" --loss $loss --embed-cache $EMB --out-dir runs/intent_routing/typed_$loss
done
python evaluation/intent_routing/logreg_baseline.py --data LocalLLaMA/typed-decisions \
    --workflow all --embed-cache $EMB
```

Reading the output of each `finetune.py` run (also in `finetune_summary.json`):

- **Zero-shot:** `epoch 0 (init) ... test acc` (`init_test.acc`), the released head
  before any update.
- **Fine-tuned:** `TEST acc` (`test.acc`), the best-validation epoch.

All other settings are the `choice` defaults: encoder frozen, heads warm-started
from the released checkpoint, AdamW lr 5e-4 with OneCycle, batch 256, up to 20
epochs, early stopping with patience 5 on a 10% validation split, soft targets
(one-hot here), seed 1234.

`logreg_baseline.py` fits StandardScaler + LogisticRegression (`max_iter=2000`)
per question on the train state embeddings from the same cache. It can only
predict labels seen in training, so it is not reported on unseen intents.

---

## Results

Accuracy. "Seen" and "unseen" refer to CLM's training data.

| Test set | n | CLM zero-shot | CLM fine-tuned (InfoNCE) | CLM fine-tuned (soft-CE) | LogReg on embeddings | SemIf (reference) |
|---|---|---|---|---|---|---|
| Banking77, seen intents | 1500 | 0.051 | 0.775 | 0.795 | **0.900** | 0.767¹ |
| Banking77, unseen intents | 1000 | 0.061 | 0.440 | 0.397 | n/a | **0.675**¹ |
| Bitext, seen intents | 1500 | 0.056 | 0.991 | 0.989 | **0.997** | 0.928¹ |
| Bitext, unseen intents | 1000 | 0.015 | 0.406 | 0.463 | n/a | **0.880**¹ |
| typed-decisions | 2000 | 0.355 | 0.686 | 0.661 | **0.737** | 0.572² |

In short:

- **Zero-shot**, the released head rarely picks the right intent among 39–77
  described options.
- **Fine-tuned on the tested intents**, the heads reach 0.78–0.99. A logistic
  regression on the same frozen embeddings scores similarly or higher (0.74–1.00), which
  speaks well of the Qwen3 embeddings; the head's advantage is that it can also
  score intents it was never trained on.
- **Fine-tuned, on held-out intents**, accuracy is 0.40–0.46.

**SemIf** ([SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev), MIT) is
an open zero-shot option-scoring model, included only as a reference point. It
gets the same message, question and label descriptions, and no training
examples.

¹ **SemIf accepts at most 16 options per question,** so sets with more labels
were scored with a two-round knockout:

1. The labels are split into the fewest near-equal chunks of 16 or fewer
   (5 chunks for Banking77, 3 for Bitext), and SemIf picks one label per chunk.
2. SemIf then picks from those chunk winners.

This approximates choosing from all labels at once. The correct label can lose
to a similar label within its chunk and never reach the final, so if anything
it works against SemIf.

² SemIf covers only the 600 `choice` questions, where always picking the most
common label scores 0.357. The other columns cover all 2000 questions (choice,
yes/no, score), so this entry is not directly comparable.

### How these numbers were computed

Our numbers were computed with the same `run_choice` code and the same state
and candidate texts as above, but with a transformers-based embedder (bf16,
same last-token pooling after the final norm, L2-normalised, last 2048 tokens
kept) instead of vLLM, so results with vLLM may differ slightly. As a check,
this setup gives 0.845 / 0.988 / 2.00 on the Quickstart example (playground
screenshot: 0.848 / 0.988 / 2.00) and 0.993 on the model-card `rank` example.

---

## Other observations

- **typed-decisions, zero-shot:** the released head (0.355) scores below always
  picking the most common label (0.483), and on 14 of its 20 question types it
  gives the same answer for every input. Score questions collapsing to one
  level looks like the behaviour already reported in issue #3.
- **Rewording did not change zero-shot routing** on typed-decisions'
  customer-service questions (0.20–0.44 across variants): the state as a JSON
  string, options written as "key: description", a plain-transcript state, and
  label keys only.

---

## Caveats

- CLM and the logistic regression are given training examples; SemIf is not.
- One seed per configuration. Gaps of a few points between the two losses are
  within noise.
- **Label noise affects every method:** most Banking77 `get_physical_card`
  examples are really PIN questions, and Bitext has four overlapping
  "talk to a person" intents (`agent`, `customer_service`, `human_agent`,
  `insurance_representative`).
- The `majority` baseline that `run_choice` prints compares option indices
  between train and test. Here train lists only seen labels and test lists all
  of them, so that figure is not meaningful for these datasets.
- typed-decisions is loaded from the Hub's current revision by default; our
  numbers used `f7a2487`.
