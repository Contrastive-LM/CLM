# Harvey projection head

The [`AbeHou/harvey-clm-head`](https://huggingface.co/AbeHou/harvey-clm-head)
model repository contains two projection heads trained on successful Harvey
traces from Parthenon and CAFL. It uses the frozen Qwen3-8B encoder. It does
not contain the encoder or the training traces.

The checkpoint has the `state_head`, `action_head`, `logit_scale`, and `cfg`
fields expected by this repository. It selects epoch 14. Its SHA-256 is
`554dde3cc8efdd424ded2a65284e620b3e0d1147cb3d5e479a899c50523a4f9d`.
Local training paths were removed from the checkpoint metadata. The weights
are identical to the selected training checkpoint.

## Data preparation

The run used the CLM release code at commit
`c2021be813c0ab80b4ea9f7f8e2db5b5cddefc4a`. The Harvey conversation
adapter converted each successful trace into causal state and next-action
pairs. It kept failed traces only for evaluation. The adapted data and the
Harvey adapter are not part of this pull request.

The pool held 310 tasks, 698 successful traces, and 32,801 state/action pairs.
Parthenon supplied 8,642 pairs. CAFL supplied 24,159 pairs. The trainer held
out 31 tasks for internal validation and used 279 tasks for updates. The
update set contained 29,455 pairs: 7,532 Parthenon and 21,923 CAFL.

For each state, apply the Qwen3-8B chat template to the causal conversation.
Keep the final 8,191 tokens. For each action, tokenize without special tokens
and keep the first 8,191 tokens. Use Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218` with last-token pooling.
Normalize each 4,096-dimensional embedding. Do not average chunks. The
repository's `train/embed_utils.py` defines this token recipe.

Store the pairs as `state_embeddings.pt`, `action_embeddings.pt`, and
`metadata.json` in one training directory. Each metadata sample needs a
`task_id` and `step_idx`. Evaluation directories also need `trajectory_id`
and a binary `reward` for each sample. The sample order must match both
embedding tensors. These files are not included because they contain Harvey
trace data.

## Fine-tuning

Download the released starting head. Then run the unchanged CLM trainer on
the prepared training embeddings:

```bash
clm-download
python train/finetune.py --task clm \
  --emb-dir /path/to/harvey/train-embeddings \
  --init-ckpt "$(clm-download)" \
  --out-dir runs/harvey \
  --batch 2048 --epochs 20 --weight-decay 0 \
  --val-frac 0.1 --patience 5 --seed 1234 \
  --sampler random --clm-select-metric within_task_top1
```

The trainer freezes the encoder and updates both projection heads and the
learned temperature. It uses masked bidirectional InfoNCE, AdamW, and the
default width and batch learning-rate rule. That rule gives
`0.002309401076758503` for this head. The scheduler is OneCycle with cosine
decay. The selected checkpoint reached 18.05% internal within-task action
retrieval at epoch 14. Training stopped at epoch 19 after five epochs without
enough improvement.

The internal split separates task IDs, but eight related task families appear
on both sides. The outer validation and test families are separate from the
full training pool.

## Evaluation

Download the head. Then run the evaluator on each prepared outer cohort:

```bash
hf download AbeHou/harvey-clm-head head.pt --local-dir heads/harvey

python evaluation/bon_eval.py \
  --embeddings-dir /path/to/harvey/validation-embeddings \
  --checkpoint heads/harvey/head.pt \
  --n 8 --window 12 --aggregation min

python evaluation/bon_eval.py \
  --embeddings-dir /path/to/harvey/test-embeddings \
  --checkpoint heads/harvey/head.pt \
  --n 8 --window 12 --aggregation min
```

The score for each trace is the minimum cosine similarity over its final 12
steps. The selector chooses the highest-scoring trace. It uses up to eight
candidates per task and includes shorter groups.

| Cohort | Released starting head | Harvey head | Random selection |
| --- | ---: | ---: | ---: |
| Validation, 52 tasks | 7/52 (13.46%) | 17/52 (32.69%) | 22.36% |
| Test, 54 tasks | 12/54 (22.22%) | 19/54 (35.19%) | 20.17% |

These cohorts contain only tasks with at least one successful candidate.
Their rates are not DeepSWE or Terminal-Bench rates. The test cohort was
inspected in earlier experiments, so it is not an untouched final test. The
run does not isolate the effects of the data source, token limit, checkpoint
selection, or trace score. CAFL supplied 74.43% of update pairs, while both
outer cohorts contain only Parthenon traces.
