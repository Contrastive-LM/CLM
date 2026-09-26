# Extended clm on harvey LAB benchmark

This PR documents CLM fine-tuning and trace selection on the Harvey LAB benchmark. The training dataset consists of successful agent traces from Parthenon and CAFL, converted into causal state/next-action pairs.

### Traces and training

- Parthenon supplied 408 successful traces from Claude Code (Sonnet 4.6 and Haiku 4.5) and Codex (GPT-5.5 and GPT-5.4 mini). CAFL supplied 290 successful traces from DeepSeek, Gemini, and GLM configurations.
- The combined pool contains 310 tasks and 32,801 causal state/next-action pairs. The trainer uses 279 tasks for updates and 31 for internal validation.
- Start from the released CLM-v0.1-8B head. Keep Qwen3-8B frozen and preserve both projection-head architectures. Use normalized last-token embeddings with the tutorial's 8,191-token input limits.
- Train with masked bidirectional InfoNCE, AdamW, batches of 2,048, and a 20-epoch OneCycle cosine schedule. The lower-rate experiment halves the warm-start learning rate to 0.0011547 and applies the logit-scale cap after each optimizer step. These changes are confined to `train/finetune.py`.

### Evaluation

Select from up to eight traces per task using mean cosine similarity over the final 12 steps.

| Head | Validation, 52 tasks | Test, 54 tasks |
| --- | ---: | ---: |
| Released starting head | 14/52 (26.92%) | 15/54 (27.78%) |
| Extended clm on harvey LAB benchmark | 18/52 (34.62%) | 19/54 (35.19%) |

The lower rate tied the temperature-only change on validation. The test cohort was inspected earlier, so its result is exploratory. Both cohorts require at least one successful candidate.

## Reproduce the experiment

The experiment used upstream commit `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`.
The [trainer patch](finetune.patch) contains the two changes described above.
It changes only `train/finetune.py` when applied. The upstream trainer and
mean evaluator are unchanged in this example pull request.

Prepare successful traces as causal conversation states and next actions.
Use Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
Apply its chat template to states and retain the final 8,191 tokens.
Tokenize actions without special tokens and retain the first 8,191 tokens.
Use normalized, 4,096-dimensional last-token embeddings without chunk averaging.
Save `state_embeddings.pt`, `action_embeddings.pt`, and `metadata.json` in
separate training, validation, and test directories. Each training sample needs
`task_id` and `step_idx`. Evaluation samples also need `trajectory_id` and
binary `reward`. Metadata and embedding rows must have the same order.

Apply the patch from the repository root and run training:

```bash
git apply examples/harvey/finetune.patch
python train/finetune.py --task clm \
  --emb-dir /path/to/harvey/train-embeddings \
  --init-ckpt "$(clm-download)" --out-dir runs/harvey \
  --batch 2048 --epochs 20 --weight-decay 0 \
  --val-frac 0.1 --patience 5 --seed 1234 \
  --sampler random --clm-select-metric within_task_top1
```

Both heads retain width 1,536, depth 3, 512 output dimensions, GELU,
layer normalization, and no residual connections. The trainer selected epoch
20 by internal within-task retrieval. Its retrieval accuracy was 17.30%,
which is separate from whole-trace selection success.

Score the selected checkpoint with the unchanged evaluator:

```bash
python evaluation/bon_eval.py \
  --embeddings-dir /path/to/harvey/validation-embeddings \
  --checkpoint runs/harvey/best_head.pt --n 8 --window 12
python evaluation/bon_eval.py \
  --embeddings-dir /path/to/harvey/test-embeddings \
  --checkpoint runs/harvey/best_head.pt --n 8 --window 12
```

The outer validation cohort contains 416 traces; test contains 431.
The evaluator retains groups with fewer than eight candidates. Outer cohorts
contain Parthenon traces and exclude task families in the training pool.
The internal training/validation split separates task IDs, but shares eight
related task families. CAFL supplies 74.43% of actual update pairs.

The unchanged trainer scored 16/52 on outer validation. The temperature
change and the lower-rate experiment each scored 18/52. The experiment loop
therefore kept the temperature change and discarded the extra learning-rate
change. The later test comparison is reported here as exploratory; it did
not improve the validation result used by that loop.

The dataset-specific conversation adapter, traces, embeddings, and trained
weights are not included. Reproducing the full result requires those inputs.
