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

## Published artifacts

- [Trained head](https://huggingface.co/AbeHou/harvey-clm-head): the epoch-20 checkpoint, training history, settings, and per-task evaluation results.
- [Data](https://huggingface.co/datasets/AbeHou/harvey-clm-data): 1,545 adapted conversations, labels, task splits, and exact cached embeddings.

The data includes 698 training traces, 416 validation traces, and 431 test traces.
The compressed conversations and cached embeddings occupy about 1.06 GB.
The head occupies about 76 MB. Each repository includes SHA-256 checksums.
`release.json` pins the artifact revisions used by the scripts.

## Reproduce the scores

Run these commands from the repository root with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 numpy==2.5.3 requests huggingface_hub
python examples/harvey/reproduce.py download
python examples/harvey/reproduce.py evaluate
```

Evaluation works on a CPU. A CUDA GPU is used when available.
The script calls the unchanged `evaluation/bon_eval.py` with `N=8` and window 12.
It checks both scores and selected trace IDs against the recorded results.
Outputs go to `runs/harvey-reproduction/`.

## Reproduce training

The experiment used upstream commit `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`.
The [trainer patch](finetune.patch) changes only `train/finetune.py`.
The script creates a separate checkout, applies the patch, and verifies its hash.
It also checks the starting checkpoint hash and the resulting task split.

```bash
python examples/harvey/reproduce.py train --output runs/harvey-training
```

This trains from the published cached embeddings and then evaluates the trained head.
It uses the recorded batch size, schedule, seed, and checkpoint selection rule.
The exact trainer command is:

```bash
python train/finetune.py --task clm \
  --emb-dir artifacts/harvey/data/embeddings/parthenon_cafl \
  --init-ckpt "$(clm-download)" --out-dir runs/harvey \
  --batch 2048 --epochs 20 --weight-decay 0 \
  --val-frac 0.1 --patience 5 --seed 1234 \
  --sampler random --clm-select-metric within_task_top1
```

Apply the patch before using this command directly.
Both heads retain width 1,536, depth 3, 512 output dimensions, GELU,
layer normalization, and no residual connections.
The published checkpoint is epoch 20, selected by internal within-task retrieval.
Its retrieval accuracy was 17.30%, separate from whole-trace selection success.
GPU and library differences can change retraining results.
The published checkpoint and cached embeddings fix the inputs for score reproduction.

## Rebuild embeddings from conversations

The data includes the exact adapted conversations used by the original encoder.
The encoder script preserves their row order and causal state/action boundaries.
It uses upstream `Recipe` tokenization with frozen Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`.
States retain the final 8,191 tokens after the chat template.
Actions retain the first 8,191 tokens without special tokens.
Pooling uses the normalized last token, without chunk averaging.

Use a separate environment for the original vLLM encoder:

```bash
python -m venv .venv-encode
source .venv-encode/bin/activate
python -m pip install vllm==0.22.1
python examples/harvey/encode.py --data artifacts/harvey/data prepare --workers 8
CUDA_VISIBLE_DEVICES=0 python examples/harvey/encode.py \
  --data artifacts/harvey/data encode --shard 0 --num-shards 1
python examples/harvey/encode.py --data artifacts/harvey/data assemble
```

Encoding requires a GPU with enough memory for Qwen3-8B and an 8,192-token context.
The original run used four A100 workers. The command above uses one GPU.
For four workers, run shards 0 through 3 with `--num-shards 4`, one visible GPU each.
Encoded batches can resume. `assemble` requires all shards to finish.
Rebuilt tensors go to `reencoded_embeddings/` and preserve the published tensors.
The original run encoded 409 million tokens; cached embeddings avoid this cost.

## Cohort and experiment details

Outer cohorts contain Parthenon traces and exclude task families in the training pool.
The evaluator retains candidate groups smaller than eight.
The internal split separates task IDs, but shares eight related task families.
CAFL supplies 74.43% of actual update pairs.

The unchanged trainer scored 16/52 on outer validation.
The temperature change and lower-rate experiment each scored 18/52.
The experiment loop kept the temperature change and discarded the extra learning-rate change.
The later test comparison is exploratory; it did not improve that validation result.

The release starts from adapted conversations and includes their success labels.
Original provider log archives and separately stored task documents are outside this package.
