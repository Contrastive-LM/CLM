# finetuning

This is an experiment to have the LLM autonomously improve CLM fine-tuning.

## Setup

Work with the user to:

1. Agree on `clm` or `choice`, a run tag, and a fixed training and evaluation
   command.
2. Create a fresh `autofinetune/<tag>` branch.
3. Read `README.md`, `train/finetune.py`, `train/adapters.py`,
   `train/embed_utils.py`, `preprocessing/hf_embeddings.py`, and
   `evaluation/bon_eval.py`.
4. Verify the dataset, checkpoint, folds, and GPU are available.
5. Create an untracked `results.tsv` containing:

   ```text
   commit	score	status	description
   ```

6. Confirm the setup with the user, then run the unchanged baseline.

## Experimentation

Run training with the agreed command and redirect output:

```bash
python train/finetune.py ... --out-dir runs/<tag>/<commit> > run.log 2>&1
```

For `clm`, run the agreed held-out evaluation afterward:

```bash
python evaluation/bon_eval.py ... --n <N> --window <K> >> run.log 2>&1
```

**You may:**

- Modify `train/finetune.py`.
- Change the head, objective, optimizer, scheduler, batching, and
  hyperparameters.

**You may not:**

- Modify any other file.
- Change the data, embeddings, splits, folds, evaluation set, `--n`, or
  `--window`.
- Install dependencies.
- Train on evaluation data or tune on the `choice` test split.

The goal is to maximize the held-out best-of-N rate for `clm` or validation
accuracy for `choice`. Prefer simpler changes when scores are equal.

The first run must always establish the unmodified baseline.

## Logging results

Append one tab-separated row after every run:

```text
commit	score	status	description
a1b2c3d	0.796500	keep	baseline
b2c3d4e	0.805300	keep	task-blocked batches
c3d4e5f	0.787600	discard	constant learning rate
d4e5f6a	0.000000	crash	double width (OOM)
```

Do not commit `results.tsv`, `run.log`, or run outputs.

## The experiment loop

LOOP FOREVER:

1. Inspect the current commit and previous results.
2. Make one focused change to `train/finetune.py`.
3. Commit it.
4. Run training and evaluation.
5. Record the result.
6. Keep the commit if the score improved.
7. Otherwise, reset to the last kept commit.
8. Continue with another idea.

If a run crashes, inspect `tail -n 80 run.log`. Fix simple bugs and rerun;
otherwise record the crash and reset. If it exceeds the agreed timeout, stop it,
record a timeout, and reset.

Once the loop begins, do not ask whether to continue. Keep experimenting until
the user interrupts you.
