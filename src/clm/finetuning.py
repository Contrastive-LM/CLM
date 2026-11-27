"""``clm.finetune``: fine-tune the CLM heads on your own data in one call.

    import clm

    result = clm.finetune("data/train.jsonl")                # or a list of dicts
    engine = result.engine()                                 # the head you just trained
    engine.rank("What causes tides on Earth?", ["The Moon's gravity.", "Photosynthesis."])

The encoder stays frozen; only the two projection heads train, so a run costs one
embedding pass over the data plus seconds to minutes of head training. This is a
thin layer over the trainer in ``clm.trainer`` (the ``clm-finetune`` command): it
writes in-memory records to a file, picks the task from their shape, finds the
embedding server and the head to warm-start from, and reads the result back.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_ENCODER = "Qwen/Qwen3-8B"
LOCAL_SERVER = "http://127.0.0.1:8090/v1"


@dataclass
class FinetuneResult:
    """What ``clm.finetune`` returns.

    Attributes:
        checkpoint: the best head on disk (``<out_dir>/best_head.pt``).
        out_dir: the run directory (checkpoints, history, summaries, cached embeddings).
        task: ``"clm"`` (state/action records) or ``"choice"`` (records with candidates).
        metric: the validation metric the best epoch was selected on: ``within_task_top1``
            or ``val_loss`` for ``clm``, ``val_acc`` for ``choice``.
        best_score: that metric at the best epoch.
        baseline_score: the same metric before training (epoch 0: the warm-start head, or a
            random one when training from scratch).
        best_epoch: 0 means no epoch beat the starting point.
        history: per-epoch validation metrics.
    """

    checkpoint: str
    out_dir: str
    task: str
    metric: str
    best_score: float
    baseline_score: float
    best_epoch: int
    history: list[dict] = field(repr=False, default_factory=list)
    embed_url: str | None = field(repr=False, default=None)
    embed_model: str | None = field(repr=False, default=None)

    @property
    def improvement(self) -> float:
        """Change in the metric from the starting point; positive is better."""
        delta = self.best_score - self.baseline_score
        return -delta if self.metric == "val_loss" else delta

    def engine(self, **kwargs: Any):
        """A ``clm.Engine`` serving this head, on the embedding server used for training
        (or the ``Engine`` default when the encoder ran in-process)."""
        from .engine import Engine
        if self.embed_url:
            kwargs.setdefault("emb_url", self.embed_url)
            kwargs.setdefault("emb_model", self.embed_model)
        return Engine(checkpoint=self.checkpoint, **kwargs)


def _records_file(dataset: Any, out_dir: str, name: str) -> str:
    """A path to ``dataset`` as a records file, writing a list of dicts under ``out_dir/data``."""
    if isinstance(dataset, (str, os.PathLike)):
        return os.fspath(dataset)
    try:
        records = [dict(r) for r in dataset]
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a list of dicts or a path to a .json / .jsonl file") from None
    path = os.path.join(out_dir, "data", f"{name}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _embedding_server(base_url: str | None, say) -> str | None:
    """``"auto"``: ``CLM_EMB_URL``, else the local vLLM server if one answers, else None
    (the encoder then runs in-process)."""
    if base_url != "auto":
        return base_url
    if os.environ.get("CLM_EMB_URL"):
        return os.environ["CLM_EMB_URL"]
    from .embed_utils import EmbedderError, served
    try:
        served(LOCAL_SERVER, timeout=2.0)
    except EmbedderError:
        say(f"[finetune] no embedding server at {LOCAL_SERVER}: running the encoder in-process with vLLM")
        return None
    return LOCAL_SERVER


def finetune(
    trainset: Any,
    valset: Any = None,
    *,
    task: str = "auto",
    model: str = DEFAULT_ENCODER,
    base_url: str | None = "auto",
    init: str | None = "auto",
    out_dir: str = "runs/finetune",
    epochs: int = 20,
    batch_size: int | None = None,
    lr: float | None = None,
    val_frac: float = 0.1,
    patience: int = 5,
    seed: int = 1234,
    verbose: bool = True,
    **options: Any,
) -> FinetuneResult:
    """Fine-tune the CLM projection heads on your own data.

    Args:
        trainset: training records: a list of dicts, or a ``.jsonl`` / ``.json`` file of
            them. The shape picks the task (see ``clm.adapters``):

            * state/action records train the ``clm`` task. Agent step traces
              (``state``, ``action``, ``task_id``, ``step_idx``) or plain pairs
              (``{"input": ..., "answer": ...}``).
            * records with candidates and a gold answer train the ``choice`` task:
              ``{"state": ..., "candidates": [...], "rewards": [1, 0, 0]}`` (or
              ``"label"``), or ``{"question": ..., "answer": ..., "negatives": [...]}``.

            A string that is not a records file is passed through to the trainer: an
            embedding dir or Hub embedding dataset for ``clm``, a typed-decisions dataset
            with ``task="choice"``.
        valset: validation records in the same form. Without one, ``val_frac`` of the
            training tasks (``clm``) or records (``choice``) is held out.
        task: ``"clm"``, ``"choice"``, or ``"auto"`` to read it off the records.
        model: the frozen encoder (a Hugging Face id).
        base_url: the vLLM pooling server that embeds with ``model``, as its ``/v1`` base
            or ``/v1/embeddings`` URL. ``"auto"`` uses ``CLM_EMB_URL``, else the server at
            ``http://127.0.0.1:8090`` if one is running, else in-process vLLM. ``None``
            always runs the encoder in-process.
        init: checkpoint to warm-start from. ``"auto"`` is the reference CLM head when
            ``model`` is Qwen3-8B (downloaded on first use) and a fresh head otherwise;
            ``None`` always trains from scratch.
        out_dir: run directory. Receives ``best_head.pt``, ``final_head.pt``, the metric
            history and the cached embeddings.
        epochs, batch_size, lr, val_frac, patience, seed: optimisation. Unset values take
            the trainer's defaults (batch 2048 for ``clm``, 256 for ``choice``).
        verbose: print the trainer's progress.
        **options: any other ``clm-finetune`` option by its Python name, for example
            ``max_len=4096``, ``served_model_name="qwen3-8b"``, ``sampler="task-blocked"``,
            ``targets="hard"``, ``loss="softce"`` or ``holdout_tasks="heldout.json"``.

    Returns:
        A ``FinetuneResult``. ``result.checkpoint`` is the best head, ``result.engine()``
        loads it for ranking, and ``result.baseline_score`` / ``result.best_score`` say
        what training bought.
    """
    from . import adapters, heads, trainer

    if task not in ("auto", "clm", "choice"):
        raise ValueError(f"task must be 'auto', 'clm' or 'choice', got {task!r}")
    say = (lambda msg: print(msg, flush=True)) if verbose else (lambda msg: None)
    os.makedirs(out_dir, exist_ok=True)

    parser = trainer.build_parser()
    known = {a.dest for a in parser._actions} - {"help", "task", "out_dir", "data", "val_data", "embed_model",
                                                 "embed_url", "init_ckpt", "epochs", "batch", "lr",
                                                 "val_frac", "patience", "seed"}
    unknown = sorted(set(options) - known)
    if unknown:
        raise TypeError(f"finetune() got unexpected options {unknown}; trainer options are {sorted(known)}")
    if any(options.get(k) for k in ("folds", "holdout_folds")):
        raise ValueError("cross-validation folds train several heads; run them with clm-finetune")

    # ---- data: a records file picks its own task; any other source is the trainer's to resolve
    source = _records_file(trainset, out_dir, "train")
    if adapters.is_records_file(source):
        found = adapters.task_of(adapters.read_records(source))
        if task not in ("auto", found):
            raise ValueError(f"task={task!r}, but {source} holds {found!r} records")
        task = found
        data_flag = "--data"
    elif source.endswith((".json", ".jsonl")):
        raise FileNotFoundError(source)
    else:
        task = "clm" if task == "auto" else task
        data_flag = "--data" if task == "choice" else (
            "--emb-dir" if os.path.isdir(source) else "--hf-dataset")
    argv = ["--task", task, "--out-dir", out_dir, data_flag, source, "--embed-model", model,
            "--epochs", str(epochs), "--val-frac", str(val_frac), "--patience", str(patience),
            "--seed", str(seed)]
    if valset is not None:
        if data_flag != "--data":
            raise ValueError("valset needs trainset to be records (a list of dicts or a .json / .jsonl file)")
        argv += ["--val-data", _records_file(valset, out_dir, "val")]

    # ---- encoder and starting head
    embed_url = _embedding_server(base_url, say)
    if embed_url:
        argv += ["--embed-url", embed_url]
    if init == "auto":
        init = (heads.default_checkpoint() or heads.download()) if model == DEFAULT_ENCODER else None
    if init:
        argv += ["--init-ckpt", os.fspath(init)]

    for flag, value in (("--batch", batch_size), ("--lr", lr)):
        if value is not None:
            argv += [flag, str(value)]
    for name, value in options.items():
        if value is None or value is False:
            continue
        flag = "--" + name.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]

    # ---- train: option errors surface as exceptions, not as a process exit
    def fail(message: str):
        raise ValueError(message)

    parser.error = fail
    args = parser.parse_args(argv)
    try:
        with contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO()):
            trainer.resolve(parser, args)
            summary = trainer.run(args)
    except SystemExit as e:
        raise ValueError(str(e)) from None

    # ---- read the outcome back
    if task == "clm":
        run = json.load(open(os.path.join(out_dir, "summary.json")))
        metric, history, best_epoch = run["select_metric"], run["history"], run["best_epoch"]
        scores = {h["epoch"]: h[metric] for h in history}
    else:
        history = json.load(open(os.path.join(out_dir, "history.json")))
        metric, best_epoch = "val_acc", summary["best_epoch"]
        scores = {h["epoch"]: h["val"]["acc"] for h in history}
    result = FinetuneResult(
        checkpoint=os.path.join(out_dir, "best_head.pt"), out_dir=out_dir, task=task, metric=metric,
        best_score=float(scores[best_epoch]), baseline_score=float(scores[0]), best_epoch=int(best_epoch),
        history=history, embed_url=args.embed_url, embed_model=args.served_model_name)
    say(f"[finetune] {metric} {result.baseline_score:.4f} -> {result.best_score:.4f} "
        f"(best epoch {result.best_epoch}) | {result.checkpoint}")
    return result
