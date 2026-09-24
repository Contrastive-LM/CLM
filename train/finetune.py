#!/usr/bin/env python3
"""Fine-tune the CLM projection heads on a frozen Qwen3-8B encoder.

Tasks (``--task``)
  clm     (state, action) step traces from DeepSWE or custom data;
          one shared in-batch group-masked InfoNCE trainer.
          ``--holdout-folds`` trains one head per fold and writes fold_spec.json
          for evaluation/bon_eval.py. ``--folds K --fold-index INDEX`` creates
          task-disjoint folds stratified by candidate/pass count before training.
  choice  typed System One questions (LocalLLaMA/typed-decisions), candidates built by
          clm.schema.build_pairs. ``--loss infonce`` (default): bidirectional in-batch InfoNCE
          over the batch's distinct option texts; ``softce``: softmax over each question's own
          candidates. Both train against the annotator distribution (``--targets soft``) or the
          gold label (``hard``); evaluation is always per question.

Embeddings (clm)
  --emb-dir DIR        embedding dir (embed_shard.py + merge_embeddings.py output)
  --hf-dataset REPO    native or parquet embedding dataset (defaults from
                       --benchmark when no CLM data source is supplied)
  --data FILE          step transitions (.json / .jsonl), embedded from scratch
choice always embeds from scratch, with offline vLLM or a vLLM pooling server
(``--embed-url``).

Checkpoints hold state_head / action_head / logit_scale / cfg.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "preprocessing"))
sys.path.insert(0, HERE)
import adapters  # noqa: E402
import embed_utils  # noqa: E402
import hf_embeddings  # noqa: E402
from clm.heads import make_head  # noqa: E402


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


# --------------------------------------------------------------------------- #
# clm
# --------------------------------------------------------------------------- #
def _make_stratified_folds(index_path: str, k: int, out_dir: str,
                           config: str | None = None) -> tuple[list[str], dict]:
    """Write task-disjoint folds balanced by candidate count and pass count."""
    data = json.load(open(index_path))
    records = data if isinstance(data, list) else data.get("rows", data.get("records"))
    if not isinstance(records, list):
        raise ValueError(f"{index_path}: expected a list or an object containing rows/records")

    by_task = collections.defaultdict(list)
    for record in records:
        row_config = (record.get("config") or record.get("job_id") or
                      record.get("effort") or "default")
        if config is not None and row_config != config:
            continue
        task = record.get("task_name") or record.get("task_id")
        reward = record.get("passed")
        if reward is None:
            reward = record.get("reward")
        if reward is None:
            reward = (record.get("rewards") or {}).get("reward")
        if task is not None and reward is not None:
            by_task[task].append(float(reward) > 0.5)
    if len(by_task) < k:
        suffix = f" for config {config!r}" if config is not None else ""
        raise ValueError(f"{index_path}: need at least {k} indexed tasks{suffix}; found {len(by_task)}")

    strata = collections.defaultdict(list)
    for task, outcomes in by_task.items():
        strata[(len(outcomes), sum(outcomes))].append(task)
    folds = [[] for _ in range(k)]
    position = 0
    for key in sorted(strata):
        for task in sorted(strata[key]):
            folds[position % k].append(task)
            position += 1

    fold_dir = os.path.join(out_dir, "folds")
    os.makedirs(fold_dir, exist_ok=True)
    paths, summary = [], {"index": os.path.abspath(index_path), "config": config, "k": k,
                          "stratification": ["candidate_count", "pass_count"], "folds": []}
    for number, tasks in enumerate(folds):
        path = os.path.join(fold_dir, f"fold{number}.json")
        tasks = sorted(tasks)
        json.dump(tasks, open(path, "w"), indent=1)
        pass_counts = collections.Counter(sum(by_task[task]) for task in tasks)
        mixed = sum(0 < sum(by_task[task]) < len(by_task[task]) for task in tasks)
        summary["folds"].append({"path": os.path.relpath(path, out_dir), "tasks": len(tasks),
                                 "mixed_tasks": mixed,
                                 "pass_count_histogram": dict(sorted(pass_counts.items()))})
        paths.append(path)
        print(f"[folds] fold{number}: {len(tasks)} tasks ({mixed} mixed/decidable)", flush=True)
    json.dump(summary, open(os.path.join(fold_dir, "manifest.json"), "w"), indent=1)
    print(f"[folds] {len(by_task)} tasks -> {fold_dir}", flush=True)
    return paths, summary


def clm_embedding_dir(args) -> str:
    """Resolve the CLM data source to an embedding dir (building it if needed)."""
    if args.emb_dir:
        if not hf_embeddings.is_embedding_dir(args.emb_dir):
            raise SystemExit(f"--emb-dir {args.emb_dir} is not an embedding dir")
        return args.emb_dir
    cache = args.embed_cache or os.path.join(args.out_dir, "embeddings")
    if args.hf_dataset:
        return hf_embeddings.download(args.hf_dataset, os.path.join(cache, _slug(args.hf_dataset)),
                                      cache_dir=args.hf_cache, split=args.hf_split)
    out = os.path.join(cache, "clm_" + _slug(os.path.splitext(os.path.basename(args.data))[0]))
    if hf_embeddings.is_embedding_dir(out):
        print(f"[clm] reusing embeddings in {out}", flush=True)
        return out
    recs = adapters.read_transitions(args.data)
    recipe = embed_utils.Recipe(args.embed_model, args.max_len)
    backend = embed_utils.make_backend(args.embed_url, args.embed_model, args.max_len, args.gpu_mem,
                                       args.served_model_name)
    print(f"[clm] embedding {len(recs)} steps from scratch", flush=True)
    se = backend.embed([recipe.state_ids(r["state"]) for r in recs])
    ae = backend.embed([recipe.text_ids(r["action"], keep="head") for r in recs])
    os.makedirs(out, exist_ok=True)
    torch.save(torch.from_numpy(se.astype(np.float16)), os.path.join(out, "state_embeddings.pt"))
    torch.save(torch.from_numpy(ae.astype(np.float16)), os.path.join(out, "action_embeddings.pt"))
    meta = {"num_samples": len(recs), "hidden_size": int(se.shape[1]), "model_name": args.embed_model,
            "max_model_len": args.max_len, "input_file": os.path.abspath(args.data),
            "samples": [{k: r.get(k) for k in hf_embeddings.META_KEYS} for r in recs]}
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"))
    return out


def _load_clm_data(path, device, keep_tasks=None):
    """Load one embedding directory, optionally restricted to a task set."""
    states = torch.load(os.path.join(path, "state_embeddings.pt"), map_location="cpu")
    actions = torch.load(os.path.join(path, "action_embeddings.pt"), map_location="cpu")
    samples = json.load(open(os.path.join(path, "metadata.json")))["samples"]
    if not (len(samples) == len(states) == len(actions)):
        raise ValueError(f"{path}: embedding and metadata lengths differ")
    if keep_tasks is not None:
        keep = torch.tensor([i for i, sample in enumerate(samples)
                             if sample["task_id"] in keep_tasks], dtype=torch.long)
        states, actions = states[keep], actions[keep]
        samples = [samples[i] for i in keep.tolist()]
    if len(samples) < 2:
        raise ValueError(f"{path}: need at least two samples after task filtering")
    tasks = sorted({sample["task_id"] for sample in samples})
    task_ids = {task: i for i, task in enumerate(tasks)}
    task_idx = torch.tensor([task_ids[sample["task_id"]] for sample in samples])
    groups = {}
    codes = torch.tensor([groups.setdefault((sample["task_id"], sample["step_idx"]), len(groups))
                          for sample in samples])
    return states.to(device), actions.to(device), codes.to(device), task_idx, samples, len(tasks)


def _clm_loss(state_head, action_head, logit_scale, states, actions, codes):
    state_z = F.normalize(state_head(states), dim=-1)
    action_z = F.normalize(action_head(actions), dim=-1)
    logits = logit_scale.exp().clamp(max=100.0) * state_z @ action_z.t()
    labels = torch.arange(len(states), device=states.device)
    same = codes.unsqueeze(0) == codes.unsqueeze(1)
    same.fill_diagonal_(False)
    forward = F.cross_entropy(logits.masked_fill(same, float("-inf")), labels)
    backward = F.cross_entropy(logits.t().masked_fill(same, float("-inf")), labels)
    return (forward + backward) / 2, logits, labels


@torch.no_grad()
def _evaluate_clm(state_head, action_head, logit_scale, data, batch):
    states, actions, codes, task_idx, _, n_tasks = data
    state_head.eval(); action_head.eval()
    n = len(states)
    losses, top1, seen = [], 0, 0
    for start in range(0, n, min(batch, n)):
        stop = min(start + batch, n)
        if stop - start < 2:
            continue
        loss, logits, labels = _clm_loss(
            state_head, action_head, logit_scale, states[start:stop].float(),
            actions[start:stop].float(), codes[start:stop])
        losses.append(loss.item())
        top1 += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)

    state_z, action_z = [], []
    for start in range(0, n, 8192):
        state_z.append(F.normalize(state_head(states[start:start + 8192].float()), dim=-1))
        action_z.append(F.normalize(action_head(actions[start:start + 8192].float()), dim=-1))
    state_z, action_z = torch.cat(state_z), torch.cat(action_z)
    within_hits = within_n = 0
    within_ranks = []
    for task in range(n_tasks):
        members = (task_idx == task).nonzero(as_tuple=True)[0][:4096]
        if len(members) < 2:
            continue
        device_members = members.to(states.device)
        similarity = state_z[device_members] @ action_z[device_members].t()
        member_codes = codes[device_members]
        same = member_codes.unsqueeze(0) == member_codes.unsqueeze(1)
        same.fill_diagonal_(False)
        similarity = similarity.masked_fill(same, float("-inf"))
        rank = (similarity > similarity.diagonal().unsqueeze(1)).sum(1)
        within_hits += (rank == 0).sum().item()
        within_n += len(members)
        within_ranks.append((rank.float() / (len(members) - 1)).mean().item())
    state_head.train(); action_head.train()
    return {"val_loss": sum(losses) / max(1, len(losses)),
            "val_top1": top1 / max(1, seen),
            "within_task_top1": within_hits / max(1, within_n),
            "within_task_meanrank": sum(within_ranks) / max(1, len(within_ranks)),
            "n_val": n}


def _clm_batches(n, batch, sampler, task_idx, generator, tasks_per_batch):
    def chunks(indices):
        return [indices[i:i + batch] for i in range(0, len(indices), batch)
                if len(indices[i:i + batch]) >= 2]

    if sampler == "random":
        return chunks(torch.randperm(n, generator=generator))
    batches, pending = [], []
    unique = task_idx.unique().tolist()
    for position in torch.randperm(len(unique), generator=generator).tolist():
        indices = (task_idx == unique[position]).nonzero(as_tuple=True)[0]
        pending.append(indices[torch.randperm(len(indices), generator=generator)])
        if len(pending) == tasks_per_batch:
            block = torch.cat(pending); pending = []
            batches.extend(chunks(block[torch.randperm(len(block), generator=generator)]))
    if pending:
        block = torch.cat(pending)
        batches.extend(chunks(block[torch.randperm(len(block), generator=generator)]))
    return batches


def _train_clm(args, emb_dir, out_dir, holdout_path):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    checkpoint = (torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
                  if args.init_ckpt else None)
    initial = (checkpoint or {}).get("cfg", {})
    width = args.width or initial.get("width", 1536)
    depth = args.depth or initial.get("depth", 3)
    head_options = dict(activation=initial.get("activation", "gelu"),
                        layernorm=initial.get("layernorm", True),
                        residual=initial.get("residual", False))

    metadata = json.load(open(os.path.join(emb_dir, "metadata.json")))
    pool_tasks = sorted({sample["task_id"] for sample in metadata["samples"]})
    holdout = set(json.load(open(holdout_path))) if holdout_path else set()
    usable = [task for task in pool_tasks if task not in holdout]
    if len(usable) < 2:
        raise ValueError("CLM training needs at least two non-held-out tasks")
    shuffled = list(usable); random.Random(args.seed).shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, round(args.val_frac * len(shuffled))))
    val_tasks, train_tasks = set(shuffled[:n_val]), set(shuffled[n_val:])
    train = _load_clm_data(emb_dir, device, train_tasks)
    val = _load_clm_data(emb_dir, device, val_tasks)
    hidden = int(train[0].shape[1])
    if val[0].shape[1] != hidden:
        raise ValueError("training and validation embedding widths differ")
    if checkpoint and initial.get("hidden_size", hidden) != hidden:
        raise ValueError("checkpoint and dataset embedding widths differ")
    json.dump({"holdout": sorted(holdout), "train_tasks": sorted(train_tasks),
               "val_tasks": sorted(val_tasks)}, open(os.path.join(out_dir, "task_split.json"), "w"))
    print(f"[clm] tasks: pool {len(pool_tasks)}, holdout {len(holdout)}, "
          f"train {len(train_tasks)}, val {len(val_tasks)}", flush=True)
    print(f"[clm] train {len(train[0])} pairs | val {len(val[0])} pairs", flush=True)

    state_head = make_head(width, depth, args.proj, hidden=hidden, **head_options).to(device)
    action_head = make_head(width, depth, args.proj, hidden=hidden, **head_options).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    if checkpoint:
        state_head.load_state_dict(checkpoint["state_head"])
        action_head.load_state_dict(checkpoint["action_head"])
        with torch.no_grad():
            logit_scale.copy_(torch.as_tensor(checkpoint["logit_scale"]).to(device))

    params = list(state_head.parameters()) + list(action_head.parameters()) + [logit_scale]
    lr = args.lr or 2e-3 * math.sqrt(1024 / width) * math.sqrt(args.batch / 1024)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    initial_batches = _clm_batches(len(train[0]), args.batch, args.sampler, train[3],
                                   torch.Generator().manual_seed(0), args.tasks_per_batch)
    if not initial_batches:
        raise ValueError("CLM training produced no batches; lower --batch or change --sampler")
    total_steps = len(initial_batches) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1, anneal_strategy="cos")

    def metric_value(metrics):
        return (-metrics["val_loss"] if args.clm_select_metric == "val_loss"
                else metrics["within_task_top1"])

    def checkpoint_blob(epoch, metrics):
        return {"state_head": state_head.state_dict(), "action_head": action_head.state_dict(),
                "logit_scale": logit_scale.detach().cpu(),
                "cfg": {"width": width, "depth": depth, "projection_dim": args.proj,
                        "hidden_size": hidden, **head_options, "init_ckpt": args.init_ckpt,
                        "sampler": args.sampler, "batch": args.batch, "lr": lr,
                        "train_dir": emb_dir}, "epoch": epoch, "metrics": metrics}

    initial_metrics = _evaluate_clm(state_head, action_head, logit_scale, val, args.batch)
    history = [{"epoch": 0, **initial_metrics}]
    best, best_epoch, stale = metric_value(initial_metrics), 0, 0
    torch.save(checkpoint_blob(0, initial_metrics), os.path.join(out_dir, "best_head.pt"))
    print(f"[clm] epoch 0 {json.dumps(initial_metrics)}", flush=True)
    started = time.time()
    scheduler_steps = 0
    for epoch in range(1, args.epochs + 1):
        batches = _clm_batches(len(train[0]), args.batch, args.sampler, train[3],
                               generator, args.tasks_per_batch)
        train_loss = 0.0
        for indices in batches:
            indices = indices.to(device)
            loss, _, _ = _clm_loss(state_head, action_head, logit_scale,
                                    train[0][indices].float(), train[1][indices].float(),
                                    train[2][indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if scheduler_steps < total_steps:
                scheduler.step()
                scheduler_steps += 1
            train_loss += loss.item()
        metrics = _evaluate_clm(state_head, action_head, logit_scale, val, args.batch)
        metrics.update(train_loss=train_loss / len(batches), epoch=epoch,
                       minutes=(time.time() - started) / 60, logit_scale=logit_scale.item())
        history.append(metrics)
        print(f"[clm] epoch {epoch} {json.dumps(metrics)}", flush=True)
        torch.save(checkpoint_blob(epoch, metrics), os.path.join(out_dir, "final_head.pt"))
        score = metric_value(metrics)
        if score > best + 1e-4:
            best, best_epoch, stale = score, epoch, 0
            torch.save(checkpoint_blob(epoch, metrics), os.path.join(out_dir, "best_head.pt"))
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[clm] early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break
    summary = {"history": history, "best_epoch": best_epoch,
               "select_metric": args.clm_select_metric, "best_score": best}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return summary


def run_clm(args) -> dict:
    emb_dir = clm_embedding_dir(args)
    fold_summary = None
    if args.folds:
        args.holdout_folds, fold_summary = _make_stratified_folds(
            args.fold_index, args.folds, args.out_dir, args.fold_config)
        pool_tasks = {sample["task_id"] for sample in
                      json.load(open(os.path.join(emb_dir, "metadata.json")))["samples"]}
        fold_tasks = set().union(*(set(json.load(open(path))) for path in args.holdout_folds))
        missing = sorted(fold_tasks - pool_tasks)
        if missing:
            preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            raise ValueError(
                f"{args.folds}-fold cross-validation needs training embeddings for every indexed task; "
                f"{len(missing)}/{len(fold_tasks)} are absent ({preview}). Supply the complete "
                "embedding store instead of a task subset.")
    runs = ([(f"fold{k}", f) for k, f in enumerate(args.holdout_folds)] if args.holdout_folds
            else [(".", args.holdout_tasks)])
    spec, results = [], {}
    for name, holdout in runs:
        out = os.path.normpath(os.path.join(args.out_dir, name))
        os.makedirs(out, exist_ok=True)
        print(f"[clm] training {name} -> {out}", flush=True)
        summ = _train_clm(args, emb_dir, out, holdout)
        results[name] = {"best_epoch": summ["best_epoch"], summ["select_metric"]: summ["best_score"]}
        if args.holdout_folds:
            spec.append({"checkpoint": f"{name}/best_head.pt", "tasks": sorted(json.load(open(holdout)))})
    if spec:
        path = os.path.join(args.out_dir, "fold_spec.json")
        json.dump(spec, open(path, "w"), indent=1)
        print(f"[clm] wrote {path}", flush=True)
    return {"embeddings": emb_dir, "folds": fold_summary, "runs": results}


# --------------------------------------------------------------------------- #
# choice
# --------------------------------------------------------------------------- #
def load_typed_rows(data: str, split: str, workflow: str, cache_dir: str | None) -> list[dict]:
    import pyarrow.parquet as pq
    if os.path.isfile(data):
        files = [data]
    else:
        root = data
        if not os.path.isdir(root):
            from huggingface_hub import snapshot_download
            root = snapshot_download(data, repo_type="dataset", cache_dir=cache_dir,
                                     allow_patterns=[f"{workflow}/*.parquet"])
        files = sorted(f for f in glob.glob(os.path.join(root, workflow, "*.parquet"))
                       if os.path.basename(f).startswith(split))
    if not files:
        raise SystemExit(f"no {split!r} parquet for workflow {workflow!r} in {data}")
    rows: list[dict] = []
    for f in files:
        rows += pq.read_table(f).to_pylist()
    return rows


class TextCache:
    """sha1(text) -> embedding, persisted as .npz."""

    def __init__(self, path: str):
        self.path, self.vecs = path, {}
        if os.path.exists(path):
            z = np.load(path)
            self.vecs = dict(zip(z["keys"].tolist(), z["vecs"]))
            print(f"[choice] embedding cache: {len(self.vecs)} texts from {path}", flush=True)

    @staticmethod
    def key(t: str) -> str:
        return hashlib.sha1(t.encode()).hexdigest()

    def missing(self, texts):
        return [t for t in dict.fromkeys(texts) if self.key(t) not in self.vecs]

    def add(self, texts, vecs):
        self.vecs.update({self.key(t): np.asarray(v, dtype=np.float16) for t, v in zip(texts, vecs)})
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        ks = list(self.vecs)
        np.savez(self.path, keys=np.array(ks), vecs=np.stack([self.vecs[k] for k in ks]).astype(np.float16))

    def __getitem__(self, t):
        return self.vecs[self.key(t)]


def run_choice(args) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from clm.heads import HIDDEN, make_head

    train_rows = load_typed_rows(args.data, "train", args.workflow, args.hf_cache)
    test_rows = load_typed_rows(args.data, "test", args.workflow, args.hf_cache)
    ids = sorted({r["id"] for r in train_rows})
    random.Random(args.seed).shuffle(ids)
    val_ids = set(ids[:max(1, int(round(args.val_frac * len(ids))))])
    splits = {"train": [r for r in train_rows if r["id"] not in val_ids],
              "val": [r for r in train_rows if r["id"] in val_ids], "test": test_rows}
    ex = {k: list(adapters.typed_decision_examples(v)) for k, v in splits.items()}
    print(f"[choice] {args.data} [{args.workflow}] rows " +
          " ".join(f"{k} {len(splits[k])}" for k in splits) + " | questions " +
          " ".join(f"{k} {len(v)}" for k, v in ex.items()), flush=True)

    cache = TextCache(os.path.join(args.embed_cache or os.path.join(args.out_dir, "embeddings"),
                                   f"choice_{_slug(args.embed_model)}_{args.max_len}.npz"))
    texts = [t for v in ex.values() for e in v for t in (e.state_text, *e.candidates)]
    todo = cache.missing(texts)
    if todo:
        recipe = embed_utils.Recipe(args.embed_model, args.max_len)
        backend = embed_utils.make_backend(args.embed_url, args.embed_model, args.max_len, args.gpu_mem,
                                           args.served_model_name)
        print(f"[choice] embedding {len(todo)} unique texts", flush=True)
        cache.add(todo, backend.embed([recipe.text_ids(t, keep="tail") for t in todo]))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    def pack(examples):  # one dense bucket per candidate count
        buckets = defaultdict(list)
        for e in examples:
            buckets[len(e.keys)].append(e)
        out = []
        for k, es in buckets.items():
            q = torch.tensor(np.stack([cache[e.state_text] for e in es]), dtype=torch.float32)
            c = torch.tensor(np.stack([[cache[t] for t in e.candidates] for e in es]), dtype=torch.float32)
            out.append((q.to(device), c.to(device),
                        torch.tensor([e.target for e in es], device=device),
                        torch.tensor([e.label for e in es], device=device), [e.qid for e in es]))
        return out

    data = {k: pack(v) for k, v in ex.items()}

    ck = torch.load(args.init_ckpt, map_location="cpu", weights_only=False) if args.init_ckpt else None
    cfg = dict(width=args.width or 1536, depth=args.depth or 3, activation="gelu", layernorm=True, residual=False)
    if ck:
        c0 = ck.get("cfg", {})
        cfg.update({k: c0[k] for k in ("width", "depth", "activation", "layernorm", "residual") if k in c0})
    proj = (ck or {}).get("cfg", {}).get("projection_dim", args.proj)
    sh = make_head(cfg["width"], cfg["depth"], proj, cfg["activation"], cfg["layernorm"], cfg["residual"]).to(device)
    ah = make_head(cfg["width"], cfg["depth"], proj, cfg["activation"], cfg["layernorm"], cfg["residual"]).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))
    if ck:
        sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
        with torch.no_grad():
            logit_scale.copy_(torch.as_tensor(ck["logit_scale"]).float().to(device))
        print(f"[choice] warm start {args.init_ckpt} {cfg}", flush=True)

    def logits(q, c):
        zq = F.normalize(sh(q), dim=-1)
        zc = F.normalize(ah(c.reshape(-1, c.shape[-1])), dim=-1).view(c.shape[0], c.shape[1], -1)
        return logit_scale.exp().clamp(max=100.0) * torch.einsum("bh,bkh->bk", zq, zc)

    def loss_of(lg, tgt, lab):
        if args.targets == "soft":
            return -(tgt * F.log_softmax(lg, -1)).sum(-1).mean()
        return F.cross_entropy(lg, lab)

    if args.loss == "infonce":  # flat train set: every text once, options as indices into it
        tr = ex["train"]
        text_id = {t: i for i, t in enumerate(dict.fromkeys(t for e in tr for t in (e.state_text, *e.candidates)))}
        emb = torch.tensor(np.stack([cache[t] for t in text_id]), dtype=torch.float32, device=device)
        kmax = max(len(e.keys) for e in tr)
        st_idx = torch.tensor([text_id[e.state_text] for e in tr], device=device)
        opt_idx = torch.tensor([[text_id[t] for t in e.candidates] + [-1] * (kmax - len(e.keys)) for e in tr],
                               device=device)
        opt_tgt = torch.tensor([(e.target if args.targets == "soft" else
                                 [float(i == e.label) for i in range(len(e.keys))]) + [0.0] * (kmax - len(e.keys))
                                for e in tr], device=device)

    def infonce_loss(idx):
        """Bidirectional in-batch InfoNCE over the batch's distinct option texts: states -> options
        against the gold distribution, options -> states against the states that hold them."""
        oi, tg = opt_idx[idx], opt_tgt[idx]
        valid = oi >= 0
        pool, col = torch.unique(oi[valid], return_inverse=True)
        cols = torch.zeros_like(oi)
        cols[valid] = col
        tgt = torch.zeros(len(idx), len(pool), device=device).scatter_add_(1, cols, tg * valid)
        zq = F.normalize(sh(emb[st_idx[idx]]), dim=-1)
        zc = F.normalize(ah(emb[pool]), dim=-1)
        lg = logit_scale.exp().clamp(max=100.0) * zq @ zc.t()
        fwd = -(tgt * F.log_softmax(lg, 1)).sum(1).mean()
        w = tgt.t()
        keep = w.sum(1) > 0
        w = w[keep] / w[keep].sum(1, keepdim=True)
        bwd = -(w * F.log_softmax(lg.t()[keep], 1)).sum(1).mean()
        return (fwd + bwd) / 2

    @torch.no_grad()
    def evaluate(split):
        sh.eval(); ah.eval()
        hit = n = 0; ce = 0.0; per = defaultdict(lambda: [0, 0])
        for q, c, tgt, lab, qids in data[split]:
            lg = logits(q, c)
            pred = lg.argmax(-1)
            ce += -(tgt * F.log_softmax(lg, -1)).sum(-1).sum().item()
            ok = (pred == lab).tolist()
            hit += sum(ok); n += len(ok)
            for qid, o in zip(qids, ok):
                per[qid][0] += o; per[qid][1] += 1
        sh.train(); ah.train()
        return {"acc": hit / max(1, n), "soft_ce": ce / max(1, n),
                "per_question": {k: round(h / t, 4) for k, (h, t) in sorted(per.items())}}

    def majority_baseline(split):
        counts = defaultdict(lambda: defaultdict(int))
        for e in ex["train"]:
            counts[e.qid][e.label] += 1
        hit = sum(e.label == max(counts[e.qid], key=counts[e.qid].get) for e in ex[split] if counts[e.qid])
        return hit / max(1, len(ex[split]))

    params = list(sh.parameters()) + list(ah.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=args.lr or 5e-4, weight_decay=args.weight_decay)
    n_train = sum(len(b[3]) for b in data["train"])
    per_epoch = (math.ceil(n_train / args.batch) if args.loss == "infonce" else
                 sum(math.ceil(len(b[3]) / args.batch) for b in data["train"]))
    steps = max(1, per_epoch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr or 5e-4, total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    base = {"majority_val": majority_baseline("val"), "majority_test": majority_baseline("test")}
    m0 = {"val": evaluate("val"), "test": evaluate("test")}
    print(f"[choice] baselines {json.dumps(base)}", flush=True)
    print(f"[choice] epoch 0 (init) val acc {m0['val']['acc']:.4f} test acc {m0['test']['acc']:.4f}", flush=True)

    def blob(epoch, metrics):
        return {"state_head": sh.state_dict(), "action_head": ah.state_dict(),
                "logit_scale": logit_scale.detach().cpu(),
                "cfg": {**cfg, "projection_dim": proj, "hidden_size": HIDDEN, "task": "choice",
                        "targets": args.targets, "loss": args.loss, "data": args.data, "workflow": args.workflow,
                        "embed_model": args.embed_model, "max_len": args.max_len,
                        "init_ckpt": os.path.basename(args.init_ckpt) if args.init_ckpt else None},
                "epoch": epoch, "metrics": metrics}

    g = torch.Generator().manual_seed(args.seed)
    best, best_ep, bad, history = m0["val"]["acc"], 0, 0, [{"epoch": 0, **m0}]
    torch.save(blob(0, m0["val"]), os.path.join(args.out_dir, "best_head.pt"))
    for ep in range(1, args.epochs + 1):
        tot = nb = 0
        if args.loss == "infonce":  # batches mix candidate counts, so draw from the flat set
            order = [(None, idx) for idx in torch.randperm(len(st_idx), generator=g).split(args.batch)]
        else:
            order = [(bi, idx) for bi, b in enumerate(data["train"])
                     for idx in torch.randperm(len(b[3]), generator=g).split(args.batch)]
            random.Random(args.seed + ep).shuffle(order)
        for bi, idx in order:
            idx = idx.to(device)
            if args.loss == "infonce":
                loss = infonce_loss(idx)
            else:
                q, c, tgt, lab, _ = data["train"][bi]
                loss = loss_of(logits(q[idx], c[idx]), tgt[idx], lab[idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        mv = evaluate("val")
        history.append({"epoch": ep, "train_loss": tot / max(1, nb), "val": mv})
        print(f"[choice] epoch {ep} loss {tot/max(1,nb):.4f} val acc {mv['acc']:.4f} "
              f"soft_ce {mv['soft_ce']:.4f}", flush=True)
        torch.save(blob(ep, mv), os.path.join(args.out_dir, "final_head.pt"))
        if mv["acc"] > best + 1e-9:
            best, best_ep, bad = mv["acc"], ep, 0
            torch.save(blob(ep, mv), os.path.join(args.out_dir, "best_head.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[choice] early stop at epoch {ep}", flush=True)
                break
    best_ck = torch.load(os.path.join(args.out_dir, "best_head.pt"), map_location="cpu", weights_only=False)
    sh.load_state_dict(best_ck["state_head"]); ah.load_state_dict(best_ck["action_head"])
    with torch.no_grad():
        logit_scale.copy_(best_ck["logit_scale"].to(device))
    mt = evaluate("test")
    print(f"[choice] best epoch {best_ep}: val acc {best:.4f} | TEST acc {mt['acc']:.4f} "
          f"(init {m0['test']['acc']:.4f}, majority {base['majority_test']:.4f}) per_question {mt['per_question']}",
          flush=True)
    json.dump(history, open(os.path.join(args.out_dir, "history.json"), "w"), indent=1)
    return {"best_epoch": best_ep, "val_acc": best, "test": mt, "init_test": m0["test"], **base,
            "train_questions": n_train}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["clm", "choice"], required=True)
    ap.add_argument("--out-dir", required=True)
    src = ap.add_argument_group("data")
    src.add_argument("--benchmark", choices=sorted(hf_embeddings.TRAIN_DATASETS), default=None,
                     help="clm artifact profile; default: deepswe")
    src.add_argument("--emb-dir", help="clm: embedding dir to train on")
    src.add_argument("--hf-dataset",
                     help="clm: published native/parquet embedding dataset (HF id or local path); "
                          f"default: {hf_embeddings.DEEPSWE_TRAIN_DATASET}")
    src.add_argument("--data", help="clm: transitions .json/.jsonl (embedded from scratch); "
                                    "choice: typed-decisions HF id, local dir or parquet file")
    src.add_argument("--workflow", default="all", help="choice: typed-decisions config")
    src.add_argument("--hf-cache", default=None, help="Hugging Face download cache")
    src.add_argument("--hf-split", default=None, help="native HF embedding subdirectory, e.g. train")
    emb = ap.add_argument_group("from-scratch embedding")
    emb.add_argument("--embed-model", default="Qwen/Qwen3-8B")
    emb.add_argument("--embed-url", default=None, help="vLLM /v1/embeddings server; offline vLLM if unset")
    emb.add_argument("--served-model-name", default=None, help="model name the server was started with")
    emb.add_argument("--max-len", type=int, default=None, help="token budget (default 8192 clm, 2048 choice)")
    emb.add_argument("--gpu-mem", type=float, default=0.85)
    emb.add_argument("--embed-cache", default=None, help="where embeddings are cached (default OUT/embeddings)")
    hd = ap.add_argument_group("heads / optimisation")
    hd.add_argument("--init-ckpt", default=None, help="warm-start checkpoint")
    hd.add_argument("--width", type=int, default=None)
    hd.add_argument("--depth", type=int, default=None)
    hd.add_argument("--proj", type=int, default=512)
    hd.add_argument("--epochs", type=int, default=20)
    hd.add_argument("--batch", type=int, default=None, help="default 2048 clm, 256 choice")
    hd.add_argument("--lr", type=float, default=None, help="default: CLM width/batch rule, 5e-4 choice")
    hd.add_argument("--weight-decay", type=float, default=0.0)
    hd.add_argument("--val-frac", type=float, default=0.1)
    hd.add_argument("--patience", type=int, default=5)
    hd.add_argument("--seed", type=int, default=1234)
    hd.add_argument("--gpu", type=int, default=0)
    clm = ap.add_argument_group("clm only")
    clm.add_argument("--holdout-folds", nargs="+", default=None, metavar="FOLD_JSON",
                    help="train one head per fold (each holds out that fold's tasks) + write fold_spec.json")
    clm.add_argument("--holdout-tasks", default=None, help="single run holding out these tasks")
    clm.add_argument("--folds", type=int, default=None, metavar="K",
                    help="generate K stratified task folds from --fold-index and train them")
    clm.add_argument("--fold-index", default=None,
                    help="trial index used by --folds; strata are candidate count and pass count")
    clm.add_argument("--fold-config", default=None,
                    help="optional config/job/effort value selected from --fold-index")
    clm.add_argument("--sampler", choices=["random", "task-blocked"], default="random")
    clm.add_argument("--tasks-per-batch", type=int, default=4)
    clm.add_argument("--clm-select-metric", choices=["within_task_top1", "val_loss"],
                     default="within_task_top1")
    ch = ap.add_argument_group("choice only")
    ch.add_argument("--targets", choices=["soft", "hard"], default="soft",
                    help="train on annotator distributions (soft) or gold labels (hard)")
    ch.add_argument("--loss", choices=["infonce", "softce"], default="infonce",
                    help="bidirectional in-batch InfoNCE over the batch's distinct option texts (infonce) "
                         "or a softmax over each question's own options (softce)")
    args = ap.parse_args()

    if args.epochs < 1 or args.patience < 1:
        ap.error("--epochs and --patience must be positive")
    if args.batch is not None and args.batch < 2:
        ap.error("--batch must be at least 2")

    if args.task == "clm":
        args.benchmark = args.benchmark or "deepswe"
        sources = sum(bool(x) for x in (args.emb_dir, args.hf_dataset, args.data))
        if sources > 1:
            ap.error("clm accepts only one of --emb-dir, --hf-dataset, --data")
        if sources == 0:
            args.hf_dataset = hf_embeddings.TRAIN_DATASETS[args.benchmark]
            args.hf_split = args.hf_split or hf_embeddings.TRAIN_SPLITS[args.benchmark]
        if sum(bool(x) for x in (args.holdout_folds, args.holdout_tasks, args.folds)) > 1:
            ap.error("--holdout-folds, --holdout-tasks, and --folds are mutually exclusive")
        if args.folds is not None and args.folds < 2:
            ap.error("--folds must be at least 2")
        if bool(args.folds) != bool(args.fold_index):
            ap.error("--folds and --fold-index must be supplied together")
        if args.fold_config and not args.folds:
            ap.error("--fold-config requires --folds")
        args.max_len = args.max_len or 8192
        args.batch = args.batch or 2048
    else:
        if args.benchmark or args.hf_split:
            ap.error("--benchmark/--hf-split are clm options")
        if not args.data:
            ap.error("choice needs --data (e.g. LocalLLaMA/typed-decisions)")
        if (args.emb_dir or args.hf_dataset or args.holdout_folds or args.holdout_tasks or
                args.folds or args.fold_index or args.fold_config):
            ap.error("--emb-dir/--hf-dataset/--holdout-*/--fold-* are clm options")
        args.max_len = args.max_len or 2048
        args.batch = args.batch or 256
    for p in (args.init_ckpt, *(args.holdout_folds or []), args.holdout_tasks, args.fold_index):
        if p and not os.path.exists(p):
            ap.error(f"file not found: {p}")
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    res = run_clm(args) if args.task == "clm" else run_choice(args)
    res["minutes"] = round((time.time() - t0) / 60, 2)
    json.dump({**res, "args": vars(args)}, open(os.path.join(args.out_dir, "finetune_summary.json"), "w"),
              indent=1, default=str)
    print(f"[done] {json.dumps(res, default=str)}", flush=True)


if __name__ == "__main__":
    main()
