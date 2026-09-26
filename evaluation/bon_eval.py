#!/usr/bin/env python3
"""Evaluate a CLM trajectory selector on any compatible best-of-N dataset.

Every benchmark follows the same path: score each step with a CLM head, use the
mean score over the final ``--window`` steps as the trajectory score, and
select the highest-scoring candidate. Candidate budget, inputs, checkpoints, and
window are explicit command-line arguments; there are no benchmark profiles or
expected-result constants. Exact score ties use uniform expectation and short
candidate groups are retained.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "preprocessing"))
import hf_embeddings  # noqa: E402
from clm.heads import HIDDEN, make_head  # noqa: E402  (released checkpoint architecture)


def load_heads(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    w = cfg.get("width", 1536); d = cfg.get("depth", 3)
    hk = dict(activation=cfg.get("activation", "gelu"),
              layernorm=cfg.get("layernorm", True),
              residual=cfg.get("residual", False))
    # the seeded init the experiments helper used is irrelevant: load_state_dict overwrites it
    hidden = cfg.get("hidden_size", HIDDEN)
    sh = make_head(w, d, cfg.get("projection_dim", 512), hidden=hidden, **hk).to(device)
    ah = make_head(w, d, cfg.get("projection_dim", 512), hidden=hidden, **hk).to(device)
    sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
    sh.eval(); ah.eval()
    return sh, ah, cfg


@torch.no_grad()
def step_scores(emb_dir, sh, ah, device, chunk=8192):
    se = torch.load(os.path.join(emb_dir, "state_embeddings.pt"), map_location="cpu")
    ae = torch.load(os.path.join(emb_dir, "action_embeddings.pt"), map_location="cpu")
    meta = json.load(open(os.path.join(emb_dir, "metadata.json")))
    out = []
    for i in range(0, len(se), chunk):
        zq = F.normalize(sh(se[i:i + chunk].float().to(device)), dim=-1)
        za = F.normalize(ah(ae[i:i + chunk].float().to(device)), dim=-1)
        out.append((zq * za).sum(-1).cpu())
    return torch.cat(out), meta["samples"]


def aggregate(scores, window):
    """Use one benchmark-independent trajectory score: final-window mean."""
    if not scores:
        raise ValueError("cannot aggregate an empty trajectory")
    tail = scores[-window:]
    return sum(tail) / len(tail)


def best_of_n(candidates, n):
    """Exact expectation for a uniform N-subset and uniform breaking of top-score ties."""
    if not 1 <= n <= len(candidates):
        raise ValueError(f"invalid candidate budget N={n} for {len(candidates)} candidates")
    blocks = defaultdict(list)
    for score, reward in candidates:
        if not math.isfinite(score) or reward not in (0, 1):
            raise ValueError("finite scores and binary outcomes are required")
        blocks[score].append(reward)
    choose = lambda count: math.comb(count, n) if count >= n else 0
    denominator = choose(len(candidates))
    selected = 0.0
    lower = 0
    for _, rewards in sorted(blocks.items()):
        probability = (choose(lower + len(rewards)) - choose(lower)) / denominator
        selected += probability * sum(rewards) / len(rewards)
        lower += len(rewards)
    passed = sum(reward for _, reward in candidates)
    return {"selected": selected, "random": passed / len(candidates),
            "oracle": 1 - choose(len(candidates) - passed) / denominator}


def selection_rate(by_task, key, n):
    """Expected selected reward per task, with outcome-blind exact tie handling."""
    hits = 0.0
    picks = {}
    for task, trials in by_task.items():
        scores = [key(trial) for trial in trials]
        budget = min(n, len(trials))
        hits += best_of_n(list(zip(scores, [int(t["passed"]) for t in trials])), budget)["selected"]
        best = max(scores)
        # keys may be (task, policy) tuples; JSON needs strings
        picks["::".join(task) if isinstance(task, tuple) else task] = [
            trial["trial_name"] for trial, score in zip(trials, scores) if score == best]
    return hits / len(by_task), picks


def load_trials_index(path):
    """Normalize common trial-index layouts to one evaluation schema."""
    data = json.load(open(path))
    records = data if isinstance(data, list) else data.get("rows", data.get("records"))
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a list or an object containing rows/records")
    rows = {}
    for record in records:
        trajectory_id = record.get("trajectory_id") or record.get("id") or record.get("trial_name")
        task = record.get("task_name") or record.get("task_id")
        reward = record.get("passed")
        if reward is None:
            reward = record.get("reward")
        if reward is None:
            reward = (record.get("rewards") or {}).get("reward")
        if trajectory_id is None or task is None or reward is None:
            continue
        if trajectory_id in rows:
            raise ValueError(f"{path}: duplicate trajectory {trajectory_id}")
        rows[trajectory_id] = {
            "trial_name": trajectory_id, "task_name": task,
            "config": record.get("config") or record.get("job_id") or record.get("effort") or "default",
            "passed": float(reward) > 0.5,
        }
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--embeddings-dir", help="merged embedding dir (embed_shard.py + merge_embeddings.py)")
    src.add_argument("--hf-dataset", help="parquet embedding dataset (HF id or local path)")
    ap.add_argument("--cache-dir", default=os.path.join(os.path.expanduser("~"), ".cache", "clm", "embeddings"))
    ap.add_argument("--hf-split", default=None, help="native HF embedding subdirectory, e.g. evaluation")
    heads = ap.add_mutually_exclusive_group(required=True)
    heads.add_argument("--checkpoint", help="one head scoring every task")
    heads.add_argument("--fold-spec",
                       help='JSON [{"checkpoint": path, "tasks": [...]}, ...]: each fold\'s '
                            "held-out tasks are scored by the head that never saw them")
    ap.add_argument("--trials-index", default=None,
                    help="trials index (download_trials.py); default: task / config / pass come from "
                         "the embedding metadata (task_id, config, reward > 0.5)")
    ap.add_argument("--tasks-file", default=None,
                    help="optional JSON task allowlist; accepts a list or an object containing "
                         "tasks/heldout_tasks")
    ap.add_argument("--n", type=int, required=True, help="candidate budget per evaluation group")
    ap.add_argument("--window", type=int, default=12,
                    help="number of final step scores included in the mean (default: 12)")
    ap.add_argument("--output")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.n < 1:
        ap.error("--n must be positive")
    if args.window < 1:
        ap.error("--window must be positive")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.hf_dataset:
        slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.hf_dataset)
        args.embeddings_dir = hf_embeddings.download(
            args.hf_dataset, os.path.join(args.cache_dir, slug), split=args.hf_split)

    # (selector name) -> list of (checkpoint, task-set or None)
    plans = {}
    if args.checkpoint:
        plans["clm"] = [(args.checkpoint, None)]
    if args.fold_spec:
        spec = json.load(open(args.fold_spec))
        spec_dir = os.path.dirname(os.path.abspath(args.fold_spec))

        def resolve(p):  # relative checkpoint paths are relative to the spec file
            if os.path.isabs(p) or os.path.exists(p):
                return p
            return os.path.join(spec_dir, p)
        plans["clm_foldwise"] = [(resolve(f["checkpoint"]), set(f["tasks"])) for f in spec]
    scored = {}
    samples = None
    for name, parts in plans.items():
        acc = None
        for ckpt, tasks in parts:
            sh, ah, cfg = load_heads(ckpt, device)
            sc, samples = step_scores(args.embeddings_dir, sh, ah, device)
            if acc is None:
                acc = torch.full_like(sc, float("nan"))
            if tasks is None:
                acc = sc
            else:
                m = torch.tensor([s["task_id"] in tasks for s in samples])
                acc[m] = sc[m]
            print(f"[bon] {name} <- {os.path.basename(os.path.dirname(ckpt))}: "
                  f"{len(sc)} steps, mean sim {sc.mean():.4f} "
                  f"(width {cfg.get('width')}, depth {cfg.get('depth')})", flush=True)
        scored[name] = acc

    if args.trials_index:
        rows = load_trials_index(args.trials_index)
    else:
        rows = {}
        for s in samples:
            rows.setdefault(s["trajectory_id"], {
                "trial_name": s["trajectory_id"], "task_name": s["task_id"],
                "config": s.get("config") or "default",
                "passed": float(s.get("reward") or 0.0) > 0.5})

    allowed_tasks = None
    if args.tasks_file:
        task_data = json.load(open(args.tasks_file))
        if isinstance(task_data, dict):
            task_data = task_data.get("tasks", task_data.get("heldout_tasks"))
        if not isinstance(task_data, list) or not all(isinstance(task, str) for task in task_data):
            raise ValueError(f"{args.tasks_file}: expected a task list or tasks/heldout_tasks object")
        allowed_tasks = set(task_data)
        if len(allowed_tasks) != len(task_data):
            raise ValueError(f"{args.tasks_file}: duplicate task names")
        indexed_tasks = {row["task_name"] for row in rows.values()}
        missing_tasks = sorted(allowed_tasks - indexed_tasks)
        if missing_tasks:
            raise ValueError(f"{args.tasks_file}: {len(missing_tasks)} tasks absent from the index: "
                             f"{missing_tasks}")
        rows = {trial: row for trial, row in rows.items() if row["task_name"] in allowed_tasks}

    # group step scores by trajectory
    per_traj = defaultdict(lambda: defaultdict(list))
    order = defaultdict(list)
    for i, s in enumerate(samples):
        tn = s["trajectory_id"]
        order[tn].append(s["step_idx"])
        for name in scored:
            per_traj[name][tn].append(float(scored[name][i]))
    for tn, steps in order.items():
        if len(set(steps)) != len(steps):
            raise ValueError(f"duplicate step_idx in trajectory {tn}")
        permutation = sorted(range(len(steps)), key=steps.__getitem__)
        order[tn] = [steps[i] for i in permutation]
        for name in scored:
            per_traj[name][tn] = [per_traj[name][tn][i] for i in permutation]

    missing_embeddings = sorted(set(rows) - set(order))
    if missing_embeddings:
        print(f"[bon] index trajectories without embeddings: {len(missing_embeddings)}", flush=True)

    by_task = defaultdict(list)
    multi = len({rows[tn]["config"] for tn in order if tn in rows}) > 1
    for tn in order:
        if tn not in rows:
            continue
        row = rows[tn]
        # one best-of-N instance per (task, policy) when several policies are scored
        by_task[(row["task_name"], row["config"]) if multi else row["task_name"]].append({
            "trial_name": tn, "passed": bool(row["passed"]),
            "scores": {name: per_traj[name][tn] for name in scored},
        })
    short_groups = {}
    for task in by_task:
        by_task[task] = sorted(by_task[task], key=lambda t: t["trial_name"])
        if len(by_task[task]) < args.n:
            short_groups["::".join(task) if isinstance(task, tuple) else task] = len(by_task[task])
    if not by_task:
        raise ValueError("no evaluation groups")

    for name in list(scored):
        bad = [t for t, v in by_task.items()
               if any(any(x != x for x in tr["scores"][name]) for tr in v)]
        if bad:
            print(f"[bon] {name}: {len(bad)} tasks not covered by any fold head "
                  f"-> dropped from that selector's report", flush=True)
            for t in bad:
                for tr in by_task[t]:
                    tr["scores"].pop(name, None)

    n_tasks = len(by_task)
    n_roll = sum(len(v) for v in by_task.values())
    random_rate = sum(sum(t["passed"] for t in v) / len(v) for v in by_task.values()) / n_tasks
    oracle = sum(best_of_n([(0.0, int(t["passed"])) for t in v], min(args.n, len(v)))["oracle"]
                 for v in by_task.values()) / n_tasks
    allpass = sum(all(t["passed"] for t in v) for v in by_task.values())
    allfail = sum(not any(t["passed"] for t in v) for v in by_task.values())
    mixed = n_tasks - allpass - allfail

    res = {
        "n_tasks": n_tasks, "n_rollouts": n_roll,
        "N": args.n,
        "aggregation": {"type": "final_window_mean", "window": args.window},
        "instance_key": "(task, config)" if multi else "task",
        "n_configs": len({rows[tn]["config"] for tn in order if tn in rows}),
        "random_pick": random_rate, "oracle_any": oracle,
        "all_pass_tasks": allpass, "all_fail_tasks": allfail, "mixed_tasks": mixed,
        "embeddings_dir": args.embeddings_dir, "checkpoint": args.checkpoint,
        "tasks_file": args.tasks_file,
        "index_trajectories_without_embeddings": missing_embeddings,
        "embedded_trajectories_outside_index": len(set(order) - set(rows)),
        "short_groups_retained": short_groups, "selectors": {},
    }
    print(f"[bon] {n_tasks} {'(task,policy) instances' if multi else 'tasks'} x "
          f"{n_roll/n_tasks:.1f} rollouts | random {random_rate:.3f} "
          f"| oracle {oracle:.3f} | decidable {mixed}", flush=True)

    for name in scored:
        sub = {t: v for t, v in by_task.items() if all(name in tr["scores"] for tr in v)}
        if not sub:
            continue
        key = lambda trial, selector=name: aggregate(trial["scores"][selector], args.window)
        rate, picks = selection_rate(sub, key, args.n)
        label = f"{name}:last_{args.window}_mean"
        res["selectors"][label] = {"rate": rate, "n_tasks": len(sub),
                                    "resolved": round(rate * len(sub)), "picks": picks}
        print(f"[bon] {label} {rate:.3%} ({round(rate * len(sub))}/{len(sub)})", flush=True)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        json.dump(res, open(args.output, "w"), indent=1)
        print(f"[bon] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
