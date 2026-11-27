"""Data adapters for the trainer (``clm.trainer``) and ``clm.finetune``.

Training data is a list of JSON objects (a ``.jsonl`` or ``.json`` file). Its
shape picks the task:

``clm``: a state and the action that goes with it. Every other action in the
batch is a negative.

    {"state": [...chat messages...], "action": "pytest -q", "task_id": "django-1234", "step_idx": 7,
     "trajectory_id": "run3", "reward": 1}                      # agent step traces
    {"input": "What causes tides?", "answer": "The Moon's gravity."}       # plain pairs

``choice``: a state and a closed set of candidates with a gold answer.

    {"state": "...", "candidates": ["patch A", "patch B", "patch C"], "rewards": [1, 0, 0]}
    {"state": "...", "candidates": ["billing", "technical"], "label": "billing"}
    {"question": "...", "answer": "Paris", "negatives": ["Lyon", "Nice"]}

* ``read_transitions`` / ``as_transitions``: ``clm`` records. Step transitions as
  written by the DeepSWE trace builder in the research repo's ``main`` branch are
  returned unchanged; plain pairs get a task and a step of their own.
* ``choice_examples``: ``choice`` records -> ``ChoiceExample``.
* ``typed_decision_examples``: System One typed questions
  (``LocalLLaMA/typed-decisions``); each (row, question) becomes a state text, option
  keys, candidate texts (``clm.schema.build_pairs``) and the gold distribution.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

from .schema import build_pairs, to_text

TRANSITION_KEYS = ("state", "action", "task_id", "step_idx")
STATE_KEYS = ("state", "input", "problem", "question", "prompt")
ACTION_KEYS = ("action", "answer", "output", "response")
CANDIDATE_KEYS = ("candidates", "options")
NEGATIVE_KEYS = ("negatives", "hard_negatives")
GOLD_KEYS = ("label", "rewards", "target")


def _first(rec: Mapping, keys) -> Any:
    for k in keys:
        if rec.get(k) is not None:
            return rec[k]
    return None


def is_records_file(path: Any) -> bool:
    return isinstance(path, str) and path.endswith((".json", ".jsonl")) and os.path.isfile(path)


def read_records(path: str) -> list[dict]:
    """Load records from a ``.jsonl`` (one object per line) or ``.json`` (array) file."""
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()] if path.endswith(".jsonl") else json.load(f)
    if not isinstance(recs, list) or not all(isinstance(r, dict) for r in recs):
        raise ValueError(f"{path}: expected a list of JSON objects")
    return recs


def task_of(records: list[dict]) -> str:
    """``"clm"`` (state/action records) or ``"choice"`` (records with candidates or typed questions)."""
    if not records:
        raise ValueError("the dataset is empty")

    def one(r: dict) -> str:
        choice = ("questions" in r and "gold" in r) or _first(r, CANDIDATE_KEYS) is not None \
            or _first(r, NEGATIVE_KEYS) is not None
        return "choice" if choice else "clm"

    tasks = {one(r) for r in records}
    if len(tasks) > 1:
        raise ValueError("the dataset mixes state/action records with candidate records; "
                         "give every record the same shape")
    return tasks.pop()


def is_messages(state: Any) -> bool:
    return isinstance(state, list) and bool(state) and all(isinstance(m, dict) and "role" in m for m in state)


def as_transitions(records: list[dict], where: str = "dataset") -> list[dict]:
    """``clm`` records -> step transitions (``state``, ``action``, ``task_id``, ``step_idx``).

    A record that already has those four keys is returned unchanged. A plain pair
    (``input`` / ``answer`` and the other names in ``STATE_KEYS`` / ``ACTION_KEYS``)
    becomes a task of its own, and pairs that share an answer share a ``group`` so
    that an identical answer is never used as a negative. Records of one
    ``task_id`` without a ``step_idx`` are numbered in order.
    """
    out, next_step = [], {}
    for i, r in enumerate(records):
        if all(k in r for k in TRANSITION_KEYS):
            out.append(r)
            continue
        state, action = _first(r, STATE_KEYS), _first(r, ACTION_KEYS)
        if state is None or action is None:
            raise ValueError(f"{where}: record {i} needs a state (one of {STATE_KEYS}) and an action "
                             f"(one of {ACTION_KEYS}); got keys {sorted(r)}")
        t = {k: v for k, v in r.items() if k not in STATE_KEYS and k not in ACTION_KEYS}
        t["state"] = state if isinstance(state, str) or is_messages(state) else to_text(state)
        t["action"] = to_text(action)
        if r.get("task_id") is None:
            t["task_id"], t["step_idx"] = f"pair-{i}", 0
            t["group"] = hashlib.sha1(t["action"].encode()).hexdigest()[:16]
        else:
            step = r.get("step_idx")
            if step is None:
                step = next_step.get(r["task_id"], 0)
                next_step[r["task_id"]] = step + 1
            t["step_idx"] = step
        t.setdefault("trajectory_id", t["task_id"])
        out.append(t)
    return out


def read_transitions(path: str) -> list[dict]:
    """Load ``clm`` records from a ``.jsonl`` / ``.json`` file as step transitions."""
    return as_transitions(read_records(path), path)


def _maybe_json(x):
    """Decode JSON-encoded objects; other strings are returned as is."""
    if isinstance(x, str):
        try:
            v = json.loads(x)
        except ValueError:
            return x
        return v if isinstance(v, (dict, list)) else x
    return x


def _key_of(label) -> str:
    if isinstance(label, bool):
        return "true" if label else "false"
    return str(label)


@dataclass
class ChoiceExample:
    qid: str
    state_text: str
    keys: list[str]
    candidates: list[str]
    target: list[float]      # gold distribution over ``keys``
    label: int               # index of the gold label in ``keys``
    group: str               # source row id
    workflow: str | None = None
    meta: dict = field(default_factory=dict)


def typed_decision_examples(rows: Iterable[dict]) -> Iterator[ChoiceExample]:
    """Rows with ``state``, ``questions`` (System One wire format) and ``gold``
    (``{qid: {"label", "probabilities"}}``) -> one ChoiceExample per answered question."""
    for r in rows:
        state, questions, gold = _maybe_json(r["state"]), _maybe_json(r["questions"]), _maybe_json(r["gold"])
        for qid, (stext, keys, cands) in build_pairs(state, questions).items():
            g = gold.get(qid) if isinstance(gold, dict) else None
            if not g or "label" not in g:
                continue
            label = _key_of(g["label"])
            if label not in keys:
                raise ValueError(f"row {r.get('id')}: gold label {label!r} not among options {keys} for {qid!r}")
            probs = {_key_of(k): float(v) for k, v in (g.get("probabilities") or {}).items()}
            target = [probs.get(k, 0.0) for k in keys]
            z = sum(target)
            target = [t / z for t in target] if z > 0 else [float(k == label) for k in keys]
            yield ChoiceExample(qid=qid, state_text=stext, keys=keys, candidates=cands, target=target,
                                label=keys.index(label), group=str(r.get("id", "")),
                                workflow=r.get("workflow"))


def _gold(rec: dict, keys: list[str], texts: list[str], where: str) -> tuple[list[float], int]:
    """Gold distribution over ``keys`` and the index of the gold label; ``([], -1)`` when no
    candidate is correct."""
    k = len(keys)
    weights = rec.get("target") if rec.get("target") is not None else rec.get("rewards")
    if weights is not None:
        vals = ([float(weights.get(key, 0.0)) for key in keys] if isinstance(weights, Mapping)
                else [max(0.0, float(v)) for v in weights])
        if len(vals) != k:
            raise ValueError(f"{where}: {len(vals)} gold values for {k} candidates")
        z = sum(vals)
        if z <= 0:
            return [], -1
        return [v / z for v in vals], max(range(k), key=vals.__getitem__)
    label = rec.get("label")
    if label is None:
        raise ValueError(f"{where}: candidates need a gold answer: add one of {GOLD_KEYS}")
    if isinstance(label, bool) or not isinstance(label, int):
        label = keys.index(str(label)) if str(label) in keys else texts.index(str(label)) \
            if str(label) in texts else -1
    if not 0 <= label < k:
        raise ValueError(f"{where}: label {rec['label']!r} is not one of the {k} candidates")
    return [float(j == label) for j in range(k)], label


def choice_examples(records: Iterable[dict], where: str = "dataset") -> Iterator[ChoiceExample]:
    """``choice`` records -> one ChoiceExample each (typed-decision rows -> one per question).

    The gold answer is ``label`` (an index, a key or the candidate text), ``rewards``
    (one number per candidate, e.g. which attempts passed) or ``target`` (a
    distribution). ``{"answer": ..., "negatives": [...]}`` is the positive followed
    by its hard negatives. Records with no correct candidate are skipped. State and
    candidate texts are built by ``clm.schema.build_pairs``, exactly as
    ``Engine.rank(state, candidates, instructions)`` builds them at inference.
    """
    for i, r in enumerate(records):
        if "questions" in r and "gold" in r:
            yield from typed_decision_examples([r])
            continue
        at = f"{where}: record {i}"
        state, cands = _first(r, STATE_KEYS), _first(r, CANDIDATE_KEYS)
        if state is None:
            raise ValueError(f"{at} needs a state (one of {STATE_KEYS}); got keys {sorted(r)}")
        rec = r
        if cands is None:
            positive, negatives = _first(r, ACTION_KEYS), _first(r, NEGATIVE_KEYS)
            if positive is None or not negatives:
                raise ValueError(f"{at} needs `candidates`, or an answer with `negatives`")
            cands, rec = [positive, *negatives], {"label": 0}
        criteria = ({str(k): v for k, v in cands.items()} if isinstance(cands, Mapping)
                    else {str(j): c for j, c in enumerate(cands)})
        if len(criteria) < 2:
            raise ValueError(f"{at} needs at least two candidates")
        question = {"type": "choice", "instructions": r.get("instructions"), "criteria": criteria}
        stext, keys, texts = build_pairs(state, {"q": question})["q"]
        target, label = _gold(rec, keys, texts, at)
        if label < 0:
            continue
        yield ChoiceExample(qid=str(r.get("qid", "choice")), state_text=stext, keys=keys, candidates=texts,
                            target=target, label=label, group=str(r.get("id", r.get("task_id", i))),
                            workflow=r.get("workflow"))

