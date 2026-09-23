"""Shared question/answer schema for the CLM System One API.

The wire format is the TypeSafe ``POST /v1/systemone`` shape: a ``state``
(string, object or array) and a map of typed ``questions`` (``noul``,
``choice``, ``score``), answered with probability distributions.  This module
holds everything that both the server and the trainer must agree on:

* how a (state, question) pair is turned into the *state text* the state head
  sees and the *candidate texts* the action head sees (``build_pairs``), and
* how per-candidate similarities become an Answer of the right type
  (``answer_from_logits``).

The CLM scores each candidate by the scaled cosine between the projected state
and the projected candidate, exactly as in the InfoNCE training objective, and
a softmax over a question's candidates is that question's distribution.
"""
from __future__ import annotations

import json
import math
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")
NOUL_KEYS = ("false", "true")


def to_text(x: Any, indent: int = 0) -> str:
    """Render a state / description that may be a string, object or array as plain text.

    The heads are trained on prose, not on JSON, so an object becomes ``key: value``
    fields (top-level fields separated by a blank line, nested ones indented) and an
    array becomes one ``- item`` line per element.  Key order is preserved.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    pad = " " * indent
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}{k}:\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}{k}: {to_text(v)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(x, (list, tuple)):
        parts = []
        for v in x:
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}-\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}- {to_text(v)}")
        return "\n".join(parts)
    return json.dumps(x, ensure_ascii=False)


def state_text(state: Any, instructions: Any) -> str:
    """Context first, question last — the layout the heads were trained on."""
    s, i = to_text(state).strip(), to_text(instructions).strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


def _norm_question(q: dict) -> dict:
    t = q.get("type")
    if t not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {t!r}; expected one of {QUESTION_TYPES}")
    return q


def candidates(q: dict) -> tuple[list[str], list[str]]:
    """-> (option keys in answer order, candidate text per option)."""
    q = _norm_question(q)
    t, crit, ins = q["type"], q.get("criteria"), to_text(q.get("instructions")).strip()
    if t == "choice":
        if not isinstance(crit, dict) or not crit:
            raise ValueError("choice question needs a non-empty 'criteria' object")
        # The action head embeds the option's own text: its description when one is
        # given, else the key.  Nothing is prefixed, so a candidate reaches the encoder
        # exactly as the caller wrote it (the heads are trained on plain answer text).
        keys = list(crit)
        texts = [to_text(crit[k]) if crit[k] not in (None, "") else k for k in keys]
        return keys, texts
    if t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError("score question needs 'criteria' as an ordered list of >= 2 levels")
        keys = [str(i) for i in range(len(crit))]
        return keys, [to_text(c) for c in crit]
    # noul: optional {"true": ..., "false": ...} descriptions; default to the statement itself
    crit = crit or {}
    texts = []
    for k in NOUL_KEYS:
        d = crit.get(k) if isinstance(crit, dict) else None
        if d in (None, ""):
            d = (f"Yes. This is true: {ins}" if k == "true" else f"No. This is false: {ins}") if ins else k
        texts.append(f"{k}: {to_text(d)}")
    return list(NOUL_KEYS), texts


def build_pairs(state: Any, questions: dict[str, dict]) -> dict[str, tuple[str, list[str], list[str]]]:
    """{qid: (state_text, option_keys, candidate_texts)} for every question.

    The state head sees ``context + question``: the state is the context and the
    question's instructions are the question, appended after a blank line.  That is
    the layout the heads are trained on, so callers should put the question in
    ``instructions`` and not repeat it inside the state.
    """
    return {qid: (state_text(state, q.get("instructions")), *candidates(q)) for qid, q in questions.items()}


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    z = sum(e)
    return [v / z for v in e]


def confidence(probs: list[float]) -> float:
    """TypeSafe-style confidence: top probability minus the mean of the rest."""
    if len(probs) < 2:
        return 1.0
    j = max(range(len(probs)), key=probs.__getitem__)
    rest = [p for i, p in enumerate(probs) if i != j]
    return max(0.0, min(1.0, probs[j] - sum(rest) / len(rest)))


def answer_from_probs(q: dict, keys: list[str], probs: list[float]) -> dict:
    """Assemble the Answer object for question ``q`` from its option distribution."""
    t = q["type"]
    probs = [float(p) for p in probs]
    dist = {k: p for k, p in zip(keys, probs)}
    if t == "noul":
        return {"type": "noul", "noul": dist["true"]}
    if t == "choice":
        j = max(range(len(probs)), key=probs.__getitem__)
        return {"type": "choice", "choice": keys[j], "confidence": confidence(probs), "probabilities": dist}
    levels = q["criteria"]
    score = sum(i * p for i, p in enumerate(probs))
    legend = {str(i): c if isinstance(c, str) else to_text(c) for i, c in enumerate(levels)}
    return {"type": "score", "score": score, "confidence": confidence(probs), "legend": legend,
            "probabilities": dist}


def answer_from_logits(q: dict, keys: list[str], logits: list[float]) -> dict:
    return answer_from_probs(q, keys, softmax([float(v) for v in logits]))


def label_of(answer: dict) -> str:
    """Discrete label of an answer (for scoring): choice key, score level, or 'true'/'false'."""
    t = answer["type"]
    if t == "choice":
        return answer["choice"]
    if t == "noul":
        return "true" if answer["noul"] >= 0.5 else "false"
    p = answer["probabilities"]
    return max(p, key=p.__getitem__)


def probabilities_of(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return dict(answer["probabilities"])
