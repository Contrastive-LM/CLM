"""Python client for the CLM System One API, shaped like ``typesafe_sdk``.

    from clm import CLMClient, Choice, Noul, Score

    client = CLMClient()                       # CLM_BASE_URL (default http://127.0.0.1:8700), CLM_API_KEY
    r = client.system_one(
        state="Customer: my invoice was charged twice and nobody answers the phone!",
        questions={
            "urgency": Noul(instructions="Is this urgent?"),
            "department": Choice(instructions="Which team should handle this?",
                                 criteria={"billing": "Charges, invoices, refunds",
                                           "technical": "Bugs and outages"}),
            "frustration": Score(instructions="How frustrated is the customer?",
                                 criteria=["Calm", "Frustrated", "Very angry"]),
        },
    )
    r.answers["urgency"].noul                  # 0.0–1.0
    r.answers["department"].choice             # "billing"
    r.answers["department"].probabilities      # {"billing": ..., "technical": ...}
    r.answers["frustration"].score             # 0.0–2.0, expected level

Questions may also be plain dicts in the wire format (``{"type": "choice", ...}``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8700"
DEFAULT_MODEL = "clm-latest"


# ----------------------------------------------------------------------------- questions
@dataclass
class Noul:
    """A yes/no question; the answer is the probability that it is true."""
    instructions: Any = None
    criteria: dict[str, Any] | None = None   # optional {"true": ..., "false": ...} descriptions

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            d["criteria"] = dict(self.criteria)
        return d


@dataclass
class Choice:
    """Pick one option; the answer is a distribution over ``criteria`` keys."""
    criteria: dict[str, Any]
    instructions: Any = None

    def to_dict(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass
class Score:
    """Rate on an ordered rubric; the answer is an expected level plus a distribution."""
    criteria: list[Any]
    instructions: Any = None

    def to_dict(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Noul | Choice | Score | dict


def question_to_dict(q: Question) -> dict:
    return q if isinstance(q, dict) else q.to_dict()


# ----------------------------------------------------------------------------- answers
@dataclass
class NoulAnswer:
    noul: float
    type: str = "noul"

    @property
    def probabilities(self) -> dict[str, float]:
        return {"false": 1.0 - self.noul, "true": self.noul}


@dataclass
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]
    type: str = "choice"


@dataclass
class ScoreAnswer:
    score: float
    confidence: float
    probabilities: dict[str, float]
    legend: dict[str, Any] = field(default_factory=dict)
    type: str = "score"


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def parse_answer(d: dict) -> Answer:
    t = d.get("type")
    if t == "noul":
        return NoulAnswer(noul=float(d["noul"]))
    if t == "choice":
        return ChoiceAnswer(choice=d["choice"], confidence=float(d["confidence"]), probabilities=dict(d["probabilities"]))
    if t == "score":
        return ScoreAnswer(score=float(d["score"]), confidence=float(d["confidence"]),
                           probabilities=dict(d["probabilities"]), legend=dict(d.get("legend", {})))
    raise ValueError(f"unknown answer type {t!r}")


@dataclass
class Usage:
    billing_units: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class SystemOneResponse:
    model: str
    answers: dict[str, Answer]
    usage: Usage
    latency_ms: float | None = None


class CLMError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status, self.message = status, message


# ----------------------------------------------------------------------------- client
class CLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 300.0,
                 model: str = DEFAULT_MODEL):
        self.base_url = (base_url or os.environ.get("CLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("CLM_API_KEY")
        self.timeout, self.model = timeout, model
        self._s = requests.Session()
        if self.api_key:
            self._s.headers["Authorization"] = f"Bearer {self.api_key}"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._s.close()

    def _post(self, path: str, body: dict) -> tuple[dict, requests.Response]:
        r = self._s.post(f"{self.base_url}{path}", json=body, timeout=self.timeout)
        if r.status_code != 200:
            try:
                msg = r.json().get("detail", r.text)
            except ValueError:
                msg = r.text
            raise CLMError(r.status_code, str(msg))
        return r.json(), r

    def system_one(self, state: Any, questions: dict[str, Question], model: str | None = None,
                   temperature: float | None = None) -> SystemOneResponse:
        """One request: every question answered against one state.  ``temperature``
        (server default 1.0) flattens (>1) or sharpens (<1) the distributions."""
        body: dict[str, Any] = {"state": state, "model": model or self.model,
                                "questions": {k: question_to_dict(q) for k, q in questions.items()}}
        if temperature is not None:
            body["temperature"] = temperature
        j, r = self._post("/v1/systemone", body)
        u = j.get("usage", {}) or {}
        return SystemOneResponse(model=j["model"], answers={k: parse_answer(a) for k, a in j["answers"].items()},
                                 usage=Usage(int(u.get("billing_units", 0) or 0), u.get("input_tokens"),
                                             u.get("output_tokens")),
                                 latency_ms=float(r.headers["X-CLM-Latency-Ms"]) if "X-CLM-Latency-Ms" in r.headers else None)

    def rank(self, context: Any, question: str | None, answers: list[str], model: str | None = None,
             temperature: float | None = None) -> list[dict]:
        """Rank ``answers`` for ``context`` + ``question``; -> [{rank, candidate, prob}] best first."""
        body: dict[str, Any] = {"context": context, "question": question, "answers": list(answers),
                                "model": model or self.model}
        if temperature is not None:
            body["temperature"] = temperature
        j, _ = self._post("/v1/rank", body)
        return j["ranked"]

    def models(self) -> list[dict]:
        r = self._s.get(f"{self.base_url}/v1/models", timeout=self.timeout)
        if r.status_code != 200:
            raise CLMError(r.status_code, r.text)
        return r.json()["models"]

    def health(self) -> bool:
        try:
            return bool(self._s.get(f"{self.base_url}/health", timeout=5).json().get("ok"))
        except Exception:  # noqa: BLE001
            return False
