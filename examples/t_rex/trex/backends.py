"""The decision model behind a T-Rex player: any System One endpoint (CLM or TypeSafe Jev).

Both receive the same state and the same Choice question, built from the planner's labels, and
return probabilities over jump, duck and run.  The request is the TypeSafe wire format, which
``clm-serve`` speaks too, so one backend class covers both; only the base URL and key differ.
"""

import math
import os
import time
from dataclasses import dataclass

from .planner import ACTIONS

PROMPTS = ("labeled", "guided")
JEV_USD_PER_TOKEN = 0.042 / 1_000_000  # jev-1.13 input price; output tokens are free.
ENDPOINTS = {
    "clm": {"name": "CLM", "base_url": ("CLM_BASE_URL", "http://127.0.0.1:8700"), "model": "clm-latest",
            "keys": ("CLM_API_KEY",), "usd_per_token": 0.0},
    "jev": {"name": "Jev", "base_url": ("TYPESAFE_BASE_URL", "https://api.typesafe.ai"), "model": "jev-latest",
            "keys": ("TYPESAFE_API_KEY", "JEV_API_KEY"), "usd_per_token": JEV_USD_PER_TOKEN},
}


def situation(plan):
    if plan.threat:
        ahead = f"{plan.threat[0].upper()}{plan.threat[1:]} ahead, {max(plan.distance, 0)} px away."
    elif plan.distance is not None and plan.ahead:
        ahead = f"{plan.ahead[0].upper()}{plan.ahead[1:]} ahead, {max(plan.distance, 0)} px away."
    else:
        ahead = "Nothing ahead."
    return f"The dino is in the air. {ahead}" if plan.airborne else ahead


def build_question(plan, prompt="labeled"):
    """The state and Choice question sent to the model. `labeled` never names the answer."""
    best, notes = plan.best, plan.notes
    facts = f"Dino runner game. {situation(plan)}"
    if prompt == "labeled":
        criteria = {
            a: f"Safe. {notes[a]}. Best."
            if a == best
            else f"Safe. {notes[a]}."
            if plan.safe[a]
            else f"Unsafe. {notes[a]}. Collision."
            for a in ACTIONS
        }
        instructions = "Choose the best safe action for the dinosaur."
    elif prompt == "guided":
        criteria = {
            a: f"Best: {notes[a]}, safe."
            if a == best
            else f"Safe: {notes[a]}."
            if plan.safe[a]
            else f"Collision: {notes[a]}."
            for a in ACTIONS
        }
        facts += f" Recommended action: {best}."
        instructions = "Which action should the dinosaur take now? Pick the recommended safe action."
    else:
        raise ValueError(f"prompt must be one of {PROMPTS}")
    return facts, {"action": {"type": "choice", "instructions": instructions, "criteria": criteria}}


@dataclass
class Answer:
    probabilities: dict
    inference_ms: float
    input_tokens: int
    model: str


def checked(probabilities):
    values = {a: float(probabilities.get(a, 0.0)) for a in ACTIONS}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError(f"Model returned an invalid probability: {probabilities}")
    return values


def decide(backend, state, questions):
    """Ask the player question and return its checked action probabilities."""
    answers, elapsed, tokens = backend.ask(state, questions)
    return Answer(checked(answers["action"]["probabilities"]), elapsed, tokens, backend.model)


def prefer_performance_cores():
    """Ask macOS to schedule the calling thread as user-interactive work. No-op elsewhere."""
    try:
        import ctypes

        ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(0x21, 0)
    except (OSError, AttributeError):
        pass


class ApiBackend:
    """One persistent HTTP connection pool to a System One endpoint; no retries, since a late
    answer is useless in real time (a failed call counts as an error in the report)."""

    def __init__(self, kind, model=None, timeout=5.0, inflight=1):
        import httpx

        spec = ENDPOINTS[kind]
        env, default = spec["base_url"]
        key = next((os.environ[k] for k in spec["keys"] if os.environ.get(k)), None)
        if kind == "jev" and not key:
            raise RuntimeError("Jev needs TYPESAFE_API_KEY (or JEV_API_KEY), e.g. in <repo>/.env")
        self.name = spec["name"]
        self.usd_per_token = spec["usd_per_token"]
        self.client = httpx.Client(
            base_url=os.environ.get(env) or default,
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=timeout,
            limits=httpx.Limits(max_connections=inflight + 1, max_keepalive_connections=inflight + 1),
        )
        self.inflight = inflight
        self.request_model = model or spec["model"]
        self.model = self.request_model
        self.detail = self.describe()

    def describe(self):
        text = f"{self.client.base_url} · {self.model}"
        return text + (f" · {self.inflight} requests in flight" if self.inflight > 1 else "")

    def ask(self, state, questions):
        started = time.perf_counter()
        response = self.client.post(
            "/v1/systemone", json={"model": self.request_model, "state": state, "questions": questions}
        )
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise RuntimeError(f"{self.name} HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        if self.model != body.get("model", self.model):
            self.model = body["model"]
            self.detail = self.describe()
        return body["answers"], elapsed, body.get("usage", {}).get("input_tokens", 0)

    def close(self):
        self.client.close()


def create(kind, *, model=None, inflight=1):
    """`inflight` is how many requests the player keeps going at once; both endpoints answer them
    in parallel."""
    if kind not in ENDPOINTS:
        raise ValueError(f"Unknown backend {kind!r}; expected one of {list(ENDPOINTS)}")
    return ApiBackend(kind, model=model, inflight=inflight)
