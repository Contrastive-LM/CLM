"""Where a player thinks: the planner, the prompt and the model call, in a process of its own.

Running each player in a separate process keeps its latency its own. In one Python process
the window, the course designer and the other player compete for the interpreter lock, and
a model call made of many short Python steps can wait hundreds of milliseconds for it.
"""

import multiprocessing
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .backends import build_question, create, decide, prefer_performance_cores
from .planner import ACTIONS, Planner


def same_effect(a, b, airborne):
    """Mid-air, up and nothing do the same thing."""
    return a == b or airborne and {a, b} <= {"jump", "run"}


def think(planner, backend, snap, timing, prompt, guarded):
    """`timing` is (first, gap, period) in frames; see Planner.plan."""
    started = time.perf_counter()
    plan = planner.plan(snap, *timing)
    plan_ms = (time.perf_counter() - started) * 1000
    state, questions = build_question(plan, prompt)
    answer = decide(backend, state, questions)
    p = answer.probabilities
    proposed = max(ACTIONS, key=lambda a: (p[a], -ACTIONS.index(a)))
    allowed = plan.safe_actions
    executed = proposed
    if guarded and allowed and proposed not in allowed:
        executed = max(allowed, key=p.__getitem__)
    return {
        "probabilities": p,
        "proposed": proposed,
        "executed": executed,
        "best": plan.best,
        "safe": plan.safe,
        "intervened": executed != proposed,
        "agreed": same_effect(proposed, plan.best, plan.airborne),
        "airborne": plan.airborne,
        "threat": plan.threat,
        "inference_ms": answer.inference_ms,
        "plan_ms": plan_ms,
        "robust": plan.robust,
        "plan_states": plan.states,
        "input_tokens": answer.input_tokens,
        "model": backend.model,
    }


def warm(planner, backend, snap, prompt, count=4):
    plan = planner.plan(snap, (1, 1), (1, 1))
    state, questions = build_question(plan, prompt)
    inflight = getattr(backend, "inflight", 1)
    if inflight > 1:  # Open every connection before play, so none pays for a handshake mid-game.
        with ThreadPoolExecutor(inflight) as pool:
            list(pool.map(lambda _: decide(backend, state, questions), range(inflight)))
    return [decide(backend, state, questions).inference_ms for _ in range(count)]


class LocalBrain:
    """Thinks in the calling process. Used by tests and short scripts."""

    def __init__(self, kind, **options):
        self.backend = create(kind, **options)
        self.planner = Planner()
        self.inflight = 1
        self.describe(self.backend)

    def describe(self, backend):
        self.name = backend.name
        self.model = backend.model
        self.detail = backend.detail
        self.usd_per_token = backend.usd_per_token

    def think(self, snap, timing, prompt, guarded):
        result = think(self.planner, self.backend, snap, timing, prompt, guarded)
        self.describe(self.backend)
        return result

    def warm(self, snap, prompt):
        return warm(self.planner, self.backend, snap, prompt)

    def close(self):
        self.backend.close()


def serve(kind, options, conn):
    prefer_performance_cores()
    sys.setswitchinterval(0.001)  # Planner threads must not hold the interpreter from the model.
    try:
        backend = create(kind, **options)
    except Exception as error:
        conn.send(("error", str(error)))
        return
    planner = Planner()
    conn.send(("ready", backend.name, backend.model, backend.detail, backend.usd_per_token))
    sending = threading.Lock()

    def handle(ticket, command, payload):
        try:
            work = think if command == "think" else warm
            reply = (ticket, "ok", work(planner, backend, *payload), backend.model, backend.detail)
        except Exception as error:
            reply = (ticket, "error", str(error)[:300], backend.model, backend.detail)
        with sending:
            conn.send(reply)

    # One thread per request in flight. A local model still answers one at a time.
    workers = max(1, options.get("inflight", 1))
    with ThreadPoolExecutor(workers, initializer=prefer_performance_cores) as pool:
        while (message := conn.recv()) is not None:
            pool.submit(handle, *message)
    backend.close()


class RemoteBrain:
    """Thinks in a child process. Any number of threads may call it at once."""

    def __init__(self, kind, **options):
        context = multiprocessing.get_context("spawn")
        self.conn, child = context.Pipe()
        self.process = context.Process(
            target=serve, args=(kind, options, child), name=f"trex-{kind}", daemon=True
        )
        self.process.start()
        child.close()
        reply = self.conn.recv()
        if reply[0] == "error":
            self.process.join(timeout=5)
            raise RuntimeError(reply[1])
        _, self.name, self.model, self.detail, self.usd_per_token = reply
        self.inflight = max(1, options.get("inflight", 1))
        self.lock = threading.Lock()
        self.tickets = 0
        self.waiting = {}
        self.reader = threading.Thread(target=self.read, name=f"brain-{kind}", daemon=True)
        self.reader.start()

    def read(self):
        while True:
            try:
                ticket, status, result, model, detail = self.conn.recv()
            except (EOFError, OSError):
                status, result = "error", "The player process stopped"
                with self.lock:
                    waiting, self.waiting = list(self.waiting.values()), None
                for box in waiting:
                    box.put((status, result))
                return
            self.model, self.detail = model, detail
            with self.lock:
                box = self.waiting.pop(ticket, None)
            if box:
                box.put((status, result))

    def call(self, command, payload):
        box = queue.SimpleQueue()
        with self.lock:
            if self.waiting is None:
                raise RuntimeError("The player process stopped")
            self.tickets += 1
            self.waiting[self.tickets] = box
            self.conn.send((self.tickets, command, payload))
        status, result = box.get()
        if status == "error":
            raise RuntimeError(result)
        return result

    def think(self, snap, timing, prompt, guarded):
        return self.call("think", (snap, timing, prompt, guarded))

    def warm(self, snap, prompt):
        return self.call("warm", (snap, prompt))

    def close(self):
        try:
            with self.lock:
                self.conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
