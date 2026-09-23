"""The decision loop that lets a model play one game, and the arena that runs games side by side.

In real time a Pilot keeps its model continuously busy: as soon as one answer is ready it
asks about the newest frame, telling the planner about the answer that has not landed yet.
Idle gaps would let the GPU clock down, and a person watching the screen does not wait
for the next frame to start thinking either. Each answer takes effect at the first frame
boundary after it arrives, so the model's full latency is part of play.

A hosted model answers requests in parallel, so its Pilot may keep several in flight,
asked a few frames apart. Every answer is still a full round trip old, but the player gets
a turn every few frames instead of one per round trip. An answer is asked on the premise
that nothing changes the keys before it lands. If another answer does, the premise is
gone and the late answer is discarded, never applied.

In lockstep the game instead freezes until each answer arrives, removing latency from play.
"""

import math
import queue
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace

from .engine import FRAME_MS, Game
from .planner import snapshot
from .safety import protect

RESTART_DELAY_MS = 1500


@dataclass
class Decision:
    seq: int
    run: int
    view_frame: int
    sent_at: float
    pending: str | None = None
    epoch: int = 0
    pending_seq: int | None = None
    probabilities: dict = field(default_factory=dict)
    proposed: str = "run"
    executed: str = "run"
    best: str = "run"
    safe: dict = field(default_factory=dict)
    intervened: bool = False
    agreed: bool = False
    airborne: bool = False
    threat: str | None = None
    inference_ms: float = 0.0
    plan_ms: float = 0.0
    robust: bool = True
    plan_states: int = 0
    input_tokens: int = 0
    error: str | None = None
    latency_ms: float = 0.0
    latency_frames: int = 0
    expected_first: tuple = (0, 1)
    expected_gap: tuple = (1, 1)
    expected_period: int | None = None
    arrival_intervened: bool = False


@dataclass(frozen=True)
class View:
    frame: int
    snap: object
    run: int
    applied: int
    epoch: int


class Pilot:
    def __init__(
        self, brain, *, guarded=True, prompt="labeled", lockstep=None, seed=7, course=None
    ):
        self.brain = brain
        self.guarded = guarded
        self.prompt = prompt
        self.lockstep = lockstep  # Frames per decision when the game waits for the model.
        self.game = Game(seed, course=course)
        self.held = "run"
        self.last_action = "run"
        self.seq = 0
        self.applied = 0
        self.accepted = 0
        self.accepted_actions = {}
        self.active = set()
        self.trace = deque(maxlen=120)
        self.events = deque(maxlen=240)
        self.event = "Ready"
        self.event_frame = 0
        self.discard_notice_frame = None
        self.survival_frames = 0
        self.seen_obstacles = set()
        # The epoch counts answers that changed the keys. An answer asked in one epoch
        # is only valid in that epoch, or the next if it knew which answer would end it.
        self.epoch = 0
        self.bump_seq = None
        self.dropped = deque(maxlen=32)
        self.inflight = 1 if lockstep else max(1, getattr(brain, "inflight", 1))
        self.last_asked = None
        self.ask_gaps = deque(maxlen=60)
        self.finished = deque(maxlen=16)
        self.view = None
        self.waiting = False
        self.closed = False
        self.cond = threading.Condition()
        self.results = queue.Queue()
        self.latency_frames = 1.0  # Frames between the view and the answer landing.
        self.interval_frames = 1.0  # Frames between consecutive answers landing.
        self.jitter_frames = 1.0  # How much later than the earliest an answer may land.
        self.typical_frames = 1.0
        self.observed = deque(maxlen=90)
        self.last_applied_frame = None
        self.last = None
        self.recent = deque(maxlen=40)
        self.since = lockstep or 0
        self.stats = Stats()
        self.threads = [
            threading.Thread(target=self.work, name=f"pilot-{brain.name}-{n}", daemon=True)
            for n in range(self.inflight)
        ]
        for thread in self.threads:
            thread.start()

    # -- Main thread -------------------------------------------------------------------
    def publish(self, frame):
        """Offer the model the current frame (called after each step, or when lockstep waits)."""
        game = self.game
        view = None
        if game.playing and not game.crashed:
            view = View(frame, snapshot(game, self.held), game.run_index, self.applied, self.epoch)
        with self.cond:
            self.view = view
            self.cond.notify_all()

    def collect(self, frame):
        """Apply every answer that has arrived, oldest first. Returns how many landed."""
        landed = 0
        while True:
            try:
                decision = self.results.get_nowait()
            except queue.Empty:
                return landed
            landed += 1
            self.active.discard(decision.seq)
            self.applied = max(self.applied, decision.seq)
            decision.latency_ms = (time.perf_counter() - decision.sent_at) * 1000
            decision.latency_frames = max(0, frame - decision.view_frame - 1)
            if not self.lockstep and decision.run == self.game.run_index and not self.game.crashed:
                self.observe_timing(decision.latency_frames, frame)
            if decision.error:
                self.stats.errors += 1
                self.stats.last_error = decision.error
                self.drop(decision, frame, "Answer failed")
                continue
            # Successful replies are billed even if their game has already ended.
            self.stats.tokens += decision.input_tokens
            self.stats.cost += decision.input_tokens * self.brain.usd_per_token
            if self.game.crashed or decision.run != self.game.run_index:
                self.drop(decision, frame, "Previous run answer discarded")
                continue
            if not self.premise_holds(decision):
                self.drop(decision, frame, "Late answer discarded")
                continue
            original = decision.executed
            if self.guarded:
                decision.executed = protect(snapshot(self.game, self.held), original)
                decision.arrival_intervened = decision.executed != original
                if decision.arrival_intervened:
                    self.stats.arrival_saves += 1
            self.accepted = decision.seq
            self.accepted_actions[decision.seq] = decision.executed
            if len(self.accepted_actions) > 64:
                del self.accepted_actions[min(self.accepted_actions)]
            self.note(
                frame,
                "Arrival shield saved it"
                if decision.arrival_intervened
                else f"Answer → {decision.executed}",
                decision,
            )
            if self.changes_keys(decision.executed):
                self.epoch += 1
                self.bump_seq = decision.seq
            self.apply(decision.executed)
            self.last = decision
            self.recent.append(decision)
            self.stats.record(decision)

    def observe_timing(self, latency, frame):
        """Forget a previous fast regime immediately, recover gradually within 16 replies."""
        latency = min(90, max(0, latency))
        if latency > self.latency_frames + max(4, self.jitter_frames * 2):
            self.observed.clear()
        self.observed.append(latency)
        recent = sorted(list(self.observed)[-16:])
        self.latency_frames = recent[max(0, len(recent) // 5)]
        self.typical_frames = recent[len(recent) // 2]
        self.jitter_frames = min(12, max(1, recent[-1] - self.latency_frames))
        if self.last_applied_frame is not None:
            gap = min(90, max(1, frame - self.last_applied_frame))
            self.interval_frames += 0.3 * (gap - self.interval_frames)
        self.last_applied_frame = frame

    @staticmethod
    def notice_priority(text):
        if "discarded" in text:
            return 1
        return 0 if text.startswith("Answer →") or text in ("Ready", "") else 2

    def note(self, frame, text, decision=None):
        priority = self.notice_priority(text)
        current_priority = self.notice_priority(self.event)
        discard = priority == 1
        cooling_down = (
            discard
            and self.discard_notice_frame is not None
            and (frame - self.discard_notice_frame < 300)
        )
        protected_notice = priority < current_priority and frame - self.event_frame < 60
        if not cooling_down and not protected_notice:
            self.event, self.event_frame = text, frame
            if discard:
                self.discard_notice_frame = frame
        event = {"frame": frame, "event": text}
        if decision is not None:
            event.update(
                {
                    "seq": decision.seq,
                    "requested_frame": decision.view_frame,
                    "proposed": decision.proposed,
                    "executed": decision.executed,
                    "expected_first": decision.expected_first,
                    "actual_frames": decision.latency_frames,
                    "actual_ms": round(decision.latency_ms, 2),
                    "error": decision.error,
                }
            )
        self.events.append(event)

    def drop(self, decision, frame, reason):
        self.dropped.append(decision.seq)
        self.stats.discarded += 1
        self.note(frame, reason, decision)

    def emergency(self, frame):
        """Separate, visible assist while an answer is in flight or has failed."""
        if not self.guarded or self.game.crashed or not self.game.playing:
            return
        action = protect(snapshot(self.game, self.held), self.held)
        if action != self.held:
            self.epoch += 1
            self.bump_seq = None
            self.apply(action)
            self.stats.emergency_saves += 1
            self.note(frame, f"Emergency shield → {action}")

    def capture_tick(self, frame):
        self.trace.append(
            {
                "frame": frame,
                "run": self.game.run_index,
                "score": self.game.score,
                "held": self.held,
                "trex": snapshot(self.game, self.held).trex,
                "trex_x": self.game.trex.x,
                "speed": self.game.speed,
                "obstacles": [(o.id, round(o.x, 1), o.y, o.label) for o in self.game.obstacles],
                "thinking": len(self.active),
            }
        )
        if self.game.playing and not self.game.crashed:
            self.survival_frames += 1
            for obstacle in self.game.obstacles:
                if obstacle.x + obstacle.width < self.game.trex.x:
                    if obstacle.id not in self.seen_obstacles:
                        self.seen_obstacles.add(obstacle.id)
                        self.note(
                            frame, "Perfect jump" if self.game.trex.jumping else "Obstacle cleared"
                        )

    def on_crash(self, frame):
        self.note(frame, "Crashed — replay")
        crash = {
            "frame": frame,
            "run": self.game.run_index,
            "score": self.game.score,
            "snapshot": asdict(snapshot(self.game, self.held)),
            "timing": self.timing(),
            "events": [e for e in self.events if frame - 120 < e["frame"] <= frame],
            "trace": [t for t in self.trace if frame - 120 < t["frame"] <= frame],
        }
        self.stats.crashes.append(crash)
        return crash

    def spectator(self, frame):
        return {
            "thinking": len(self.active),
            "last_action": self.last_action,
            "event": ""
            if self.notice_priority(self.event) == 1 and frame - self.event_frame >= 60
            else self.event,
            "event_frame": self.event_frame,
            "survival_frames": self.survival_frames,
            "emergency_saves": self.stats.emergency_saves,
            "arrival_saves": self.stats.arrival_saves,
        }

    def premise_holds(self, decision):
        """Whether the keys are as the planner assumed when this answer was asked for."""
        if decision.seq <= self.accepted or decision.pending_seq in self.dropped:
            return False
        if decision.pending_seq is not None and (
            self.accepted_actions.get(decision.pending_seq) != decision.pending
        ):
            return False
        if decision.epoch == self.epoch:
            return True
        return (
            self.epoch == decision.epoch + 1 and self.bump_seq == decision.pending_seq is not None
        )

    def changes_keys(self, action):
        if action == "jump":
            # A jump tap also releases down, even when already airborne.
            return not self.game.trex.jumping or self.held != "run"
        return action != self.held

    def apply(self, action):
        self.last_action = action
        game = self.game
        if action == "jump":
            if not game.trex.jumping:
                if game.trex.ducking:
                    game.release_duck()
                game.press_jump()
            self.held = "run"
        else:
            self.held = action

    def enforce(self):
        """Hold the keys the last answer asked for, frame by frame."""
        game, trex = self.game, self.game.trex
        if not game.playing or game.crashed:
            return
        if self.held == "duck":
            if trex.jumping:
                if not trex.speed_drop:
                    game.press_duck()
            elif not trex.ducking:
                game.press_duck()
        elif trex.speed_drop or not trex.jumping and trex.ducking:
            game.release_duck()

    def reset_run(self):
        self.held = "run"
        self.last_action = "run"
        self.event = "Ready"
        self.last_asked = None  # The pause between runs is not a gap between questions.
        self.epoch += 1  # Anything still in flight was asked about the run that ended.
        self.bump_seq = None
        self.last_applied_frame = None
        self.since = self.lockstep or 0
        self.waiting = False
        self.observed.clear()
        self.ask_gaps.clear()
        self.survival_frames = 0
        self.trace.clear()
        self.seen_obstacles.clear()
        with self.cond:
            self.view = None
            self.finished.clear()
            self.cond.notify_all()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.brain.close()

    # -- Worker threads ----------------------------------------------------------------
    @property
    def stagger(self):
        """Frames between questions, so the answers in flight land evenly spread."""
        # At most four questions per round trip: enough turns, and a hosted API's rate limit
        # is respected. Extra request slots only keep that schedule regular.
        return min(12, max(1, round(max(1.0, self.typical_frames) / min(self.inflight, 4))))

    def next_view(self):
        """Wait for a view to ask about; returns it with the one answer known to land first.

        With one request in flight, the next question is asked the moment an answer
        finishes, about the newest frame, in lockstep only once that answer has landed.
        With several, questions are asked `stagger` frames apart.
        """
        with self.cond:
            while not self.closed:
                view = self.view
                if view is not None:
                    unapplied = [d for d in self.finished if d.seq > view.applied]
                    pending = unapplied[0] if len(unapplied) == 1 else None
                    fresh = view.frame != self.last_asked
                    if self.inflight > 1:
                        due = (
                            self.last_asked is None or view.frame - self.last_asked >= self.stagger
                        )
                        ready = due and len(unapplied) <= 1
                    else:
                        ready = (not unapplied and fresh) or (pending and not self.lockstep)
                    if ready:
                        if self.last_asked is not None and view.frame > self.last_asked:
                            self.ask_gaps.append(view.frame - self.last_asked)
                        self.last_asked = view.frame
                        self.seq += 1
                        self.active.add(self.seq)
                        return view, pending, self.seq
                self.cond.wait(0.05)
        return None, None, None

    def timing(self):
        """(first, gap, period) for Planner.plan, in frames."""
        if self.lockstep:
            return ((0, 0), (self.lockstep, self.lockstep), None)
        landing = (self.latency_frames, self.latency_frames + self.jitter_frames)
        if self.inflight == 1:
            # Answers follow one another, so the gaps between them equal their landing times.
            return (landing, landing, None)
        return (landing, landing, self.longest_wait)

    @property
    def longest_wait(self):
        """Frames the player may go without a question, from the gaps seen lately. A question
        needs a free request slot, so this can exceed the stagger."""
        gaps = sorted(self.ask_gaps)
        seen = gaps[int(0.95 * (len(gaps) - 1))] if len(gaps) >= 10 else 2 * self.stagger
        return min(24, max(self.stagger, seen), 3 * self.stagger + 2)

    def work(self):
        while True:
            view, pending, seq = self.next_view()
            if view is None:
                return
            decision = Decision(seq, view.run, view.frame, time.perf_counter())
            decision.epoch = view.epoch
            if pending:
                decision.pending, decision.pending_seq = pending.executed, pending.seq
            try:
                snap = replace(view.snap, pending=decision.pending)
                timing = self.timing()
                decision.expected_first, decision.expected_gap, decision.expected_period = timing
                result = self.brain.think(snap, timing, self.prompt, self.guarded)
                result.pop("model", None)
                vars(decision).update(result)
            except Exception as error:  # Keep playing on a failed call; count it.
                decision.error = str(error)[:200]
                time.sleep(0.25)
            with self.cond:
                self.finished.append(decision)
                self.cond.notify_all()
            self.results.put(decision)


class Stats:
    def __init__(self):
        self.decisions = 0
        self.interventions = 0
        self.agreements = 0
        self.errors = 0
        self.discarded = 0
        self.last_error = None
        self.tokens = 0
        self.cost = 0.0
        self.latency = deque(maxlen=600)
        self.inference = deque(maxlen=600)
        self.times = deque(maxlen=120)
        self.scores = []
        self.all_latency = []
        self.all_inference = []
        self.all_plan = []
        self.best_effort = 0
        self.arrival_saves = 0
        self.emergency_saves = 0
        self.crashes = []

    def record(self, decision):
        self.decisions += 1
        self.best_effort += not decision.robust
        self.all_plan.append(decision.plan_ms)
        self.interventions += decision.intervened
        self.agreements += decision.agreed
        self.latency.append(decision.latency_ms)
        self.inference.append(decision.inference_ms)
        self.all_latency.append(decision.latency_ms)
        self.all_inference.append(decision.inference_ms)
        self.times.append(time.perf_counter())

    @property
    def rate(self):
        if len(self.times) < 2:
            return 0.0
        span = time.perf_counter() - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0

    @staticmethod
    def percentile(values, q):
        if not values:
            return None
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    def summary(self, game, brain):
        finished = self.scores
        return {
            "model": brain.model,
            "where": brain.detail,
            "runs_finished": len(finished),
            "deaths": game.deaths,
            "scores": finished,
            "current_score": game.score if not game.crashed else None,
            "best_score": max(finished + [game.score]),
            "mean_finished_score": round(statistics.mean(finished), 1) if finished else None,
            "decisions": self.decisions,
            "latency_ms_p50": rounded(self.percentile(self.all_latency, 0.5)),
            "latency_ms_p95": rounded(self.percentile(self.all_latency, 0.95)),
            "model_ms_p50": rounded(self.percentile(self.all_inference, 0.5)),
            "planner_ms_p50": rounded(self.percentile(self.all_plan, 0.5)),
            "planner_ms_p95": rounded(self.percentile(self.all_plan, 0.95)),
            "best_effort_decisions": self.best_effort,
            "shield_interventions": self.interventions,
            "arrival_saves": self.arrival_saves,
            "emergency_saves": self.emergency_saves,
            "crash_traces": len(self.crashes),
            "agreement_with_planner": round(self.agreements / self.decisions, 3)
            if self.decisions
            else None,
            "answers_discarded": self.discarded,
            "errors": self.errors,
            "last_error": self.last_error,
            "input_tokens": self.tokens,
            "usd": round(self.cost, 6),
        }


def rounded(value):
    return None if value is None else round(value, 1)


class Pacer:
    """Turns wall time into 60 FPS game frames.

    If the host stalls, the missed time is dropped rather than replayed in a burst: a
    burst would run the games blind, with no chance for an answer to land in between.
    Dropped time is reported, so a stalled host is visible in the results.
    """

    def __init__(self, limit=3):
        self.limit = limit
        self.last = time.perf_counter()
        self.owed = 0.0
        self.dropped_ms = 0.0

    def frames(self, paused=False):
        now = time.perf_counter()
        if not paused:
            self.owed += (now - self.last) * 1000
        self.last = now
        steps = int(self.owed // FRAME_MS)
        self.owed -= steps * FRAME_MS
        if steps > self.limit:
            self.dropped_ms += (steps - self.limit) * FRAME_MS
            steps = self.limit
        return steps


class Arena:
    """Games advanced together on one clock, each driven by its own Pilot."""

    def __init__(self, pilots, director=None, lockstep=None, round_seconds=0, rounds=0):
        self.pilots = pilots
        self.director = director
        self.lockstep = lockstep
        self.frame = 0
        self.started = False
        self.paused = False
        self.pacer = None
        self.game_frames = [0] * len(pilots)
        from .match import Match

        self.match = (
            Match([p.brain.name for p in pilots], round_seconds, rounds) if round_seconds else None
        )
        self.finished = False
        self.crash_events = []

    def advance(self):
        """Run the frames that are due. Call this often; returns how many ran."""
        if self.pacer is None:
            self.pacer = Pacer()
        steps = self.pacer.frames(self.paused or not self.started or self.finished)
        for _ in range(steps):
            self.tick()
        return steps

    def start(self):
        for pilot in self.pilots:
            pilot.game.press_jump()
            if not self.lockstep:
                pilot.publish(self.frame)
        self.started = True

    def demand_course(self):
        if not self.director:
            return
        ahead = self.director.lookahead
        for pilot in self.pilots:
            game = pilot.game
            self.director.demand(game.run_index, game.obstacle_index + ahead)
            self.director.demand(game.run_index + 1, ahead)

    def tick(self):
        """Advance every game by one 60 FPS frame of wall time."""
        if not self.started or self.paused or self.finished:
            return
        self.frame += 1
        self.demand_course()
        for i, pilot in enumerate(self.pilots):
            game = pilot.game
            if game.crashed:
                pilot.collect(self.frame)
                game.step()
                if game.crash_ms >= RESTART_DELAY_MS:
                    pilot.stats.scores.append(game.score)
                    pilot.reset_run()
                    game.restart()
                    pilot.publish(self.frame)
                continue
            if self.lockstep:
                self.tick_lockstep(i, pilot)
                continue
            pilot.collect(self.frame)
            pilot.emergency(self.frame)
            pilot.enforce()
            game.step()
            self.game_frames[i] += 1
            pilot.capture_tick(self.frame)
            if game.crashed:
                self.crash_events.append({"player": pilot.brain.name, **pilot.on_crash(self.frame)})
            pilot.publish(self.frame)
        if self.match and self.match.advance(
            self.frame,
            [p.game.score for p in self.pilots],
            [p.stats.arrival_saves + p.stats.emergency_saves for p in self.pilots],
        ):
            for pilot in self.pilots:
                game = pilot.game
                game.high_score = max(game.high_score, math.ceil(game.distance))
            self.finished = self.match.finished
            if self.finished:
                for pilot in self.pilots:
                    with pilot.cond:
                        pilot.view = None
                return
            # A new round starts both players on the same new course, regardless of deaths.
            run = max(p.game.run_index for p in self.pilots) + 1
            for pilot in self.pilots:
                if not pilot.game.crashed:
                    pilot.stats.scores.append(pilot.game.score)
                pilot.reset_run()
                pilot.game.run_index = run - 1
                pilot.game.crashed = True
                pilot.game.restart()
                pilot.publish(self.frame)

    def tick_lockstep(self, i, pilot):
        """The game freezes while an answer is pending, then plays at most one frame per tick."""
        game = pilot.game
        if pilot.waiting:
            if not pilot.collect(self.frame):
                return
            pilot.waiting = False
        if game.playing and pilot.since >= self.lockstep:
            pilot.since = 0
            pilot.waiting = True
            pilot.publish(self.frame)
            return
        pilot.emergency(self.frame)
        pilot.enforce()
        game.step()
        pilot.since += 1
        self.game_frames[i] += 1
        pilot.capture_tick(self.frame)
        if game.crashed:
            self.crash_events.append({"player": pilot.brain.name, **pilot.on_crash(self.frame)})

    def report(self):
        out = {
            "frames": self.frame,
            "seconds": round(self.frame * FRAME_MS / 1000, 1),
            "mode": f"lockstep ({self.lockstep} frames per decision)"
            if self.lockstep
            else "real time",
            "host_stall_seconds_dropped": round(self.pacer.dropped_ms / 1000, 2)
            if self.pacer
            else 0,
            "players": {
                p.brain.name: {
                    **p.stats.summary(p.game, p.brain),
                    "game_time_ratio": round(self.game_frames[i] / self.frame, 3)
                    if self.frame
                    else None,
                    "shield": p.guarded,
                    "prompt": p.prompt,
                    "requests_in_flight": p.inflight,
                }
                for i, p in enumerate(self.pilots)
            },
        }
        if self.match:
            out["match"] = {**self.match.state(self.frame), "rounds": self.match.history}
        out["crash_count"] = len(self.crash_events)
        if self.director:
            d = self.director
            lat = sorted(d.latencies)
            out["course"] = {
                "designer": d.name,
                "designed": d.designed,
                "fallbacks": d.fallbacks,
                "errors": d.errors,
                "design_ms_p50": rounded(lat[len(lat) // 2]) if lat else None,
                "usd": round(d.tokens * d.backend.usd_per_token, 6),
            }
        return out
