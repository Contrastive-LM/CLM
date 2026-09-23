"""Exact-physics safety analysis for the T-Rex autopilot.

The planner sees only obstacles already on screen. It predicts their motion with the
engine's own arithmetic and searches the dino's future key presses at a fixed decision
grain. It accounts for decision latency: the action being chosen now only takes effect
when the model's answer arrives, and stays held until the next answer.

Its output labels every action safe or unsafe and names the action with the best timing
margin. Both models receive exactly these labels; the optional shield uses the same
safe set to veto a model's unsafe choice.
"""

import math
import sys
from dataclasses import dataclass, field

from .engine import (
    ACCELERATION,
    DROP_VELOCITY,
    FRAME_MS,
    GROUND_Y,
    INITIAL_JUMP_VELOCITY,
    MAX_SPEED,
    TREX_BOXES,
    TREX_HEIGHT,
    TREX_WIDTH,
    WIDTH,
    jump_step,
    obstacle_step,
)

ACTIONS = ("jump", "duck", "run")
# The original times jump physics by the current animation's frame rate, so a dino that is
# mid-air in its ducking or running animation falls more slowly.
TIMEBASE = {"jump": FRAME_MS, "duck": 1000 / 8, "run": 1000 / 12}
STATUS = {"ducking": "duck", "jumping": "jump"}

# Trex state: (y, velocity, jumping, ducking, speed_drop, reached_min_height, status).
GROUND = (GROUND_Y, 0.0, False, False, False, False, "run")
# The original's reset() animates before clearing the drop flag, so landing from a speed
# drop always leaves the dino ducking until the down key is released.
DROP_LANDING = (GROUND_Y, 0.0, False, True, False, False, "duck")
TREX_TUPLES = {k: tuple((b.x, b.y, b.width, b.height) for b in v) for k, v in TREX_BOXES.items()}


@dataclass(frozen=True)
class ObstacleView:
    id: int
    label: str
    bird: bool
    x: float
    y: float
    width: float
    height: float
    offset: float
    boxes: tuple


@dataclass(frozen=True)
class Snapshot:
    trex: tuple
    trex_x: int
    speed: float
    obstacles: tuple
    held: str
    playing: bool
    pending: str | None = None


def snapshot(game, held):
    t = game.trex
    views = tuple(
        ObstacleView(
            o.id,
            o.label,
            o.kind.name == "pterodactyl",
            o.x,
            o.y,
            o.width,
            o.kind.height,
            o.speed_offset,
            tuple((b.x, b.y, b.width, b.height) for b in o.boxes),
        )
        for o in game.obstacles
        if not o.remove and o.x < WIDTH
    )
    state = (
        t.y,
        t.velocity,
        t.jumping,
        t.ducking,
        t.speed_drop,
        t.reached_min_height and t.jumping,  # Only meaningful mid-jump; start_jump resets it.
        STATUS.get(t.status, "run"),
    )
    return Snapshot(state, t.x, game.speed, views, held, game.playing and not game.crashed)


# -- Pure trex transitions, mirroring the engine and the pilot's key handling ---------------
def enforce(s, held):
    """The pilot's per-frame key state, applied before each engine step."""
    y, v, jumping, ducking, drop, reached, status = s
    if held == "duck":
        if jumping:
            if not drop:
                drop, v = True, 1.0
        elif not ducking:
            ducking, status = True, "duck"
    elif s is GROUND:
        return s
    elif drop or not jumping and ducking:
        # Releasing the down key ends a speed drop and any ducking animation.
        drop = False
        ducking = False
        if status == "duck":
            status = "run"
    return (y, v, jumping, ducking, drop, reached, status)


def impulse_jump(s, speed):
    if s[2]:
        return s
    return (s[0], INITIAL_JUMP_VELOCITY - speed / 10, True, False, False, False, "jump")


def trex_frame(s):
    y, v, jumping, ducking, drop, reached, status = s
    if not jumping:
        return s
    y, v, r, ended = jump_step(y, v, drop, TIMEBASE[status])
    reached = reached or r
    if ended and reached and v < DROP_VELOCITY:
        v = DROP_VELOCITY
    if y > GROUND_Y:
        return DROP_LANDING if drop else GROUND
    return (y, v, jumping, ducking, drop, reached, status)


def end_of_frame(s):
    y, v, jumping, ducking, drop, reached, status = s
    if drop and y == GROUND_Y:
        # The original's setDuck(true) toggles: it un-ducks a dino that is already ducking.
        if status == "duck":
            return (y, v, jumping, False, False, reached, "run")
        return (y, v, jumping, True, False, reached, "duck")
    return s


class World:
    """Where the first (collision-checked) obstacle is at each future frame."""

    def __init__(self, snap, horizon):
        self.trex_x = snap.trex_x
        self.horizon = horizon
        self.first = {}
        self.speed = {}
        self.passed = {}
        self.birds = any(o.bird for o in snap.obstacles)
        obstacles = [[o, o.x] for o in snap.obstacles]
        speed = snap.speed
        for k in range(1, horizon + 1):
            self.speed[k] = speed
            for item in obstacles:
                item[1] -= obstacle_step(speed, item[0].offset)
                if item[1] + item[0].width <= self.trex_x + 1:
                    self.passed.setdefault(item[0].id, k)
            obstacles = [item for item in obstacles if item[1] + item[0].width > 0]
            self.first[k] = self.compile(obstacles[0], self.trex_x) if obstacles else None
            if speed < MAX_SPEED:
                speed += ACCELERATION
        self.cache = {}

    @staticmethod
    def compile(item, trex_x):
        o, x = item
        ox, oy = x + 1, o.y + 1
        outer = (ox, oy, o.width - 2, o.height - 2)
        # Horizontal overlap depends only on the world frame, not the searched dino
        # state. Most frames are clear: skip box construction and cache entries there.
        overlaps = trex_x + 1 < ox + outer[2] and trex_x + TREX_WIDTH - 1 > ox
        inner = tuple((bx + ox, by + oy, bw, bh) for bx, by, bw, bh in o.boxes) if overlaps else ()
        return o, outer, inner

    def hit(self, k, y, ducking):
        entry = self.first.get(k)
        if entry is None or not entry[2]:
            return False
        key = (k, y, ducking)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = False
        _, (_, oy, _, oh), inner = entry
        tx, ty = self.trex_x + 1, y + 1
        if ty < oy + oh and TREX_HEIGHT - 2 + ty > oy:
            for bx, by, bw, bh in TREX_TUPLES[ducking]:
                ax, ay = bx + tx, by + ty
                for cx, cy, cw, ch in inner:
                    if ax < cx + cw and ax + bw > cx and ay < cy + ch and bh + ay > cy:
                        result = True
                        break
                if result:
                    break
        self.cache[key] = result
        return result


@dataclass
class Plan:
    safe: dict
    best: str
    airborne: bool
    threat: str | None
    ahead: str | None
    distance: int | None
    eta_ms: float | None
    notes: dict = field(default_factory=dict)
    states: int = 0
    robust: bool = True
    subjects: tuple = (None, None)

    @property
    def safe_actions(self):
        return [a for a in ACTIONS if self.safe[a]]


def after(action):
    """The key state an action leaves held: a jump is a tap, so nothing stays pressed."""
    return "run" if action == "jump" else action


class Search:
    """A safety game between the player and the timing of its own answers.

    Each answer lands some frames after the previous one; the gap is anywhere in a range,
    chosen adversarially. A position is winning if some action is winning for every gap,
    and a position past the horizon is won. This is exactly the situation the player is
    in when its next answer is being planned, so a move labelled safe always leaves a safe
    move for the next answer, whenever that lands.

    Memoised on (frame, trex state, held keys, zero gap allowed, lock). Two answers can
    land on the same frame (a zero gap) but not three. A lock restricts choices until a
    frame: ("duck", k) keeps ducking and ("jump", k) forbids another jump, which lets the
    planner ask whether one manoeuvre by itself handles an obstacle.
    """

    def __init__(self, world, gap, horizon):
        self.world = world
        self.gap = gap
        self.horizon = horizon
        self.memo = {}

    def run_frames(self, s, action, start, count):
        hit = self.world.hit
        for k in range(start + 1, start + count + 1):
            if s is GROUND and action != "duck":  # The common case: just keep running.
                if hit(k, GROUND_Y, False):
                    return None
                continue
            s = trex_frame(enforce(s, action))
            if hit(k, s[0], s[3]):
                return None
            s = end_of_frame(s)
        return s

    def press(self, s, action, k):
        return (
            impulse_jump(s, self.world.speed.get(k + 1, self.world.speed[1]))
            if action == "jump"
            else s
        )

    def options(self, s, held, k, lock):
        """Follow-up moves the proof may use. Fewer than the player has, which only makes
        labels more cautious: a fast drop is never released mid-air, and the dino ducks
        on the ground only when a bird is in sight."""
        if s[2] and self.gap[0] <= 2:
            # Up does nothing mid-air, and a quick player's answer lands while still airborne.
            allowed = ("duck",) if held == "duck" else ("run", "duck")
        elif s[2]:
            # A slow player's jump, ordered mid-air, lands after touchdown: that is how it
            # clears obstacles that follow closely.
            allowed = ("duck",) if held == "duck" else ("run", "jump", "duck")
        elif self.world.birds or held == "duck":
            allowed = ACTIONS
        else:
            allowed = ("jump", "run")
        if lock and k < lock[1]:
            allowed = ("duck",) if lock[0] == "duck" else tuple(a for a in allowed if a != "jump")
        return sorted(allowed, key=lambda a: a != held)

    def lands(self, s, held, action, t, lo, hi, lock=None):
        """Whether `action` wins when it lands anywhere from `lo` to `hi` frames after `t`."""
        s = self.run_frames(s, held, t, lo)
        for g in range(lo, hi + 1):
            if s is None:
                return False
            if t + g >= self.horizon:
                return True
            if not self.wins(t + g, self.press(s, action, t + g), after(action), g > 0, lock):
                return False
            if g < hi:
                s = self.run_frames(s, held, t + g, 1)
        return True

    def wins(self, t, s, held, zero_ok=True, lock=None):
        if t >= self.horizon:
            return True
        key = (t, s, held, zero_ok, lock)
        if key in self.memo:
            return self.memo[key]
        lo, hi = self.gap
        if not zero_ok and lo == 0:
            lo, hi = 1, hi + 1  # After two answers on one frame the next needs a fresh view.
        self.memo[key] = result = any(
            self.lands(s, held, a, t, lo, hi, lock) for a in self.options(s, held, t, lock)
        )
        return result


class StaggeredSearch(Search):
    """The same game for a player that keeps several requests in flight.

    Questions go out every few frames, whenever a request slot is free, so the wait for
    the next question varies: it is at most `period` frames. Each answer lands within
    `gap = (lo, hi)` frames of its own question.

    An answer that keeps the keys as they are changes nothing, so play reaches the next
    question, whenever that is: waiting is only winning if every frame at which the next
    question may go out is winning. An answer that changes the keys may land anywhere in
    its range; the answers asked meanwhile assumed the old keys and are discarded, so the
    next usable question goes out one to `period` frames after the landing.
    """

    def __init__(self, world, gap, horizon, period):
        super().__init__(world, gap, horizon)
        self.period = period

    def waits(self, s, held, t, soonest, lock):
        """Whether every frame from `soonest` to `period` after t wins, holding the keys.

        The next question is asked from a later frame's view, so `soonest` is at least 1
        and time always advances between decisions."""
        for d in range(self.period + 1):
            if t + d >= self.horizon:
                return True
            if d >= soonest and not self.wins(t + d, s, held, lock):
                return False
            if d < self.period:
                s = self.run_frames(s, held, t + d, 1)
                if s is None:
                    return False
        return True

    def lands(self, s, held, action, t, lo, hi, lock=None):
        if action == held:
            return self.waits(s, held, t, 1, lock)
        s = self.run_frames(s, held, t, lo)
        for g in range(lo, hi + 1):
            if s is None:
                return False
            if t + g >= self.horizon:
                return True
            if not self.waits(self.press(s, action, t + g), after(action), t + g, 1, lock):
                return False
            if g < hi:
                s = self.run_frames(s, held, t + g, 1)
        return True

    def wins(self, t, s, held, lock=None):
        if t >= self.horizon:
            return True
        key = (t, s, held, lock)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = False  # A position that leads back to itself is not a win.
        self.memo[key] = result = any(
            self.lands(s, held, a, t, *self.gap, lock) for a in self.options(s, held, t, lock)
        )
        return result


DUCK_LEAD_FRAMES = 14
RECURSION_LIMIT = 20_000
QUICK_PLAYER_FRAMES = 6


class Planner:
    def __init__(self, max_horizon=150):
        self.max_horizon = max_horizon
        # The search recurses a few Python frames per game frame of look-ahead.
        if sys.getrecursionlimit() < RECURSION_LIMIT:
            sys.setrecursionlimit(RECURSION_LIMIT)

    def horizon(self, snap):
        frames = 12
        for o in snap.obstacles:
            step = max(1, obstacle_step(snap.speed, o.offset))
            frames = max(frames, math.ceil((o.x + o.width - snap.trex_x) / step) + 6)
        return min(frames, self.max_horizon)

    def plan(self, snap, first, gap, period=None):
        """Label each action for an answer that lands `first = (lo, hi)` frames from now.

        Later answers follow at gaps within `gap`. A player with several requests in flight
        passes `period`, the longest wait between its questions; `gap` is then how long
        each of its answers takes (see StaggeredSearch).

        If no action survives every timing, the labels fall back to a best-effort choice,
        and `robust` records the downgrade.
        """
        plan, search, context = self.plan_once(snap, first, gap, period)
        if plan.safe_actions or context is None:
            return plan
        # Best effort. Judge each landing time of each action by whether play can go on if
        # later answers arrive at their usual time, and prefer the action that survives the
        # most landing times. Waiting scores full marks for as long as acting later works,
        # so a change of keys is only advised once it is the better bet.
        start, held, lo, hi = context
        usual = round((search.gap[0] + search.gap[1]) / 2)
        if period:
            relaxed = StaggeredSearch(search.world, (usual, usual), search.horizon, 1)
        else:
            relaxed = Search(search.world, (max(1, usual),) * 2, search.horizon)
        cover = {
            a: sum(relaxed.lands(start, held, a, 0, g, g) for g in range(lo, hi + 1))
            for a in ACTIONS
        }
        most = max(cover.values())
        if most:
            # Ties go to keeping the keys as they are: it costs nothing and wastes no answers.
            order = sorted(ACTIONS, key=lambda a: (-cover[a], a != held, a != "run"))
            plan.safe = {a: cover[a] == most for a in ACTIONS}
            plan.best = order[0]
            plan.notes = describe(plan, *plan.subjects)
        plan.robust = False
        return plan

    @staticmethod
    def whole(values):
        lo, hi = (max(0, int(round(v))) for v in values)
        return lo, max(1, lo, hi)

    def plan_once(self, snap, first, gap, period=None):
        if not snap.obstacles:
            airborne = bool(snap.trex[2])
            plan = Plan(dict.fromkeys(ACTIONS, True), "run", airborne, None, None, None, None)
            plan.notes = describe(plan, None, None)
            return plan, None, None
        lo, hi = (max(0, int(round(v))) for v in first)
        gap = self.whole(gap)
        horizon = max(self.horizon(snap), hi + 2 * gap[1] + 2)
        world = World(snap, horizon)
        if period:
            search = StaggeredSearch(world, gap, horizon, max(1, int(round(period))))
        else:
            search = Search(world, gap, horizon)

        # An earlier answer may still be waiting to be applied; it lands first. Then the
        # world keeps moving under the held keys while this answer is in flight.
        start, held = snap.trex, snap.held
        if start == GROUND:
            start = GROUND  # The shared instance enables the fast path in run_frames.
        if snap.pending:
            start, held = search.press(start, snap.pending, 0), after(snap.pending)
        arrival = search.run_frames(start, held, 0, lo)
        airborne = bool(arrival[2]) if arrival else bool(start[2])
        safe = {a: search.lands(start, held, a, 0, lo, hi) for a in ACTIONS}

        # Threat: what the dino would hit if it never pressed anything again.
        threat = crash = None
        if arrival is not None:
            s = arrival
            for k in range(lo + 1, horizon + 1):
                s = trex_frame(enforce(s, "duck" if s[4] else "run"))
                if world.hit(k, s[0], s[3]):
                    crash, threat = k, world.first[k][0]
                    break
                s = end_of_frame(s)
        ahead = threat or next((o for o in snap.obstacles if o.x + o.width > snap.trex_x), None)
        best = self.recommend(safe, airborne, threat, crash, start, arrival, held, lo, hi, search)
        if not safe[best] and any(safe.values()):
            # Never recommend a move the shield would veto.
            best = next(a for a in ("run", "jump", "duck") if safe[a])
        distance = eta = None
        if ahead is not None:
            travelled = sum(
                obstacle_step(world.speed.get(k, snap.speed), ahead.offset)
                for k in range(1, lo + 1)
            )
            distance = int(ahead.x - travelled - (snap.trex_x + TREX_WIDTH))
        if crash is not None:
            eta = (crash - lo) * FRAME_MS
        plan = Plan(
            safe=safe,
            best=best,
            airborne=airborne,
            threat=threat.label if threat else None,
            ahead=ahead.label if ahead else None,
            distance=distance,
            eta_ms=eta,
            states=len(search.memo),
        )
        plan.subjects = (threat, ahead)
        plan.notes = describe(plan, threat, ahead)
        return plan, search, (start, held, lo, hi)

    def recommend(self, safe, airborne, threat, crash, start, arrival, held, lo, hi, search):
        """The move with the most tolerance for a late answer.

        An answer never lands earlier than planned, only later. So the best time to start a
        manoeuvre is the first moment it clears the threat by itself for every expected
        landing time: all of its timing window then lies ahead as margin.
        """
        if airborne:
            # A fast drop lands the dino sooner, but it is a change of keys. A player whose
            # next answer is many frames away should only spend one when it has to.
            quick = search.gap[1] <= QUICK_PLAYER_FRAMES and not getattr(search, "period", None)
            if safe["duck"] and (
                not safe["run"] or quick and not self.clearing(arrival, lo, search)
            ):
                return "duck"
            return "run"
        if threat is None:
            return "run"
        until = search.world.passed.get(threat.id, search.horizon)
        # Ducking works from any distance, so only crouch once the bird is close.
        near = crash - lo <= hi - lo + DUCK_LEAD_FRAMES
        for action in ("duck", "jump"):
            if action == "duck" and not (threat.bird and near):
                continue
            if safe[action] and search.lands(start, held, action, 0, lo, hi, (action, until)):
                return action
        if safe["run"]:
            return "run"
        return next((a for a in ("jump", "duck") if safe[a]), "jump")

    @staticmethod
    def clearing(arrival, latency, search):
        """Whether the current arc is still carrying the dino over an obstacle."""
        s, k = arrival, latency
        while s[2] and k < search.horizon:
            s = trex_frame(enforce(s, "run"))
            k += 1
        return any(latency < frame <= k + 2 for frame in search.world.passed.values())


def describe(plan, threat, ahead):
    label = threat.label if threat else ahead.label if ahead else None
    notes = {}
    for a in ACTIONS:
        safe = plan.safe[a]
        if plan.airborne:
            if a == "duck":
                notes[a] = "Drops fast to land sooner" if safe else f"Drops onto the {label}"
            else:
                notes[a] = "Keeps flying" if safe else f"Lands on the {label}"
        elif not safe:
            notes[a] = f"Hits the {label}" if label else "Collides"
        elif a == "run":
            if threat is None:
                notes[a] = "Keeps running"
            else:
                notes[a] = "Waits; acts later" if plan.best == "run" else "Waits; must act soon"
        elif a == "jump":
            if threat is None:
                notes[a] = "Jumps for no reason"
            elif plan.best == "run":
                notes[a] = "Jumps too early"
            else:
                notes[a] = f"Clears the {label}"
        else:
            notes[a] = (
                f"Passes under the {label}" if threat and threat.bird else "Crouches; no benefit"
            )
    return notes
