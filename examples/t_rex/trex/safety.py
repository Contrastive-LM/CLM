"""Bounded arrival checks, independent of the recursive request planner.

Reversible held keys get a 12-frame collision check. Jump and airborne key changes
are checked through the imminent obstacle clearing, capped at 42 frames, because
an unsafe takeoff or drop may cause a collision long after the immediate window.
A rejected early takeoff may wait if a bounded delayed jump clears the obstacle.
This explicit extra assist cannot prove safety beyond its recovery horizon.
"""

from .planner import ACTIONS, Search, World, after


def protect(snap, action, urgent=12, horizon=42):
    if not snap.playing or not snap.obstacles:
        return action
    # Passed obstacles remain on screen until their right edge leaves the canvas.
    # They must stay in World to match engine collision ordering, but must not
    # shorten the recovery horizon for the next obstacle ahead of the dino.
    first = next((o for o in snap.obstacles if o.x + o.width > snap.trex_x + 1), None)
    if first is None:
        return action
    search = Search(World(snap, horizon), (1, 1), horizon)
    # One tap need not clear the following obstacle: a fresh tap after landing
    # may be necessary. Check beyond the first obstacle with bounded recovery.
    recovery = min(horizon, max(urgent + 6, search.world.passed.get(first.id, horizon) + 6))
    # Preserve an already-clear jump arc. Starting a drop can be reversible in
    # isolation, yet a later release changes its physics again and destroys the
    # timing margin. Only shorten this arc when holding it would actually fail.
    if action == "duck" and snap.trex[2] and snap.held == "run":
        if search.run_frames(snap.trex, "run", 0, recovery) is not None:
            return "run"
    motion_change = action != snap.held and (snap.trex[2] or action == "jump")
    check = recovery if motion_change else urgent
    state = search.press(snap.trex, action, 0)
    if search.run_frames(state, after(action), 0, check) is not None:
        return action
    for alternate in (snap.held, "jump", "duck", "run"):
        if alternate == action or alternate not in ACTIONS:
            continue
        state = search.press(snap.trex, alternate, 0)
        if search.run_frames(state, after(alternate), 0, recovery) is not None:
            return alternate
    # Holding run can be safe even though it eventually hits the obstacle. Prove
    # one later jump is viable before vetoing a premature takeoff. This searches
    # at most twelve fixed trajectories, never a recursive tree of decisions.
    if motion_change and snap.held == "run" and action != "run":
        state = snap.trex
        for delay in range(1, min(urgent, recovery) + 1):
            state = search.run_frames(state, snap.held, delay - 1, 1)
            if state is None:
                break
            jump = search.press(state, "jump", delay)
            if search.run_frames(jump, "run", delay, recovery - delay) is not None:
                return "run"
    if motion_change:
        # When every fixed trajectory loses, avoid shortening the remaining
        # reaction window. A subsequent emergency move may still recover.
        def survival(candidate):
            state = search.press(snap.trex, candidate, 0)
            for frame in range(1, recovery + 1):
                state = search.run_frames(state, after(candidate), frame - 1, 1)
                if state is None:
                    return frame - 1
            return recovery

        if survival(snap.held) > survival(action):
            return snap.held
    return action
