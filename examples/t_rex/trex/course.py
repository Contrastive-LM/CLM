"""A shared difficulty curve that stays within the original obstacle rules."""

from dataclasses import replace

from .engine import RandomCourse


def phase(game):
    # Birds become legal at speed 8.5; never advertise them before they can spawn.
    if game.obstacle_index < 3 or game.speed < 7:
        return {"name": "Warm-up", "gap_floor": 0.85, "hint": "Easy jumps, room to recover"}
    if game.speed < 8.5:
        return {"name": "Sprint", "gap_floor": 0.55, "hint": "Build a clean survival streak"}
    if game.speed < 9.1:
        return {"name": "Bird attack", "gap_floor": 0.45, "hint": "Watch the height: jump or duck"}
    return {"name": "Final challenge", "gap_floor": 0.25, "hint": "Mixed obstacles, tighter timing"}


def shape(spec, game):
    stage = phase(game)
    size = 1 if stage["name"] == "Warm-up" else spec.size
    return replace(spec, size=size, gap=max(spec.gap, stage["gap_floor"]))


class StagedCourse:
    """Deterministic random course with recovery space and themed bird sequences."""

    def __init__(self, seed):
        self.base = RandomCourse(seed)

    def spec(self, game, index, source="random"):
        spec = self.base.spec(game, index, source)
        if phase(game)["name"] == "Bird attack" and not game.too_many("pterodactyl"):
            spec = replace(spec, kind="pterodactyl", size=1, y=(75, 50, 100)[index % 3])
        return shape(spec, game)
