"""Round scoring independent of model timing and rendering."""

import math

from .engine import FRAME_MS


class Match:
    """Distance earned in a fixed interval; deaths cost time, not past distance."""

    def __init__(self, names, seconds=60, rounds=0):
        self.names = list(names)
        self.length = max(1, round(seconds * 1000 / FRAME_MS))
        self.limit = rounds
        self.number = 1
        self.start = 0
        self.points = [0] * len(names)
        self.previous = [0] * len(names)
        self.assist_start = [0] * len(names)
        self.assists = [0] * len(names)
        self.wins = [0] * len(names)
        self.history = []
        self.finished = False
        self.last_result = None

    def advance(self, frame, scores, assists=None):
        if self.finished:
            return False
        if assists is not None:
            self.assists = [total - start for total, start in zip(assists, self.assist_start)]
        for i, score in enumerate(scores):
            self.points[i] += max(0, score - self.previous[i])
            self.previous[i] = score
        if frame - self.start < self.length:
            return False
        best = max(self.points)
        winners = [i for i, score in enumerate(self.points) if score == best]
        reason = "distance"
        if len(winners) > 1:
            fewest = min(self.assists[i] for i in winners)
            winners = [i for i in winners if self.assists[i] == fewest]
            reason = "fewer live saves"
        winner = winners[0] if len(winners) == 1 else None
        if winner is not None:
            self.wins[winner] += 1
        self.last_result = {
            "round": self.number,
            "points": self.points.copy(),
            "winner": self.names[winner] if winner is not None else None,
            "reason": reason if winner is not None else "tie",
            "assists": self.assists.copy(),
            "frame": frame,
        }
        self.history.append(self.last_result)
        self.finished = bool(self.limit and self.number >= self.limit)
        if not self.finished:
            self.number += 1
            self.start = frame
            self.points = [0] * len(self.names)
            self.previous = [0] * len(self.names)
            self.assist_start = list(assists) if assists is not None else [0] * len(self.names)
            self.assists = [0] * len(self.names)
        return True

    def state(self, frame):
        points = self.last_result["points"] if self.finished else self.points
        ordered = sorted(points, reverse=True)
        lead = ordered[0] - ordered[1] if len(ordered) > 1 else 0
        leader = self.names[points.index(ordered[0])] if lead else None
        remaining = max(0, self.length - (frame - self.start))
        return {
            "round": self.number,
            "remaining": math.ceil(remaining * FRAME_MS / 1000),
            "points": points.copy(),
            "assists": self.assists.copy(),
            "wins": self.wins.copy(),
            "leader": leader,
            "lead": lead,
            "result": self.last_result,
            "finished": self.finished,
            "round_seconds": round(self.length * FRAME_MS / 1000, 3),
            "champion": self.names[self.wins.index(max(self.wins))]
            if self.finished and self.wins.count(max(self.wins)) == 1
            else None,
        }
