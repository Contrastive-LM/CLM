#!/usr/bin/env python3
"""Chrome T-Rex runner, 60 seconds per seed: does the model keep the dinosaur alive?

    python examples/t_rex/run.py --model clm                 # 5 seeds x 60 s, real time, shield on
    python examples/t_rex/run.py --model jev
    python examples/t_rex/run.py --model clm --no-shield     # the model's answer stands even when labelled unsafe
    python examples/t_rex/run.py --model clm --lockstep 6    # game waits for every answer (latency removed)

A deterministic physics planner labels each action (jump / duck / run) safe or unsafe for the
moment the model's answer will land, and the model reads those labels in a Choice:

    state:    Dino runner game. 2 large cacti ahead, 96 px away.
    question: Choose the best safe action for the dinosaur.
      jump: Safe. Clears the 2 large cacti. Best.
      duck: Unsafe. Hits the 2 large cacti. Collision.
      run:  Unsafe. Hits the 2 large cacti. Collision.

The game runs at 60 FPS in real time; the model's own latency is part of play (several requests
stay in flight, asked a few frames apart).  A run lasts ``--duration`` seconds on one seeded course
(the original game's obstacle rules); a crash restarts the course after 1.5 s.  "Survived" means
zero deaths in the window.  By default the harness's shield is on, as upstream ships it: an answer
the planner labelled unsafe is replaced by the model's most probable safe action, and an emergency
check can act before a collision; the report counts those interventions.  The report per seed also carries score, decisions, answer latency and
how often the model agreed with the planner's best move.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import load_env, write_json  # noqa: E402
from trex.brain import RemoteBrain  # noqa: E402
from trex.course import StagedCourse  # noqa: E402
from trex.engine import FPS, FRAME_MS  # noqa: E402
from trex.pilot import Arena, Pilot  # noqa: E402
from trex.planner import snapshot  # noqa: E402


def warm_up(arena, log=print):
    """Measure the player's answer latency before the start, as the upstream harness does."""
    for pilot in arena.pilots:
        samples = pilot.brain.warm(snapshot(pilot.game, "run"), pilot.prompt)
        typical = sorted(samples)[len(samples) // 2] / FRAME_MS
        pilot.latency_frames = int(min(samples) / FRAME_MS)
        pilot.typical_frames = typical
        pilot.interval_frames = max(1.0, typical)
        pilot.jitter_frames = max(1.0, max(samples) / FRAME_MS + 1 - pilot.latency_frames)
        log(f"{pilot.brain.name}: {pilot.brain.detail}, warm answer {min(samples):.0f} ms")


def play(kind: str, seed: int, duration: float, *, shield: bool, lockstep: int | None, inflight: int,
         course_style: str, prompt: str) -> dict:
    brain = RemoteBrain(kind, inflight=1 if lockstep else inflight)
    pilot = Pilot(brain, guarded=shield, prompt=prompt, lockstep=lockstep, seed=seed,
                  course=StagedCourse(seed) if course_style == "staged" else None)
    arena = Arena([pilot], lockstep=lockstep)
    try:
        warm_up(arena)
        arena.start()
        frames = int(duration * FPS)
        while (arena.game_frames[0] if lockstep else arena.frame) < frames:
            arena.advance()
            pilot.game.drain_events()
            time.sleep(0.002)
        report = arena.report()
    finally:
        pilot.close()
    p = report["players"][brain.name]
    game = pilot.game
    return {"seed": seed, "survived": game.deaths == 0, "deaths": game.deaths,
            "best_score": p["best_score"], "scores": p["scores"] + ([game.score] if not game.crashed else []),
            "decisions": p["decisions"], "agreement_with_planner": p["agreement_with_planner"],
            "latency_ms_p50": p["latency_ms_p50"], "latency_ms_p95": p["latency_ms_p95"], "model_ms_p50": p["model_ms_p50"],
            "answers_discarded": p["answers_discarded"], "errors": p["errors"], "last_error": p["last_error"],
            "best_effort_decisions": p["best_effort_decisions"], "shield_interventions": p["shield_interventions"],
            "arrival_saves": p["arrival_saves"], "emergency_saves": p["emergency_saves"],
            "input_tokens": p["input_tokens"], "game_seconds": round(arena.game_frames[0] / FPS, 1),
            "wall_seconds": report["seconds"], "host_stall_seconds_dropped": report["host_stall_seconds_dropped"],
            "model": p["model"], "endpoint": p["where"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("clm", "jev"), required=True)
    ap.add_argument("--seeds", type=int, default=5, help="runs; run i uses --seed + i")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds per run")
    ap.add_argument("--no-shield", action="store_true",
                    help="execute the model's answer even when the planner labelled it unsafe")
    ap.add_argument("--lockstep", type=int, default=None, metavar="FRAMES",
                    help="freeze the game while the model answers, then play FRAMES frames per decision")
    ap.add_argument("--inflight", type=int, default=6, help="requests kept in flight in real time (1-8)")
    ap.add_argument("--course-style", choices=("original", "staged"), default="original",
                    help="original: the Chrome game's obstacle rules; staged: upstream's easier phased course")
    ap.add_argument("--prompt", choices=("labeled", "guided"), default="labeled")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    load_env()
    if not 1 <= args.inflight <= 8:
        ap.error("--inflight takes 1 to 8")

    results, summary = [], {}
    mode = f"lockstep{args.lockstep}" if args.lockstep else "realtime"
    shield = not args.no_shield
    tag = f"{args.model}_{mode}{'' if shield else '_noshield'}"
    for i in range(args.seeds):
        seed = args.seed + i
        row = play(args.model, seed, args.duration, shield=shield, lockstep=args.lockstep, inflight=args.inflight,
                   course_style=args.course_style, prompt=args.prompt)
        results.append(row)
        print(f"[{tag} seed={seed}] survived={row['survived']} deaths={row['deaths']} best_score={row['best_score']} "
              f"decisions={row['decisions']} agreement={row['agreement_with_planner']} latency_p50={row['latency_ms_p50']}ms "
              f"model_p50={row['model_ms_p50']}ms errors={row['errors']}", flush=True)
        summary = {"model": results[0]["model"], "endpoint": results[0]["endpoint"], "mode": mode, "shield": shield,
                   "course_style": args.course_style, "prompt": args.prompt, "duration_s": args.duration,
                   "seeds": len(results), "survived": sum(r["survived"] for r in results),
                   "deaths": sum(r["deaths"] for r in results),
                   "mean_best_score": round(statistics.fmean(r["best_score"] for r in results), 1),
                   "mean_decisions": round(statistics.fmean(r["decisions"] for r in results), 1),
                   "mean_agreement_with_planner": round(statistics.fmean(r["agreement_with_planner"] or 0 for r in results), 3),
                   "latency_ms_p50_median": statistics.median(r["latency_ms_p50"] or 0 for r in results),
                   "model_ms_p50_median": statistics.median(r["model_ms_p50"] or 0 for r in results),
                   "errors": sum(r["errors"] for r in results), "answers_discarded": sum(r["answers_discarded"] for r in results),
                   "shield_interventions": sum(r["shield_interventions"] + r["arrival_saves"] + r["emergency_saves"] for r in results),
                   "inflight": 1 if args.lockstep else args.inflight}
        write_json(args.out or str(HERE / "results" / f"{tag}.json"), {"summary": summary, "results": results})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.setswitchinterval(0.001)
    raise SystemExit(main())
