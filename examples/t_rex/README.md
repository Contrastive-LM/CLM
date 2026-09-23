# T-Rex runner: CLM vs Jev

The same served head, unchanged, against TypeSafe's hosted Jev (`jev-latest`) on Chrome's offline
dinosaur game, played in real time. CLM's `POST /v1/systemone` speaks the TypeSafe wire format, so
both players send the same request through one client ([`examples/common.py`](../common.py), built on
`clm.CLMClient`); only the base URL and API key differ.

**Task.** Keep the dinosaur alive for 60 seconds. [`trex/engine.py`](trex/engine.py) is a deterministic
Python clone of Chromium's dino game (same constants, jump physics, collision boxes, obstacle rules
and speed curve; from [laya-vs-jev](https://github.com/virajbhartiya/laya-vs-jev), Apache-2.0),
advanced in fixed 60 FPS steps in real time. A physics planner labels each action (`jump`, `duck`,
`run`) safe or unsafe for the moment the model's answer will land, given the answer latency it has
observed for this player, and names the one with the best timing margin. The model reads those
labels; its highest-probability action is executed when the answer lands. Several requests stay in
flight, asked a few frames apart, so a player gets a turn every few frames rather than once per
round trip. Five seeded courses (the original game's obstacle rules), 60 s each; a crash restarts
the course after 1.5 s. "Survived" = zero deaths in the window. The harness's shield is on, as
upstream ships it: an answer the planner labelled unsafe is replaced by the model's most probable
safe action, and an emergency check can act before a collision. The report counts every such
intervention, so the survival number measures the combined system and the agreement and
intervention rows measure the model.

**Request.** One Choice per decision, identical for both models:

```
state:    Dino runner game. 2 large cacti ahead, 96 px away.
question: Choose the best safe action for the dinosaur.
  jump: Safe. Clears the 2 large cacti. Best.
  duck: Unsafe. Hits the 2 large cacti. Collision.
  run:  Unsafe. Hits the 2 large cacti. Collision.
```

**Results.**

| | CLM v0.1 | TypeSafe Jev |
|---|---|---|
| survived 60 s (5 seeds) | **5/5** | **5/5** |
| deaths | 0 | 0 |
| best score (game distance meter; 697 = 60 s without a death) | 697.0 | 697.0 |
| decisions per 60 s | 3342 | 1119 |
| answers that were the planner's best move | 65.8% | 98.7% |
| shield interventions per seed | 806, 893, 975, 1101, 1108 | 9, 6, 5, 6, 2 |
| model latency p50 | 2.6 ms | 131.9 ms |
| answer-to-effect latency p50 (request out to answer applied) | 16.5 ms | 149.8 ms |
| failed calls | 0 | 0 |

Both keep the dinosaur alive on every seed, but not equally on their own. Jev picked the planner's
best move in 98.7% of its answers and the shield stepped in 28 times over five minutes of play;
CLM agreed 65.8% of the time and the shield stepped in 4883 times. Asked offline about 859 distinct
planner situations from a planner-driven game, CLM contradicts the labelled best move mostly when
the right answer is *run* ("Safe. Waits; acts later. Best.": 250 of 709 such situations, 34 of them
for an action labelled Unsafe), preferring an option whose description sounds like an action
("Clears the low bird", "Drops fast to land sooner") over the one marked *Best*; Jev reads the
markers. CLM answers eight times faster: its server-side time is about 3 ms because a run's few
hundred distinct situations are soon all in `clm-serve`'s vector cache, so it got a turn every
frame or two and Jev every three to six frames (3342 vs 1119 decisions a minute).

CLM run on 2026-09-23 with `clm-latest` = `CLM_v0.1-8B.pt` (Qwen3-8B encoder on one RTX 4090);
Jev run on 2026-09-22 with `jev-latest` (answered as `jev-1.13.0`). Latencies are client-side per
request.

**Reproduce.**

```bash
pip install -r requirements.txt
# from the repo root, with clm-serve running (see Quickstart in the main README); for Jev put TYPESAFE_API_KEY=... in <repo>/.env
python examples/t_rex/run.py --model clm                 # 5 seeds x 60 s, real time, shield on
python examples/t_rex/run.py --model jev
python examples/t_rex/run.py --model clm --no-shield     # the model's answer stands
python examples/t_rex/run.py --model clm --lockstep 6    # game freezes while the model answers
```

`CLM_BASE_URL` (default `http://127.0.0.1:8700`), `CLM_API_KEY`, `TYPESAFE_BASE_URL` and
`TYPESAFE_MODEL` override the endpoints. Each player runs in its own process
(`trex/brain.py`) so the game loop cannot steal its time. `results/<model>_realtime.json` holds
the per-seed report (deaths, score, decisions, answer latency, agreement with the planner's best
move, discarded late answers, errors). Real-time runs depend on the host:
`host_stall_seconds_dropped` in the report says how much wall time the game had to skip.
`--no-shield` and `--lockstep` are knobs for your own experiments, not part of the table.
