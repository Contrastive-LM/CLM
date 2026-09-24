"""A deterministic clone of Chrome's offline T-Rex runner, played by a System One model.

``engine.py`` (the game), ``planner.py`` / ``safety.py`` (exact-physics labels for each action,
given the player's answer latency), ``pilot.py`` (the real-time decision loop), ``brain.py`` (the
player process) and ``course.py`` come from https://github.com/virajbhartiya/laya-vs-jev (Apache-2.0;
game rules and constants from Chromium, BSD-3-Clause).  ``backends.py`` talks to CLM or Jev.
"""
