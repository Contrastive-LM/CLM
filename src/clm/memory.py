"""macOS / unified-memory pressure helpers for CLM MLX loads.

Loading Qwen3-8B next to a resident 30B+ Metal server gets jetsam'd.  Call
``require_free_memory`` before ``mlx_lm.load`` so we fail loud instead of dying
mid-fetch with an empty log.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class MemSnapshot:
    total_bytes: int
    free_bytes: int          # free + speculative + purgeable (best-effort reclaimable)
    pressure: str            # "normal" | "warn" | "critical" | "unknown"
    top: list[tuple[str, int]]  # (command, rss_bytes) worst offenders

    @property
    def free_gib(self) -> float:
        return self.free_bytes / (1 << 30)

    @property
    def total_gib(self) -> float:
        return self.total_bytes / (1 << 30)


def _sysctl_int(name: str) -> int | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL).strip()
        return int(out)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def _vm_pages() -> dict[str, int]:
    """Parse ``vm_stat`` page counts (macOS)."""
    try:
        raw = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    page = 4096
    m = re.search(r"page size of (\d+)", raw)
    if m:
        page = int(m.group(1))
    counts: dict[str, int] = {"_page": page}
    for line in raw.splitlines():
        mm = re.match(r"Pages ([^:]+):\s+(\d+)", line)
        if mm:
            counts[mm.group(1).strip().lower()] = int(mm.group(2).rstrip("."))
    return counts


def _pressure_level() -> str:
    try:
        raw = subprocess.check_output(["memory_pressure"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    low = raw.lower()
    if "critical" in low:
        return "critical"
    if "warn" in low:
        return "warn"
    if "normal" in low:
        return "normal"
    return "unknown"


def _top_rss(n: int = 8) -> list[tuple[str, int]]:
    try:
        raw = subprocess.check_output(["ps", "-axo", "rss,comm"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows: list[tuple[str, int]] = []
    for line in raw.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            rss_kb = int(parts[0])
        except ValueError:
            continue
        rows.append((parts[1], rss_kb * 1024))
    rows.sort(key=lambda kv: -kv[1])
    return rows[:n]


def snapshot() -> MemSnapshot:
    total = _sysctl_int("hw.memsize") or (16 << 30)
    pages = _vm_pages()
    page = pages.get("_page", 16384)
    # reclaimable-ish: free + speculative + purgeable
    free_pages = (pages.get("free", 0) + pages.get("speculative", 0) + pages.get("purgeable", 0))
    free = free_pages * page
    return MemSnapshot(total_bytes=total, free_bytes=free, pressure=_pressure_level(), top=_top_rss())


def format_snapshot(snap: MemSnapshot) -> str:
    lines = [
        f"memory: {snap.free_gib:.1f} GiB reclaimable / {snap.total_gib:.1f} GiB total "
        f"(pressure={snap.pressure})",
        "top RSS:",
    ]
    for cmd, rss in snap.top[:6]:
        lines.append(f"  {rss / (1 << 30):5.1f} GiB  {cmd}")
    return "\n".join(lines)


def estimate_model_gib(model_id: str) -> float:
    """Rough unified-memory need for load + first forward (weights + activations)."""
    override = os.environ.get("CLM_MLX_NEED_GIB")
    if override:
        return float(override)
    low = model_id.lower().replace("4bit", "").replace("8bit", "").replace("3bit", "")
    if "0.6b" in low or "0.5b" in low:
        return 2.0
    if "1.7b" in low or "1.5b" in low:
        return 3.0
    if "27b" in low or "32b" in low or "35b" in low:
        return 28.0
    if "14b" in low or "9b" in low:
        return 14.0
    if "8b" in low:
        return 10.0  # 4-bit ~4.5 GiB weights + headroom under a busy Metal heap
    if "4b" in low:
        return 6.0
    return 10.0


class MemoryPressureError(RuntimeError):
    pass


def require_free_memory(need_gib: float, *, wait_s: float = 0.0, poll_s: float = 5.0,
                        log: Callable[[str], None] | None = print) -> MemSnapshot:
    """Block until reclaimable memory >= need_gib, or raise MemoryPressureError.

    ``wait_s=0`` means check once and fail immediately (safe default under jetsam).
    """
    _log = log or (lambda _m: None)
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        snap = snapshot()
        if snap.free_gib >= need_gib and snap.pressure != "critical":
            _log(f"[clm-mem] ok: need {need_gib:.1f} GiB, have {snap.free_gib:.1f} GiB "
                 f"(pressure={snap.pressure})")
            return snap
        msg = (
            f"[clm-mem] need {need_gib:.1f} GiB reclaimable, have {snap.free_gib:.1f} GiB "
            f"(pressure={snap.pressure}). Free Metal/RSS before loading.\n"
            f"{format_snapshot(snap)}"
        )
        if time.monotonic() >= deadline:
            raise MemoryPressureError(msg)
        _log(msg + f"\n[clm-mem] waiting up to {deadline - time.monotonic():.0f}s ...")
        time.sleep(poll_s)


class MemoryWatchdog:
    """Background logger: warn if free memory collapses during a long load/download."""

    def __init__(self, min_free_gib: float, interval_s: float = 3.0,
                 log: Callable[[str], None] | None = print):
        self.min_free_gib = min_free_gib
        self.interval_s = interval_s
        self._log = log or (lambda _m: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.trips = 0

    def start(self) -> "MemoryWatchdog":
        self._thread = threading.Thread(target=self._run, name="clm-mem-watch", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            snap = snapshot()
            if snap.free_gib < self.min_free_gib or snap.pressure == "critical":
                self.trips += 1
                self._log(f"[clm-mem] WATCHDOG trip #{self.trips}: {format_snapshot(snap)}")
