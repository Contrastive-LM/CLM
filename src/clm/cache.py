"""A reserved device arena for the vectors an agent loop keeps asking about again.

An agent asks about a changing state but a mostly fixed set of actions, and it
often revisits states it has already seen.  Neither their embeddings nor their
projections change while the head does not, so they are worth keeping next to the
head rather than recomputing (or copying from host) on every request.

The arena is claimed once at start-up, the way vLLM claims its KV cache: one flat
allocation whose size comes from a budget -- a fraction of the device's memory
(``0.02``), or an absolute size (``512MiB``).  Pools of different widths are
carved out of that one allocation (512-d projections for the heads, 4096-d
encoder embeddings for the raw ablation), so nothing grows afterwards and a
long-running server cannot drift into an out-of-memory kill.

Entries are keyed by a namespace and the text.  The namespace names the head and
its generation, so several served heads share one arena, and a head that
hot-reloads simply stops matching rows its previous weights produced;
least-recently-used eviction reclaims them.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any, Callable

DEFAULT_BUDGET = "0.02"          # fraction of total device memory, vLLM-style
_UNITS = {"": 1, "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9,
          "KIB": 1 << 10, "MIB": 1 << 20, "GIB": 1 << 30}


class CacheDisabled(Exception):
    pass


def parse_budget(spec: Any, total_bytes: int) -> int:
    """``0.02`` -> 2% of the device; ``512MiB`` / ``2GB`` -> that many bytes; ``0`` -> off."""
    if spec is None or spec == "":
        spec = DEFAULT_BUDGET
    if isinstance(spec, (int, float)):
        spec = repr(spec)
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*", str(spec))
    if not m:
        raise ValueError(f"cache budget {spec!r} is not a fraction (0.02) or a size (512MiB)")
    value, unit = float(m.group(1)), m.group(2).upper()
    if unit not in _UNITS:
        raise ValueError(f"unknown size unit {m.group(2)!r}; use B, KB, MB, GB, KiB, MiB or GiB")
    if not unit:
        if not 0 <= value < 1:
            raise ValueError("a bare number is a fraction of device memory and must be in [0, 1); "
                             "give a unit (512MiB) for an absolute size")
        return int(value * total_bytes)
    return int(value * _UNITS[unit])


class Pool:
    """One width of vector inside the arena: a fixed row count, LRU."""

    def __init__(self, buffer, dim: int):
        self.buffer, self.dim = buffer, dim
        self.capacity = buffer.shape[0]
        self.slots: OrderedDict[str, int] = OrderedDict()
        self.free: list[int] = list(range(self.capacity - 1, -1, -1))
        self.hits = self.misses = self.evictions = 0

    def claim(self, key: str) -> int:
        if self.free:
            slot = self.free.pop()
        else:
            _, slot = self.slots.popitem(last=False)     # least recently used
            self.evictions += 1
        self.slots[key] = slot
        return slot

    def stats(self) -> dict:
        asked = self.hits + self.misses
        return {"dim": self.dim, "capacity": self.capacity, "used": len(self.slots),
                "reserved_mb": round(self.capacity * self.dim * 4 / 10**6, 1),
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "hit_rate": round(self.hits / asked, 4) if asked else None}


class VectorArena:
    """A single device allocation, carved into per-width pools, never grown."""

    def __init__(self, device: str = "cuda", budget: Any = None, dtype: str = "float32"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
        elif self.device.type == "mps":
            total = torch.mps.recommended_max_memory()
            free = max(0, total - torch.mps.driver_allocated_memory())
        else:  # a CPU arena is still bounded, it just cannot ask the driver for a budget
            free = total = 8 << 30
        want = parse_budget(budget, total)
        if want <= 0:
            raise CacheDisabled("arena disabled by budget 0")
        self.item = torch.empty((), dtype=self.dtype).element_size()
        # Leave the encoder and the heads room: never take more than 90% of what is free now.
        self.reserved_bytes = min(want, int(free * 0.9))
        self.flat = torch.zeros(self.reserved_bytes // self.item, device=self.device, dtype=self.dtype)
        self.reserved_mb = round(self.flat.numel() * self.item / 10**6, 1)
        self.cursor = 0
        self.pools: dict[int, Pool] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- reservation
    def reserve(self, dim: int, share: float) -> Pool | None:
        """Carve ``share`` of the arena into rows of ``dim``. Call at start-up, once per width."""
        with self._lock:
            if dim in self.pools:
                return self.pools[dim]
            rows = int(self.flat.numel() * share) // dim
            if rows < 1:
                return None                       # this width does not fit its share; bypass it
            end = self.cursor + rows * dim
            if end > self.flat.numel():
                rows = (self.flat.numel() - self.cursor) // dim
                if rows < 1:
                    return None
                end = self.cursor + rows * dim
            pool = Pool(self.flat[self.cursor:end].view(rows, dim), dim)
            self.cursor = end
            self.pools[dim] = pool
            return pool

    # ---------------------------------------------------------------- lookup
    def get(self, namespace: str, dim: int, texts: list[str], compute: Callable[[list[str]], Any]) -> Any:
        """-> [len(texts), dim] device tensor; ``compute`` fills the misses, in order."""
        pool = self.pools.get(dim)
        if pool is None:
            return compute(texts)
        keys = [f"{namespace}\x00{t}" for t in texts]
        with self._lock:
            missing, missing_keys = [], []
            for text, key in zip(texts, keys):
                slot = pool.slots.get(key)
                if slot is None:
                    if key not in missing_keys:
                        missing.append(text); missing_keys.append(key)
                else:
                    pool.slots.move_to_end(key)
                    pool.hits += 1
        if missing:
            pool.misses += len(missing)
            vectors = compute(missing)               # outside the lock: this is the slow path
            with self._lock:
                for key, vector in zip(missing_keys, vectors):
                    pool.buffer[pool.claim(key)] = vector
        with self._lock:
            # Resolve under the lock: a concurrent request may have evicted a row and
            # handed the slot to another key since it was filled.
            rows = [pool.slots.get(k) for k in keys]
            if any(r is None for r in rows):
                return compute(texts)                # thrashing; answer without the arena
            index = self.torch.as_tensor(rows, device=self.device, dtype=self.torch.long)
            return pool.buffer.index_select(0, index)   # copies while the lock is held

    # ---------------------------------------------------------------- reporting
    def stats(self) -> dict:
        with self._lock:
            pools = {str(dim): p.stats() for dim, p in sorted(self.pools.items())}
        asked = sum(p["hits"] + p["misses"] for p in pools.values())
        hits = sum(p["hits"] for p in pools.values())
        return {"device": str(self.device), "reserved_mb": self.reserved_mb,
                "hit_rate": round(hits / asked, 4) if asked else None, "pools": pools}
