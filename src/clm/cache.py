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

Supports ``cuda`` / ``cpu`` (torch) and ``mlx`` (Metal unified memory).
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

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
        if isinstance(buffer, list):
            self.capacity = len(buffer)
        else:
            self.capacity = int(buffer.shape[0])
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


def _device_memory(device: str) -> tuple[int, int]:
    """-> (free_bytes, total_bytes) best-effort."""
    kind = device.split(":")[0]
    if kind == "cuda":
        import torch
        return torch.cuda.mem_get_info(torch.device(device))
    if kind == "mlx":
        try:
            import mlx.core as mx
            info = getattr(mx, "metal", None)
            if info is not None and hasattr(info, "device_info"):
                di = info.device_info()
                total = int(di.get("memory_size", di.get("max_recommended_working_set_size", 16 << 30)))
                used = int(di.get("current_allocated_size", 0)) if "current_allocated_size" in di else 0
                return max(total - used, total // 4), total
        except Exception:  # noqa: BLE001
            pass
        total = 16 << 30
        return total, total
    return 8 << 30, 8 << 30


class VectorArena:
    """A single device allocation, carved into per-width pools, never grown."""

    def __init__(self, device: str = "cuda", budget: Any = None, dtype: str = "float32"):
        self.device_str = device
        self.backend = "mlx" if device.split(":")[0] == "mlx" else "torch"
        free, total = _device_memory(device)
        want = parse_budget(budget, total)
        if want <= 0:
            raise CacheDisabled("arena disabled by budget 0")
        self.reserved_bytes = min(want, int(free * 0.9))
        self.item = 4  # float32
        self.capacity_elems = self.reserved_bytes // self.item
        self.cursor = 0
        self.pools: dict[int, Pool] = {}
        self._lock = threading.Lock()
        self.reserved_mb = round(self.capacity_elems * self.item / 10**6, 1)

        if self.backend == "mlx":
            import mlx.core as mx
            self.mx = mx
            # Per-pool 2D buffers (MLX views of a 1D slab are not always writable aliases).
            self.flat = None
            self.device = "mlx"
        else:
            import torch
            self.torch = torch
            self.device = torch.device(device)
            self.dtype = getattr(torch, dtype)
            self.item = torch.empty((), dtype=self.dtype).element_size()
            self.capacity_elems = self.reserved_bytes // self.item
            self.reserved_mb = round(self.capacity_elems * self.item / 10**6, 1)
            self.flat = torch.zeros(self.capacity_elems, device=self.device, dtype=self.dtype)

    # ---------------------------------------------------------------- reservation
    def reserve(self, dim: int, share: float) -> Pool | None:
        """Carve ``share`` of the arena into rows of ``dim``. Call at start-up, once per width."""
        with self._lock:
            if dim in self.pools:
                return self.pools[dim]
            rows = int(self.capacity_elems * share) // dim
            if rows < 1:
                return None
            end = self.cursor + rows * dim
            if end > self.capacity_elems:
                rows = (self.capacity_elems - self.cursor) // dim
                if rows < 1:
                    return None
                end = self.cursor + rows * dim
            if self.backend == "mlx":
                # Slot list of rows (MLX has no efficient in-place 2D row write).
                buf = [None] * rows
            else:
                buf = self.flat[self.cursor:end].view(rows, dim)
            pool = Pool(buf, dim)
            self.cursor = end
            self.pools[dim] = pool
            return pool

    def _row(self, vectors, i: int):
        """Extract row ``i`` from a batch returned by ``compute``."""
        return vectors[i]

    def _store_row(self, pool: Pool, slot: int, vector) -> None:
        if self.backend == "mlx":
            row = vector if hasattr(vector, "dtype") else self.mx.array(np.asarray(vector, dtype=np.float32))
            self.mx.eval(row)
            pool.buffer[slot] = row
        else:
            pool.buffer[slot] = vector

    def _gather(self, pool: Pool, rows: list[int]):
        if self.backend == "mlx":
            stacked = self.mx.stack([pool.buffer[r] for r in rows], axis=0)
            self.mx.eval(stacked)
            return stacked
        index = self.torch.as_tensor(rows, device=self.device, dtype=self.torch.long)
        return pool.buffer.index_select(0, index)

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
            vectors = compute(missing)
            with self._lock:
                for i, key in enumerate(missing_keys):
                    slot = pool.claim(key)
                    self._store_row(pool, slot, self._row(vectors, i))
        with self._lock:
            rows = [pool.slots.get(k) for k in keys]
            if any(r is None for r in rows):
                return compute(texts)
            return self._gather(pool, rows)  # type: ignore[arg-type]

    # ---------------------------------------------------------------- reporting
    def stats(self) -> dict:
        with self._lock:
            pools = {str(dim): p.stats() for dim, p in sorted(self.pools.items())}
        asked = sum(p["hits"] + p["misses"] for p in pools.values())
        hits = sum(p["hits"] for p in pools.values())
        return {"device": str(self.device), "reserved_mb": self.reserved_mb,
                "hit_rate": round(hits / asked, 4) if asked else None, "pools": pools}
