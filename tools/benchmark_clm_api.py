"""Measure CLM API latency with empty, partly reused, and warm caches.

Start a separate, fresh clm-serve process before running this script. The
Qwen3-8B embeddings server may stay loaded. The default empty-cache check
guards against accidentally publishing timings from an already used API.
"""

import argparse
import http.client
import json
import math
import statistics
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit


CHOICES = {
    "billing": "Charges, invoices, and refunds",
    "technical": "Bugs and outages",
    "sales": "Product plans and purchases",
}
INSTRUCTIONS = "Which team should handle this customer request?"


def question(choices):
    return {"department": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": choices}}


def percentile(values, fraction):
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def summary(rows):
    return {
        "requests": len(rows),
        "server_ms_median": round(statistics.median(row["server_ms"] for row in rows), 1),
        "server_ms_p95": round(percentile([row["server_ms"] for row in rows], 0.95), 1),
        "wall_ms_median": round(statistics.median(row["wall_ms"] for row in rows), 1),
        "wall_ms_p95": round(percentile([row["wall_ms"] for row in rows], 0.95), 1),
        "input_tokens_median": statistics.median(row["input_tokens"] for row in rows),
    }


class Client:
    def __init__(self, base_url):
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.path not in ("", "/"):
            raise ValueError("base URL must be the CLM API origin, for example http://127.0.0.1:8702")
        conn_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        self.conn = conn_type(url.hostname, url.port, timeout=120)

    def health(self):
        self.conn.request("GET", "/health")
        response = self.conn.getresponse()
        data = response.read()
        if response.status != 200:
            raise RuntimeError(f"health returned HTTP {response.status}: {data[:300]!r}")
        return json.loads(data)

    def ask(self, payload):
        started = time.perf_counter()
        self.conn.request("POST", "/v1/systemone", body=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        response = self.conn.getresponse()
        data = response.read()
        wall_ms = (time.perf_counter() - started) * 1000
        if response.status != 200:
            raise RuntimeError(f"systemone returned HTTP {response.status}: {data[:300]!r}")
        result = json.loads(data)
        return {
            "wall_ms": round(wall_ms, 3),
            "server_ms": float(response.getheader("X-CLM-Latency-Ms")),
            "input_tokens": result["usage"]["input_tokens"],
            "choice": result["answers"]["department"]["choice"],
        }

    def close(self):
        self.conn.close()


def run(base_url, samples, allow_warm_cache=False):
    if samples < 2:
        raise ValueError("samples must be at least 2")
    client = Client(base_url)
    run_id = uuid.uuid4().hex[:12]
    try:
        before = client.health()
        if not before.get("ok") or not before.get("embedder"):
            raise RuntimeError("CLM API or encoder is not healthy")
        pools = ((before.get("cache") or {}).get("pools") or {}).values()
        if not allow_warm_cache and (not before.get("cache") or any(pool["used"] for pool in pools)):
            raise RuntimeError("CLM cache is not empty; start a fresh clm-serve process")

        def payload(state, choices):
            return {"model": "clm-latest", "state": state, "questions": question(choices)}

        first = client.ask(payload(f"Customer {run_id}-first: My invoice was charged twice; please refund it.", CHOICES))
        if first["input_tokens"] <= 0:
            raise RuntimeError("the first request did not reach the encoder")

        full_misses = []
        for i in range(samples):
            tag = f"{run_id}-{i:04d}"
            unique_choices = {key: f"{description}. Case {tag}." for key, description in CHOICES.items()}
            row = client.ask(payload(f"Customer case {tag}: My invoice has an extra charge; please refund it.",
                                     unique_choices))
            if row["input_tokens"] <= 0:
                raise RuntimeError(f"expected uncached embeddings for case {tag}")
            full_misses.append(row)

        # Populate the fixed choice texts once, outside the measured state-only phase.
        client.ask(payload(f"Customer {run_id}-choices: My invoice was charged twice.", CHOICES))
        state_only = []
        last_payload = None
        for i in range(samples):
            last_payload = payload(f"Customer case {run_id}-state-{i:04d}: My invoice has an extra charge; please refund it.",
                                   CHOICES)
            row = client.ask(last_payload)
            if row["input_tokens"] <= 0:
                raise RuntimeError(f"expected a fresh state embedding for case {i}")
            state_only.append(row)

        cached = []
        for _ in range(samples):
            row = client.ask(last_payload)
            if row["input_tokens"] != 0:
                raise RuntimeError("expected zero encoder tokens for a cached repeat")
            cached.append(row)
        return {
            "run_id": run_id,
            "base_url": base_url,
            "samples_per_phase": samples,
            "method": "first request after fresh API; fresh states and choices; fresh states with fixed choices; identical cached repeat",
            "health_before": before,
            "health_after": client.health(),
            "first_after_reset": first,
            "phases": {
                "all_texts_uncached": {"summary": summary(full_misses), "samples": full_misses},
                "fresh_state_fixed_choices": {"summary": summary(state_only), "samples": state_only},
                "identical_request_cached": {"summary": summary(cached), "samples": cached},
            },
        }
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="origin of a freshly started clm-serve process")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-warm-cache", action="store_true", help="for smoke tests only")
    args = parser.parse_args()
    result = run(args.base_url, args.samples, args.allow_warm_cache)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print("| Cache condition | n | Server median | Server p95 | Local wall median | Local wall p95 | Encoder tokens median |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    first = result["first_after_reset"]
    print(f"| First request after reset | 1 | {first['server_ms']:.1f} ms | — | {first['wall_ms']:.1f} ms | — | {first['input_tokens']} |")
    for name, phase in result["phases"].items():
        s = phase["summary"]
        print(f"| {name.replace('_', ' ')} | {s['requests']} | {s['server_ms_median']:.1f} ms | "
              f"{s['server_ms_p95']:.1f} ms | {s['wall_ms_median']:.1f} ms | "
              f"{s['wall_ms_p95']:.1f} ms | {s['input_tokens_median']:g} |")


if __name__ == "__main__":
    main()
