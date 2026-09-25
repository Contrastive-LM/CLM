"""CLM System One API server (FastAPI).

    clm-serve --port 8700 --emb-url http://127.0.0.1:8090/v1/embeddings
    export CLM_API_KEY=...      # required once this server is reachable from another machine
                                # (a non-loopback --host, or --cors); requests then need
                                # "Authorization: Bearer <key>"

    POST /v1/systemone   {"state": ..., "model": "clm-latest", "questions": {id: Question},
                          "temperature": 1.0}          -> {"model", "answers": {id: Answer}, "usage"}
    POST /v1/rank        {"context": ..., "question": ..., "answers": [...]}
                                                       -> {"model", "ranked": [{rank, candidate, prob}]}
    GET  /v1/models      -> {"models": [{"name", "description", "release_date"}]}
    GET  /health         -> {"ok": true, ...}
    GET  /               -> the playground: a zero-dependency web UI for the endpoint
                            above (``--no-ui`` to leave it off)

Question / Answer objects follow the TypeSafe wire schema (noul / choice /
score), so a request written for TypeSafe replays here unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
import time

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .embedder import EmbedderError
from .engine import DEFAULT_MODEL, Engine, ModelNotFound
from .heads import DEFAULT_CKPT_DIR, HF_FILE, default_device, download

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


_UI_FILES = ("index.html", "app.css", "app.js")


def _asset_stamp() -> str:
    """Newest mtime across the UI files, as a short hex tag for their URLs."""
    return format(int(max(os.path.getmtime(os.path.join(STATIC_DIR, f)) for f in _UI_FILES)), "x")


def _index_html() -> str:
    """index.html with ?v=<stamp> on its assets, so a browser that cached an
    older app.css/app.js cannot render the new page with the old styling."""
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    v = _asset_stamp()
    return html.replace('href="app.css"', f'href="app.css?v={v}"').replace('src="app.js"', f'src="app.js?v={v}"')


class _RevalidatingStatic(StaticFiles):
    """Static files that must be revalidated, so upgrading clm cannot leave a
    stale playground in the browser cache.  The ETag still answers 304."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(engine: Engine, api_key: str | None = None, ui: bool = True, cors: bool = False) -> FastAPI:
    """The API, plus the playground at ``/`` unless ``ui=False``.

    ``cors=True`` allows browser requests from any origin (and exposes the
    latency header), which a playground served from somewhere else needs.  It
    is off by default: an API key travels in a header the browser would then
    be free to send from any page.
    """
    app = FastAPI(title="CLM System One API", version="0.1.0")
    app.state.engine = engine
    if cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"],
                           allow_headers=["*"], allow_credentials=False, expose_headers=["X-CLM-Latency-Ms"])

    def auth(authorization: str | None):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid API key")

    @app.get("/health")
    def health():
        out = {"ok": True, "embedder": engine.embedder.healthy(),
               "models": [m["name"] for m in engine.models()],
               "cache": engine.arena.stats() if engine.arena else None}
        if getattr(engine, "mock", False):      # tools/playground_mock.py: the UI warns about fake numbers
            out["mock"] = True
        return out

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        auth(authorization)
        return {"models": engine.models()}

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"body is not JSON: {e}") from e
        if not isinstance(body, dict) or "state" not in body or not isinstance(body.get("questions"), dict):
            raise HTTPException(422, "body must be {state, model, questions}")
        try:
            temperature = float(body.get("temperature", 1.0))
        except (TypeError, ValueError) as e:
            raise HTTPException(422, "temperature must be a number") from e
        t0 = time.perf_counter()
        try:
            out = await asyncio.get_running_loop().run_in_executor(
                None, engine.answer, body["state"], body["questions"], body.get("model") or DEFAULT_MODEL, temperature)
        except ModelNotFound as e:
            raise HTTPException(422, str(e.args[0])) from e
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise HTTPException(422, f"invalid request: {e}") from e
        except EmbedderError as e:
            raise HTTPException(502, str(e)) from e
        return JSONResponse(out, headers={"X-CLM-Latency-Ms": f"{(time.perf_counter() - t0) * 1000:.1f}"})

    @app.post("/v1/rank")
    async def rank(request: Request, authorization: str | None = Header(default=None)):
        """{context, question, answers[, model, temperature]} -> answers ranked best first.

        The plain form of the same primitive: the state head sees ``context + question``,
        the action head sees each answer verbatim.
        """
        auth(authorization)
        try:
            body = await request.json()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"body is not JSON: {e}") from e
        if not isinstance(body, dict) or not isinstance(body.get("answers"), list) or not body["answers"]:
            raise HTTPException(422, "body must be {context, question, answers: [..]}")
        if not all(isinstance(a, str) and a for a in body["answers"]):
            raise HTTPException(422, "answers must be non-empty strings")
        try:
            temperature = float(body.get("temperature", 1.0))
        except (TypeError, ValueError) as e:
            raise HTTPException(422, "temperature must be a number") from e
        t0 = time.perf_counter()
        try:
            ranked = await asyncio.get_running_loop().run_in_executor(
                None, lambda: engine.rank(body.get("context") or "", body["answers"], body.get("question"),
                                          body.get("model") or DEFAULT_MODEL, temperature))
        except ModelNotFound as e:
            raise HTTPException(422, str(e.args[0])) from e
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise HTTPException(422, f"invalid request: {e}") from e
        except EmbedderError as e:
            raise HTTPException(502, str(e)) from e
        return JSONResponse({"model": body.get("model") or DEFAULT_MODEL, "ranked": ranked},
                            headers={"X-CLM-Latency-Ms": f"{(time.perf_counter() - t0) * 1000:.1f}"})

    # mounted last: the routes above shadow it, everything else is the static UI
    if ui and os.path.isdir(STATIC_DIR):
        @app.get("/", include_in_schema=False)
        @app.get("/index.html", include_in_schema=False)
        def playground():
            return HTMLResponse(_index_html(), headers={"Cache-Control": "no-cache"})

        app.mount("/", _RevalidatingStatic(directory=STATIC_DIR, html=True), name="playground")

    return app


def is_loopback(host: str) -> bool:
    """Whether ``host`` is reachable from this machine only.

    ``localhost`` and the loopback IPs are; ``0.0.0.0`` and ``::`` — every
    interface, which is how uvicorn reads them — are not, and neither is a
    hostname or an empty string, which resolves to this machine's own addresses.
    The caller only uses this to decide whether an API key is mandatory, so
    anything we cannot parse without network I/O counts as reachable.
    """
    host = (host or "").strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("CLM_HOST", "127.0.0.1"),
                    help="interface to bind (0.0.0.0 for every one); the default serves this machine "
                         "only, so the API is not reachable from the network.  Environment: CLM_HOST")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLM_PORT", 8700)))
    ap.add_argument("--emb-url", default=os.environ.get("CLM_EMB_URL", "http://127.0.0.1:8090/v1/embeddings"))
    ap.add_argument("--emb-model", default=os.environ.get("CLM_EMB_MODEL", "qwen3-8b"))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("CLM_EMB_MAX_TOKENS", 2048)),
                    help="truncate texts to this many tokens before embedding (the embedder's max-model-len)")
    ap.add_argument("--ckpt", default=os.environ.get("CLM_CKPT"),
                    help=f"checkpoint served as clm-latest (default: {DEFAULT_CKPT_DIR}/{HF_FILE}, downloaded if missing)")
    ap.add_argument("--ckpt-dir", default=None, help="also serve every *.pt in this directory under its file stem")
    ap.add_argument("--model", action="append", default=[], metavar="NAME=PATH", help="serve an extra checkpoint as NAME")
    ap.add_argument("--device", default=None,
                    help="device for the heads (cpu or cuda; default: cuda when available, CLM_DEVICE overrides)")
    ap.add_argument("--action-cache", default=None, metavar="BUDGET",
                    help="GPU memory reserved at start-up for reused state and action vectors: a fraction "
                         "of the device (0.02, the default) or a size (512MiB); 0 disables it. "
                         "Environment: CLM_ACTION_CACHE")
    ap.add_argument("--no-download", action="store_true", help="fail instead of downloading the reference head")
    ap.add_argument("--no-ui", action="store_true", help="do not serve the playground at /")
    ap.add_argument("--cors", action="store_true",
                    help="allow browser requests from any origin (needed to drive this server from a "
                         "playground served elsewhere)")
    ap.add_argument("--allow-anonymous", action="store_true",
                    help="serve without CLM_API_KEY even though the server is reachable beyond this "
                         "machine (a non-loopback --host, or --cors)")
    args = ap.parse_args()

    api_key = os.environ.get("CLM_API_KEY")
    exposed = args.cors or not is_loopback(args.host)
    if exposed and not api_key:
        if not args.allow_anonymous:
            raise SystemExit(
                f"refusing to serve: --host {args.host} / --cors makes this server reachable from other "
                "machines, and CLM_API_KEY is not set, so anyone who can reach the port can call the "
                "model.  Set CLM_API_KEY=... (clients then send \"Authorization: Bearer <key>\"), or "
                "pass --allow-anonymous if that exposure is intended.")
        print("[clm] warning: no CLM_API_KEY; serving anonymously on a reachable interface", flush=True)

    from .embedder import Embedder
    ckpt = args.ckpt
    if not ckpt and not args.no_download:
        ckpt = download()
    extra = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=PATH, got {spec!r}")
        extra[name] = path
    device = args.device or default_device()
    engine = Engine(Embedder(args.emb_url, args.emb_model, max_tokens=args.max_tokens), checkpoint=ckpt,
                    models=extra, checkpoint_dir=args.ckpt_dir, device=device,
                    action_cache=args.action_cache)
    if not engine.heads and not extra:
        raise SystemExit("no checkpoint: pass --ckpt or let clm-serve download the reference head")
    app = create_app(engine, api_key, ui=not args.no_ui, cors=args.cors)
    print(f"[clm] models {[m['name'] for m in engine.models()]} on {device}", flush=True)
    print(f"[clm] embedder {args.emb_url} ({args.emb_model}) {'up' if engine.embedder.healthy() else 'NOT REACHABLE'}; "
          f"auth {'on' if api_key else 'off'}", flush=True)
    arena = engine.arena
    if arena:
        pools = " + ".join(f"{p['capacity']:,}x{p['dim']}d" for p in arena.stats()["pools"].values())
        print(f"[clm] vector cache {arena.reserved_mb} MB reserved on {arena.device} ({pools})", flush=True)
    else:
        print("[clm] vector cache off", flush=True)
    print(f"[clm] POST http://{args.host}:{args.port}/v1/systemone", flush=True)
    if not args.no_ui:
        host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
        print(f"[clm] playground http://{host}:{args.port}/", flush=True)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
