"""Offline wiring tests for the encoder API key. No server, encoder, model or GPU."""

import sys
import types

import clm.embedder
import clm.server

ENCODER_KEY = "sk-encoder-test"
FRONT_KEY = "sk-front-test"


class _FakeEmbedder:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def healthy(self):
        return True


class _FakeEngine:
    def __init__(self, embedder, **kwargs):
        self.embedder, self.kwargs = embedder, kwargs
        self.heads = [object()]     # a checkpoint is loaded, so main() does not exit
        self.arena = None

    def models(self):
        return [{"name": "clm-latest"}]


def _serve(monkeypatch):
    """Run ``clm.server.main()`` with everything that reaches a model or a socket stubbed."""
    captured = {}

    def embedder(*args, **kwargs):
        captured["embedder"] = kwargs
        return _FakeEmbedder(*args, **kwargs)

    def create_app(engine, api_key=None, **kwargs):
        captured["api_key"] = api_key
        return object()

    monkeypatch.setattr(clm.embedder, "Embedder", embedder)     # main() imports it lazily
    monkeypatch.setattr(clm.server, "Engine", _FakeEngine)
    monkeypatch.setattr(clm.server, "create_app", create_app)
    monkeypatch.setattr(clm.server, "download", lambda: "ckpt.pt")
    monkeypatch.setattr(clm.server, "default_device", lambda: "cpu")
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: None))
    monkeypatch.setattr(sys, "argv", ["clm-serve"])
    clm.server.main()
    return captured


def test_encoder_key_reaches_the_embedder(monkeypatch, capsys):
    monkeypatch.setenv("CLM_EMB_API_KEY", ENCODER_KEY)
    captured = _serve(monkeypatch)
    assert captured["embedder"]["api_key"] == ENCODER_KEY


def test_startup_reports_encoder_auth_without_printing_the_key(monkeypatch, capsys):
    monkeypatch.setenv("CLM_EMB_API_KEY", ENCODER_KEY)
    _serve(monkeypatch)
    out = capsys.readouterr().out
    assert "encoder auth on" in out
    assert ENCODER_KEY not in out


def test_unset_key_keeps_the_unauthenticated_encoder(monkeypatch, capsys):
    monkeypatch.delenv("CLM_EMB_API_KEY", raising=False)
    captured = _serve(monkeypatch)
    assert captured["embedder"]["api_key"] is None
    assert "encoder auth off" in capsys.readouterr().out


def test_empty_key_keeps_the_unauthenticated_encoder(monkeypatch, capsys):
    monkeypatch.setenv("CLM_EMB_API_KEY", "")
    assert _serve(monkeypatch)["embedder"]["api_key"] is None


def test_front_and_encoder_keys_stay_independent(monkeypatch, capsys):
    monkeypatch.setenv("CLM_API_KEY", FRONT_KEY)
    monkeypatch.setenv("CLM_EMB_API_KEY", ENCODER_KEY)
    captured = _serve(monkeypatch)
    assert captured["api_key"] == FRONT_KEY
    assert captured["embedder"]["api_key"] == ENCODER_KEY
    out = capsys.readouterr().out
    assert FRONT_KEY not in out and ENCODER_KEY not in out


def test_embedder_sends_the_key_as_a_bearer_header():
    assert clm.embedder.Embedder(api_key=ENCODER_KEY).session.headers["Authorization"] == f"Bearer {ENCODER_KEY}"


def test_embedder_sends_no_auth_header_without_a_key():
    assert "Authorization" not in clm.embedder.Embedder().session.headers
