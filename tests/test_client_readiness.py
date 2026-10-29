"""Offline contracts for ``CLMClient`` liveness vs readiness. No server, no encoder."""

import requests

from clm.client import CLMClient


class _Response:
    """A stand-in for ``requests.Response``: enough for the health helpers."""

    def __init__(self, payload=None, status_code=200, bad_json=False):
        self._payload, self.status_code, self._bad_json = payload, status_code, bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class _Session:
    """Records every GET and answers with one fixed response (or one fixed error)."""

    def __init__(self, response=None, error=None):
        self.response, self.error = response, error
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        return self.response


def client(session):
    c = CLMClient(base_url="http://clm.test")
    c._s = session
    return c


def test_liveness_stays_true_when_the_encoder_is_down():
    """The compatibility case from the issue: `ok` true, `embedder` false."""
    session = _Session(_Response({"ok": True, "embedder": False, "models": ["clm-latest"]}))
    assert client(session).health() is True
    assert session.calls == ["http://clm.test/health"]


def test_readiness_requires_both_liveness_and_the_encoder():
    session = _Session(_Response({"ok": True, "embedder": True, "models": ["clm-latest"]}))
    c = client(session)
    assert c.health() is True
    assert c.ready() is True


def test_encoder_down_is_not_ready():
    session = _Session(_Response({"ok": True, "embedder": False}))
    assert client(session).ready() is False


def test_dead_server_is_neither_live_nor_ready():
    session = _Session(_Response({"ok": True, "embedder": True}, status_code=502))
    c = client(session)
    assert c.health_info() == {}
    assert c.health() is False
    assert c.ready() is False


def test_unparsable_body_is_neither_live_nor_ready():
    session = _Session(_Response(bad_json=True))
    c = client(session)
    assert c.health_info() == {}
    assert c.health() is False
    assert c.ready() is False


def test_transport_failure_is_neither_live_nor_ready():
    session = _Session(error=requests.ConnectionError("connection refused"))
    c = client(session)
    assert c.health_info() == {}
    assert c.health() is False
    assert c.ready() is False


def test_missing_or_malformed_encoder_field_is_not_ready():
    """An absent field and a truthy non-boolean are both read conservatively."""
    for payload in ({"ok": True}, {"ok": True, "embedder": None}, {"ok": True, "embedder": "true"},
                    {"ok": True, "embedder": 1}):
        assert client(_Session(_Response(payload))).ready() is False


def test_a_non_object_payload_is_neither_live_nor_ready():
    session = _Session(_Response(["ok"]))
    c = client(session)
    assert c.health_info() == {}
    assert c.health() is False
    assert c.ready() is False


def test_health_info_passes_the_payload_through():
    payload = {"ok": True, "embedder": True, "models": ["clm-latest"], "cache": {"hit_rate": 0.5}}
    assert client(_Session(_Response(payload))).health_info() == payload
