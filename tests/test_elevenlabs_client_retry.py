"""Tests for ElevenLabsClient's retry/backoff behavior.

Real network calls are never made here — requests.get is monkeypatched to
return scripted responses, and time.sleep is monkeypatched to a no-op so
these run instantly regardless of backoff delay. What's under test is the
retry DECISION (which statuses get retried, which don't, and that retries
actually stop and raise once exhausted), not real timing or real HTTP.
"""

import pytest
import requests

from agent_config_judge.elevenlabs_client import ElevenLabsApiError, ElevenLabsClient


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.headers = headers or {}
        self.text = str(json_body)

    def json(self):
        return self._json_body


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # Retries must not actually pause the test suite — only the decision to
    # retry (and how many times) is under test here.
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)


def _client(max_retries=4) -> ElevenLabsClient:
    return ElevenLabsClient(api_key="test-key", max_retries=max_retries, retry_base_delay_secs=0.001)


def test_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return _FakeResponse(429)
        return _FakeResponse(200, {"agents": []})

    monkeypatch.setattr(requests, "get", fake_get)
    client = _client()
    result = client._get("/v1/convai/agents")

    assert result == {"agents": []}
    assert calls["n"] == 3  # two 429s, then the successful third attempt


def test_does_not_retry_on_404(monkeypatch):
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        return _FakeResponse(404, {"detail": "not found"})

    monkeypatch.setattr(requests, "get", fake_get)
    client = _client()

    with pytest.raises(ElevenLabsApiError):
        client._get("/v1/convai/agents/does-not-exist")
    assert calls["n"] == 1  # no retry on a non-retryable status


def test_does_not_retry_on_401(monkeypatch):
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        return _FakeResponse(401, {"detail": "bad key"})

    monkeypatch.setattr(requests, "get", fake_get)
    client = _client()

    with pytest.raises(ElevenLabsApiError):
        client._get("/v1/convai/agents")
    assert calls["n"] == 1  # retrying won't fix a bad api_key


def test_raises_after_retries_exhausted_on_persistent_5xx(monkeypatch):
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        return _FakeResponse(503)

    monkeypatch.setattr(requests, "get", fake_get)
    client = _client(max_retries=2)

    with pytest.raises(ElevenLabsApiError):
        client._get("/v1/convai/agents")
    assert calls["n"] == 3  # first attempt + 2 retries, then give up


def test_retries_on_connection_error_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.exceptions.ConnectionError("dropped")
        return _FakeResponse(200, {"agents": []})

    monkeypatch.setattr(requests, "get", fake_get)
    client = _client()
    result = client._get("/v1/convai/agents")

    assert result == {"agents": []}
    assert calls["n"] == 2


def test_honors_retry_after_header(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda secs: sleep_calls.append(secs))
    calls = {"n": 0}

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(429, headers={"Retry-After": "0.05"})
        return _FakeResponse(200, {"agents": []})

    monkeypatch.setattr(requests, "get", fake_get)
    client = _client()
    client._get("/v1/convai/agents")

    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 0.05  # base delay honored (jitter only adds on top)
