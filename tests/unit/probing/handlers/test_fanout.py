"""Unit tests for torch profiler cluster fan-out."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from probing.handlers.fanout import discover_peer_addrs, fanout_start


def _response(payload, status=200):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_discover_peer_addrs_excludes_local_rank(monkeypatch):
    monkeypatch.setenv("PROBING_PORT", "9700")
    monkeypatch.setenv("RANK", "1")
    nodes = {
        "nodes": [
            {"host": "h0", "addr": "10.0.0.1:9700", "rank": 0},
            {"host": "h1", "addr": "10.0.0.2:9700", "rank": 1},
            {"host": "h2", "addr": "10.0.0.3:9700", "rank": 2},
        ]
    }
    monkeypatch.setattr(
        "probing.handlers.fanout.urlopen",
        lambda url, timeout=3.0: _response(nodes),
    )
    assert discover_peer_addrs() == ["10.0.0.1:9700", "10.0.0.3:9700"]


def test_discover_peer_addrs_empty_without_port(monkeypatch):
    monkeypatch.delenv("PROBING_PORT", raising=False)
    assert discover_peer_addrs() == []


def test_discover_peer_addrs_ignores_non_numeric_rank(monkeypatch):
    monkeypatch.setenv("PROBING_PORT", "9700")
    monkeypatch.setenv("RANK", "0")
    nodes = {
        "nodes": [
            {"host": "h0", "addr": "10.0.0.1:9700", "rank": "not-a-rank"},
        ]
    }
    monkeypatch.setattr(
        "probing.handlers.fanout.urlopen",
        lambda url, timeout=3.0: _response(nodes),
    )
    assert discover_peer_addrs() == ["10.0.0.1:9700"]


def test_fanout_start_requests_each_peer(monkeypatch):
    monkeypatch.setenv("PROBING_PORT", "9700")
    monkeypatch.setenv("RANK", "0")
    nodes = {
        "nodes": [
            {"host": "h0", "addr": "10.0.0.1:9700", "rank": 0},
            {"host": "h1", "addr": "10.0.0.2:9700", "rank": 1},
        ]
    }

    calls = []

    def fake_urlopen(url, timeout=8.0):
        calls.append((url, timeout))
        return _response({"success": True})

    monkeypatch.setattr(
        "probing.handlers.fanout.urlopen", fake_urlopen
    )

    summary = fanout_start(steps=7, trigger="http", analysis="roofline")
    assert summary["peers_attempted"] == 1
    assert summary["peers_ok"] == 1
    assert len(calls) == 2  # node discovery + one peer request
    peer_url = calls[1][0]
    assert "/apis/pythonext/pytorch/profile/start?" in peer_url
    assert "steps=7" in peer_url
    assert "cluster=false" in peer_url
    assert "analysis=roofline" in peer_url


def test_fanout_start_counts_peer_json_failure(monkeypatch):
    monkeypatch.setenv("PROBING_PORT", "9700")
    monkeypatch.setenv("RANK", "0")
    nodes = {
        "nodes": [
            {"host": "h0", "addr": "10.0.0.1:9700", "rank": 0},
            {"host": "h1", "addr": "10.0.0.2:9700", "rank": 1},
        ]
    }

    responses = {
        "http://127.0.0.1:9700/apis/nodes?offset=0&limit=10000": nodes,
        "http://10.0.0.2:9700/apis/pythonext/pytorch/profile/start?": {"success": False, "error": "already running"},
    }

    def fake_urlopen(url, timeout=8.0):
        for prefix, payload in responses.items():
            if url.startswith(prefix):
                return _response(payload)
        return _response({"success": True})

    monkeypatch.setattr(
        "probing.handlers.fanout.urlopen", fake_urlopen
    )

    summary = fanout_start(steps=7, trigger="http", analysis="roofline")
    assert summary["peers_attempted"] == 1
    assert summary["peers_ok"] == 0
    assert summary["peers_failed"] == 1


def test_fanout_start_treats_missing_success_as_failure(monkeypatch):
    monkeypatch.setenv("PROBING_PORT", "9700")
    monkeypatch.setenv("RANK", "0")
    nodes = {
        "nodes": [
            {"host": "h0", "addr": "10.0.0.1:9700", "rank": 0},
            {"host": "h1", "addr": "10.0.0.2:9700", "rank": 1},
        ]
    }

    def fake_urlopen(url, timeout=8.0):
        if "/apis/nodes" in url:
            return _response(nodes)
        return _response({"error": "already running", "traceback": "..."})

    monkeypatch.setattr(
        "probing.handlers.fanout.urlopen", fake_urlopen
    )

    summary = fanout_start(steps=7, trigger="http", analysis="roofline")
    assert summary["peers_attempted"] == 1
    assert summary["peers_ok"] == 0
    assert summary["peers_failed"] == 1
