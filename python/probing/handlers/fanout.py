"""Cluster fan-out for on-demand torch profiler captures.

``pytorch/profile/start`` is local by design (it drives the in-process
``ProfilerController``). In a torchrun job the operator can request fan-out
with ``cluster=true`` (or by setting
``PROBING_TORCH_PROFILER_CLUSTER_FANOUT=1``); this module then discovers peers
from the local ``GET /apis/nodes`` registry and asks each rank to start its own
capture with ``cluster=false``. Every rank keeps an independent
``python.profile_capture`` and the operator aggregates them with
``cluster query``, matching roofline-backends.zh.md section 9.

This lives in ``probing.handlers`` rather than ``torch_profiler`` because it is
HTTP control-plane fan-out for the Python extension routes. The Rust
``probing-server`` ``cluster_fanout`` service fans out SQL/dataframe queries and
is not wired to invoke Python handler functions; bridging it would require a
server rebuild, so the handler layer keeps a minimal stdlib-only transport here.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

logger = logging.getLogger(__name__)

FANOUT_ENV = "PROBING_TORCH_PROFILER_CLUSTER_FANOUT"
_NODES_PAGE_LIMIT = 10_000


def cluster_fanout_enabled() -> bool:
    return os.environ.get(FANOUT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _local_nodes_url() -> str | None:
    port = os.environ.get("PROBING_PORT", "").strip()
    if not port:
        return None
    return f"http://127.0.0.1:{port}/apis/nodes?offset=0&limit={_NODES_PAGE_LIMIT}"


def _rank_matches(node_rank: Any, local_rank: int) -> bool:
    try:
        return int(node_rank) == local_rank
    except (TypeError, ValueError):
        return False


def discover_peer_addrs(timeout_s: float = 3.0) -> list[str]:
    """Return reachable peer ``host:port`` strings, excluding this global rank."""
    url = _local_nodes_url()
    if url is None:
        return []
    try:
        with urlopen(url, timeout=timeout_s) as response:
            document = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.debug("cluster fan-out node discovery failed: %s", exc)
        return []

    local_rank = _env_int("RANK")
    peers: list[str] = []
    for node in document.get("nodes") or []:
        addr = (node.get("addr") or "").strip()
        if not addr:
            continue
        if local_rank is not None and _rank_matches(node.get("rank"), local_rank):
            continue
        peers.append(addr)
    return peers


def _get_json(url: str, timeout_s: float) -> tuple[int | None, Any]:
    try:
        with urlopen(url, timeout=timeout_s) as response:
            status = getattr(response, "status", 200)
            body = response.read().decode("utf-8", "replace")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:400]
    except (URLError, TimeoutError, OSError) as exc:
        return None, str(exc)

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = body[:400]
    return status, payload


def _peer_succeeded(status: int | None, payload: Any) -> bool:
    if status != 200:
        return False
    if isinstance(payload, dict) and payload.get("success") is False:
        return False
    return True


def fanout_start(
    steps: int,
    trigger: str,
    analysis: str | None,
    timeout_s: float = 8.0,
) -> dict[str, Any]:
    peers = discover_peer_addrs()
    results: list[dict[str, Any]] = []
    for addr in peers:
        query = urlencode(
            {
                "steps": steps,
                "trigger": trigger,
                "analysis": analysis or "",
                "cluster": "false",
            }
        )
        status, payload = _get_json(
            f"http://{addr}/apis/pythonext/pytorch/profile/start?{query}",
            timeout_s,
        )
        results.append(
            {
                "addr": addr,
                "status": status,
                "success": _peer_succeeded(status, payload),
                "response": payload,
            }
        )

    ok = sum(1 for item in results if item.get("success"))
    failed = len(results) - ok
    return {
        "peers_attempted": len(results),
        "peers_ok": ok,
        "peers_failed": failed,
        "results": results,
    }


def fanout_stop(timeout_s: float = 8.0) -> dict[str, Any]:
    peers = discover_peer_addrs()
    results: list[dict[str, Any]] = []
    for addr in peers:
        status, payload = _get_json(
            f"http://{addr}/apis/pythonext/pytorch/profile/stop?cluster=false",
            timeout_s,
        )
        results.append(
            {
                "addr": addr,
                "status": status,
                "success": _peer_succeeded(status, payload),
                "response": payload,
            }
        )

    ok = sum(1 for item in results if item.get("success"))
    failed = len(results) - ok
    return {
        "peers_attempted": len(results),
        "peers_ok": ok,
        "peers_failed": failed,
        "results": results,
    }
