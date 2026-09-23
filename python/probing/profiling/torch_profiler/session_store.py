"""In-process store for profile captures and hotspot rows (no memtable)."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Optional


def _max_sessions() -> int:
    raw = os.environ.get("PROBING_TORCH_PROFILER_MAX_SESSIONS", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        return 8
    return max(value, 1)


@dataclass
class CaptureRecord:
    capture_id: str
    local_step: int = -1
    global_step: int = -1
    rank: int = -1
    world_size: int = -1
    role: str = ""
    trigger: str = ""
    steps_profiled: int = 0
    wall_us: int = 0
    started_at_us: int = 0
    ended_at_us: int = 0
    status: str = "running"
    truncated: bool = False
    event_count: int = 0
    error: str = ""
    analysis: str = "none"
    roofline_quality: str = "unavailable"
    roofline_counter_events: int = 0
    roofline_associated_kernels: int = 0
    roofline_unassociated_kernels: int = 0
    roofline_missing_metrics: str = "[]"
    roofline_parser_version: str = ""
    counter_backend: str = "none"
    device_vendor: str = ""
    device_model: str = ""
    device_arch: str = ""


@dataclass
class HotspotRecord:
    capture_id: str
    local_step: int = -1
    global_step: int = -1
    rank: int = -1
    bucket_kind: str = "other"
    bucket_name: str = ""
    self_us: int = 0
    wall_us: int = 0
    calls: int = 0
    pct_of_capture: float = 0.0
    module_hint: str = ""


@dataclass
class CounterRecord:
    capture_id: str
    local_step: int = -1
    global_step: int = -1
    rank: int = -1
    role: str = ""
    kernel_name: str = ""
    op_name: str = ""
    top_level_op: str = ""
    bottom_level_op: str = ""
    op_stack: str = "[]"
    calls: int = 0
    duration_us: int | None = 0
    flops: int | None = 0
    dram_bytes: int | None = 0
    metrics: str = "{}"


@dataclass
class RooflineRecord:
    capture_id: str
    local_step: int = -1
    global_step: int = -1
    rank: int = -1
    role: str = ""
    op_name: str = ""
    kernel_name: str = ""
    calls: int = 0
    self_duration_us: int = 0
    flops: int = 0
    dram_bytes: int = 0
    arithmetic_intensity: float | None = None
    achieved_flops: float | None = None
    achieved_bytes_per_sec: float | None = None
    peak_flops: float | None = None
    peak_bytes_per_sec: float | None = None
    peak_flops_kind: str = "fp16_tensor_dense"
    boundedness: float | None = None
    bottleneck: str = "unknown"
    data_quality: str = "partial"


@dataclass
class SessionStore:
    """Bounded in-memory captures + hotspot fact rows."""

    max_sessions: int = field(default_factory=_max_sessions)
    _captures: list[CaptureRecord] = field(default_factory=list)
    _hotspots: list[HotspotRecord] = field(default_factory=list)
    _counters: list[CounterRecord] = field(default_factory=list)
    _rooflines: list[RooflineRecord] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def add_capture(
        self,
        capture: CaptureRecord,
        hotspots: list[HotspotRecord],
        counters: list[CounterRecord] | None = None,
        rooflines: list[RooflineRecord] | None = None,
    ) -> None:
        with self._lock:
            self._captures.append(capture)
            self._hotspots.extend(hotspots)
            self._counters.extend(counters or [])
            self._rooflines.extend(rooflines or [])
            overflow = len(self._captures) - self.max_sessions
            if overflow > 0:
                drop_ids = {c.capture_id for c in self._captures[:overflow]}
                self._captures = self._captures[overflow:]
                self._hotspots = [
                    h for h in self._hotspots if h.capture_id not in drop_ids
                ]
                self._counters = [
                    c for c in self._counters if c.capture_id not in drop_ids
                ]
                self._rooflines = [
                    r for r in self._rooflines if r.capture_id not in drop_ids
                ]

    def replace_capture(
        self,
        capture: CaptureRecord,
        hotspots: list[HotspotRecord],
        counters: list[CounterRecord] | None = None,
        rooflines: list[RooflineRecord] | None = None,
    ) -> None:
        """Replace an existing capture (same id) with its compiled rows."""
        with self._lock:
            for index, existing in enumerate(self._captures):
                if existing.capture_id != capture.capture_id:
                    continue
                self._captures[index] = capture
                self._hotspots = [
                    h for h in self._hotspots if h.capture_id != capture.capture_id
                ]
                self._counters = [
                    c for c in self._counters if c.capture_id != capture.capture_id
                ]
                self._rooflines = [
                    r for r in self._rooflines if r.capture_id != capture.capture_id
                ]
                self._hotspots.extend(hotspots)
                self._counters.extend(counters or [])
                self._rooflines.extend(rooflines or [])
                return
            self.add_capture(capture, hotspots, counters, rooflines)

    def captures(self) -> list[CaptureRecord]:
        with self._lock:
            return list(self._captures)

    def hotspots(self) -> list[HotspotRecord]:
        with self._lock:
            return list(self._hotspots)

    def counters(self) -> list[CounterRecord]:
        with self._lock:
            return list(self._counters)

    def rooflines(self) -> list[RooflineRecord]:
        with self._lock:
            return list(self._rooflines)

    def latest_capture_id(self) -> Optional[str]:
        with self._lock:
            if not self._captures:
                return None
            return self._captures[-1].capture_id

    def clear(self) -> None:
        with self._lock:
            self._captures.clear()
            self._hotspots.clear()
            self._counters.clear()
            self._rooflines.clear()


_STORE: Optional[SessionStore] = None
_STORE_LOCK = threading.Lock()


def get_session_store() -> SessionStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = SessionStore()
        return _STORE


def reset_session_store_for_tests() -> None:
    """Clear captures/hotspots between tests (not for production)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.clear()


def roofline_peaks(
    backend: str | None = None,
    device_arch: str | None = None,
) -> tuple[float | None, float | None]:
    if backend == "rocm":
        raw = os.environ.get("PROBING_TORCH_ROOFLINE_ROCM_PEAKS_JSON", "").strip()
    else:
        raw = os.environ.get("PROBING_TORCH_ROOFLINE_PEAKS_JSON", "").strip()
    if not raw:
        return None, None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("peaks JSON must be an object")
        if backend is not None and parsed.get("backend") not in {None, backend}:
            raise ValueError(
                f"peaks JSON backend {parsed.get('backend')!r} does not match {backend!r}"
            )
        if device_arch is not None and parsed.get("device_arch") not in {None, device_arch}:
            raise ValueError(
                "peaks JSON device_arch "
                f"{parsed.get('device_arch')!r} does not match {device_arch!r}"
            )
        if isinstance(parsed.get("peaks"), dict):
            entry = parsed["peaks"].get("fp16_tensor_dense")
        else:
            entry = parsed.get("fp16_tensor_dense")
        if not isinstance(entry, dict):
            raise ValueError("peaks JSON missing fp16_tensor_dense")
        peak_flops = entry.get("peak_flops")
        peak_bytes = entry.get("peak_bytes_per_sec")
        if not isinstance(peak_flops, (int, float)) or not isinstance(
            peak_bytes, (int, float)
        ):
            raise ValueError("peaks must be numbers")
        if peak_flops == 0 or peak_bytes == 0:
            import logging

            logging.getLogger(__name__).debug(
                "roofline peaks are uncalibrated (zero); treating as unavailable"
            )
            return None, None
        if peak_flops < 0 or peak_bytes < 0:
            raise ValueError("peaks must be non-negative numbers")
        return float(peak_flops), float(peak_bytes)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        import logging

        logging.getLogger(__name__).warning(
            "invalid roofline peaks JSON: %s", exc
        )
        return None, None
