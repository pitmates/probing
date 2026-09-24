"""Vendor backends for roofline counter acquisition.

The CUDA path keeps the existing in-process Kineto/CUPTI flow. The ROCm path
adds detection/probing, launcher wrapper rendering, and offline artifact import. Tuning lives
in ``PROBING_TORCH_ROOFLINE_CONFIG`` (see ``.config``); legacy per-vendor env
variables remain as fallbacks.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .rocm_metrics import ROCM_DEFAULT_METRICS
from .rocm_runner import sidecar_command, sidecar_enabled
from .config import load_roofline_config

ROCM_PROBE_CMD_ENV = "PROBING_TORCH_ROOFLINE_ROCPROF_PROBE_CMD"


def _rocm_probe_command() -> Optional[str]:
    config = load_roofline_config()
    if config.probe_cmd:
        return config.probe_cmd
    return os.environ.get(ROCM_PROBE_CMD_ENV, "").strip() or None


def _run_rocm_probe(command: str, timeout_s: float = 15.0) -> tuple[bool, str]:
    """Run an operator-configured dry-run probe; return (ok, diagnostic)."""
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"rocprofiler probe timed out after {timeout_s}s"
    except Exception as exc:
        return False, f"failed to run rocprofiler probe: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return (
            False,
            f"rocprofiler probe exited with {completed.returncode}: {detail[:300]}",
        )
    return True, ""


@dataclass(frozen=True)
class BackendInfo:
    vendor: str
    device_model: str
    device_arch: str
    counter_source: str
    diagnostic: str = ""


@dataclass(frozen=True)
class CapabilityResult:
    backend: BackendInfo
    status: str
    error: str = ""
    experimental_config: Any = None


CUDA_DEFAULT_METRICS: tuple[str, ...] = (
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)


def _device_properties(torch_module: Any) -> Any:
    if not torch_module.cuda.is_available():
        return None
    try:
        return torch_module.cuda.get_device_properties(0)
    except Exception:
        return None


def detect_backend(torch_module: Any) -> BackendInfo:
    hip_version = getattr(getattr(torch_module, "version", None), "hip", None)
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    props = _device_properties(torch_module)
    model = ""
    arch = ""

    if isinstance(props, object):
        model_value = getattr(props, "name", "")
        model = model_value if isinstance(model_value, str) else ""

    if isinstance(hip_version, str) and hip_version.strip():
        vendor = "amd"
        source = "rocm"
        arch_value = getattr(props, "gcnArchName", None)
        if isinstance(arch_value, str) and arch_value.strip():
            arch = arch_value
        elif (
            isinstance(getattr(props, "major", None), int)
            and isinstance(getattr(props, "minor", None), int)
        ):
            arch = f"gfx{getattr(props, 'major')}{getattr(props, 'minor')}"
        else:
            arch = "unknown"
    elif isinstance(cuda_version, str) and cuda_version.strip():
        vendor = "nvidia"
        source = "cuda"
        if isinstance(getattr(props, "major", None), int) and isinstance(
            getattr(props, "minor", None), int
        ):
            arch = f"sm_{getattr(props, 'major')}{getattr(props, 'minor')}"
        else:
            arch = "unknown"
    else:
        try:
            cuda_available = bool(torch_module.cuda.is_available())
        except Exception:
            cuda_available = False
        if cuda_available:
            vendor = "nvidia"
            source = "cuda"
            arch = "unknown"
        else:
            vendor = "unknown"
            source = "none"
            arch = "unknown"

    return BackendInfo(
        vendor=vendor,
        device_model=model,
        device_arch=arch,
        counter_source=source,
    )


def selected_backend(torch_module: Any) -> BackendInfo:
    detected = detect_backend(torch_module)
    config = load_roofline_config()
    requested = config.backend
    if requested == "auto":
        requested = os.environ.get("PROBING_TORCH_ROOFLINE_BACKEND", "auto").strip().lower()
    if requested in {"", "auto"}:
        return detected
    if requested == "cuda" and detected.counter_source != "cuda":
        return BackendInfo(
            vendor=detected.vendor,
            device_model=detected.device_model,
            device_arch=detected.device_arch,
            counter_source="none",
            diagnostic="PROBING_TORCH_ROOFLINE_BACKEND=cuda requested on a non-CUDA device",
        )
    if requested == "rocm" and detected.counter_source != "rocm":
        return BackendInfo(
            vendor=detected.vendor,
            device_model=detected.device_model,
            device_arch=detected.device_arch,
            counter_source="none",
            diagnostic="PROBING_TORCH_ROOFLINE_BACKEND=rocm requested on a non-ROCm device",
        )
    return detected


class RooflineBackend(ABC):
    """Vendor-specific roofline backend.

    CUDA owns the Kineto/CUPTI experimental config, capability probe, and
    counter compilation. ROCm owns detection, launcher wrapper rendering, and offline artifact
    import (tuned by ``PROBING_TORCH_ROOFLINE_CONFIG``).
    """

    def __init__(self, info: BackendInfo) -> None:
        self.info = info
        self._capability_error = ""
        self._runtime_error = ""

    @abstractmethod
    def metrics(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def probe(self, torch_module: Any) -> CapabilityResult:
        raise NotImplementedError

    @abstractmethod
    def build_profiler_kwargs(self, base_kwargs: dict[str, Any], torch_module: Any) -> dict[str, Any]:
        raise NotImplementedError

    def probe_capabilities(self, torch_module: Any) -> CapabilityResult:
        return self.probe(torch_module)

    def note_collection_error(self, message: str) -> None:
        """Record a runtime collection failure surfaced when the capture compiles."""
        if message and not self._runtime_error:
            self._runtime_error = message

    @staticmethod
    def detect(torch_module: Any) -> BackendInfo:
        return detect_backend(torch_module)

    def platform_peaks(self, info: Optional[BackendInfo] = None) -> tuple[float | None, float | None]:
        from .session_store import roofline_peaks

        return roofline_peaks(
            info.counter_source if info is not None else None,
            info.device_arch if info is not None else None,
        )

    @abstractmethod
    def compile_counter_rows(
        self,
        raw_events: list[Any],
        *,
        capture_id: str,
        local_step: int,
        global_step: int,
        rank: int,
        role: str,
        timeline_events: Any = None,
    ) -> Any:
        raise NotImplementedError


class CudaRooflineBackend(RooflineBackend):
    def __init__(self, info: BackendInfo) -> None:
        super().__init__(info)
        self._experimental_config: Any = None

    def metrics(self) -> tuple[str, ...]:
        raw = os.environ.get("PROBING_TORCH_ROOFLINE_METRICS", "").strip()
        if raw:
            metrics = tuple(item.strip() for item in raw.split(",") if item.strip())
            if metrics:
                return metrics
        return CUDA_DEFAULT_METRICS

    def probe(self, torch_module: Any) -> CapabilityResult:
        experimental_config = getattr(torch_module.profiler, "_ExperimentalConfig", None)
        if experimental_config is None:
            raise RuntimeError(
                "roofline counters require a PyTorch version with "
                "torch.profiler._ExperimentalConfig"
            )
        try:
            self._experimental_config = experimental_config(
                profiler_metrics=list(self.metrics()),
                profiler_measure_per_kernel=True,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "roofline counters require a compatible PyTorch/CUPTI runtime"
            ) from exc
        if not torch_module.cuda.is_available():
            raise RuntimeError("roofline counters require an available CUDA device")
        activities = [
            torch_module.profiler.ProfilerActivity.CPU,
            torch_module.profiler.ProfilerActivity.CUDA,
        ]
        try:
            probe = torch_module.profiler.profile(
                activities=activities,
                experimental_config=self._experimental_config,
            )
        except Exception as exc:
            raise RuntimeError(
                "roofline capability check failed: PyTorch/Kineto rejected "
                "CUPTI Range Profiler or the requested metrics"
            ) from exc
        try:
            probe.__enter__()
        except Exception as exc:
            raise RuntimeError(
                "roofline capability check failed: PyTorch/Kineto rejected "
                "CUPTI Range Profiler or the requested metrics"
            ) from exc
        finally:
            try:
                probe.__exit__(None, None, None)
            except Exception:
                pass
        return CapabilityResult(
            backend=self.info,
            status="ok",
            experimental_config=self._experimental_config,
        )

    def build_profiler_kwargs(self, base_kwargs: dict[str, Any], torch_module: Any) -> dict[str, Any]:
        if self._experimental_config is None:
            self.probe(torch_module)
        kwargs = dict(base_kwargs)
        kwargs["experimental_config"] = self._experimental_config
        return kwargs

    def compile_counter_rows(
        self,
        raw_events: list[Any],
        *,
        capture_id: str,
        local_step: int,
        global_step: int,
        rank: int,
        role: str,
        timeline_events: Any = None,
    ) -> Any:
        del timeline_events
        from .adaptor import _compile_counter_rows

        return _compile_counter_rows(
            raw_events,
            capture_id=capture_id,
            local_step=local_step,
            global_step=global_step,
            rank=rank,
            role=role,
            peaks=self.platform_peaks(self.info),
        )


class RocmRooflineBackend(RooflineBackend):
    def metrics(self) -> tuple[str, ...]:
        config = load_roofline_config()
        if config.metrics:
            return config.metrics
        raw = os.environ.get("PROBING_TORCH_ROOFLINE_ROCM_METRICS", "").strip()
        metrics = tuple(item.strip() for item in raw.split(",") if item.strip())
        if metrics:
            return metrics
        return ROCM_DEFAULT_METRICS

    def probe(self, torch_module: Any) -> CapabilityResult:
        del torch_module
        if not sidecar_enabled():
            self._capability_error = (
                "rocm roofline wrapper collection is disabled (PROBING_TORCH_ROOFLINE_CONFIG "
                "enabled=false or PROBING_TORCH_ROOFLINE_ROCM_PROFILE=0)"
            )
            return CapabilityResult(
                backend=self.info,
                status="unavailable",
                error=self._capability_error,
            )
        command = sidecar_command()
        if not command:
            self._capability_error = (
                "rocm roofline wrapper collection is enabled but no rocprof command "
                "template is configured"
            )
            return CapabilityResult(
                backend=self.info,
                status="unavailable",
                error=self._capability_error,
            )
        probe_command = _rocm_probe_command()
        if probe_command:
            probe_ok, probe_error = _run_rocm_probe(probe_command)
            if not probe_ok:
                self._capability_error = (
                    "rocprofiler capability probe failed: " + probe_error
                )
                return CapabilityResult(
                    backend=self.info,
                    status="unavailable",
                    error=self._capability_error,
                )
        self._capability_error = ""
        return CapabilityResult(
            backend=self.info,
            status="ok",
        )

    def build_profiler_kwargs(self, base_kwargs: dict[str, Any], torch_module: Any) -> dict[str, Any]:
        return dict(base_kwargs)

    def compile_counter_rows(
        self,
        raw_events: list[Any],
        *,
        capture_id: str,
        local_step: int,
        global_step: int,
        rank: int,
        role: str,
        timeline_events: Any = None,
    ) -> Any:
        from .adaptor import (
            _RooflineCompileResult,
            join_rocm_rows_with_timeline,
        )
        from .rocm_sidecar import build_counter_records, parse_counter_artifact

        if not raw_events:
            return _RooflineCompileResult(
                counters=[],
                rooflines=[],
                quality="unavailable",
                counter_events=0,
                associated_kernels=0,
                unassociated_kernels=0,
                missing_metrics=[],
                error=self._runtime_error
                or self._capability_error
                or "no rocm counter artifacts were produced",
            )
        try:
            rows = parse_counter_artifact(raw_events)
        except (ValueError, TypeError) as exc:
            return _RooflineCompileResult(
                counters=[],
                rooflines=[],
                quality="unavailable",
                counter_events=0,
                associated_kernels=0,
                unassociated_kernels=0,
                missing_metrics=[],
                error=f"rocm artifact parse failed: {exc}",
            )
        if not rows:
            return _RooflineCompileResult(
                counters=[],
                rooflines=[],
                quality="unavailable",
                counter_events=0,
                associated_kernels=0,
                unassociated_kernels=0,
                missing_metrics=[],
                error="rocm artifact contained no counter rows",
            )
        rows = join_rocm_rows_with_timeline(rows, timeline_events or [])
        counters, missing_metrics = build_counter_records(
            rows,
            capture_id=capture_id,
            local_step=local_step,
            global_step=global_step,
            rank=rank,
            role=role,
        )
        associated = sum(1 for row in rows if row.get("op_name"))
        unassociated = len(rows) - associated
        rooflines: list[Any] = []
        quality = "partial"
        error = ""
        if any(counter.flops is not None for counter in counters):
            peaks = self.platform_peaks(self.info)
            if peaks[0] is None or peaks[1] is None:
                error = (
                    "ROCm FLOP weights are calibrated but platform peaks are "
                    "missing or uncalibrated; roofline efficiency was not computed"
                )
            else:
                quality = "ok" if not missing_metrics and not unassociated else "partial"
                rooflines = _build_rocm_roofline_records(
                    counters,
                    peaks=peaks,
                    quality=quality,
                    capture_id=capture_id,
                    local_step=local_step,
                    global_step=global_step,
                    rank=rank,
                    role=role,
                )
                if not rooflines:
                    quality = "partial"
                    if not error:
                        error = (
                            "ROCm counter rows are present but no roofline rows "
                            "could be derived (missing duration/flops/bytes)"
                        )
        return _RooflineCompileResult(
            counters=counters,
            rooflines=rooflines,
            quality=quality,
            counter_events=len(counters),
            associated_kernels=associated,
            unassociated_kernels=unassociated,
            missing_metrics=missing_metrics,
            error=error,
        )


class UnavailableRooflineBackend(RooflineBackend):
    def __init__(self, info: BackendInfo) -> None:
        super().__init__(info)

    def metrics(self) -> tuple[str, ...]:
        return ()

    def probe(self, torch_module: Any) -> CapabilityResult:
        self._capability_error = self.info.diagnostic or (
            "roofline counters are not supported on this device"
        )
        return CapabilityResult(
            backend=self.info,
            status="unavailable",
            error=self._capability_error,
        )

    def build_profiler_kwargs(self, base_kwargs: dict[str, Any], torch_module: Any) -> dict[str, Any]:
        return dict(base_kwargs)

    def compile_counter_rows(
        self,
        raw_events: list[Any],
        *,
        capture_id: str,
        local_step: int,
        global_step: int,
        rank: int,
        role: str,
        timeline_events: Any = None,
    ) -> Any:
        del raw_events, capture_id, local_step, global_step, rank, role, timeline_events
        from .adaptor import _RooflineCompileResult

        return _RooflineCompileResult(
            counters=[],
            rooflines=[],
            quality="unavailable",
            counter_events=0,
            associated_kernels=0,
            unassociated_kernels=0,
            missing_metrics=[],
            error=self._capability_error or "roofline counters are not supported on this device",
        )


def _build_rocm_roofline_records(
    counters: list[Any],
    *,
    peaks: tuple[float | None, float | None],
    quality: str,
    capture_id: str,
    local_step: int,
    global_step: int,
    rank: int,
    role: str,
) -> list[Any]:
    """Derive ROCm roofline efficiency rows only when calibration is present."""
    from .adaptor import _balanced_threshold
    from .session_store import RooflineRecord

    peak_flops, peak_bytes = peaks
    if peak_flops is None or peak_bytes is None:
        return []
    threshold = _balanced_threshold()
    rooflines: list[Any] = []
    for counter in counters:
        if (
            counter.flops is None
            or counter.dram_bytes is None
            or counter.duration_us is None
            or counter.duration_us <= 0
        ):
            continue
        arithmetic_intensity = (
            counter.flops / counter.dram_bytes
            if counter.dram_bytes > 0
            else None
        )
        duration_sec = counter.duration_us / 1_000_000
        achieved_flops = counter.flops / duration_sec
        achieved_bytes = counter.dram_bytes / duration_sec
        boundedness: float | None = None
        bottleneck = "unknown"
        if peak_flops and peak_bytes:
            compute_eff = achieved_flops / peak_flops
            memory_eff = achieved_bytes / peak_bytes
            boundedness = min(compute_eff, memory_eff)
            if abs(compute_eff - memory_eff) < (1.0 - threshold):
                bottleneck = "balanced"
            elif compute_eff > memory_eff:
                bottleneck = "compute"
            else:
                bottleneck = "memory"
        rooflines.append(
            RooflineRecord(
                capture_id=capture_id,
                local_step=local_step,
                global_step=global_step,
                rank=rank,
                role=role,
                op_name=counter.op_name,
                kernel_name=counter.kernel_name,
                calls=counter.calls,
                self_duration_us=counter.duration_us,
                flops=counter.flops,
                dram_bytes=counter.dram_bytes,
                arithmetic_intensity=arithmetic_intensity,
                achieved_flops=achieved_flops,
                achieved_bytes_per_sec=achieved_bytes,
                peak_flops=peak_flops,
                peak_bytes_per_sec=peak_bytes,
                boundedness=boundedness,
                bottleneck=bottleneck,
                data_quality=quality,
            )
        )
    return rooflines


detect = detect_backend


def create_roofline_backend(torch_module: Any) -> RooflineBackend:
    info = selected_backend(torch_module)
    if info.counter_source == "cuda":
        return CudaRooflineBackend(info)
    if info.counter_source == "rocm":
        return RocmRooflineBackend(info)
    return UnavailableRooflineBackend(info)
