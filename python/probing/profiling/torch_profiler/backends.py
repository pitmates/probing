"""Vendor backends for roofline counter acquisition.

The CUDA path keeps the existing in-process Kineto/CUPTI flow. The ROCm path
is detected and probed so captures can report diagnostic metadata, but v1 does
not yet implement the external ``rocprofiler`` sidecar collector.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .rocm_metrics import ROCM_DEFAULT_METRICS


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


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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


def _rocprofiler_path() -> Optional[str]:
    configured = os.environ.get("PROBING_TORCH_ROOFLINE_ROCPROF_PATH", "").strip()
    if configured:
        return configured if os.path.isfile(configured) else None
    candidates = (
        "/opt/dtk-26.04/rocprofiler/bin/rocprofv2",
        "/opt/dtk-26.04/rocprofiler/bin/rocprof",
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("rocprofv2") or shutil.which("rocprof")


class RooflineBackend(ABC):
    """Vendor-specific roofline backend.

    CUDA owns the Kineto/CUPTI experimental config, capability probe, and
    counter compilation. ROCm owns detection and the future rocprofiler
    sidecar path; until that path is implemented it returns an explicit
    unavailable result.
    """

    def __init__(self, info: BackendInfo) -> None:
        self.info = info
        self._capability_error = ""

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
    ) -> Any:
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
        raw = os.environ.get("PROBING_TORCH_ROOFLINE_ROCM_METRICS", "").strip()
        metrics = tuple(item.strip() for item in raw.split(",") if item.strip())
        if metrics:
            return metrics
        return ROCM_DEFAULT_METRICS

    def probe(self, torch_module: Any) -> CapabilityResult:
        enabled = _env_flag("PROBING_TORCH_ROOFLINE_ROCM_PROFILE")
        tool_path = _rocprofiler_path()
        if not enabled:
            self._capability_error = (
                "rocm roofline sidecar is disabled; set "
                "PROBING_TORCH_ROOFLINE_ROCM_PROFILE=1 to enable the experimental path"
            )
            return CapabilityResult(
                backend=self.info,
                status="unavailable",
                error=self._capability_error,
            )
        if not tool_path:
            self._capability_error = (
                "rocprofiler/rocprofv2 was not found; set "
                "PROBING_TORCH_ROOFLINE_ROCPROF_PATH"
            )
            return CapabilityResult(
                backend=self.info,
                status="unavailable",
                error=self._capability_error,
            )
        self._capability_error = (
            "rocm roofline sidecar is detected but collection is not implemented yet"
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
    ) -> Any:
        from .adaptor import _RooflineCompileResult
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
                error=self._capability_error
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
                error=f"rocm sidecar artifact parse failed: {exc}",
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
                error="rocm sidecar artifact contained no counter rows",
            )
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
        return _RooflineCompileResult(
            counters=counters,
            rooflines=[],
            quality="partial",
            counter_events=len(counters),
            associated_kernels=associated,
            unassociated_kernels=unassociated,
            missing_metrics=missing_metrics,
            error="",
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
    ) -> Any:
        del raw_events, capture_id, local_step, global_step, rank, role
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


detect = detect_backend


def create_roofline_backend(torch_module: Any) -> RooflineBackend:
    info = selected_backend(torch_module)
    if info.counter_source == "cuda":
        return CudaRooflineBackend(info)
    if info.counter_source == "rocm":
        return RocmRooflineBackend(info)
    return UnavailableRooflineBackend(info)
