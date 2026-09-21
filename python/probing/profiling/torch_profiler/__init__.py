"""On-demand ``torch.profiler`` / Kineto capture and virtual SQL tables.

This package is independent from :mod:`probing.profiling.torch_probe`: it owns
short capture sessions and bounded in-process results, not long-running module
hooks or memtables. Starting either path does not configure the other.
"""

from .backends import (
    BackendInfo,
    CapabilityResult,
    CudaRooflineBackend,
    RocmRooflineBackend,
    RooflineBackend,
    create_roofline_backend,
)
from .controller import ProfilerController, get_controller, profiler_status
from .sql import profile_counter_rows, profile_roofline_rows
from .session_store import SessionStore, get_session_store

__all__ = [
    "BackendInfo",
    "CapabilityResult",
    "CudaRooflineBackend",
    "RocmRooflineBackend",
    "RooflineBackend",
    "ProfilerController",
    "SessionStore",
    "create_roofline_backend",
    "get_controller",
    "get_session_store",
    "profiler_status",
]
