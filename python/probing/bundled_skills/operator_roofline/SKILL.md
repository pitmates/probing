# Operator Roofline

Use this skill when a short-window `torch.profiler` capture should classify GPU
kernels and their associated PyTorch operators as compute, memory, or balanced.

## Prerequisites

- Start a capture with `analysis=roofline`, or set the process default
  `PROBING_TORCH_PROFILER_ANALYSIS=roofline`.
- Use a PyTorch build with `torch.profiler._ExperimentalConfig` and CUPTI Range
  Profiler support.

## ROCm / DCU (DRAM-only)

On ROCm, counters come from a full-run `rocprofv2` wrapper instead of a Kineto
short window. v1 collects DRAM counters only, so `flops`, `achieved_flops`, and
`arithmetic_intensity` are `NULL`; `bottleneck` is `memory` and `boundedness` is
the memory efficiency. Calibrate `PROBING_TORCH_ROOFLINE_CONFIG.peaks` with the
device DRAM bandwidth before reading bandwidth utilization.

## Interpretation

- `data_quality != "ok"` means the roofline conclusion is partial, truncated, or unavailable.
- Memory-bound operators benefit from layout, fusion, and access-pattern work.
- Compute-bound operators with low efficiency need kernel implementation work.
- `flops IS NULL` means the DRAM-only mode (ROCm v1) or an uncalibrated
  compute path; those rows support bandwidth / memory-efficiency analysis only.
