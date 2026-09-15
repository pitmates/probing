# Operator Roofline

Use this skill when a short-window `torch.profiler` capture should classify GPU
kernels and their associated PyTorch operators as compute, memory, or balanced.

## Prerequisites

- Start a capture with `analysis=roofline`, or set the process default
  `PROBING_TORCH_PROFILER_ANALYSIS=roofline`.
- Use a PyTorch build with `torch.profiler._ExperimentalConfig` and CUPTI Range
  Profiler support.

## Interpretation

- `data_quality != "ok"` means the roofline conclusion is partial, truncated, or unavailable.
- Memory-bound operators benefit from layout, fusion, and access-pattern work.
- Compute-bound operators with low efficiency need kernel implementation work.
- Missing peaks (`boundedness IS NULL`) support FLOPs and arithmetic-intensity analysis only.
