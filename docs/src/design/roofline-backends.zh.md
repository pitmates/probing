# Roofline 多厂商后端设计

状态：目标设计（Draft）。本文描述 CUDA/CUPTI 之外的 roofline counter 后端抽象，
不改变现有 `roofline.md` 描述的 CUDA v1 契约。

关联文档：

- [Operator Roofline](roofline.zh.md)：当前 CUPTI 采集、算子关联、SQL 表和 skill 契约。
- [Profiling and tracing](profiling.zh.md)：TorchProbe 与短窗口 profiler 的总体边界。

## 1. 目标

- 将 roofline counter 采集从“CUPTI 特例”抽象为可插拔厂商后端。
- 支持 NVIDIA/CUDA、AMD/ROCm（含 DCU），并为未来厂商 GPU 预留接口。
- 保持 `python.profile_capture` / `python.profile_counter` /
  `python.profile_roofline` 的 SQL 契约稳定，skill 查询无需按厂商分叉。
- 能力不足时返回可诊断的 `unavailable`，禁止用 `with_flops=True` 估算值冒充 counter roofline。

## 2. 现状问题

当前实现把 CUPTI 假设散布在三个地方：

- `controller.py`：固定 `torch.profiler._ExperimentalConfig`，并检查 `torch.cuda.is_available()`。
- `adaptor.py`：只识别 `cuda_profiler_range`，FLOPs/DRAM 公式硬编码 NVIDIA SASS 指标。
- `session_store.py`：峰值配置只支持 `fp16_tensor_dense`，没有厂商/架构维度。

在 DCU/ROCm 上，`torch.cuda` 可能为真，但没有 CUPTI counter，因此 capture 完成但
`roofline_quality=unavailable`、`roofline_counter_events=0`。

## 3. 后端抽象

Python 侧新增一个薄接口，CUDA 后端保持现有逻辑不变：

```text
RooflineBackend
├── detect() -> BackendInfo
├── probe_capabilities() -> CapabilityResult
├── build_profiler_kwargs() -> dict
├── compile_counter_rows(events_or_artifacts, ctx) -> _RooflineCompileResult
└── platform_peaks(info) -> (peak_flops, peak_bytes)
```

`BackendInfo` 至少包含：

```text
vendor: nvidia | amd | other
device_model: A100 | MI250X | ...
device_arch: sm_80 | gfx942 | ...
counter_source: cuda | rocm | none
```

`CapabilityResult` 分为：

- `ok`：可以启动 counter capture。
- `unavailable`：当前平台不支持，给出可诊断错误。
- `degraded`：部分 metric 可用，capture 结果应标记 `partial`。

现有 CUPTI 代码迁入 `cuda` backend；`rocm` backend 只负责 AMD 指标解析和换算；
SQL 聚合、roofline 公式、`SessionStore` 写入保持共享。

## 4. 能力探测

能力探测回答“平台能力”，不回答“具体跑了哪些 kernel”：

- 识别厂商、设备型号和架构。
- 选择对应 metric catalog 和 peak 档位。
- 验证 metric 集合被当前 profiler/驱动接受。
- 验证能产生 counter 事件，而不仅是能构造 profiler 对象。

具体 kernel 名称只能在真实 capture 后由解析器得到。探测阶段不应运行训练 kernel。

探测结果写入 capture 元数据，便于诊断：

```text
counter_backend = cuda | rocm | none
device_vendor   = nvidia | amd | ...
device_model    = ...
device_arch     = ...
```

未来新增厂商时，只新增一个 vendor probe 和 metric catalog，不修改 roofline 核心公式。

## 5. ROCm/DCU 适配路径

优先做 spike，根据结果二选一：

### 5.1 进程内 Kineto 路径

- 验证 PyTorch ROCm 版本是否能产生 AMD counter 事件。
- 若能，扩展 adaptor，识别 `hip_profiler_range` 或等价 category。
- 优点：生命周期与现有 `torch.profiler` 一致，改动小。
- 缺点：依赖 PyTorch/Kineto/ROCm 组合是否真正暴露 metric。

### 5.2 离线 `rocprofiler` sidecar 路径

- 由 L2 collector 在捕获窗口内运行 `rocprofiler`，输出 CSV/JSON。
- 捕获结束后按 `capture_id`、时间戳或 `correlation_id` 与 Kineto timeline join。
- 优点：metric 完整，兼容性更可控。
- 缺点：多进程生命周期、fan-out 和文件清理更复杂。

## 6. Metric 映射草案

ROCm 指标名必须在目标 DCU 型号上验证。初始草案：

- FLOPs：`SQ_INSTS_VALU`、`SQ_INSTS_VALU_MFMA_MAC_F32` 等，按指令类型加权。
- DRAM：`TCC_EA_RDREQ` / `TCC_EA_WRREQ`，乘以 burst size 得到字节数。
- Kernel 时长：`DurationNs` / `KernelDuration`。
- 算子关联：优先 `correlation_id`，缺失时按时间戳回退最近 CPU op launch。

禁止把 NVIDIA SASS 指标名直接搬给 ROCm，也禁止用估算 FLOPs 标记为 `ok`。

## 7. 峰值配置

现有 `PROBING_TORCH_ROOFLINE_PEAKS_JSON` 只支持 `fp16_tensor_dense`。
扩展为厂商/架构/精度可解析的结构：

```json
{
  "backend": "rocm",
  "device_arch": "gfx942",
  "peaks": {
    "fp16_tensor_dense": {
      "peak_flops": 312000000000000,
      "peak_bytes_per_sec": 1600000000000
    }
  }
}
```

向后兼容：不提供 `backend` 时按当前 CUDA v1 语义解析。

## 8. SQL 与配置

- 新增 capture 元数据列：`counter_backend`、`device_vendor`、`device_model`、`device_arch`。
- 表 `python.profile_counter` / `python.profile_roofline` 的结构保持稳定。
- 新增 env：
  - `PROBING_TORCH_ROOFLINE_BACKEND=auto|cuda|rocm`
  - `PROBING_TORCH_ROOFLINE_ROCM_METRICS`
  - `PROBING_TORCH_ROOFLINE_ROCM_PEAKS_JSON`
  - `PROBING_TORCH_ROOFLINE_ROCPROF_PATH`
- 同步更新 `env-vars`、`sql-tables`、`semantic_catalog` 和 `operator_roofline` skill。

## 9. 多 rank

- 现有 `profile/start` 只作用于接收请求的进程。
- ROCm sidecar 需要按 rank 启动，复用 torchrun cluster 的 rank endpoint 做 fan-out。
- 每 rank 的 `profile_capture` 保持独立，查询时通过 `cluster query` 聚合。
- 不在 SQL 层合成一条假全局 capture。

## 10. 实施阶段

1. Phase 0：DCU/ROCm spike，验证 Kineto 或 `rocprofiler` 的真实事件、指标和关联能力。
2. Phase 1：抽出 `RooflineBackend`，现有 CUDA 路径回归保持不变。
3. Phase 2：实现 `rocm` backend 的采集与解析。
4. Phase 3：能力探测、峰值配置、capture 元数据落库。
5. Phase 4：单元测试、fixture parity、ROCm E2E 标记 `slow`。

## 11. 测试

- 单元：vendor 探测、metric 映射、单位换算、峰值解析。
- Fixture：用离线 rocprofiler 输出验证 counter、关联和 roofline 数值。
- 回归：确保 CUDA 路径不因后端抽象发生行为变化。
- E2E：在真实 ROCm/DCU 环境验证完整链路。

## 12. 风险

- DCU 不同型号/ROCm 版本 metric 名差异大。
- ROCm 是否由 Kineto 直接暴露 counter 尚未验证。
- `rocprofiler` 多 rank 并发采集需要控制互斥与文件生命周期。
- 算子关联精度依赖 `correlation_id` 在真实训练栈中的可用性。