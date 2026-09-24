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
- `degraded`：预留状态，表示部分 metric 可用；v1 实现目前只产出 `ok` / `unavailable`，暂不产出 `degraded`。

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

Phase 0 已确认当前 DCU（`gfx936` / HCU，4 × 80 CU、8 Shader Engine、1500 MHz）无法通过 PyTorch/Kineto 进程内路径取得 roofline counter。Kineto 只提供 kernel 时间线，不暴露 `cuda_profiler_range` / `hip_profiler_range`，也没有 metric args。因此 v1 采用 `rocprofiler` sidecar；进程内路径只保留诊断记录，不用于 counter 采集。

### 5.1 进程内 Kineto 路径（已验证，不可用）

- PyTorch 2.9.0 / ROCm 6.3.26093 会接受 `_ExperimentalConfig` 指标，但 `profiler.events()` 中无 counter 事件，metric keys 为空。
- Chrome trace 可见 `kernel`、`cuda_runtime`、`gpu_memset`，但没有 `cuda_profiler_range` 或 `hip_profiler_range`。
- 结论：v1 不再投资该路径；能力探测固定写入 `counter_backend=rocm`，sidecar 未就绪时返回 `unavailable`。

### 5.2 离线 `rocprofiler` sidecar 路径（v1 主路径）

- 工具链：`/opt/dtk-26.04/rocprofiler/bin/rocprof`、`rocprofv2`；metric 定义在 `lib/rocprofiler/metrics.xml` 与 `gfx_metrics.xml`。
- L2 collector 在捕获窗口内启动 `rocprofiler`，按 rank 独立采集，输出 CSV/JSON 后按 `capture_id`、时间戳或 `correlation_id` 与 Kineto timeline join。
- 优点：metric 完整，兼容性更可控。
- 缺点：多进程生命周期、fan-out 和文件清理更复杂。
- 当前风险：`rocprofv2 --plugin file` 尚未产出端到端 counter 文件；旧 `rocprof` 已进入 metric 分组，但首次验证因 `Context Create failed` 中止。实现时 sidecar 必须标为 experimental，并保留 `unavailable` 降级。

## 6. Metric 映射草案

以下名称已在 `gfx936` 的 `rocprofv2 --list-counters` 中确认存在，但数值转换仍须 fixture / E2E 验证。

- 指令计数：`SQ_INSTS_VALU`、`SQ_INSTS_SALU`、`SQ_INSTS_VMEM_RD`、`SQ_INSTS_VMEM_WR`、`SQ_INSTS_VMEM`、`SQ_INSTS_MMOP`。
- DRAM 字节：
  - `TCC_EA_RDREQ_32B`、`TCC_EA_RDREQ`
  - `TCC_EA_WRREQ_64B`、`TCC_EA_WRREQ`
  - 公式草案：`read_bytes = 32 * RDREQ_32B + 64 * (RDREQ - RDREQ_32B)`，write 同理。
- Kernel 时长：优先取 `DurationNs` / `KernelDuration`；缺失时不得用 CPU 时间冒充 device 时长。
- 算子关联：优先 `correlation_id`，缺失时按时间戳回退到最近 launch，不采用“最后一个 CPU op”或虚假关联计数。
- 禁止把 NVIDIA SASS 指标名直接映射到 ROCm，也禁止用估算 FLOPs 标记为 `ok`。

## 7. 峰值配置

现有 `PROBING_TORCH_ROOFLINE_PEAKS_JSON` 只支持 `fp16_tensor_dense`。扩展为 vendor / arch / precision 可解析结构；`gfx936` 先使用占位值，待 DCU 规格确认后填入。

```json
{
  "backend": "rocm",
  "device_arch": "gfx936",
  "peaks": {
    "fp16_tensor_dense": {
      "peak_flops": 0,
      "peak_bytes_per_sec": 0
    }
  }
}
```

- `0` 表示未校准，禁止参与 roofline 效率结论，只能用于能力 / 数据完整性展示。
- 不提供 `backend` 时按当前 CUDA v1 语义解析。

## 8. SQL 与配置

- 新增 capture 元数据列：`counter_backend`、`device_vendor`、`device_model`、`device_arch`。
- 表 `python.profile_counter` / `python.profile_roofline` 的结构保持稳定。
- 新增 env（推荐只用一个 `PROBING_TORCH_ROOFLINE_CONFIG`，以下旧版变量保留为 fallback）：
  - `PROBING_TORCH_ROOFLINE_CONFIG`：内联 JSON 或文件路径，聚合 `backend` / `rocm_enabled` / `rocprof_cmd` / `probe_cmd` / `metrics` / `peaks` / `flop_weights`
  - `PROBING_TORCH_ROOFLINE_BACKEND=auto|cuda|rocm`
  - `PROBING_TORCH_ROOFLINE_ROCM_METRICS`
  - `PROBING_TORCH_ROOFLINE_ROCM_PEAKS_JSON`
  - `PROBING_TORCH_ROOFLINE_ROCPROF_CMD`
  - `PROBING_TORCH_ROOFLINE_ROCPROF_PROBE_CMD`（可选 dry-run 能力探测）
  - `PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON`（显式指令→FLOP 校准）
  - `PROBING_TORCH_ROOFLINE_ROCM_PROFILE=0|1`（experimental sidecar 开关；新版默认开启）
  - `PROBING_TORCH_PROFILER_CLUSTER_FANOUT=0|1`（`profile/start` 按 rank fan-out）
- 同步更新 `env-vars`、`sql-tables`、`semantic_catalog` 和 `operator_roofline` skill。

## 9. 多 rank

- 现有 `profile/start` 默认只作用于接收请求的进程；`cluster` 为三态：缺省走 `PROBING_TORCH_PROFILER_CLUSTER_FANOUT`，`cluster=true` 强制 fan-out，`cluster=false` 强制本地。fan-out 时从本地 `GET /apis/nodes` 发现 peer，并逐个以 `profile/start?cluster=false` 触发，避免 peer 再次递归 fan-out。
- 每 rank 的 `profile_capture` 保持独立，查询时通过 `cluster query` 聚合。
- 不在 SQL 层合成一条假全局 capture。

## 10. 实施阶段

1. Phase 0（已完成）：DCU/ROCm spike，确认 Kineto 不可用，选定 `rocprofiler` sidecar。
2. Phase 1（已完成，代码）：抽取 `RooflineBackend`，CUDA 路径回归保持不变。
3. Phase 2（已完成，代码）：实现 `rocm` capability probe 与 metric catalog，capture 元数据落库；sidecar 采集标为 experimental。
4. Phase 3（已完成，代码）：补齐 `rocprofiler` 调用、fan-out、输出解析与文件清理；真实 counter 验证见 Phase 5。
5. Phase 4（已完成，代码）：单测、fixture parity、ROCm E2E 标 `slow`。
6. Phase 5（真实环境待验证）：在 `gfx936`/DCU 上跑通 sidecar 端到端输出，确定 `rocprof` 正确参数组合并回填峰值 / FLOP 权重校准值。用 `python -m probing.profiling.torch_profiler.rocm_e2e_spike` 先验证离线解析链路，再接入训练短窗口。

## 11. 测试

- 单元：vendor 探测、metric 映射、单位换算、峰值解析、backend 选择。
- Fixture：用离线 rocprofiler CSV/JSON 验证 counter 解析、算子关联和 roofline 数值。
- 回归：确保 CUDA 路径不因后端抽象发生行为变化。
- E2E：在真实 `gfx936`/DCU 环境验证完整链路，标 `slow`；先运行 `rocm_e2e_spike` 确认 `rocprof` 参数与 CSV/JSON 解析，再替换 fixture 中的离线样例。

## 12. 风险

- `gfx936` 不同 ROCm/DTK 版本 metric 名称可能变化。
- `rocprofv2 --plugin file` 当前未产出端到端结果，需先验证正确参数组合。
- 多 pass 采集受硬件计数器限制，counter 分组和 session 管理复杂。
- 多 rank 并发 sidecar 需要互斥与文件生命周期控制。
- 算子关联精度依赖 `correlation_id` 是否在真实训练栈可用。
