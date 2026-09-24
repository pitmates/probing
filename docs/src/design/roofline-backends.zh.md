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

Phase 0 已确认当前 DCU（`gfx936` / HCU，4 × 80 CU、8 Shader Engine、1500 MHz）无法通过 PyTorch/Kineto 进程内路径取得 roofline counter。Kineto 只提供 kernel 时间线，不暴露 `cuda_profiler_range` / `hip_profiler_range`，也没有 metric args。因此 v1 采用 `rocprofiler` 采集；进程内路径只保留诊断记录，不用于 counter 计数。

关键约束（DTK 26.04 验证）：`/opt/dtk-26.04/rocprofiler/bin/rocprof` 与 `rocprofv2` 都是 wrap-launch 采集器——命令形态为 `rocprof[v2] -i <pmc> -d <out> <app>`，采集器自己拉起被测程序，不支持 `--pid` / `--target-process` attach 到已运行训练进程。因此“训练运行中按需 start/stop 的捕获窗口”在 rocprof 工具链上不可直接实现，v1 改为“启动时全程采集 + 事后离线统计”。

### 5.1 进程内 Kineto 路径（已验证，不可用）

- PyTorch 2.9.0 / ROCm 6.3.26093 会接受 `_ExperimentalConfig` 指标，但 `profiler.events()` 中无 counter 事件，metric keys 为空。
- Chrome trace 可见 `kernel`、`cuda_runtime`、`gpu_memset`，但没有 `cuda_profiler_range` 或 `hip_profiler_range`。
- 结论：v1 不再投资该路径；能力探测固定写入 `counter_backend=rocm`，wrapper 采集链路未就绪时返回 `unavailable`。

### 5.2 全程采集 + 离线统计（v1 主路径）

v1 采用两段式模型，采集与统计解耦。

**采集期（训练启动时包裹）**

- 由 launcher / probing 将采集命令套在训练进程外层，覆盖整个训练生命周期；每个 rank 产出独立 counter artifact（CSV/JSON）。
- 基准命令形态：

```text
rocprof -i {pmc} --timestamp on -d {output} <train_cmd>
```

- `{pmc}` 为仅含 DRAM 四计数器与 `SQ_INSTS_*` 的 metric 文件；launcher 渲染时令 `{output}={artifact_dir}/rank{rank}/{launch_ts}`，与 offline import 的发现规则一致。`--timestamp on` 保留 device/steady 时间戳供关联与切片。
- v1 先单卡单进程验证 `rocprof -i pmc.txt --timestamp on -d out python bench.py`；多卡 `torchrun` 注入层级（包整个 `torchrun`，或 launcher 逐 rank 包 worker）是 v1 主路径专项，见 Phase 5b。
- 模板渲染必须走参数列表或 `shlex.quote`，禁止把 `{pmc}`/`{output}`/`{app}` 直接字符串拼接；路径或训练命令可能含空格/引号。

**统计期（offline import）**

- probing 进程内 collector 用 `rocm_sidecar.parse_counter_artifact()` 解析 artifact，产出 `python.profile_counter` 事实行，`flops` 保持 `NULL`。v1 的真实 rocprof 输出是裸 CSV（走 `parse_counter_csv`）；JSON 分支要求 `format=probing-rocm-sidecar-v1` envelope，目前只用于本库 fixture 与 parity 测试，不代表裸 rocprof JSON 可直接 import。
- 再在进程内用 `join_rocm_rows_with_timeline()` 把 counter 行与当次 capture 的 Kineto `profiler.events()`（`timeline_events`）关联：优先 `correlation_id`，缺失时按时间戳回退，得到 `op_stack`。
- 随后用 `rocm_metrics` 换算 DRAM bytes（`TCC_EA_*`）与 FLOPs（`SQ_INSTS_*`，未校准为 `NULL`），写 `python.profile_roofline`。
- 职责边界：`parse_counter_artifact()` 只做 artifact 规范化；kernel→op 关联由 collector 内的 `join_rocm_rows_with_timeline()` 完成，落库后不再二次 SQL JOIN。`python.torch_trace` 是 TorchProbe 的模块级采样表，不是 Kineto kernel timeline，不参与本关联。

**step 切片**

- 全程采集不等于整段都必须进 roofline，但 v1 的窗口边界用 `profile_capture.started_at_us/ended_at_us` 近似，不按 step 切。
- per-step 区间归并后置：`TorchStepTiming` 只有 `step_duration_sec`，没有 step 起止 wall-clock，当前无法按 step 切片；要支持需先补“step 起止时间戳”依赖，列为后续项。
- 长训练分区：v1 先只做单卡短 bench，允许整段 artifact；命名 `{artifact_dir}/rank{rank}/{launch_ts}/*`，rotation / 大小上限 / 断点续跑留到多卡与长训练专项。

### 5.3 `profile/start` 语义（v1 降级）

- CUDA：维持 `profile/start?steps=N` 进程内按需 start/stop。
- ROCm v1：counter 由全程 wrapper 采集；`profile/start` 仍开短窗口 `torch.profiler`（Kineto），只取 CPU op timeline 与 `external_id` 供关联，不从 Kineto 取 counter，随后触发 offline import/finalize。artifact 未产出/不可达时不伪造 counter，`roofline_quality=unavailable` 并在 `profile_capture.error` 附原因。
- 方法 2（进程内 ROCProfiler/HSA 原生采集，依赖 `librocprofiler` C API 与 `_core` 编译）作为恢复“训练中按需 start/stop”的后路，v1 不实施，但保留 `RooflineBackend` 的采集策略替换点。

返回语义（当前实现已按此口径，Phase 3 沿用）：`profile/status` 的 `running` / `steps_target` / `steps_completed` / `finalizing` 仅对应进程内 Kineto 短窗口；`capture.status` 表示该 timeline 窗口是否成功，`roofline_quality` 独立表示 counter 质量，两者不混用。

| 场景 | `profile/start` | `capture.status` | `roofline_quality` | `profile_roofline` | `capture_id` |
| --- | --- | --- | --- | --- | --- |
| artifact 就绪且已校准 | `success=true` | `completed` | `ok`（无 `missing_metrics` / `unassociated`） | 有值 | 有值 |
| artifact 就绪但未校准（`flops` 全 `NULL`） | `success=true` | `completed` | `partial` | 空 | 有值 |
| artifact 缺失 / 解析失败 | `success=true` | `completed`（`error` 附原因） | `unavailable` | 空 | 有值 |
| start 阶段失败（窗口未建立） | `success=false` | 无 capture 行 | — | — | 空 |
| 窗口已建、finalize / `__exit__` 失败 | `success=true` | `failed` | `unavailable`（按失败点） | 空 | 有值 |

### 5.4 数据流

```text
训练启动 / launcher
   rocprof -i {pmc} --timestamp on -d {artifact_dir}/rank{rank}/{launch_ts} <train_cmd>
        └─ 全程采集 ─▶ {artifact_dir}/rank{rank}/{launch_ts}/*.csv|json （counter artifact）
                              │
                              ▼  rocm_sidecar.parse_counter_artifact()
                     python.profile_counter          （kernel / op / op_stack / calls / duration / metrics，flops=NULL）
                              │
                              ▼  进程内 join_rocm_rows_with_timeline(correlation_id → 时间戳回退)
                     当次 Kineto profile events        （CPU op stack 来源）
                              │
                              ▼  rocm_metrics 换算（DRAM bytes + FLOPs）
                     python.profile_roofline
```

### 5.5 offline import 契约（v1，Phase 3 已实现）

- 触发入口：`profile/start?analysis=roofline&artifact_dir=…` 标记窗口；finalize（自动或 `profile/stop`）时扫描 artifact 目录并 import。`artifact_dir` 缺省取 `PROBING_TORCH_ROOFLINE_ARTIFACT_DIR`，再回退 `rocprof_cmd` 的 `{output}`。
- 发现 / 命名：`{artifact_dir}/rank{rank}/{launch_ts}/*`；`capture_id` 在 offline import 时才生成并与窗口关联，文件名不预取 `capture_id`。import 按 mtime 从新到旧遍历候选，跳过解析失败或零行的 stray 组件文件，取第一个产出 counter 行的 artifact；真实后缀与时间戳列名以 Phase 5 的 rocprof 输出为准。
- 回收：import 成功后删除临时 artifact；`PROBING_TORCH_ROOFLINE_KEEP_ARTIFACTS=1` 保留现场。
- finalized 门控：全程采集下进程内 finalize 可能与仍在写盘的 rocprof 竞态。import 前需 `PROBING_TORCH_ROOFLINE_FINALIZED=1`（或 config `finalized: true`）确认采集已结束；未确认时降级 `unavailable` 并附原因，不消费半截文件。`rocm_e2e_spike --artifact-dir` 是训后 import 入口，自动以 `finalized=True` 调用。
- v1 不新增 `profile_capture.artifact_path` 列，artifact 位置由目录约定承载；确需跨进程 / 延迟 import 时再补列与 CLI/HTTP 契约。
- HTTP 表面变化已同步 `probing/server/API.md`；`tests/regression/spec/api_spec.json` 不记录查询参数，无需为 `artifact_dir` 增列。

## 6. Metric 映射草案

以下名称已在 `gfx936` 的 `rocprofv2 --list-counters` 中确认存在，但数值转换仍须 fixture / E2E 验证。

- 指令计数：`SQ_INSTS_VALU`、`SQ_INSTS_SALU`、`SQ_INSTS_VMEM_RD`、`SQ_INSTS_VMEM_WR`、`SQ_INSTS_VMEM`、`SQ_INSTS_MMOP`。
- DRAM 字节：
  - `TCC_EA_RDREQ_32B`、`TCC_EA_RDREQ`
  - `TCC_EA_WRREQ_64B`、`TCC_EA_WRREQ`
  - 公式草案：`read_bytes = 32 * RDREQ_32B + 64 * (RDREQ - RDREQ_32B)`，write 同理。
- Kernel 时长：优先取 `DurationNs` / `KernelDuration`；缺失时不得用 CPU 时间冒充 device 时长。
- 算子关联：`correlation_id` 来自 Kineto 短窗口的 CPU op `external_id`；rocprof 原生 artifact 通常不带它，回退到「kernel 名 + 时间戳最近 launch」，不采用“最后一个 CPU op”或虚假关联计数。
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
  - `PROBING_TORCH_ROOFLINE_CONFIG`：内联 JSON 或文件路径，聚合 `backend` / `rocm_enabled` / `rocprof_cmd` / `probe_cmd` / `metrics` / `peaks` / `flop_weights` / `artifact_dir` / `keep_artifacts` / `finalized`
  - `PROBING_TORCH_ROOFLINE_BACKEND=auto|cuda|rocm`
  - `PROBING_TORCH_ROOFLINE_ROCM_METRICS`
  - `PROBING_TORCH_ROOFLINE_ROCM_PEAKS_JSON`
  - `PROBING_TORCH_ROOFLINE_ROCPROF_CMD`（wrapper 模板，占位符 `{pmc}`/`{output}`/`{app}`，默认含 `--timestamp on`；已由 sidecar 模板 `{output}`/`{pid}` breaking 变更而来）
  - `PROBING_TORCH_ROOFLINE_ROCPROF_PROBE_CMD`（wrapper probe：验证最小 kernel 能否产出 counter，不再做 attach 式 dry-run 探测）
  - `PROBING_TORCH_ROOFLINE_ROCM_FLOP_WEIGHTS_JSON`（显式指令→FLOP 校准）
  - `PROBING_TORCH_ROOFLINE_ROCM_PROFILE=0|1`（wrapper 采集开关；旧名 sidecar 保留兼容，新版默认开启）
  - `PROBING_TORCH_ROOFLINE_ARTIFACT_DIR` / `PROBING_TORCH_ROOFLINE_KEEP_ARTIFACTS` / `PROBING_TORCH_ROOFLINE_FINALIZED`（offline import 目录 / 保留现场 / 采集完成门控）
  - `PROBING_TORCH_PROFILER_CLUSTER_FANOUT=0|1`（`profile/start` 按 rank fan-out）
- 同步更新 `env-vars`、`sql-tables`、`semantic_catalog` 和 `operator_roofline` skill。

## 9. 多 rank

- 现有 `profile/start` 默认只作用于接收请求的进程；`cluster` 为三态：缺省走 `PROBING_TORCH_PROFILER_CLUSTER_FANOUT`，`cluster=true` 强制 fan-out，`cluster=false` 强制本地。fan-out 时从本地 `GET /apis/nodes` 发现 peer，并逐个以 `profile/start?cluster=false` 触发，避免 peer 再次递归 fan-out。
- 每 rank 的 `profile_capture` 保持独立，查询时通过 `cluster query` 聚合。
- 不在 SQL 层合成一条假全局 capture。
- ROCm v1：训练期 counter 由 launcher 在启动时采集；`profile/start` fan-out 只触发各 rank 的 offline import/finalize，artifact 逐 rank 解析、逐 rank 落库。

## 10. 实施阶段

1. Phase 0（已完成）：DCU/ROCm spike，确认 Kineto 不可用，并确认 DTK `rocprof`/`rocprofv2` 只支持 wrap-launch、无 `--pid` attach，据此选定“全程采集 + 离线统计”。
2. Phase 1（已完成，代码）：抽取 `RooflineBackend`，CUDA 路径回归保持不变。
3. Phase 2（已完成，代码）：实现 `rocm` capability probe 与 metric catalog、`PROBING_TORCH_ROOFLINE_CONFIG` 配置收敛；capture 元数据落库。
4. Phase 3（已完成，代码）：v1 采集已从“窗口内 sidecar”改为 wrapper：`wrap_command()` 渲染含 `{pmc}`/`{output}`/`{app}` 且默认 `--timestamp on` 的命令；`import_artifact_rows()` 实现 offline import（artifact → `profile_counter` → 进程内 `join_rocm_rows_with_timeline` → `profile_roofline`）与临时 artifact 回收。
5. Phase 4（已完成，代码）：离线 fixture/单测/dry-run；`rocm_e2e_spike` 验证 CSV/JSON 解析与 wrapper 命令模板（不实际采集）。本机无 `_core`，pytest 未运行，待真实环境执行。
6. Phase 5（真实 DCU 待验证，单卡）：单卡小 bench 跑通 `rocprof -i pmc.txt --timestamp on -d out <app>` 端到端 counter，确认 metric 名 / 时间戳列 / artifact 后缀，回填峰值 / FLOP 权重。
7. Phase 5b（多卡注入层级，v1 主路径专项）：验证“包整个 `torchrun`”与“launcher 逐 rank 包 worker”的 counter 归属与落盘，确定 rank → artifact 子目录映射；这是单卡阶段验证不了的独立专项。
8. 后续（暂不实施）：方法 2 进程内 ROCProfiler/HSA 原生采集，恢复“训练中按需 start/stop”；保留 `RooflineBackend` 采集策略替换点。

## 11. 测试

- 单元：vendor 探测、metric 映射、单位换算、峰值解析、backend 选择。
- Fixture：用离线 rocprofiler CSV/JSON 验证 counter 解析、算子关联和 roofline 数值。
- 回归：确保 CUDA 路径不因后端抽象发生行为变化。
- E2E：在真实 `gfx936`/DCU 环境验证完整链路，标 `slow`；先运行 `rocm_e2e_spike` 确认 `rocprof` 参数与 CSV/JSON 解析，再替换 fixture 中的离线样例。

## 12. 风险

- `gfx936` 不同 ROCm/DTK 版本 metric 名称可能变化。
- DTK `rocprof/rocprofv2` 无 `--pid` attach，v1 只能全程采集；长训练时 artifact 体积与解析成本随运行时长增长，需按窗口 / capture 分区落盘。
- `--timestamp on` 在不同 DTK 版本的输出列名（`BeginNs/EndNs` vs `Start_Timestamp/End_Timestamp`）可能不同，`rocm_sidecar` 需做列名探测；且 rocprof 的 device/steady 时间戳需与 host wall-clock 做一次时钟域/epoch 对齐，否则时间戳回退关联与窗口切片错位。
- 多卡 `torchrun` 下 wrapper 注入层级（包 `torchrun` vs 包每 rank python）未经真实验证，counter 到 rank 的归属可能错配。
- 多 pass 采集受硬件计数器限制，counter 分组和 session 管理复杂。
- 算子关联精度依赖 Kineto timeline 的 `external_id` 可用性；rocprof artifact 通常无 correlation_id，fallback 到 kernel 名 + 时间戳最近 launch 时精度下降。
