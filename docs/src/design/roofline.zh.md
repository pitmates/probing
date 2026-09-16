# 算子 Roofline 建模

状态：已实现（CUPTI 环境依赖能力探测；E2E 需支持 CUPTI Range Profiler）。

本方案描述 Probing 如何在现有短窗口 `torch.profiler` 路径上增加算子级
roofline 统计。建模方法参考 Holistic Trace Analysis（HTA）的
CUPTI Counter Analysis：采集 CUPTI Range Profiler 事件，将 GPU kernel 的
硬件计数关联到 CPU 侧 PyTorch 算子，再基于 FLOPs、DRAM 访存字节数与
kernel 执行时间计算算术强度和性能效率。

## 1. 背景与目标

HTA 的 CUPTI Counter Analysis 是离线分析：先把 Kineto trace 写入
Chrome JSON 文件，再由 Python 进程加载 DataFrame、构建 CallGraph、过滤
`cuda_profiler_range` 事件并关联算子。Probing 是在线诊断：数据必须在被观测
训练进程内生成，受控写入本地 store，并立即通过 SQL 查询。为弥合这个 gap，
本方案不把“HTA 解析器”嵌入 server，而是把 HTA 方法拆成可在线执行的流水线，
并让算子关联发生在事件仍在内存中的 capture 结束点。

| HTA 离线环节 | Probing 在线化方式 | 保持不变的语义 |
|------------|-------------------|---------------|
| 读取 Chrome JSON / symbol table | 复用进程内 `torch.profiler.events()`，不落盘、不二次加载 | 事件名称、类别、计数 args |
| 重建整份 trace 的 CallGraph | 在 capture 窗口内，用 `ExternalID` / launch 关联构建局部算子栈 | kernel 属于哪个 CPU op |
| 事后全量 DataFrame 分析 | capture finalize 时增量聚合为固定 schema 行 | FLOPs、DRAM bytes、AI 公式 |
| 分析产物只存在 Notebook / DataFrame | 写入 `SessionStore` 虚拟表，立即供 SQL / skill 查询 | 算子级 roofline 结论 |
| 用户另行采集 trace | 由现有 on-demand profiler controller 控制短窗口 | 独立于 TorchProbe 采样 |

这个转换的关键约束是：**采集仍在目标训练进程内、数据只在 capture 结束时
物化一次**。因此它不是连续 profiling，不会按 hook 反复扫描事件树；在线性
体现在“短窗口采集结束后立即可查”，而不是每个 micro-step 实时更新 roofline。

## 2. HTA 离线方法与 Probing 在线化

现有 `python.profile_capture` / `python.profile_hotspot` 已经能回答
“哪个 kernel 或算子耗时最长”，但无法区分算子是计算受限还是访存受限：

- FLOPs（由 SASS 浮点指令计算，FMA 记为 2，普通乘加记为 1）；
- DRAM 字节数（`dram__bytes_read.sum` + `dram__bytes_write.sum`）；
- 算术强度（FLOPs / DRAM bytes）；
- 计算吞吐（FLOPs / 时间）与访存吞吐（bytes / 时间）。

## 3. 数据流与模块归属

Python 侧物化为 `python.profile_counter` 与 `python.profile_roofline`。前者
保留 kernel 级原始计数事实，后者提供 SQL JOIN / 聚合友好的算子级结论表。
两者都只追加，不修改现有 hotspot 表，保持向后兼容。

为了复用现有注册边界，Rust 侧新增两个固定 schema 的
`RecordBatchBuilder`（`profile_counter.rs`、`profile_roofline.rs`），在
`profile_sql.rs` 同层注册，并在 `tbls.rs` 中暴露：

- `python.profile_counter`；
- `python.profile_roofline`。

不新增 HTTP endpoint，不做 server 特例；所有交互仍通过现有
`POST /query`、CLI query 或 `GET /apis/pythonext/tables/*`。

## 4. 数据模型

### 4.1 `python.profile_counter`

一行对应一个 capture 内同一 `(kernel, op_signature)` 组合的聚合结果：

| 列 | 类型 | 说明 |
|----|------|------|
| `capture_id` | `utf8` | 关联 `python.profile_capture` |
| `local_step` / `global_step` / `rank` / `role` | — | 训练坐标 |
| `kernel_name` | `utf8` | GPU kernel 名 |
| `op_name` | `utf8` | 关联的 PyTorch CPU 算子名 |
| `top_level_op` | `utf8` | 算子栈最外层 |
| `bottom_level_op` | `utf8` | 算子栈最内层 |
| `op_stack` | `utf8` | 算子栈 JSON 数组，仅用于诊断 |
| `calls` | `int64` | kernel 调用次数 |
| `duration_us` | `int64` | kernel 总时长（µs） |
| `flops` | `int64` | 由 SASS 指令计数计算的 FLOPs |
| `dram_bytes` | `int64` | DRAM 读写字节数 |
| `metrics` | `utf8` | 额外 metric 的 JSON 对象；仅包含请求并成功采集、且不属于默认 SASS/DRAM 字段的 metric，例如 `{"sm__warps_active.avg.pct_of_peak_sustained_active": 0.42}` |

### 4.2 `python.profile_roofline`

一行对应同一 capture 内 `(op_signature, kernel_name)` 的 roofline 结论：

| 列 | 类型 | 说明 |
|----|------|------|
| `capture_id` | `utf8` | 关联 `python.profile_capture` |
| `local_step` / `global_step` / `rank` / `role` | — | 训练坐标 |
| `op_name` | `utf8` | 算子名 |
| `kernel_name` | `utf8` | GPU kernel 名 |
| `calls` | `int64` | 调用次数 |
| `self_duration_us` | `int64` | kernel 总时长 |
| `flops` | `int64` | FLOPs 总量 |
| `dram_bytes` | `int64` | DRAM 访存总量 |
| `arithmetic_intensity` | `float64` | FLOPs / DRAM bytes |
| `achieved_flops` | `float64` | FLOPs / duration |
| `achieved_bytes_per_sec` | `float64` | bytes / duration |
| `peak_flops` | `float64` | 参考精度下的理论计算峰值 |
| `peak_bytes_per_sec` | `float64` | 平台理论显存带宽 |
| `peak_flops_kind` | `utf8` | 峰值口径，如 `fp16_tensor_dense` |
| `boundedness` | `float64` | `min(FLOPs_eff, Bytes_eff)` |
| `bottleneck` | `utf8` | `compute` / `memory` / `balanced` |
| `data_quality` | `utf8` | 计数质量标签 |

定义：

| 指标 | 公式 |
|------|------|
| `FLOPs_eff` | `achieved_flops / peak_flops` |
| `Bytes_eff` | `achieved_bytes_per_sec / peak_bytes_per_sec` |
| `boundedness` | `min(FLOPs_eff, Bytes_eff)` |
| `bottleneck` | 效率更接近峰值的一侧：`FLOPs_eff > Bytes_eff` 为 compute，反之为 memory |

在默认阈值 `0.9` 下，`|FLOPs_eff - Bytes_eff| < 0.1` 时标记 `balanced`。
若任一 peak 缺失，则 `boundedness` 设为 `NULL`，`bottleneck` 标记为
`unknown`，对应行保留用于 FLOPs / AI 分析。

### 4.3 平台峰值

第一版仅通过 `PROBING_TORCH_ROOFLINE_PEAKS_JSON` 显式配置平台峰值，不在
Python collector 内查询 GPU collector，避免跨 collector 直接调用。

第一版不做按 kernel 精度自动匹配峰值。默认参考口径为
`fp16_tensor_dense`；稀疏 Tensor Core、INT8、纯 FP32 或 double 精度 kernel
不能直接把该比值解释为绝对硬件利用率。后续版本可通过
`PROBING_TORCH_ROOFLINE_PEAKS_JSON` 配置多个精度档位，再由 kernel 名 /
SASS 指令组成推断档位。

若未配置该环境变量，则降级为部分 roofline，仅输出 FLOPs / AI，不输出效率结论。

`PROBING_TORCH_ROOFLINE_PEAKS_JSON` 第一版 schema 固定为：

```json
{
  "fp16_tensor_dense": {
    "peak_flops": 312000000000000,
    "peak_bytes_per_sec": 1600000000000
  }
}
```

第一版只读取 `fp16_tensor_dense` 一个 key；JSON 中存在其他精度 key 时必须
忽略并记录 debug 日志，不做自动匹配。该 schema 是 v1 契约；若未来支持多精度
档位，属于新增能力，不能静默改变现有 key 语义。

## 5. 采集与编译流程

### 5.1 开启 CUPTI 计数

`torch.profiler._ExperimentalConfig` 为实验接口。实现时必须通过能力探测
避免绑定特定 PyTorch 版本：

```python
experimental_config = getattr(torch.profiler, "_ExperimentalConfig", None)
if experimental_config is None:
    raise RuntimeError(
        "roofline counters require a PyTorch version with "
        "torch.profiler._ExperimentalConfig"
    )
```

仅存在 `_ExperimentalConfig` 不代表当前 PyTorch 构建支持 CUPTI Range
Profiler。实现必须把以下条件作为能力探测，而不是假设版本号即可：

- PyTorch / Kineto 构建包含 CUPTI Range Profiler 支持；
- 当前 GPU 与驱动支持请求的 metric；
- profiler 启动后能产生 `cuda_profiler_range` 事件；
- 不处于只允许单 process / 单 profiler 的 CUPTI 限制场景。

支持的 PyTorch / CUDA / driver 组合由实现 PR 的环境矩阵给出；能力探测
失败时返回可诊断错误，不 fallback 到 `with_flops=True` 估算值。

### 5.2 采集开销

Roofline 不是在现有 hook 中“额外保存几个字段”就能得到的。现有
optimizer-step hook 只负责开关短窗口 `torch.profiler`；打开 CUPTI
Range Profiler 后，GPU 会为被测 kernel 采集硬件计数，这会改变 profiler
采集模式并引入额外开销。开销来自两层：

- **采集期**：CUPTI per-kernel counter range、counter 寄存器复用、
  kernel 序列化 / 重放或驱动侧聚合，具体开销依赖 GPU、驱动、指标集与
  kernel 数量；
- **finalize 期**：Probing 在内存中扫描 `profiler.events()`、构建局部
  算子关联并聚合两张表的行；成本与事件数成正比。

因此 CUPTI roofline 默认关闭，只允许显式短窗口启用；它不并入 TorchProbe
的采样 step，也不改变 TorchProbe 的 hook 顺序。若用户只开启普通
`torch.profiler`，Probing 只解析已有事件，不额外开启计数；缺少 CUPTI
counter 时输出 `data_quality="unavailable"`，不能用 `with_flops=True`
提供的估算 FLOPs 冒充 DRAM roofline。

实现 PR 必须报告同一 workload 在以下三种状态下的 step 时间：
`PROBING_TORCH_PROFILER` 普通短窗口、附加 roofline metrics 的短窗口、
未开启 profiler 的基线。只要求给出相对开销，不承诺与普通 profiler 完全
相同。

分析能力按 capture 选择，而不是占用一个独立的全局布尔开关：

- 调用方在 `profile/start` 请求中传 `analysis=roofline`；
- 对分布式 / 批量启动场景，`PROBING_TORCH_PROFILER_ANALYSIS` 提供默认值；
- 请求参数优先于环境变量，环境变量默认 `none`；
- 后续通信、流水线、overlap 等短窗口分析继续使用 `analysis` 列表，
  不再为每类工具新增顶层布尔开关。

这个归属的理由是：CUPTI counter 在构造 profiler 前就会改变采集模式，其
生命周期与单次 capture 一致，而不是与 TorchProbe 或整个进程一致。把
能力挂到 `PROBING_TORCH_PROFILING` 会把深钻开销隐式带入长期采样；把
每类分析都做成进程级开关则会让配置面随着诊断工具数量线性膨胀。

能力配置：

- `PROBING_TORCH_ROOFLINE_METRICS`：覆盖默认 CUPTI 指标集；
- `PROBING_TORCH_ROOFLINE_BALANCED_THRESHOLD`：`balanced` 阈值；
- `PROBING_TORCH_ROOFLINE_PEAKS_JSON`：GPU 平台峰值配置。
- `PROBING_TORCH_ROOFLINE_MAX_EVENTS`：roofline 事件上限，默认 `200000`。

默认指标集：

```text
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
dram__bytes_read.sum
dram__bytes_write.sum
```

### 5.3 FLOPs 计算

FMA 类指令（`ffma`、`hfma`、`dfma`）按每条指令 2 FLOPs 计算，普通
乘法/加法指令按每条 1 FLOPs 计算。第一版只采用 SASS thread-instruction
口径，与 HTA 保持一致。

已知限制：Tensor Core MMA 指令是 warp-level 指令，一条指令覆盖的 FLOPs
由 HMMA shape 决定，不能与 thread-level SASS 计数直接换算或相加。因此
Tensor Core kernel 的 FLOPs 在该口径下可能被显著低估；其 AI 与 compute
efficiency 只能用于同口径比较，不能解释为绝对硬件利用率。

### 5.4 编译流程

Roofline 采用“在线采集、窗口内编译”的两段式执行：

1. **在线采集（目标训练进程内）**：现有 optimizer-step hook 驱动
   `torch.profiler`；CUPTI Range Profiler 事件先留在 profiler 对象中，
   训练主循环不做逐事件处理；
2. **窗口内编译（capture finalize）**：`ProfilerController._finalize_capture()`
   在 `profiler.__exit__()` 后调用 `compile_from_profiler()`。后者一次性
   读取 `profiler.events()`，同一输入独立产出 hotspot、counter、roofline
   三类行；roofline 是 `cuda_profiler_range` 子集的衍生结果，不阻塞
   hotspot 编译。

finalize 阶段执行：

1. 过滤 `cat == "cuda_profiler_range"` 的 CUPTI 计数事件；
2. 通过 `ExternalID` 与 `cpu_op` 事件关联，构建 `op_stack`；
3. 计算每个 kernel 的 FLOPs 与 DRAM bytes；
4. 聚合生成 `profile_counter` 行；
5. 获取平台峰值，计算 roofline 效率；
6. 将 hotspot、counter、roofline 行一次性追加到 `SessionStore`，共享同一
   capture 生命周期。

实现上，`compile_from_profiler()` 的返回值扩展为
`(CaptureRecord, list[HotspotRecord], list[CounterRecord], list[RooflineRecord])`，
`SessionStore.add_capture()` 同步接收四类数据；仍然只属于
`python/probing/profiling/torch_profiler/` 这一 L2 collector，不引入
server 依赖。

若 CUPTI 计数不可用，`compile_from_profiler()` 应正常生成原有 hotspot
结果，不因 roofline 缺失导致 capture 失败。

### 5.5 在线生命周期与降级

- **窗口化**：CUPTI 计数只随显式 / skill 触发的短窗口 capture 打开，
  不并入 TorchProbe 的采样 step，不改变 TorchProbe hook 顺序与开销模型；
- **即时可查**：finalize 完成后，`python.profile_counter` 和
  `python.profile_roofline` 立即出现在现有虚拟表查询中；
- **生命周期**：新表与 `python.profile_hotspot` 共用 `capture_id`，随
  `SessionStore` 的 `max_sessions` 一起淘汰，不引入第二套保留策略；
- **事件上限**：roofline 使用独立的
  `PROBING_TORCH_ROOFLINE_MAX_EVENTS`。达到上限时整个 roofline 结果标记
  `data_quality="truncated"`，并视为不可用于 parity 结论；不做截取子集后
  继续计算完整 roofline 的降级。普通 hotspot 仍可使用现有
  `PROBING_TORCH_PROFILER_MAX_EVENTS` 逻辑。
- **失败降级**：缺少 `_ExperimentalConfig`、CUPTI 指标或平台峰值时，
  capture 仍完成并写入可诊断的错误 / 缺失标记，不影响已有 hotspot 表；
- **进程边界**：编译结果只通过虚拟表契约暴露，server 不导入 Python
  分析内部类型，也不解析 HTA 文件格式。

### 5.6 与离线分析的一致性保证

一致性目标不是复用 HTA 代码，而是保证：**同一个 profiler session 中的
kernel、counter、算子关联和 roofline 数值，与导出 Chrome trace 后由
HTA 等价算法得到的结果一致**。实现按四层保证：

1. **事件源一致**：只使用 `profiler.events()` 中尚未二次加工的事件。
   禁止从 `key_averages()` 提取 counter，因为 `key_averages()` 会先聚合、
   丢失 `args` 与调用栈，天然无法与离线 trace 对齐。
   现有 `compile_from_profiler()` 会优先用 `key_averages()` 生成 hotspot、
   失败时 fallback 到 `events()`；该行为只允许继续服务 `profile_hotspot`。
   `profile_counter` / `profile_roofline` 必须强制读取 `profiler.events()`，
   与 hotspot 输入源显式分叉，不得复用 `key_averages()` 聚合结果。
2. **事件语义一致**：过滤、字段读取、时长单位与 HTA 对齐：
   - `cat == "cuda_profiler_range"`；
   - counter 从事件 `args` 读取；
   - FLOPs 使用同一 SASS 指令表，FMA 为 2，非 FMA 为 1；
   - kernel 与算子关联优先使用 `ExternalID`，不可用时退回 launch 顺序；
   - 表内时间统一为 µs，转换必须发生在边界处，不在中间层反复换算。
3. **可审计质量**：`profile_capture` 或 `data_quality` 必须暴露：
   - counter 事件数；
   - 成功关联算子的 kernel 数；
   - 未关联 kernel 数；
   - 缺失的必需 metric；
   - 解析器版本。
   解析器版本必须同时包含 SASS 指令表对齐的 HTA 版本。当前对齐版本为
   `52f86de6bdbdaa996102bc017429283bf2b24b9d`；该 9 行指令表是从 HTA
   手动同步的常量。上游 HTA 增加或修改指令时，Probing 必须显式更新
   该常量并更新版本记录，不允许运行时隐式合并。

   出现未关联事件或缺失 metric 时不能静默丢弃；至少标记
   `data_quality="partial"`，完全无法解析时标记 `unavailable`。
4. **Parity 验收**：测试中导出同一 profiler session 的 Chrome trace：
   - 离线路径解析 trace；
   - 在线路径解析 `profiler.events()`；
   - 断言 kernel 名、counter 名、counter 值、调用次数、时长、算子栈、
     FLOPs、DRAM bytes 与 roofline 指标一致。

   CUPTI counter 均为整型计数，要求精确相等；时间受导出精度影响时，
   parity 测试必须先固定 fixture 精度，不允许用宽泛容忍度掩盖换算差异。
   该测试只在 CI fixture 中解析文件，生产路径仍不落盘。

若 parity 失败，视为阻断缺陷；运行时只能降级并标记质量，不得对缺失
counter、未关联 kernel 或异常峰值做插值、补零或静默过滤。

## 6. Skill 工作流

新增 bundled skill `operator_roofline`：

```yaml
tables:
  - python.profile_capture
  - python.profile_roofline
```

Skill 输出：

1. `roofline_summary`：按算子聚合 top N 的算术强度与瓶颈分布；
2. `compute_bottleneck_ops`：计算受限算子；
3. `memory_bottleneck_ops`：访存受限算子；
4. `balanced_ops`：近似平衡的算子。

示例 SQL：

```sql
WITH latest_capture AS (
  SELECT capture_id
  FROM python.profile_capture
  WHERE status = 'completed'
  ORDER BY ended_at_us DESC
  LIMIT 1
)
SELECT
  op_name,
  SUM(flops) AS total_flops,
  SUM(dram_bytes) AS total_bytes,
  SUM(flops) / NULLIF(SUM(dram_bytes), 0) AS arithmetic_intensity,
  SUM(flops) * 1e6 / NULLIF(SUM(self_duration_us), 0) AS achieved_flops
FROM python.profile_roofline
WHERE capture_id = (SELECT capture_id FROM latest_capture)
GROUP BY op_name
ORDER BY total_flops DESC
```

跨 kernel / 跨调用聚合时不得对逐行 `boundedness` 取简单平均值；若需要
聚合效率，必须先聚合 FLOPs、bytes 与 duration，再用聚合后的吞吐除以同一
peak 口径重新计算。

Skill 解释规则：

- 若无 `python.profile_counter` 数据，提示需要 `analysis=roofline`
  （或全局默认 `PROBING_TORCH_PROFILER_ANALYSIS=roofline`）并使用
  支持 CUPTI Range Profiler 的 PyTorch 版本；
- 若多数算子 memory bound，建议优化访存布局或算子融合；
- 若 compute bound 且效率较低，建议关注 kernel 实现；
- 若峰值缺失，只输出 FLOPs 与 AI，不给出瓶颈结论。

## 7. 测试策略

| 测试 | 层级 | 内容 |
|------|------|------|
| CUPTI 事件解析 | Python 单元 | 使用模拟 `torch.profiler.events()` 验证 `cuda_profiler_range` 提取、FLOPs 计算、算子关联 |
| Roofline 公式 | Python 单元 | 验证 AI、吞吐、boundedness、`balanced` 分类与 NULL 传播 |
| 离线 parity | Python 单元 | 在测试 fixture 中导出同一 profiler session 的 Chrome trace，解析离线路径与在线 `profiler.events()` 路径并精确比对 counter / 关联 / roofline 结果；生产路径不落盘 |
| 能力探测 | Python 单元 | 验证 `_ExperimentalConfig` 缺失、CUPTI Range 不支持、metric 被拒绝时的错误与降级标记 |
| Rust schema | Rust 单元 | 验证 `profile_counter` / `profile_roofline` schema 与列 |
| SQL 虚拟表 | Python 回归 | 验证表注册、SQL 查询与文档元数据 |
| E2E | Python 回归 | 在支持 CUPTI 的环境下验证完整 capture -> counter -> roofline 链路 |

E2E 测试需标记 `slow` 或通过环境变量跳过，避免 CI 硬依赖 CUDA 环境。

## 8. 文档更新

实现时需同步更新：

- `docs/src/reference/sql-tables.md`；
- `docs/src/reference/env-vars.md`；
- `docs/src/design/profiling.md`；
- `python/probing/bundled_skills/operator_roofline/SKILL.md`；
- `python/probing/bundled_skills/catalog.yaml`。
