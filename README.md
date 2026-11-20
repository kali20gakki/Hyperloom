# Hyperloom 昇腾 NPU 适配版

> 本仓库是 [Hyperloom](https://github.com/AMD-AGI/Hyperloom) 的昇腾 NPU 适配分支。英文版说明见
> [docs/ascend/README.md](docs/ascend/README.md)；上游项目原始 README 保留在
> [README.upstream.md](README.upstream.md)。

> **完整跑通指南（英文，逐步命令 + 排错表）**：[docs/ascend/RUNBOOK.md](docs/ascend/RUNBOOK.md)

状态：**bring-up 阶段**。测量、分析与决策三条链路已经能在昇腾 NPU 上跑通；内核层与量化尚未接通。
本文说明能跑什么、怎么跑、改了哪些地方，以及还缺什么。

## 1. 当前可运行的部分

| 阶段 | 状态 | 说明 |
|---|---|---|
| 设备发现、占用门禁、轮次环境 | **可运行** | 经 `hyperloom_common.device_backend` 走 `npu-smi`；AMD 路径不变，由 `HYPERLOOM_DEVICE_BACKEND=amd` 选择 |
| aarch64 上的工具 staging | **可运行** | `StagingProfile.INFERENCEX` 只铺 InferenceX 子集；钉死的 AgentX 客户端 CPython 是 x86_64 专用，自行拉起服务的轮次不再需要它 |
| 单轮基准（`inference_suite run`） | **可运行** | 自己拉起 `vllm serve`（vLLM-Ascend）、压测、写 `report.json` |
| 容量测量与形态对比（TP1 / TP2） | **可运行** | 见 §6 |
| 读 trace 派生候选 | **昇腾上可运行** | `analysis_route=msagent`：模型通过 msprof-mcp 与 msAgent profiler 技能读 `torch_npu` 采集；候选按 `recoverable_share_pct`（占比 × 可优化空间）排序，不再按名字。TraceLens 仍是 AMD 路由 |
| Profile 采集 | **可运行** | vLLM-Ascend 自带的 `torch_npu.profiler` 封装，由 `--profiler-config` 与 `/start_profile`、`/stop_profile` 驱动 |
| 算子优化 | **委托单轮闭环** | `ascend_kernel_agent` 读排序后的候选与 CANNBot 技能树，改 serving 源码树、产出补丁；补丁必须通过“应用 / 导入 / 起服”三道门。一次真实闭环测出：该候选仅占 2.8% 设备时间，实测在噪声内 |
| 量化 | **可运行** | msModelSlim 经 agfsm 技能树，由前置（`--quantize`）驱动或采纳已有 checkpoint（`HYPERLOOM_QUANTIZED_MODEL`）；8 卡上比 bf16+默认值再 +1.3% |
| KEEP/REVERT 判定（`hyperloom optimize`） | **端到端跑通** | 一次 2 小时会话测出基线（1519.8 tok/s）和 5 个种子候选，全部 REVERT；栈为空，因为没有候选越过 1% 阈值（见 §6） |

## 2. 方案

Hyperloom 的控制面（相位链、动作目录、PolicyGate、会话状态、报告）本来与硬件无关。AMD 专属的部分集中在
四个接缝上；本适配的做法是在每个接缝后面加一个实现，而不是在接缝内部写分支：

1. **设备后端**——设备枚举、可见卡掩码、占用判定、遥测。`hyperloom_common.device_backend` 回答 `amd` 或
   `ascend`，`hyperloom_common.ascend_smi` 读 `npu-smi`。
2. **Staging 分档**——一轮需要哪些钉死的工具。自行拉起服务的轮次只读 InferenceX 子集，不需要 AgentX
   客户端环境。
3. **分析路由**——把 trace 拆成 breakdown 与内核候选。契约（`analysis.md`、`kernel_candidates.json`、
   roofline 目标）不变，只换背后的产出者。
4. **内核委托**——整条流水线或单算子优化。GEAK 没有 NPU 支持，因此昇腾会话走单算子/委托路线，而不是
   GEAK 的 e2e 工作流。

两个主机事实决定了改动的形状：

- 空闲的昇腾 NPU 会报出数 GiB 驱动预留 HBM，且不属于任何进程；集群监控还可能在每张卡上持有 122 MiB 的
  context。因此占用判定沿用 AMD 规则：**进程的常驻内存超过 1 GiB**（或读不到内存）才算占用。
- 轮次的服务端只继承 `config.server.env_inherit`，所以 `PYTHONPATH` 必须保留 CANN 的 `site-packages`
  （`acl` 模块在那里），多卡运行必须设置 `HCCL_SOCKET_IFNAME`。

## 3. 仓库改动

| 文件 | 改动 |
|---|---|
| `src/hyperloom_common/device_backend.py` | 新增：`amd` / `ascend` 选择，`HYPERLOOM_DEVICE_BACKEND` 覆盖 |
| `src/hyperloom_common/ascend_smi.py` | 新增：解析 `npu-smi info`——卡、健康、功耗/温度/HBM、进程表 |
| `src/inference_suite/runtime/gpus.py` | 轮次的设备门禁在昇腾上走 `npu-smi`；1 GiB 占用规则 |
| `src/inference_suite/staging.py` | `StagingProfile`（`agentx` / `inferencex`）贯穿 `stage`、`verify`、`ensure_staged`；子集缓存会拒绝 AgentX 轮次 |
| `src/inference_suite/cli.py` | `stage --profile {agentx,inferencex}` |
| `src/inference_suite/run.py` | 轮次按自身负载类型对应的档位校验 |
| `src/inference_suite/__init__.py` | 导出 `StagingProfile`、`profile_for` |
| `src/inference_suite/tests/` | `conftest.py` 把套件里的 AMD 用例钉在 amd 后端；新增 `test_ascend_npu_gate.py`、`test_staging_profile.py` |

AMD 行为不变：不设 `HYPERLOOM_DEVICE_BACKEND` 时，有 `rocm-smi`/`amd-smi` 的主机保持 AMD 路径，AMD 用例
也自行钉住该后端。

## 4. 环境要求

- 昇腾 910B 级硬件（在 910B3、每卡 64 GiB HBM 上验证），CANN 9.x。
- 与所配 vLLM 匹配的 vLLM-Ascend。本次 bring-up 使用 vLLM 0.30.0 + vllm-ascend 0.30.0 + torch 2.10.0 +
  torch_npu 2.10.0.post7 + transformers 5.14.1，并 source 过 `ASCEND_HOME_PATH`。
- Hyperloom 自身不需要 `torch`：`inference_suite` 跑在一个装有 PyYAML、`aiohttp`、`transformers` 的解释器
  里（它自己运行压测客户端），并把 `server.framework_python` 指向 vLLM-Ascend 的解释器。

### 环境补丁（仅 bring-up 用，不在仓库内）

vllm-ascend 0.30.0 的 cp310 wheel 有三处本机无法直接运行的写法。以下都是 site-packages 补丁，刻意不进仓库；
长期解法是 Python 3.12 环境与装有 `nnal/atb` 的主机。

- `patch_indexer_kv_dtype.py` / `patch_kv_cache_dtype.py` 里的 `typing.Literal[*args]`（3.11+ 语法）→ 改为
  传元组，适配 Python 3.10。
- 模型补丁会主动 import `fla_npu`（GDN/KDA 算子，本机没有）→ 打桩，被调用时报错。
- 没有 `nnal/atb` 时缺 `libatb.so` → ATB 注册与预热改为记录日志并跳过。
- Profiler 导出 `[Text, Db]`，便于基于 DB 的分析工具读取采集结果。

## 5. 使用方式

```bash
export HYPERLOOM_DEVICE_BACKEND=ascend
export ASCEND_RT_VISIBLE_DEVICES=0
source "$ASCEND_HOME_PATH/set_env.sh"

# 1. 铺设一轮要读的工具。NPU 主机上铺子集档。
python -m inference_suite stage --tool-cache /path/to/tools --profile inferencex

# 2. 描述一次启动。该文件只被读取、不会被执行：轮次自己拉起服务。
cat > serve.sh <<'SH'
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve /models/Qwen3-8B --tensor-parallel-size 1 --max-model-len 4096 --dtype bfloat16
SH
python -m inference_suite resolve --query query.json --gpus 1 \
    --out base_launch.json --user-file serve.sh

# 3. 跑一轮并读 report.json。
python -m inference_suite run --config config.yaml --run-dir runs/base-1
```

`query.json` 是 `LaunchQuery` 文档（`model_id`、`framework`、`board`、`workload: fixed`、`concurrencies`、
`tp`、`isl`、`osl`）。`config.yaml` 是 `inference-suite-config/1` 文档；昇腾上值得设置的是
`server.framework_python`（vLLM-Ascend 解释器）、`server.gpus`、`paths.tool_cache` 与
`paths.tool_manifest_sha256`（`stage` 打印）、`telemetry.gpu.enabled: false`（没有 `amd-smi`），以及
`server.env.inherit`——必须包含 `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME`、`LD_LIBRARY_PATH`、`PATH`、
`PYTHONPATH` 和 `ASCEND_RT_VISIBLE_DEVICES`。

### 多卡（TP2）

```bash
# serve-tp2.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_SOCKET_IFNAME=<承载 HCCL 的网卡>   # 不设会 hcclGetRootInfo 失败（error 19）
vllm serve /models/Qwen3-8B --tensor-parallel-size 2 --max-model-len 4096 \
    --dtype bfloat16 --gpu-memory-utilization 0.75   # TP2 下 0.85 会在 sampler 处 OOM
```

然后用 `--gpus 2` 和一个 `tp` 为 `2` 的 `LaunchQuery` 做 resolve。

### 昇腾分析路由

一次昇腾采集由模型通过 `msprof-mcp` 与 msAgent profiler 技能来读：

```bash
export HYPERLOOM_MSAGENT_SKILLS=/path/to/msagent/skills/profiler
export HYPERLOOM_MSPROF_MCP=/path/to/mcp-venv/bin/msprof-mcp   # 一个装了 mcp<2 的解释器
export OPENAI_BASE_URL=... OPENAI_API_KEY=... CODEX_MODEL=<模型>
python -m hyperloom.orchestrator.trace_analysis.msagent_analysis \
    --trace-dir <轮次 trace 目录> --out <分析输出目录> --framework vllm
```

它会找到 `ASCEND_PROFILER_OUTPUT`，对数据库与 CSV 跑一轮 agent，按契约校验模型写出的 JSON，并发布
`analysis.md` 与 `kernel_candidates.json`（后者是编排器读的
`hot_kernels` / `routable_kernels` / `skipped_kernels` / `task_groups` 结构）。`trace_analyze` 请求带
`analysis_route=msagent` 时进入同一模块，命令行与协调器路径共用一份实现。

### Profile 采集

昇腾上的 profile 轮次先写出原始 `*_ascend_pt`（`PROF_*`、`FRAMEWORK/`），解析由
`torch_npu.profiler.profiler.analyse` 完成；执行器会跑这一步，再把采集记为这一轮的 trace
（`capture_format: ascend_pt`、`capture_parsed`），未指定路由的 `trace_analyze` 走 msagent 路由。
CANN 用主机自己的安装（本机是 `source /home/smoke/Ascend/ascend-toolkit/set_env.sh`）：它会把 `bisheng`
放进 `PATH`，否则 vLLM 报 `HAS_TRITON=False`，vllm-ascend 不注册 Qwen3 需要的 Triton 算子。默认采集窗口是按 AMD 的
长跑算的，短轮次里 profiler 根本不会启动，需要手动收窄：

```bash
export HYPERLOOM_PROFILE_MAX_ITERS=64    # 采集步数；设了它延时默认回到 8
export HYPERLOOM_PROFILE_DELAY_ITERS=1   # 采集前跳过的步数
```

### 决策环

```bash
export HYPERLOOM_DEVICE_BACKEND=ascend
export KERNEL_AGENT_ENV=<文件>                    # install.sh 本来会写的 kernel-agent env
export INFERENCE_SUITE_TOOL_CACHE=<已铺设的缓存>

python -m hyperloom optimize --framework vllm --model <模型路径> \
    --reference-script serve.sh --tp 1 --conc 64 --isl 1024 --osl 1024 \
    --max-hours 1.0 --no-kernel --no-enable-roofline --enablement off
```

原先要靠环境变量绕过的几个主机侧问题，现在由代码处理：构建探针在昇腾上问的是 `torch_npu` 能否看到
NPU（不再是“是不是 ROCm 构建”）、NPU 主机上 Ray 默认关闭、启动时丢弃 httpx 解析不了的 `no_proxy`
条目、板卡通过 `npu-smi` 自动识别。codex 自带的沙箱需要 bubblewrap：装它（`apt install bubblewrap`），
或在专用容器里设 `HYPERLOOM_CODEX_SANDBOX_MODE=bypass`。

编排与评审角色跑在网关模型上（`INFERENCE_OPTIMIZER_AGENT_BACKEND=codex`、`CODEX_MODEL=<模型>`）。要活得比
启动它的 shell 更久的会话，用 systemd 托管——下面 §6 那次跑满 2 小时的会话就是这么起的（普通
`tmux`/`setsid` 在这台机器上会在预检阶段被回收）：

```bash
sudo systemd-run --unit=hl-keep --collect --working-directory=<仓库> ./run-keep.sh
journalctl -u hl-keep -f            # 脚本自己的日志在脚本旁边
```

单卡基线一次 15–20 分钟（加载模型、预热轮 + 测量轮、精度评测），每个候选再加约 7 分钟，所以还要试网格的
会话需要两小时量级的预算。

## 6. 端到端流程速查

五个阶段的速查；每步的完整参数、产物与排错见 [docs/ascend/RUNBOOK.md](docs/ascend/RUNBOOK.md)，
本文件的 §5 保留各环节的详细用法。

```bash
# 0. 环境：CANN（自带 bisheng）、venv、网关、msAgent/msprof、可选 CANNBot / msModelSlim 技能树
source <cann>/set_env.sh; export HYPERLOOM_DEVICE_BACKEND=ascend; export CODEX_MODEL=<网关模型>

# 1. 一轮基准（先确认形态能起服）
inference_suite resolve --query query.json --gpus 8 --out base_launch.json --user-file serve-8card.sh
inference_suite run --config config.yaml --run-dir <run dir>

# 2. 优化会话（放 systemd 里，防止被回收）
systemd-run --unit=hl-run --collect --working-directory=<repo> ./run-keep.sh

# 3. 分析（profile 采集会自动 analyse；也可显式跑）
python -m hyperloom.orchestrator.trace_analysis.msagent_analysis --trace-dir <capture> --out <dir>

# 4. 算子（CANNBot 委托，三道门：应用 / 导入 / 起服）
python -m hyperloom.orchestrator.actions.executors.ascend_kernel_agent \
  --candidates <dir>/kernel_candidates.json --repo <vllm-ascend 源码树> --out <dir> \
  --serve-model <模型> --port 18070

# 5. 量化（前置驱动 msModelSlim，或采纳已有 checkpoint）
python -m hyperloom optimize ... --quantize "w8a8"
export HYPERLOOM_QUANTIZED_MODEL=<checkpoint>
```

## 7. 实测结果

Qwen3-8B，ISL/OSL 1024，并发 64，128 请求，单节点，prefix cache 保持启动脚本的原样。由
`inference_suite run` 自身测得：

| 形态 | 输出吞吐 | 平均 TTFT | 平均 TPOT | 完成情况 |
|---|---|---|---|---|
| TP1，单卡 | 1521.5 tok/s | 1733 ms | 40.20 ms | 128/128，0 失败 |
| TP2，双卡 | 2044.6 tok/s | 1756 ms | 29.45 ms | 128/128，0 失败 |

TP2 相对 TP1：**输出吞吐 +34.4%，TPOT −26.7%，TTFT +1.3%**。TP1 形态在丢弃一轮预热后，用压测客户端连跑
的重复性为 ±0.1%；上表每个形态各跑了一轮。

对 TP1 形态抓的 `torch_npu` profile 解释了天花板：decode 的
GEMM（`aclnnMatmul_MatMulV2`）占设备时间 79%，平均 `aic_mac_ratio` 仅 0.16——Cube 空闲，瓶颈是从 HBM 流
动权重；prefill GEMM 能到 0.86。不改变“每 token 搬运字节数”的框架旋钮都没有效果：调度器的 reserve-ISL
开关是中性（1434.5 → 1434.6 tok/s），`TASK_QUEUE_ENABLE=2` 在图捕获下被拒绝。有效的杠杆是摊薄或缩小权重
流——张量并行（见上）与权重量化（尚未跑）。

### 4 小时实跑，8 卡 DP8（Qwen3-4B）

同一形态给了 4 小时预算，协调器跑了两轮网格并复测：**基线 10413.4 → 11356.5 tok/s，累计 `cumulative_gain_validated`
9.06%，栈长 2，精度 0.9219**。两条 KEEP 都是“默认值不适配这个形态”，不是调参：

| 改动 | 增益 |
|---|---:|
| `--additional-config '{"scheduler_config":{"enable_balance_scheduling":true}}'` | +1.38% |
| 其上再 `--no-scheduler-reserve-full-isl` + `VLLM_USE_V2_MODEL_RUNNER=1` | +7.57% |

同一轮里四个经典旋钮全部为负（reserve-ISL 单独 −0.3%、`max-num-seqs 512` −3.8%、`batched-tokens 16k` −3.6%、
关 prefix cache −6.0%）。要点：**单个杠杆在噪声带内、组合起来才是 9%**；两次会话的基线本身就从 9934 漂到
10413，所以单次 <5% 的增益不能当结论。详见
[docs/ascend/experiments/2026-11-19-qwen3-4b-dp8.md](docs/ascend/experiments/2026-11-19-qwen3-4b-dp8.md)。

### 量化叠加（8 卡 DP8，W8A8）

用 msModelSlim 在本机导出的 W8A8 checkpoint，叠加上一节的三件套默认值，再跑一轮 2.5 小时会话：

| 形态 | 吞吐 | 相对 bf16 基线 | GSM8K |
|---|---:|---:|---:|
| bf16 基线 | 10413.4 | — | 0.9219 |
| bf16 + 三默认值 | 11356.5 | +9.06% | 0.9219 |
| W8A8 + 三默认值 | 11507.1 | +10.5% | 0.9062 |
| W8A8 + 默认值 + `--prefill-schedule-interval 2` | **11942.6** | **+14.7%** | 0.8906 |

量化这个形态只买到 **+1.3%**（4B 在 DP8/并发 256 下权重字节已不是瓶颈），代价是 GSM8K 掉 1.6 个点（1/64 样本）；
会话又 KEEP 了一个**调度**开关（`--prefill-schedule-interval 2`，+3.78%）。细节见实验记录。

### 决策环实跑（2 小时，单卡 910B3）

基线 **1519.8 tok/s**（320 请求，GSM8K 0.90625），随后 5 个种子候选。全部 REVERT：栈为空、
`cumulative_gain_val` 为 0，会话以 `stop_reason=time_exhausted` 收尾。

| 候选 | 吞吐 | 相对基线 | 判决 |
|---|---:|---:|---|
| 基线 | 1519.8 | — | — |
| `v00` decode-only 图模式（`cudagraph_mode=FULL_DECODE_ONLY`） | 1474.0 | −3.0% | REVERT |
| `v01` `--no-scheduler-reserve-full-isl` | 1511.7 | −0.5% | REVERT（`gain_below_threshold`） |
| `v02` `--max-num-seqs 128` | 1513.9 | −0.4% | REVERT |
| `v03` `--max-num-batched-tokens 16384` | 1524.7 | +0.3% | REVERT |
| `v04` `--no-enable-prefix-caching` | 1533.9 | +0.9% | REVERT（低于 1% 阈值） |

结合上面的形态对比：这个负载上**参数层已经饱和**——最好的候选落在噪声带内，能改变数字的是“每 token 搬运
的字节数”：张量并行（已测）与权重量化（尚未跑）。

8 卡 Qwen3-4B 这次会话的独立记录（环境、逐候选数据、两个 KEEP、期间的主机事故）见
[docs/ascend/experiments/2026-11-19-qwen3-4b-dp8.md](docs/ascend/experiments/2026-11-19-qwen3-4b-dp8.md)。

## 8. 已知限制

- 昇腾分析路由需要 agent CLI、模型网关，以及一个装了 `mcp<2` 的解释器来跑 msprof-mcp；这些都不在默认安装里。
- 还没有 KEEP：这个负载上参数层已饱和（见 §6）。收益要靠量化或换部署形态，两者都尚未接通。
- 没有内核后端：`kernelforge` 不支持 Ascend C 或 Triton-Ascend，GEAK 也没有 NPU 路径。
- 没有量化集成（msModelSlim），没有多机（单节点之外的 HCCL/拓扑）。
- AgentX 客户端环境是 x86_64 专用（`pins.AGENTX_PYTHON_BUILD`、`client_env.LOCK_PLATFORMS`），因此
  AgentX 与 MLPerf 负载无法在 aarch64 主机上 staging。
- CI 不覆盖这些内容：没有 NPU runner。

## 9. 下一步任务

1. **量化的定位**——msModelSlim 链路已通（8 卡上 W8A8 比 bf16+默认值再 +1.3%），收益量级取决于模型与形态。
   下一步：换 8B/32B 或更高并发量一遍，确认量化真正的杠杆位置；同时把精度评测从 64 样本提到 128/256。
2. **算子的调度与候选质量**——委托 turn 已能产出通过三道门（应用/导入/起服）的补丁，但 KERNEL 阶段还没有调用
   方，且当前候选（2.8% 设备时间）天花板只有零点几个百分点。下一步：接上阶段调度，并让分析侧优先给 GEMM
   tiling / PagedAttention 这类大占比候选（排序已按 `recoverable_share_pct`）。
3. **把张量并行作为部署形态**——TP2 已测 +34.4%，但它不是“同形态优化”；让会话能提出它（现在 base 固定
   TP1）并按正常判决保留。
4. **分析路由加固**——还差报告端在等的 `kernel_roofline.json` 旁挂文件，以及没有 agent/网关时的确定性回退。
5. **aarch64 上的 AgentX**——补 aarch64 解释器 pin 与配套客户端 lock，使 `agentx`/`mlperf` 负载能在 arm
   主机上 staging。
6. **CI**——把 `npu-smi` 解析与 staging 分档的夹具测试放进默认选择，并加一个带标记的 NPU runner 集成任务。
