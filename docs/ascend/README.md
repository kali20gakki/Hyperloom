# Hyperloom on Ascend NPU

> 中文版（仓库首页）：[README.md](../../README.md)

Status: **bring-up spike**. The measurement, analysis and decision paths run on Ascend NPUs today; the kernel
and quantization layers are not wired. This document states what runs, how to run it, what changed in the tree, and
what is still missing.

## 1. What runs

| Stage | State | Notes |
|---|---|---|
| Device discovery, occupancy gate, round environment | **works** | `npu-smi` through `hyperloom_common.device_backend`; the AMD path is unchanged and selected by `HYPERLOOM_DEVICE_BACKEND=amd` |
| Tool staging on aarch64 | **works** | `StagingProfile.INFERENCEX` stages only the InferenceX subset; the pinned AgentX client CPython is x86_64-only and is no longer required by a round that drives the server itself |
| One benchmark round (`inference_suite run`) | **works** | boots `vllm serve` (vLLM-Ascend), loads it, writes `report.json` |
| Capacity measurement and shape comparison (TP1 vs TP2) | **works** | see §6 |
| Reading a trace and deriving candidates | **works on Ascend** | `analysis_route=msagent`: a model reads the `torch_npu` capture through msprof-mcp under the msAgent profiler skills. TraceLens stays the AMD route |
| Profile capture | **works** | vLLM-Ascend's own `torch_npu.profiler` wrapper, driven by `--profiler-config` and `/start_profile`/`/stop_profile` |
| Kernel optimization | **not wired** | no Ascend kernel backend in `kernelforge`; kernel work is meant to be delegated (CANNBot workflows) |
| KEEP/REVERT decisions (`hyperloom optimize`) | **runs end to end** | a 2 h session measured the baseline (1519.8 tok/s) and five seeded candidates, and reverted all five; the stack stayed empty because none cleared the 1% threshold (§6) |

## 2. Approach

Hyperloom's control plane (phase chain, action catalogue, policy gate, session state, reporting) is already
hardware-agnostic. What is AMD-specific sits in four seams, and this adaptation adds an implementation behind
each one rather than branching inside them:

1. **Device backend** — device enumeration, the visible-device mask, occupancy, telemetry.
   `hyperloom_common.device_backend` answers `amd` or `ascend`; `hyperloom_common.ascend_smi` reads `npu-smi`.
2. **Staging profile** — which pinned tools a round needs. A round that drives the server itself reads only
   InferenceX's subset, so it does not need the AgentX client environment.
3. **Analysis route** — frame a trace into a breakdown and kernel candidates. The contract (`analysis.md`,
   `kernel_candidates.json`, roofline targets) is unchanged; only the producer behind it is new.
4. **Kernel delegate** — whole-pipeline or per-kernel optimization. GEAK has no NPU support, so an Ascend
   session uses per-kernel/delegated work instead of GEAK's e2e workflow.

Two host facts drove the shape of the change:

- An idle Ascend NPU reports several GiB of driver-reserved HBM that no process owns, and a cluster monitor
  may hold a 122 MiB context on every device. Occupancy therefore follows the AMD rule: a process holds a
  device only when its **resident memory** exceeds 1 GiB (or when it reports none).
- A round's server inherits only `config.server.env_inherit`, so `PYTHONPATH` must still carry CANN's
  `site-packages` (the `acl` module lives there) and `HCCL_SOCKET_IFNAME` must be set for multi-card runs.

## 3. What changed in the tree

| File | Change |
|---|---|
| `src/hyperloom_common/device_backend.py` | new: `amd` / `ascend` selection, `HYPERLOOM_DEVICE_BACKEND` override |
| `src/hyperloom_common/ascend_smi.py` | new: `npu-smi info` parsing — cards, health, power/temperature/HBM, process table |
| `src/inference_suite/runtime/gpus.py` | the round's device gate dispatches to `npu-smi` on an Ascend host; 1 GiB hold rule |
| `src/inference_suite/staging.py` | `StagingProfile` (`agentx` / `inferencex`) threaded through `stage`, `verify`, `ensure_staged`; a subset cache refuses an AgentX round |
| `src/inference_suite/cli.py` | `stage --profile {agentx,inferencex}` |
| `src/inference_suite/run.py` | a round verifies against the profile of its workload |
| `src/inference_suite/__init__.py` | exports `StagingProfile`, `profile_for` |
| `src/inference_suite/tests/` | `conftest.py` pins the AMD backend for the suite's AMD tests; `test_ascend_npu_gate.py`, `test_staging_profile.py` |

The AMD behaviour is unchanged: without `HYPERLOOM_DEVICE_BACKEND`, a host with `rocm-smi`/`amd-smi` keeps the
AMD path, and the existing AMD tests pin that backend themselves.

## 4. Requirements

- Ascend 910B-class hardware (validated on 910B3, 64 GiB HBM per card), CANN 9.x.
- A vLLM-Ascend install matching the vLLM it plugs into. The spike used vLLM 0.30.0 + vllm-ascend 0.30.0 +
  torch 2.10.0 + torch_npu 2.10.0.post7 + transformers 5.14.1, with `ASCEND_HOME_PATH` sourced.
- Hyperloom's own packages need no `torch`: run `inference_suite` under an interpreter that has PyYAML,
  `aiohttp` and `transformers` (it runs the load client itself), and point `server.framework_python` at the
  vLLM-Ascend interpreter.

### Environment workarounds (spike only, not part of the tree)

vllm-ascend 0.30.0's cp310 wheel ships three constructs this host cannot run as-is. These are site-package
patches, deliberately kept out of the repository; the durable fixes are a Python 3.12 environment and a host
with the `nnal/atb` package installed.

- `typing.Literal[*args]` (3.11+ syntax) in `patch_indexer_kv_dtype.py` / `patch_kv_cache_dtype.py` → pass the
  tuple instead, for Python 3.10.
- `fla_npu` (GDN/KDA ops, absent here) is imported eagerly by a model patch → a stub that raises if called.
- `libatb.so` is absent without `nnal/atb` → ATB registration and warm-up log and skip instead of failing.
- Profiler: export `[Text, Db]` so DB-based analysis tools can read the capture.

## 5. Usage

```bash
export HYPERLOOM_DEVICE_BACKEND=ascend
export ASCEND_RT_VISIBLE_DEVICES=0
source "$ASCEND_HOME_PATH/set_env.sh"

# 1. Stage the tools a round reads. On an NPU host the subset profile is the one that stages.
python -m inference_suite stage --tool-cache /path/to/tools --profile inferencex

# 2. Describe the launch once. The file is read, never run: a round starts the server itself.
cat > serve.sh <<'SH'
export ASCEND_RT_VISIBLE_DEVICES=0
vllm serve /models/Qwen3-8B --tensor-parallel-size 1 --max-model-len 4096 --dtype bfloat16
SH
python -m inference_suite resolve --query query.json --gpus 1 \
    --out base_launch.json --user-file serve.sh

# 3. Run one round and read report.json.
python -m inference_suite run --config config.yaml --run-dir runs/base-1
```

`query.json` is a `LaunchQuery` document (`model_id`, `framework`, `board`, `workload: fixed`, `concurrencies`,
`tp`, `isl`, `osl`). `config.yaml` is an `inference-suite-config/1` document; the fields worth setting on NPU
are `server.framework_python` (the vLLM-Ascend interpreter), `server.gpus`, `paths.tool_cache` +
`paths.tool_manifest_sha256` (printed by `stage`), `telemetry.gpu.enabled: false` (no `amd-smi`), and
`server.env.inherit` — which must include `ASCEND_HOME_PATH`, `ASCEND_TOOLKIT_HOME`, `LD_LIBRARY_PATH`,
`PATH`, `PYTHONPATH` and `ASCEND_RT_VISIBLE_DEVICES`.

### Multi-card (TP2)

```bash
# serve-tp2.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_SOCKET_IFNAME=<the NIC that carries HCCL>   # without it hcclGetRootInfo fails (error 19)
vllm serve /models/Qwen3-8B --tensor-parallel-size 2 --max-model-len 4096 \
    --dtype bfloat16 --gpu-memory-utilization 0.75      # 0.85 OOMs in the sampler under TP2
```

Then resolve with `--gpus 2` and a `LaunchQuery` whose `tp` is `2`.

### Analysis route (Ascend)

An Ascend capture is read by a model through the `msprof-mcp` tools and the msAgent profiler skills:

```bash
export HYPERLOOM_MSAGENT_SKILLS=/path/to/msagent/skills/profiler
export HYPERLOOM_MSPROF_MCP=/path/to/mcp-venv/bin/msprof-mcp   # an interpreter with mcp<2
export OPENAI_BASE_URL=... OPENAI_API_KEY=... CODEX_MODEL=<model>
python -m hyperloom.orchestrator.trace_analysis.msagent_analysis \
    --trace-dir <round trace directory> --out <analysis directory> --framework vllm
```

It discovers the `ASCEND_PROFILER_OUTPUT` directory, runs one agent turn over the database and CSVs, checks the JSON
the model wrote against the report contract, and publishes `analysis.md` and `kernel_candidates.json` (the latter with
the `hot_kernels` / `routable_kernels` / `skipped_kernels` / `task_groups` shape the orchestrator reads). A requested
`trace_analyze` with `analysis_route=msagent` enters the same module, so the coordinator path and the direct command
share one implementation.

### Profile capture

A profile round on an Ascend host writes `*_ascend_pt/ASCEND_PROFILER_OUTPUT` under the round's trace directory;
the executor records that directory as the round's trace (`capture_format: ascend_pt`) and a `trace_analyze` that
names no route goes to the msagent one. The default capture window is computed for a much longer AMD run, so the
profiler never starts inside a short NPU round; bound it:

```bash
export HYPERLOOM_PROFILE_MAX_ITERS=64    # capture steps; setting it also resets the delay default to 8
export HYPERLOOM_PROFILE_DELAY_ITERS=1   # steps to skip before capturing
```

### Decision loop

```bash
export HYPERLOOM_DEVICE_BACKEND=ascend
export KERNEL_AGENT_ENV=<file>                   # the kernel-agent env install.sh normally writes
export INFERENCE_SUITE_TOOL_CACHE=<staged cache>

python -m hyperloom optimize --framework vllm --model <model path> \
    --reference-script serve.sh --tp 1 --conc 64 --isl 1024 --osl 1024 \
    --max-hours 1.0 --no-kernel --no-enable-roofline --enablement off
```

The host details this used to need variables for are now decided by the code: the build probe asks whether
`torch_npu` sees an NPU instead of whether torch is a ROCm build, Ray is off by default on an NPU host,
`no_proxy` entries httpx cannot parse are dropped at startup, and the board is read through `npu-smi`.
Codex's own sandbox needs bubblewrap: install it (`apt install bubblewrap`), or set
`HYPERLOOM_CODEX_SANDBOX_MODE=bypass` on a dedicated container.

The orchestration and critic roles run on the gateway model (`INFERENCE_OPTIMIZER_AGENT_BACKEND=codex`,
`CODEX_MODEL=<model>`). A session that has to outlive the shell it was started from runs under systemd — that is how
the completed run in §6 was made, after a plain `tmux`/`setsid` launch died in preflight here:

```bash
sudo systemd-run --unit=hl-keep --collect --working-directory=<repo> ./run-keep.sh
journalctl -u hl-keep -f            # the script's own log sits beside it
```

A baseline of one card costs 15–20 minutes (model load, one warmup plus measured round, and the eval), and each
candidate costs about 7 minutes more, so a session that is to try a grid needs a couple of hours.

## 6. Measured results

Qwen3-8B, ISL/OSL 1024, concurrency 64, 128 prompts, one node, prefix caching as the launch leaves it.
Measured by `inference_suite run` itself:

| Shape | Output tok/s | Mean TTFT | Mean TPOT | Completed |
|---|---|---|---|---|
| TP1, one card | 1521.5 | 1733 ms | 40.20 ms | 128/128, 0 failed |
| TP2, two cards | 2044.6 | 1756 ms | 29.45 ms | 128/128, 0 failed |

TP2 over TP1: **+34.4% output throughput, −26.7% TPOT, +1.3% TTFT**. The TP1 shape reproduced to ±0.1%
across back-to-back benchmark-client rounds once one warm round was discarded; the engine rounds above ran once
per shape.

A `torch_npu` profile of the TP1 shape explains the ceiling: decode GEMMs (`aclnnMatmul_MatMulV2`) take 79% of
device time with an average `aic_mac_ratio` of 0.16, i.e. the Cube is idle and the weights stream from HBM.
Prefill GEMMs reach 0.86. Framework knobs that do not touch bytes moved per token did nothing: a scheduler
reserve-ISL toggle was neutral (1434.5 → 1434.6 tok/s), and `TASK_QUEUE_ENABLE=2` is refused under graph
capture. The levers that do are the ones that spread or shrink the weight stream — tensor parallel (above) and
weight quantization (not yet run).

### Decision-loop run (2 h, one 910B3)

Baseline **1519.8 tok/s** (320 prompts, GSM8K 0.90625), then the five seeded candidates. Every one was reverted:
the stack stayed empty, `cumulative_gain_val` is 0, and the session closed with `stop_reason=time_exhausted`.

| Candidate | tok/s | vs baseline | Verdict |
|---|---:|---:|---|
| baseline | 1519.8 | — | — |
| `v00` decode-only graph (`cudagraph_mode=FULL_DECODE_ONLY`) | 1474.0 | −3.0% | REVERT |
| `v01` `--no-scheduler-reserve-full-isl` | 1511.7 | −0.5% | REVERT (`gain_below_threshold`) |
| `v02` `--max-num-seqs 128` | 1513.9 | −0.4% | REVERT |
| `v03` `--max-num-batched-tokens 16384` | 1524.7 | +0.3% | REVERT |
| `v04` `--no-enable-prefix-caching` | 1533.9 | +0.9% | REVERT (below the 1% threshold) |

Read with the shape comparison above, this says the parameter layer is saturated for this workload: the best
candidate lands inside the noise band, and the levers that move the number are transfers per token — tensor parallel
(measured) and weight quantization (not yet run).

A standalone record of the eight-card Qwen3-4B session — setup, per-candidate numbers, the two keeps, and the host
incidents around it — is in [experiments/2026-11-19-qwen3-4b-dp8.md](experiments/2026-11-19-qwen3-4b-dp8.md).

## 7. Known limitations

- The Ascend analysis route needs an agent CLI, a gateway and an `mcp<2` interpreter for msprof-mcp; none of that is
  in the default install.
- No candidate has been kept: the parameter layer is saturated on this workload (§6). A gain needs quantization
  or a different deployment shape, and neither is wired yet.
- No kernel backend: `kernelforge` has no Ascend C or Triton-Ascend language support, and GEAK has no NPU path.
- No quantization integration (msModelSlim), no multi-node (HCCL/VLLM topology beyond one node).
- The AgentX client environment is x86_64-only (`pins.AGENTX_PYTHON_BUILD`, `client_env.LOCK_PLATFORMS`), so
  AgentX and MLPerf workloads cannot stage on an aarch64 host.
- CI does not exercise any of this: there is no NPU runner.

## 8. Next steps

1. **Quantization** — msModelSlim W8A8 weights through vLLM-Ascend, gated on the accuracy eval. This is the
   lever the profile supports and the one the saturated parameter layer leaves: acceptance is a kept candidate
   above the 1% threshold with accuracy inside the eval's tolerance.
2. **Tensor parallel as a deployment shape** — TP2 is measured at +34.4% but is not a same-shape optimization;
   make the session able to propose it (the base launches at TP1 today) and keep it through the normal verdict.
3. **Analysis route hardening** — the route exists and a live capture produced `analysis.md` and six candidates,
   one routable. What remains: the roofline sidecar the report writer expects (`kernel_roofline.json`) and a
   deterministic fallback when no agent or gateway is configured.
4. **Kernel delegation** — hand a hot operator to the CANNBot Triton-Ascend / Ascend C workflows, integrate the
   result, and measure it end to end. Acceptance: one operator with a measured speedup and unchanged accuracy.
5. **AgentX on aarch64** — add an aarch64 interpreter pin and a matching client lock so `agentx`/`mlperf`
   workloads stage on arm hosts.
6. **CI** — fixture-driven tests for `npu-smi` parsing and the staging profiles in the default selection, plus a
   marked integration job on an NPU runner.
