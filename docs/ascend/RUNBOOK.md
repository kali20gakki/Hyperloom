# Running Hyperloom on Ascend

A runbook for the whole path: one round, an optimization session, the analysis route, a delegated kernel turn, and
quantization. It assumes the host has already been brought up as described in [README.md](README.md); every stage
below is exercised — the support matrix at the end says which parts were measured and which only tested.

Placeholders: `<repo>`, `<venv>`, `<model>`, `<qwen3-4b>`, `<cann-setenv>`, `<tool-cache>`, `<gateway>`. Keep the real
values in one environment file and source it from every shell — the launcher, the agent turns and the rounds all read
the same variables.

## 1. What you need

| Piece | Why |
|---|---|
| A complete CANN install with `bisheng` on `PATH` | `torch_npu` loads and vLLM reports `HAS_TRITON=True`; without the compiler vllm-ascend silently skips its Triton ops and a Qwen3 serve dies on `qkv_rmsnorm_rope` |
| A Python environment with vLLM 0.30 + vllm-ascend 0.30 + torch 2.10 + torch_npu | the serving stack and the framework interpreter |
| An OpenAI-compatible agent gateway | the coordinator, the critic, the analysis turn, the kernel turn |
| A staged InferenceX subset | `inference_suite`'s benchmark client; `StagingProfile.INFERENCEX` on aarch64 |
| msAgent profiler skills + an `mcp<2` interpreter with msprof-mcp | the analysis route |
| CANNBot skill tree (optional) | the kernel turn's capability |
| msModelSlim checkout or a venv with it installed, plus the agfsm skill tree (optional) | the quantization turn |
| A GSM8K cache (`~/.cache/huggingface`) | the accuracy gate runs offline on many clusters |

## 2. Environment

```bash
# <cann-setenv>: the install's own set_env.sh. This is also what puts bisheng on PATH.
source <cann-setenv>

export PATH=<venv>/bin:$PATH
export HYPERLOOM_DEVICE_BACKEND=ascend
export ASCEND_RT_VISIBLE_DEVICES=0                     # widen to 0,1,... for a multi-card shape
export HCCL_SOCKET_IFNAME=<the node's RDMA/ethernet interface>   # required once collectives span cards
export USER_DATA_PATH=<sessions root>

# The agent gateway (OpenAI-compatible; some gateways need a routing header).
export OPENAI_BASE_URL=<gateway>
export OPENAI_API_KEY=<key>
export OPENAI_CUSTOM_HEADERS='x-opencode-session: <label>'
export CODEX_MODEL=<model>
export INFERENCE_OPTIMIZER_AGENT_BACKEND=codex

# Analysis route
export HYPERLOOM_MSAGENT_SKILLS=<msagent profiler skills>
export HYPERLOOM_MSPROF_MCP=<mcp venv>/bin/msprof-mcp

# Kernel turn (optional)
export HYPERLOOM_CANNBOT_SKILLS=<cannbot checkout or a curated dir of skill links>

# Quantization turn (optional)
export HYPERLOOM_AGFSM_SKILLS=<agfsm skills>
export HYPERLOOM_MSMODELSLIM=<msmodelslim checkout>
export HYPERLOOM_MSMODELSLIM_PYTHON=<venv with msmodelslim>
export HYPERLOOM_QUANTIZE_ENABLED=1
```

Preflight by hand before a long run:

```bash
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
python -c "from vllm.triton_utils import HAS_TRITON; print('HAS_TRITON', HAS_TRITON)"   # must be True on Ascend
npu-smi info | sed -n '/Process id/,$p' | head            # a card held by another process blocks every round
```

## 3. One round

Resolve a base launch from a serve script, then run one round into an empty directory:

```bash
inference_suite resolve --query <query.json> --gpus <N> --out <base_launch.json> --user-file <serve script>
inference_suite run --config <config.yaml> --run-dir <run dir>
```

`<query.json>` carries the shape (`model_id`, `framework`, `board`, `concurrencies`, `tp`, `isl`, `osl`); `<config.yaml>`
carries the base ref, model path, port, GPU ids, concurrency and prompts. A round boots the server, loads it, measures,
and writes `report.json` plus an `eval/` directory when the accuracy gate runs.

## 4. An optimization session

```bash
python -m hyperloom optimize \
  --framework vllm --model <qwen3-4b> \
  --reference-script <serve script> \
  --tp 1 --conc 256 --isl 1024 --osl 1024 --max-model-len 8192 --gpus-per-node 8 \
  --max-hours 2.0 --no-kernel --no-enable-roofline --enablement off \
  --orch-prompt <steering text file>
```

- `--no-kernel` keeps the session to parameters (the kernel layer is a delegated turn, see §6).
- A baseline on eight cards costs ~10 minutes including the accuracy gate; each candidate ~7 more. Budget
  `--max-hours` accordingly — a session under ~40 minutes never reaches `explore`.
- Run it where it cannot be reaped: `systemd-run --unit=hl-run --collect --working-directory=<repo> <runner script>`.
  A plain `tmux`/`setsid` start was killed in preflight on the tested host.
- The run's own log is the runner's stdout; the session directory holds everything else.

## 5. Analysis route

`--no-kernel` sessions do not request a trace analysis, but the profile round does capture. To analyze explicitly:

```bash
# torch_npu writes a raw *_ascend_pt tree; analyse() turns it into ASCEND_PROFILER_OUTPUT (the DB and CSVs)
python -c "from torch_npu.profiler.profiler import analyse; analyse('<capture dir>')"

python -m hyperloom.orchestrator.trace_analysis.msagent_analysis \
  --trace-dir <capture dir> --out <out dir> --framework vllm --timeout-sec 1200
```

The turn writes `analysis.md`, `kernel_candidates.json` and `msagent_analysis.json`. The profile executor and the
analysis route both run `analyse` themselves when it has not been run, so a session does not depend on the manual step.

**Candidates are ranked by `recoverable_share_pct` = `share_pct x expected_gain_pct`** — the device time a candidate
could actually take back — then by absolute time. The prompt asks for that ranking and for structural changes (GEMM
tiling/blocking or a weight layout; a PagedAttention or producer-consumer change) over micro-optimizations.

## 6. Kernel turn

The kernel layer is a headless agent turn whose capability comes from a CANNBot skill tree. It needs a **git tree** it
may patch — a checkout of the serving stack, not an installed package.

```bash
python -m hyperloom.orchestrator.actions.executors.ascend_kernel_agent \
  --candidates <out dir>/kernel_candidates.json \
  --repo <vllm-ascend checkout> \
  --out <kernel turn dir> \
  --serve-model <qwen3-4b> --port 18070 --timeout-sec 1800
```

The turn writes `kernel_patch.diff` and `kernel_agent_report.json`, and a patch is accepted only when:

1. it applies to the tree (`git apply --check`), which also rejects a turn that left its edits in the working tree;
2. every module it edits imports in the framework interpreter (`import_gate`);
3. a server booted from a patched copy reaches `/health` (`serve_gate`, `--serve-model`).

Without `--serve-model` only the first two run. The agent has no NPU in its sandbox, so it cannot measure: the gates
prove the tree still works and the session's round measures the effect. With `serve_gate` a turn that breaks
initialization is refused instead of costing a measurement.

## 7. Quantization

Three ways in, from most to least automatic:

```bash
# a) the prelude runs the msModelSlim turn before the loop
python -m hyperloom optimize ... --quantize "quantize this Qwen3 dense model to W8A8 with msModelSlim"

# b) adopt a checkpoint that already exists (validated, not re-quantized)
export HYPERLOOM_QUANTIZED_MODEL=<checkpoint>
python -m hyperloom optimize ... --quantize "w8a8"

# c) quantize by hand, then serve the checkpoint
cd <a directory you own>            # msModelSlim writes features.npy into the working directory
PYTHONPATH=<msmodelslim checkout> <msmodelslim venv>/bin/python \
  example/Qwen/quant_qwen.py --model_path <model> --save_directory <out> \
  --device_type npu --model_type qwen3 --w_bit 8 --a_bit 8 --trust_remote_code True
vllm serve <out> --quantization ascend
```

The prelude fails closed: a turn that cannot produce a checkpoint stops the run (exit 4) instead of optimizing the
un-quantized model. An `ok` checkpoint must carry `config.json` **and** `quant_model_description.json` or a
`quant_model_weight_*.safetensors`.

## 8. Reading the results

| Artifact | What it answers |
|---|---|
| `state.json` | current phase, `baseline_tput`, `current_best_tput`, `optimization_stack` |
| `reports/final.json` | `stop_reason`, `baseline_tput`, `cumulative_gain_validated`, `current_best` |
| `reports/optimization_journal.json` | per-candidate KEEP/REVERT, gain, fingerprint |
| `reports/final.md` | the same in prose |
| `runs/<action>/...` | per round: `server.log`, `report.json`, `eval/stdout.log` |
| `runs/profile/.../*_ascend_pt/ASCEND_PROFILER_OUTPUT` | the parsed capture |

A gain is only real when `cumulative_gain_validated` moves: the session re-measures the kept stack. Treat a single-run
gain under ~5% as unresolved — the baseline itself moved 5% between two identical sessions.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libhccl.so: cannot open shared object file` | the toolkit's `latest` link points at an incomplete install | source the complete install's `set_env.sh`, or pin it |
| `Triton not installed or not compatible`, then `qkv_rmsnorm_rope` missing | `bisheng` not on `PATH`, so vLLM reports `HAS_TRITON=False` | put the compiler's `bin` on `PATH` (it usually comes with `set_env.sh`) |
| `No module named 'acl'` | `PYTHONPATH` lost the CANN python path | source the toolkit's `set_env.sh` rather than hand-assembling paths |
| `Couldn't reach 'openai/gsm8k'` in the eval | no Hub access, no cache | pre-populate `~/.cache/huggingface` |
| `host_gpus_unavailable` | a leftover `VLLM::EngineCore` holds the card | `npu-smi info` the process table, kill by pid |
| `NPU out of memory` at startup | `--gpu-memory-utilization` above what a clean card leaves | 0.85 on a 64 GiB card |
| a candidate dies with `libatb.so` | the attention variant needs ATB, absent on this host | install ATB or exclude those variants |
| the session disappears in preflight | it was started under a shell that got reaped | `systemd-run` |
| the kernel turn's patch does not apply | the agent left its edits in the tree | the prompt asks it to revert; the gate refuses the patch |
| the patched tree does not serve | the patch breaks initialization, which an import cannot see | `--serve-model` runs the serve gate |

## 10. Support matrix

| Capability | State |
|---|---|
| Device discovery, occupancy gate, rounds, accuracy gate | measured |
| Parameter exploration + KEEP/REVERT | measured (8 cards, Qwen3-4B: +9.06% from three defaults) |
| Profile capture and automatic `analyse` | measured |
| Analysis route (msagent) with ranked candidates | measured (7 candidates ranked by recoverable share) |
| Kernel turn | measured once end to end: patch applied, imported, served, measured (no gain — a 2.8%-of-device-time kernel) |
| Quantization (msModelSlim, checkpoint adoption) | measured (W8A8 served; +1.3% over bf16-with-defaults on eight cards) |
| Multi-node, AgentX/MLPerf workloads, CI on NPU | not supported yet |
