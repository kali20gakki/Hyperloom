# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent-driven Ascend trace analysis: the msprof MCP tools and the msAgent profiler skills, under a model.

An Ascend capture is a ``*_ascend_pt`` directory holding ``ASCEND_PROFILER_OUTPUT``: a SQLite database (when the
profiler exported ``ExportType.Db``), ``kernel_details.csv``, ``op_statistic.csv`` and ``profiler_info_0.json``.
The AMD route parses a trace with TraceLens; here a model reads the same evidence through msprof-mcp and the msAgent
profiler skills, and returns its findings as strict JSON. This module owns the contract and the plumbing around that
turn — discovering the capture, prompting the agent, validating what it wrote, and publishing the two artifacts the
orchestrator already reads (``analysis.md`` and ``kernel_candidates.json``).

The agent is an external CLI (``codex exec`` today), configured from the environment: ``OPENAI_BASE_URL``,
``OPENAI_API_KEY``, ``OPENAI_CUSTOM_HEADERS`` and ``CODEX_MODEL``, with the MCP server shared out of its own
interpreter (``HYPERLOOM_MSPROF_MCP``, default ``msprof-mcp``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.io import JsonFormat, atomic_write_json
from hyperloom_common.timeutil import now_iso

from ._kernel_partition import build_kernel_candidates_document

log = logging.getLogger(__name__)

#: What a capture directory must hold for the agent to have evidence to read.
CAPTURE_OUTPUT_DIR = "ASCEND_PROFILER_OUTPUT"
CAPTURE_DB = "ascend_pytorch_profiler_0.db"
CAPTURE_KERNELS = "kernel_details.csv"
CAPTURE_OPS = "op_statistic.csv"
CAPTURE_CONFIG = "profiler_info_0.json"

#: The file the agent writes its findings to, beside the report and the candidate document.
REPORT_FILENAME = "msagent_analysis.json"

#: The msAgent skills an Ascend analysis reads. Each is a directory under the skills root holding ``SKILL.md``.
ANALYSIS_SKILLS = ("msagent-profiler-breakdown", "ascend-computation-analysis")

REPORT_SCHEMA = "hyperloom-msagent-analysis/1"
CANDIDATES_SCHEMA = "hyperloom-kernel-candidates/1"

#: Environment variables that configure the turn: the agent CLI, the MCP server command, and the skills root.
AGENT_ENV = "HYPERLOOM_ANALYSIS_AGENT"
SKILLS_ROOT_ENV = "HYPERLOOM_MSAGENT_SKILLS"
MCP_COMMAND_ENV = "HYPERLOOM_MSPROF_MCP"

#: How much of each skill's ``SKILL.md`` the prompt carries, so a large skill cannot swamp the evidence.
SKILL_CHARS = 6000


#: The suffix torch_npu gives a capture directory (``rank0_..._ascend_pt``).
_ASCEND_CAPTURE_SUFFIX = "_ascend_pt"


class AnalysisError(RuntimeError):
    """The capture, the agent turn, or what it wrote does not meet the contract."""


@dataclass(frozen=True)
class Capture:
    """The evidence one profile round left, resolved to the files the agent reads."""

    root: Path
    output: Path
    database: Path | None
    kernels: Path
    ops: Path | None
    config: Path | None


def is_ascend_capture(trace_dir: Path | str) -> bool:
    """Whether ``trace_dir`` names a torch_npu profiler capture: ``*_ascend_pt`` or the output directory itself."""
    path = Path(trace_dir)
    if path.name == CAPTURE_OUTPUT_DIR or path.name.endswith(_ASCEND_CAPTURE_SUFFIX):
        return True
    try:
        return path.is_dir() and (path / CAPTURE_OUTPUT_DIR).is_dir()
    except OSError:
        return False


#: ``torch_npu.profiler.profiler.analyse`` turns a raw ``*_ascend_pt`` tree into ``ASCEND_PROFILER_OUTPUT``; it
#: runs in the framework's interpreter and can take a couple of minutes for a large capture.
ANALYSE_TIMEOUT_SEC = 1800.0
_DIRECT_PARSER = "from torch_npu.profiler.profiler import analyse; import sys; analyse(sys.argv[1])"


def ensure_parsed(capture_dir: Path | str, *, python: str | None = None) -> Path:
    """Parse the raw capture at ``capture_dir`` when it holds no ``ASCEND_PROFILER_OUTPUT`` yet.

    The profiler wrapper writes ``PROF_*`` trees and leaves parsing to ``analyse``; without that step the capture has
    no database or CSVs to read. Returns the capture directory (now parsed) or raises ``AnalysisError``.
    """
    capture = Path(capture_dir)
    if (capture / CAPTURE_OUTPUT_DIR).is_dir():
        return capture
    interpreter = python or os.environ.get("HYPERLOOM_FRAMEWORK_PYTHON") or sys.executable
    log.info("msagent analysis: parsing %s with %s", capture, Path(interpreter).name)
    try:
        proc = subprocess.run(
            [interpreter, "-c", _DIRECT_PARSER, str(capture)],
            capture_output=True,
            text=True,
            timeout=ANALYSE_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnalysisError(f"torch_npu analyse did not run on {capture}: {exc}") from exc
    if proc.returncode != 0 and not (capture / CAPTURE_OUTPUT_DIR).is_dir():
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise AnalysisError(f"torch_npu analyse failed on {capture}: " + " | ".join(detail))
    if not (capture / CAPTURE_OUTPUT_DIR).is_dir():
        raise AnalysisError(f"torch_npu analyse left no {CAPTURE_OUTPUT_DIR} under {capture}")
    return capture


def discover_capture(trace_dir: Path) -> Capture:
    """The Ascend capture under ``trace_dir``: the deepest ``ASCEND_PROFILER_OUTPUT`` holding a kernel CSV.

    Raises:
        AnalysisError: No such directory, or it lacks the kernel details the analysis stands on.
    """
    candidates = [path for path in Path(trace_dir).rglob(CAPTURE_OUTPUT_DIR) if (path / CAPTURE_KERNELS).is_file()]
    if not candidates:
        raise AnalysisError(
            f"{trace_dir} holds no {CAPTURE_OUTPUT_DIR} with a {CAPTURE_KERNELS}; "
            "an Ascend profile round writes one under its trace directory"
        )
    output = max(candidates, key=lambda path: (path / CAPTURE_KERNELS).stat().st_mtime)
    database = output / CAPTURE_DB
    return Capture(
        root=output.parent,
        output=output,
        database=database if database.is_file() else None,
        kernels=output / CAPTURE_KERNELS,
        ops=(output / CAPTURE_OPS) if (output / CAPTURE_OPS).is_file() else None,
        config=(output / CAPTURE_CONFIG) if (output / CAPTURE_CONFIG).is_file() else None,
    )


def skills_root() -> Path | None:
    """The msAgent skills directory named by the environment, or ``None`` when it is unset."""
    raw = (os.environ.get(SKILLS_ROOT_ENV) or "").strip()
    return Path(raw) if raw else None


def _skill_text(root: Path | None, name: str) -> str:
    if root is None:
        return ""
    path = root / name / "SKILL.md"
    try:
        return path.read_text(encoding="utf-8")[:SKILL_CHARS]
    except OSError:
        log.warning("the msAgent skill %s is not readable at %s", name, path)
        return ""


def build_prompt(capture: Capture, *, framework: str, skills: Path | None, out_dir: Path) -> str:
    """The one turn's prompt: the evidence paths, the tools, the skills, and the exact output contract."""
    evidence = [
        f"- database (SQLite, preferred when present): {capture.database}"
        if capture.database
        else "- database: absent",
        f"- kernel details CSV: {capture.kernels}",
        f"- operator statistics CSV: {capture.ops}" if capture.ops else "- operator statistics CSV: absent",
        f"- profiler config JSON: {capture.config}" if capture.config else "- profiler config JSON: absent",
    ]
    skill_blocks = "\n\n".join(
        f'<skill name="{name}">\n{text}\n</skill>' for name in ANALYSIS_SKILLS if (text := _skill_text(skills, name))
    )
    report = out_dir / REPORT_FILENAME
    return f"""You are an Ascend NPU performance analyst. Read one profiler capture through the msprof-mcp tools and \
report what limits this workload, with evidence.

## Evidence

{chr(10).join(evidence)}

## Tools

The `msprof-mcp` MCP server is connected. Use it rather than guessing: `get_profiler_config`, `analyze_kernel_details`,
`analyze_op_statistic`, `execute_sql` (when the database is present), `msprof_analyze_advisor`. Read a CSV directly
only when a tool cannot, and say so.

## Skills

{skill_blocks or "(skills root not configured; follow the tool descriptions)"}

## Task

1. Decide which side binds the workload: compute, weight bandwidth, dispatch, or communication. Use the `aic_*`
   pipeline ratios and total device time, not intuition.
2. Rank the kernels by the device time each could actually recover: `share_pct` times the fraction of its own time the
   optimization takes back (`expected_gain_pct`). A kernel at 2% of device time with 80% headroom matters more than one
   at 30% with 2%; do not sort by name or by absolute time alone. For each, say what it is, its time and share, its core
   type and MAC ratio, why it costs what it costs, and which optimization would change it — preferring the structural
   change (GEMM tiling/blocking or weight layout for a matrix-bound profile, a PagedAttention or producer-consumer
   change for an attention-bound one) over a micro-optimization of the same kernel.
3. Estimate the roofline for this chip: BF16 peak TFLOP/s and HBM GB/s, then the achieved figures from the data.
4. Write your findings to `{report}` as strict JSON with exactly this shape:

{{
  "summary": "one paragraph",
  "scenario": {{"framework": "{framework}", "task_type": "inference",
               "bottleneck_class": "compute | memory_bound_decode | memory_bound_prefill | dispatch | communication"}},
  "hot_kernels": [
    {{"name": "<kernel as the profiler spells it>", "device_time_ms": 0.0, "share_pct": 0.0,
      "core_type": "<AI_CORE | MIX_AIC | AI_VECTOR_CORE | AICPU>", "category": "gemm | attention | norm | activation | moe | elementwise | other",
      "mac_ratio": 0.0, "evidence": "...",
      "optimization": "<the change; prefer GEMM tiling/blocking, a weight layout, a PagedAttention or producer-consumer change over a micro-optimization>",
      "expected_gain_pct": 0.0,
      "source_kind": "library | custom"}}
  ],
  "roofline": {{"peak_tflops_bf16": 0.0, "hbm_gbps": 0.0, "achieved_tflops": 0.0, "achieved_bw_gbps": 0.0}},
  "findings": [{{"severity": "info | warn", "finding": "...", "evidence": "..."}}],
  "confidence": "high | medium | low"
}}

Rules: every number must come from the tools' output; do not invent kernels, times or ratios. `source_kind` is
`library` for a vendor/aclnn kernel that cannot be rewritten and `custom` for an in-tree Ascend C or Triton-Ascend
kernel. Write nothing else to `{out_dir}` besides that JSON and an `analysis.md` summary of at most 400 lines.
"""


def agent_argv(prompt: str, workdir: Path, *, timeout_sec: float, mcp_command: str | None) -> list[str]:
    """The headless agent command: the configured gateway, the model, and the msprof MCP server."""
    cli = (os.environ.get(AGENT_ENV) or "codex").strip()
    if cli != "codex":
        raise AnalysisError(f"{AGENT_ENV}={cli!r} is not a supported analysis agent (codex)")
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    model = (os.environ.get("CODEX_MODEL") or "").strip()
    if not base_url or not model:
        raise AnalysisError("the analysis agent needs OPENAI_BASE_URL and CODEX_MODEL")
    argv = [
        cli,
        "exec",
        "--skip-git-repo-check",
        "--cd",
        str(workdir),
        # The turn writes its report and candidate document, so the sandbox must allow the workspace.
        "-s",
        "workspace-write",
        "-c",
        "model_provider=opencode",
        "-c",
        "model_providers.opencode.name=opencode",
        "-c",
        f"model_providers.opencode.base_url={base_url}",
        "-c",
        "model_providers.opencode.env_key=OPENAI_API_KEY",
        "-c",
        "model_providers.opencode.wire_api=responses",
        "-c",
        f"model={model}",
    ]
    for header in (os.environ.get("OPENAI_CUSTOM_HEADERS") or "").splitlines():
        name, _, value = header.partition(":")
        if name.strip() and value.strip():
            argv += ["-c", f"model_providers.opencode.http_headers.{name.strip()}={value.strip()}"]
    if mcp_command:
        argv += [
            "-c",
            f"mcp_servers.msprof-mcp.command={mcp_command}",
            # A non-interactive turn cannot answer an approval prompt; the analysis tools are read-only.
            "-c",
            'mcp_servers.msprof-mcp.default_tools_approval_mode="approve"',
        ]
    argv.append(prompt)
    return argv


def _run_agent(argv: Sequence[str], *, timeout_sec: float, log_path: Path) -> int:
    """Run the agent once, streaming its output to ``log_path``; returns the exit status."""
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            done = subprocess.run(  # nosec B603 - argv assembled here, no shell.
                list(argv), stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout_sec, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise AnalysisError(f"the analysis agent did not finish within {timeout_sec:.0f}s") from exc
    return done.returncode


def validate_report(document: object) -> dict[str, Any]:
    """The agent's report, checked for the fields the candidates stand on.

    Raises:
        AnalysisError: When the document is not an object with a non-empty ``hot_kernels`` list whose entries carry a
            name and a positive time.
    """
    if not isinstance(document, dict):
        raise AnalysisError("the report is not a JSON object")
    kernels = document.get("hot_kernels")
    if not isinstance(kernels, list) or not kernels:
        raise AnalysisError("the report names no hot kernel")
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, dict) or not str(kernel.get("name") or "").strip():
            raise AnalysisError(f"hot_kernels[{index}] names no kernel")
        time_ms = kernel.get("device_time_ms")
        if not isinstance(time_ms, (int, float)) or isinstance(time_ms, bool) or time_ms <= 0:
            raise AnalysisError(f"hot_kernels[{index}] ({kernel.get('name')}) reports no positive device_time_ms")
    return document


def _recoverable_share_pct(row: Mapping[str, Any]) -> float:
    """The share of total device time a candidate could recover: its share times the fraction it can take back.

    ``expected_gain_pct`` is the kernel's own recoverable fraction; an unsized kernel keeps its plain share so it is
    not buried under the sized ones.
    """
    try:
        share = float(row.get("pct_of_total") or 0.0)
    except (TypeError, ValueError):
        share = 0.0
    try:
        headroom = float(row.get("expected_gain_pct")) if row.get("expected_gain_pct") is not None else 100.0
    except (TypeError, ValueError):
        headroom = 100.0
    return round(share * max(0.0, min(headroom, 100.0)) / 100.0, 4)


def candidates_from_report(report: Mapping[str, Any], *, capture: Capture, framework: str) -> list[dict[str, Any]]:
    """The router's candidate rows, ranked by the device time each could recover (not by name)."""
    rows: list[dict[str, Any]] = []
    for kernel in report.get("hot_kernels") or ():
        source_kind = str(kernel.get("source_kind") or "library").strip().lower()
        rows.append(
            {
                "kernel_name": str(kernel["name"]).strip(),
                "category": str(kernel.get("category") or "other").strip().lower(),
                "core_type": kernel.get("core_type"),
                "duration_ms": float(kernel["device_time_ms"]),
                "pct_of_total": kernel.get("share_pct"),
                "mac_ratio": kernel.get("mac_ratio"),
                "source_type": source_kind,
                "patchable": source_kind == "custom",
                "framework": framework,
                "evidence": kernel.get("evidence"),
                "optimization": kernel.get("optimization"),
                "expected_gain_pct": kernel.get("expected_gain_pct"),
                "capture": str(capture.output),
            }
        )
    for row in rows:
        row["recoverable_share_pct"] = _recoverable_share_pct(row)
    rows.sort(key=lambda row: (-row["recoverable_share_pct"], -(row.get("duration_ms") or 0.0), row["kernel_name"]))
    return rows


def _render_analysis_md(report: Mapping[str, Any], capture: Capture) -> str:
    scenario = report.get("scenario") or {}
    roofline = report.get("roofline") or {}
    lines = [
        "# Ascend trace analysis",
        "",
        f"Capture: `{capture.output}`",
        "",
        str(report.get("summary") or "").strip(),
        "",
        "## Scenario",
        "",
        f"- framework: {scenario.get('framework', 'unknown')}",
        f"- task type: {scenario.get('task_type', 'inference')}",
        f"- bottleneck class: {scenario.get('bottleneck_class', 'unknown')}",
        f"- confidence: {report.get('confidence', 'unknown')}",
        "",
        "## Hot kernels",
        "",
        "| kernel | time (ms) | share | core | category | mac ratio |",
        "|---|---|---|---|---|---|",
    ]
    for kernel in report.get("hot_kernels") or ():
        lines.append(
            f"| `{kernel.get('name')}` | {kernel.get('device_time_ms')} | {kernel.get('share_pct')}% | "
            f"{kernel.get('core_type')} | {kernel.get('category')} | {kernel.get('mac_ratio')} |"
        )
    lines += ["", "## Roofline", ""]
    lines += [f"- {name}: {value}" for name, value in roofline.items()] or ["- not reported"]
    findings = report.get("findings") or []
    if findings:
        lines += ["", "## Findings", ""]
        for finding in findings:
            severity = str(finding.get("severity") or "info").upper()
            lines.append(f"- **{severity}** {finding.get('finding')} — {finding.get('evidence')}")
    lines.append("")
    return "\n".join(lines)


def run_analysis(
    trace_dir: Path,
    out_dir: Path,
    *,
    framework: str = "vllm",
    timeout_sec: float = 3600.0,
) -> dict[str, Any]:
    """Analyse one capture under an agent and write ``analysis.md`` and ``kernel_candidates.json`` into ``out_dir``.

    Raises:
        AnalysisError: The capture is missing, the agent turn fails or times out, or what it wrote breaks the contract.
    """
    if (Path(trace_dir).name.endswith(_ASCEND_CAPTURE_SUFFIX) or is_ascend_capture(trace_dir)) and not (
        Path(trace_dir) / CAPTURE_OUTPUT_DIR
    ).is_dir():
        ensure_parsed(trace_dir)
    capture = discover_capture(trace_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skills = skills_root()
    prompt = build_prompt(capture, framework=framework, skills=skills, out_dir=out_dir)
    prompt_path = out_dir / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    argv = agent_argv(
        prompt, out_dir, timeout_sec=timeout_sec, mcp_command=(os.environ.get(MCP_COMMAND_ENV) or "").strip() or None
    )
    started = time.time()
    returncode = _run_agent(argv, timeout_sec=timeout_sec, log_path=out_dir / "agent.log")
    report_path = out_dir / REPORT_FILENAME
    try:
        document = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"the agent wrote no readable {REPORT_FILENAME} (exit {returncode}): {exc}") from exc
    report = validate_report(document)
    candidates = candidates_from_report(report, capture=capture, framework=framework)
    candidates_path = out_dir / "kernel_candidates.json"
    atomic_write_json(
        candidates_path,
        build_kernel_candidates_document(
            {
                "schema": CANDIDATES_SCHEMA,
                "route": "msagent",
                "framework": framework,
                "capture": str(capture.output),
                "generated_at": now_iso(),
            },
            candidates,
            [],
            is_routable=lambda candidate: bool(candidate.get("patchable")),
        ),
        fmt=JsonFormat(indent=2, trailing_newline=True),
    )
    analysis_md = out_dir / "analysis.md"
    agent_md = out_dir / "analysis.md"
    if not agent_md.is_file():
        analysis_md.write_text(_render_analysis_md(report, capture), encoding="utf-8")
    return {
        "status": "ok",
        "route": "msagent",
        "capture": str(capture.output),
        "analysis_md": str(analysis_md),
        "kernel_candidates": str(candidates_path),
        # The handler's artifact contract, so the request path needs no adapter.
        "trace_report_path": str(analysis_md),
        "candidates_path": str(candidates_path),
        "artifact_paths": {"kernel_candidates": str(candidates_path), "trace_report_path": str(analysis_md)},
        "hot_kernels": len(candidates),
        "duration_sec": round(time.time() - started, 1),
        "agent_exit": returncode,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hyperloom.orchestrator.trace_analysis.msagent_analysis",
        description="Analyse an Ascend profiler capture with msprof-mcp and the msAgent profiler skills.",
    )
    parser.add_argument("--trace-dir", type=Path, required=True, help="the round's trace directory")
    parser.add_argument("--out", type=Path, required=True, help="where the artifacts are written")
    parser.add_argument("--framework", default="vllm")
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    args = parser.parse_args(argv)
    try:
        summary = run_analysis(args.trace_dir, args.out, framework=args.framework, timeout_sec=args.timeout_sec)
    except AnalysisError as exc:
        print(json.dumps({"status": "error", "route": "msagent", "error": str(exc)}))
        return 1
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
