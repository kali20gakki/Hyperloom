# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The agent-driven Ascend route: capture discovery, the turn's command, and the artifacts it publishes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis import msagent_analysis as route

_REPORT = {
    "summary": "Decode is weight-bandwidth bound.",
    "scenario": {"framework": "vllm", "task_type": "inference", "bottleneck_class": "memory_bound_decode"},
    "hot_kernels": [
        {
            "name": "aclnnMatmul_MatMulV2",
            "device_time_ms": 14110.2,
            "share_pct": 79.0,
            "core_type": "AI_CORE",
            "category": "gemm",
            "mac_ratio": 0.16,
            "evidence": "79% of device time at a 0.16 MAC ratio",
            "optimization": "quantize the weights",
            "expected_gain_pct": 40.0,
            "source_kind": "library",
        },
        {
            "name": "split_qkv_rmsnorm_rope_kernel",
            "device_time_ms": 390.0,
            "share_pct": 2.2,
            "core_type": "AI_VECTOR_CORE",
            "category": "norm",
            "mac_ratio": 0.0,
            "evidence": "in-tree fused kernel",
            "optimization": "tile the vector loop",
            "expected_gain_pct": 1.0,
            "source_kind": "custom",
        },
    ],
    "roofline": {"peak_tflops_bf16": 294.91, "hbm_gbps": 800.0, "achieved_tflops": 3.1, "achieved_bw_gbps": 420.0},
    "findings": [{"severity": "warn", "finding": "Cube idles during decode", "evidence": "aic_mac_ratio 0.16"}],
    "confidence": "high",
}


def _capture(tmp_path: Path) -> Path:
    output = tmp_path / "capture" / "rank0_abcdef_ascend_pt" / route.CAPTURE_OUTPUT_DIR
    output.mkdir(parents=True)
    (output / route.CAPTURE_KERNELS).write_text("Device_id,Name\n", encoding="utf-8")
    (output / route.CAPTURE_OPS).write_text("Device_id,OP Type\n", encoding="utf-8")
    (output / route.CAPTURE_DB).write_bytes(b"")
    return tmp_path / "capture"


def test_discover_capture_finds_the_output_dir_and_its_files(tmp_path: Path) -> None:
    capture = route.discover_capture(_capture(tmp_path))

    assert capture.output.name == route.CAPTURE_OUTPUT_DIR
    assert capture.kernels.name == route.CAPTURE_KERNELS
    assert capture.database is not None
    assert capture.ops is not None


def test_discover_capture_refuses_a_directory_with_no_capture(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(route.AnalysisError, match="no ASCEND_PROFILER_OUTPUT"):
        route.discover_capture(tmp_path)


def test_a_report_without_hot_kernels_is_refused() -> None:
    with pytest.raises(route.AnalysisError, match="names no hot kernel"):
        route.validate_report({"summary": "x", "hot_kernels": []})


def test_a_kernel_without_a_positive_time_is_refused() -> None:
    document = {"hot_kernels": [{"name": "k", "device_time_ms": 0}]}

    with pytest.raises(route.AnalysisError, match="positive device_time_ms"):
        route.validate_report(document)


def test_only_custom_kernels_are_routable(tmp_path: Path) -> None:
    capture = route.discover_capture(_capture(tmp_path))
    report = route.validate_report(json.loads(json.dumps(_REPORT)))

    rows = route.candidates_from_report(report, capture=capture, framework="vllm")

    assert [row["patchable"] for row in rows] == [False, True]
    assert rows[0]["kernel_name"] == "aclnnMatmul_MatMulV2"
    assert rows[0]["duration_ms"] == 14110.2


def test_the_agent_command_carries_the_gateway_model_mcp_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("CODEX_MODEL", "deepseek-v4.1-flash")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "x-opencode-session: probe")

    argv = route.agent_argv("prompt", Path("/tmp/w"), timeout_sec=1.0, mcp_command="msprof-mcp")

    assert "exec" in argv and "--skip-git-repo-check" in argv
    assert "model=deepseek-v4.1-flash" in argv
    assert "model_providers.opencode.base_url=https://gateway.example/v1" in argv
    assert argv[argv.index("-s") + 1] == "workspace-write"
    assert "mcp_servers.msprof-mcp.command=msprof-mcp" in argv
    assert any(arg.startswith("mcp_servers.msprof-mcp.default_tools_approval_mode") for arg in argv)
    assert "model_providers.opencode.http_headers.x-opencode-session=probe" in argv


def test_the_agent_command_refuses_an_unsupported_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(route.AGENT_ENV, "not-an-agent")

    with pytest.raises(route.AnalysisError, match="not a supported analysis agent"):
        route.agent_argv("prompt", Path("/tmp/w"), timeout_sec=1.0, mcp_command=None)


def test_run_analysis_publishes_the_two_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _capture(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("CODEX_MODEL", "deepseek-v4.1-flash")

    def fake_agent(argv: list[str], *, timeout_sec: float, log_path: Path) -> int:
        (out / route.REPORT_FILENAME).write_text(json.dumps(_REPORT), encoding="utf-8")
        return 0

    monkeypatch.setattr(route, "_run_agent", fake_agent)
    summary = route.run_analysis(capture, out, framework="vllm", timeout_sec=5.0)

    assert summary["status"] == "ok"
    analysis_md = Path(summary["analysis_md"])
    assert analysis_md.is_file() and "Hot kernels" in analysis_md.read_text(encoding="utf-8")
    document = json.loads(Path(summary["kernel_candidates"]).read_text(encoding="utf-8"))
    assert document["route"] == "msagent"
    assert [row["patchable"] for row in document["hot_kernels"]] == [False, True]
    assert document["routable_kernels"] == [document["hot_kernels"][1]]
    assert document["skipped_kernels"] == [document["hot_kernels"][0]]


def test_a_raw_capture_is_parsed_with_torch_npus_analyse(tmp_path):
    """The profiler leaves PROF_* trees; `ensure_parsed` runs `analyse` in the framework's interpreter."""
    capture = tmp_path / "rank0_1_ascend_pt"
    (capture / "PROF_000001_x").mkdir(parents=True)

    stub = tmp_path / "analyse-stub.sh"
    stub.write_text('#!/bin/sh\nmkdir -p "$3/ASCEND_PROFILER_OUTPUT"\n', encoding="utf-8")
    stub.chmod(0o755)

    assert route.ensure_parsed(capture, python=str(stub)) == capture
    assert (capture / route.CAPTURE_OUTPUT_DIR).is_dir()


def test_an_analyse_that_leaves_nothing_raises_with_the_call_in_the_message(tmp_path):
    capture = tmp_path / "rank0_1_ascend_pt"
    capture.mkdir()

    with pytest.raises(route.AnalysisError):
        route.ensure_parsed(capture, python="/bin/false")
