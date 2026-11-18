# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An Ascend profile round's capture is its trace: ``*_ascend_pt/ASCEND_PROFILER_OUTPUT``, not a Kineto file."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import profile as pf
from hyperloom.orchestrator.actions.executors import trace_analyze as ta
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute


def _ascend_workspace(tmp_path: Path) -> Path:
    output = tmp_path / "profile" / "rank0_42_20261118_ascend_pt" / pf._ASCEND_OUTPUT_DIR
    output.mkdir(parents=True)
    (output / "kernel_details.csv").write_text("Device_id,Name\n", encoding="utf-8")
    return tmp_path


def test_the_round_records_the_ascend_capture_as_its_trace(tmp_path: Path) -> None:
    workspace = _ascend_workspace(tmp_path)
    result: dict[str, object] = {}

    pf._record_workspace_traces(result, str(workspace), "vllm", preflight=None)

    trace_dir = Path(str(result["trace_dir"]))
    assert trace_dir.name.endswith("_ascend_pt")
    assert result["capture_format"] == "ascend_pt"
    assert result["main_trace_path"] == str(trace_dir)
    assert result.get("status") != "failed"


def test_a_workspace_with_neither_trace_nor_capture_still_fails(tmp_path: Path) -> None:
    result: dict[str, object] = {}

    pf._record_workspace_traces(result, str(tmp_path), "vllm", preflight=None)

    assert result["status"] == "failed"
    assert result["error_class"] == "no_trace_files"


def test_an_ascend_capture_defaults_the_analysis_to_the_msagent_route(tmp_path: Path) -> None:
    capture = next(_ascend_workspace(tmp_path).rglob("*_ascend_pt"))

    assert ta._default_route(str(capture)) is AnalysisRoute.MSAGENT
    assert ta._default_route(str(tmp_path / "rank0.pt.trace.json.gz")) is AnalysisRoute.AGENT
