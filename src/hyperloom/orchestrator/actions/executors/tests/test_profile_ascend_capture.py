# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An Ascend profile round's capture is its trace: ``*_ascend_pt/ASCEND_PROFILER_OUTPUT``, not a Kineto file."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import profile as pf
from hyperloom.orchestrator.actions.executors import trace_analyze as ta
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute


def _ascend_workspace(tmp_path: Path, *, parsed: bool = True) -> Path:
    capture = tmp_path / "profile" / "rank0_42_20261118_ascend_pt"
    if parsed:
        output = capture / pf._ASCEND_OUTPUT_DIR
        output.mkdir(parents=True)
        (output / "kernel_details.csv").write_text("Device_id,Name\n", encoding="utf-8")
    else:
        # What torch_npu leaves before `analyse`: the raw PROF_* tree and its metadata, nothing readable yet.
        (capture / "PROF_000001_20261118120000_0000001ABCDEF").mkdir(parents=True)
        (capture / "profiler_info_0.json").write_text("{}", encoding="utf-8")
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


def test_a_raw_capture_is_parsed_before_it_is_recorded(tmp_path: Path) -> None:
    """torch_npu leaves a PROF_* tree; the round runs `analyse` so the analysis route has a database to read."""
    workspace = _ascend_workspace(tmp_path, parsed=False)

    def _fake_analyse(argv, **_kwargs):
        capture = Path(argv[-1])
        (capture / pf._ASCEND_OUTPUT_DIR).mkdir(exist_ok=True)
        return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    result: dict[str, object] = {}
    original = pf._parse_ascend_capture
    pf._parse_ascend_capture = lambda capture, **_kw: original(capture, runner=_fake_analyse)
    try:
        pf._record_workspace_traces(result, str(workspace), "vllm", preflight=None)
    finally:
        pf._parse_ascend_capture = original

    assert result["capture_format"] == "ascend_pt"
    assert result["capture_parsed"] is True
    capture = Path(str(result["main_trace_path"]))
    assert (capture / pf._ASCEND_OUTPUT_DIR).is_dir()


def test_an_unparseable_capture_is_still_recorded_for_the_analysis_route(tmp_path: Path) -> None:
    workspace = _ascend_workspace(tmp_path, parsed=False)

    def _failing_analyse(argv, **_kwargs):
        return type("Proc", (), {"returncode": 1, "stdout": "", "stderr": "no profiler data"})()

    result: dict[str, object] = {}
    original = pf._parse_ascend_capture
    pf._parse_ascend_capture = lambda capture, **_kw: original(capture, runner=_failing_analyse)
    try:
        pf._record_workspace_traces(result, str(workspace), "vllm", preflight=None)
    finally:
        pf._parse_ascend_capture = original

    assert result["capture_parsed"] is False
    assert result["trace_dir"] and Path(str(result["trace_dir"])).name.endswith("_ascend_pt")


def test_a_parsed_rank_is_preferred_over_a_raw_one(tmp_path: Path) -> None:
    """Multi-rank runs leave one capture per rank; the round should not re-parse what a rank already parsed."""
    workspace = tmp_path
    raw = workspace / "rank0_raw_ascend_pt"
    raw.mkdir()
    (raw / "PROF_1").mkdir()
    parsed = workspace / "rank1_parsed_ascend_pt"
    (parsed / pf._ASCEND_OUTPUT_DIR).mkdir(parents=True)

    assert pf._ascend_capture(workspace) == parsed
