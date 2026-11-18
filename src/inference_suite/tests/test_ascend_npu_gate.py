# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The NPU side of the round's device gate: npu-smi parsing, and the free/held decision built on it."""

from __future__ import annotations

import pytest

from hyperloom_common.ascend_smi import NpuProcess, NpuSmiError, npu_processes, npu_readings
from inference_suite.contracts.report import FailedStep
from inference_suite.runtime import gpus
from inference_suite.runtime.watch import Stop

_TABLE = """\
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.1.1                              Version: 26.1.1                                                      |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B3               | OK            | 96.7                 48                      0    / 0                |
| 0                         | 0000:C1:00.0  | 0                    0    / 0                3472 / 65536            |
+===========================+===============+======================================================================+
| 1     910B3               | OK            | 91.3                 49                      0    / 0                |
| 0                         | 0000:C2:00.0  | 37                   0    / 0                40960 / 65536           |
+===========================+===============+======================================================================+
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
| No running processes found in NPU 0                                                                              |
+------------------------------------------------------------------------------------------------------------------+
| 1       0                 | 4242          | python             | 37000                 | NA                      |
+------------------------------------------------------------------------------------------------------------------+
"""


def test_readings_name_each_npu_and_its_hbm() -> None:
    readings = npu_readings(_TABLE)

    assert [reading.index for reading in readings] == [0, 1]
    assert readings[0].name == "910B3"
    assert readings[0].health == "OK"
    assert readings[0].hbm_used_mb == 3472.0
    assert readings[0].hbm_total_mb == 65536.0
    assert readings[1].aicore_pct == 37.0


def test_an_idle_npu_is_free_however_much_hbm_the_driver_reserved() -> None:
    """An idle 910B reports GiB of driver-reserved HBM, so the process table, not the HBM reading, decides."""
    holders = npu_processes(_TABLE)

    assert 0 not in holders
    (holder,) = holders[1]
    assert (holder.pid, holder.name, holder.memory_mb) == (4242, "python", 37000.0)


def test_a_table_without_a_card_row_is_an_error() -> None:
    with pytest.raises(NpuSmiError):
        npu_readings("| nothing here |\n")


def _patch(monkeypatch: pytest.MonkeyPatch, count: int, holders: dict[int, list[NpuProcess]]) -> None:
    monkeypatch.setattr(gpus, "npu_count", lambda: count)
    monkeypatch.setattr(gpus, "npu_processes", lambda: holders)


def test_free_npus_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, 8, {})

    assert gpus.await_free_npus((0,), gate=_Gate()) is None


def test_a_held_npu_refuses_with_the_holder(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, 8, {0: [NpuProcess(7, "vllm", 37_000.0)]})
    # A held device is waited on for the release window; this test states the refusal, not the wait.
    monkeypatch.setattr(gpus, "RELEASE_WAIT_SEC", 0.0)

    stop = gpus.await_free_npus((0,), gate=_Gate())

    assert isinstance(stop, Stop)
    assert stop.step is FailedStep.GPUS_UNAVAILABLE
    assert "pid 7 (vllm)" in stop.detail


def test_a_context_holder_does_not_hold_the_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """A node monitor opens every device and keeps 122 MiB; that is not a workload to measure beside."""
    _patch(monkeypatch, 8, {0: [NpuProcess(99, "python_monitor.py", 122.0)]})

    assert gpus.await_free_npus((0,), gate=_Gate()) is None


def test_a_process_reporting_no_memory_holds_the_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable cell is not evidence of an idle card, the same rule the AMD gate applies."""
    _patch(monkeypatch, 8, {0: [NpuProcess(99, "unknown", None)]})
    monkeypatch.setattr(gpus, "RELEASE_WAIT_SEC", 0.0)

    stop = gpus.await_free_npus((0,), gate=_Gate())

    assert isinstance(stop, Stop)


def test_an_invisible_npu_refuses_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, 2, {})

    stop = gpus.await_free_npus((5,), gate=_Gate())

    assert isinstance(stop, Stop)
    assert "not visible" in stop.detail


def test_the_round_gate_dispatches_to_npus_on_an_npu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(gpus, "is_ascend", lambda: True)
    monkeypatch.setattr(gpus, "await_free_npus", lambda devices, gate: calls.append(tuple(devices)))

    gpus.await_free_gpus((0,), gate=_Gate())

    assert calls == [(0,)]


class _Gate:
    """A round gate that never trips, so the wait loop runs its first pass only."""

    def check(self, now: float) -> None:
        return None
