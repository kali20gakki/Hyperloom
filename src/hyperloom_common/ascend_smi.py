# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only ``npu-smi`` primitives: which NPUs exist, how they read, and which processes the process table names.

``npu-smi info`` prints two lines per NPU (a card line and its chip line) and a process table below them, all in
fixed-width ASCII cells separated by ``|``. The table is the only thing this module trusts: an idle 910B reports
several GiB of driver-reserved HBM that no process owns, so the HBM column says nothing about occupancy. Whether a
process holds the NPU is the caller's rule, decided from the resident memory the table reports for it.
"""

from __future__ import annotations

import re
import subprocess  # nosec B404 - fixed read-only npu-smi invocations.
from dataclasses import dataclass

__all__ = [
    "NpuProcess",
    "NpuReading",
    "NpuSmiError",
    "npu_count",
    "npu_processes",
    "npu_readings",
    "read_npu_smi",
    "run_npu_smi",
]

_READ_TIMEOUT_SEC = 30.0
_BUS_ID = re.compile(r"^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_NO_PROCESS = re.compile(r"No running processes found in NPU\s+(\d+)")


class NpuSmiError(RuntimeError):
    """``npu-smi`` could not be run or its output could not be read."""


@dataclass(frozen=True)
class NpuProcess:
    """One process the ``npu-smi`` table names on an NPU; ``memory_mb`` is the resident memory it reports, or
    ``None`` when the cell does not read as a number."""

    pid: int
    name: str
    memory_mb: float | None = None

    def describe(self) -> str:
        return f"pid {self.pid} ({self.name})"


@dataclass(frozen=True)
class NpuReading:
    """One NPU as one ``npu-smi info`` table row reports it; a field the table did not print is ``None``."""

    index: int
    name: str
    health: str
    power_w: float | None = None
    temp_c: float | None = None
    aicore_pct: float | None = None
    hbm_used_mb: float | None = None
    hbm_total_mb: float | None = None


def run_npu_smi(args: tuple[str, ...] = (), *, timeout_sec: float = _READ_TIMEOUT_SEC) -> str:
    """Run ``npu-smi <args>`` and return its stdout.

    Raises:
        NpuSmiError: ``npu-smi`` is missing, exits non-zero, or times out.
    """
    try:
        done = subprocess.run(  # nosec B603 B607 - fixed argument vector, no shell.
            ["npu-smi", *args],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NpuSmiError(f"npu-smi {' '.join(args)} could not run: {exc}") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip()[:200]
        raise NpuSmiError(f"npu-smi {' '.join(args)} exited {done.returncode}: {detail}")
    return done.stdout


def read_npu_smi(*, timeout_sec: float = _READ_TIMEOUT_SEC) -> str:
    """The ``npu-smi info`` table."""
    return run_npu_smi(("info",), timeout_sec=timeout_sec)


def _cells(line: str) -> list[str]:
    return list(line.strip().strip("|").split("|"))


def _numbers(text: str) -> list[float]:
    return [float(token) for token in _NUMBER.findall(text)]


def _reading(index: int, head: str, chip: str) -> NpuReading:
    """One NPU out of its card line and chip line."""
    head_cells, chip_cells = _cells(head), _cells(chip)
    name = (head_cells[0].split() or ["", ""])[1] if len(head_cells) >= 1 else ""
    health = head_cells[1].strip() if len(head_cells) >= 2 else ""
    head_numbers = _numbers(head_cells[2]) if len(head_cells) >= 3 else []
    chip_numbers = _numbers(chip_cells[2]) if len(chip_cells) >= 3 else []
    power = head_numbers[0] if len(head_numbers) >= 1 else None
    temp = head_numbers[1] if len(head_numbers) >= 2 else None
    aicore = chip_numbers[0] if len(chip_numbers) >= 1 else None
    hbm_used = chip_numbers[3] if len(chip_numbers) >= 5 else None
    hbm_total = chip_numbers[4] if len(chip_numbers) >= 5 else None
    return NpuReading(
        index=index,
        name=name,
        health=health,
        power_w=power,
        temp_c=temp,
        aicore_pct=aicore,
        hbm_used_mb=hbm_used,
        hbm_total_mb=hbm_total,
    )


def npu_readings(text: str | None = None) -> list[NpuReading]:
    """Every NPU the ``npu-smi info`` table (``text``, or a fresh read when ``None``) lists, in index order.

    Raises:
        NpuSmiError: When no card line could be parsed, which means the table is not the one this parser reads.
    """
    body = read_npu_smi() if text is None else text
    readings: list[NpuReading] = []
    head: tuple[int, str] | None = None
    for line in body.splitlines():
        if "Process id" in line:
            # The card table ended: the process table below it repeats the same two-int cell shape.
            break
        cells = _cells(line)
        if len(cells) < 3:
            continue
        first = cells[0].split()
        if len(first) == 2 and first[0].isdigit() and first[1] and not _BUS_ID.match(cells[1].strip()):
            if head is not None:
                readings.append(_reading(head[0], head[1], ""))
            head = (int(first[0]), line)
        elif len(first) == 1 and first[0].isdigit() and _BUS_ID.match(cells[1].strip()) and head is not None:
            readings.append(_reading(head[0], head[1], line))
            head = None
    if head is not None:
        readings.append(_reading(head[0], head[1], ""))
    if not readings:
        raise NpuSmiError("npu-smi info reported no NPU card row")
    return readings


def npu_count() -> int:
    """How many NPUs the host has, as ``npu-smi info`` lists them."""
    return len(npu_readings())


def npu_processes(text: str | None = None) -> dict[int, list[NpuProcess]]:
    """Each NPU the process table names a process on, mapped to what the row says about it.

    An NPU the table reports as idle is absent rather than mapped to an empty list, so the caller can tell "no process
    table row" from "an empty one".
    """
    body = read_npu_smi() if text is None else text
    holders: dict[int, list[str]] = {}
    in_table = False
    for line in body.splitlines():
        if "Process id" in line:
            in_table = True
            continue
        idle = _NO_PROCESS.search(line)
        if idle is not None:
            holders.pop(int(idle.group(1)), None)
            in_table = True
            continue
        if not in_table:
            continue
        cells = _cells(line)
        if len(cells) < 3:
            continue
        first = cells[0].split()
        if len(first) != 2 or not (first[0].isdigit() and first[1].isdigit()):
            continue
        pid_cells = cells[1].split()
        if not pid_cells or not pid_cells[0].isdigit():
            continue
        name = cells[2].strip() or "unnamed"
        memory = _numbers(cells[3]) if len(cells) >= 4 else []
        holders.setdefault(int(first[0]), []).append(NpuProcess(int(pid_cells[0]), name, memory[0] if memory else None))
    return holders
