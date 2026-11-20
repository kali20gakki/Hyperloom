# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The quantization prelude on Ascend: it runs the msModelSlim turn and fails closed."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import ascend_quantize_agent as quantize_agent
from hyperloom.orchestrator.phases import quantization_request_handlers as handlers


def test_an_ascend_turn_returns_its_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(handlers, "is_ascend", lambda: True, raising=False)
    calls: list[Path] = []

    def fake_run(model: Path, out: Path, **_kw: object) -> dict[str, str]:
        calls.append(out)
        checkpoint = out / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        return {"status": "ok", "checkpoint_path": str(checkpoint), "method": "w8a8"}

    monkeypatch.setattr(quantize_agent, "run_quantize_agent", fake_run)
    monkeypatch.setattr(quantize_agent, "skills_root", lambda: None)

    result = asyncio.run(
        handlers.run_quantization_prelude_async(prompt="w8a8", source_model=str(tmp_path / "model"), workspace=tmp_path)
    )

    assert result.endswith("checkpoint")
    assert calls and calls[0] == tmp_path / "ascend"


def test_a_skipped_ascend_turn_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(handlers, "is_ascend", lambda: True, raising=False)
    monkeypatch.setattr(
        quantize_agent,
        "run_quantize_agent",
        lambda *_a, **_kw: {"status": "skipped", "checkpoint_path": ""},
    )
    monkeypatch.setattr(quantize_agent, "skills_root", lambda: None)

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(
            handlers.run_quantization_prelude_async(
                prompt="w8a8", source_model=str(tmp_path / "model"), workspace=tmp_path
            )
        )

    assert excinfo.value.code == handlers.QUANTIZATION_FAILED_EXIT_CODE
