# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ascend-specific preflight behaviour: the build probe's kind, and the no_proxy entries httpx cannot parse."""

from __future__ import annotations

import os

import pytest

from hyperloom_common.env import sanitize_no_proxy
from hyperloom.inference_optimizer.cli.framework_probe import _accelerator_kind


def test_the_build_probe_is_the_npu_one_on_an_ascend_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_DEVICE_BACKEND", "ascend")

    assert _accelerator_kind() == "NPU"


def test_the_build_probe_stays_the_rocm_one_on_an_amd_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_DEVICE_BACKEND", "amd")

    assert _accelerator_kind() == "ROCm"


def test_a_bracketed_no_proxy_entry_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "localhost,127.0.0.1,[::1]")

    assert sanitize_no_proxy() is True
    assert "localhost,127.0.0.1" == os.environ["NO_PROXY"]
    assert "localhost,127.0.0.1" == os.environ["no_proxy"]


def test_a_clean_no_proxy_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.delenv("no_proxy", raising=False)

    assert sanitize_no_proxy() is False


def test_a_no_proxy_of_only_bracketed_entries_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "[::1]")

    assert sanitize_no_proxy() is True
    assert "NO_PROXY" not in os.environ
    assert "no_proxy" not in os.environ
