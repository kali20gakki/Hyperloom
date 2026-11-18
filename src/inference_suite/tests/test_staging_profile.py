# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The two staging profiles: a subset-only cache serves the loads that drive the server, and refuses the AgentX ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_suite import staging
from inference_suite.client_env import ClientEnvError
from inference_suite.contracts.config import WorkloadKind
from inference_suite.contracts.report import FailedStep


def test_profile_for_maps_the_agentx_loads_to_the_full_staging() -> None:
    assert staging.profile_for(WorkloadKind.AGENTX) is staging.StagingProfile.AGENTX
    assert staging.profile_for(WorkloadKind.MLPERF) is staging.StagingProfile.AGENTX
    for kind in (WorkloadKind.FIXED, WorkloadKind.NONE, WorkloadKind.SCRIPTABLE):
        assert staging.profile_for(kind) is staging.StagingProfile.INFERENCEX


def _subset_files(partial: Path) -> None:
    """The two files the subset profile requires, as the pinned commit's extraction would lay them out."""
    for relative in (staging.CLIENT_FILE, staging.GSM8K_TASK):
        path = partial / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")


def _stage_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cache staged for the subset profile, from a source that need not exist."""
    monkeypatch.setattr(staging, "_extract_pinned_subset", lambda source, pin, partial: _subset_files(partial))
    cache = tmp_path / "tools"
    cache.mkdir()
    staging.stage(tmp_path / "source", cache, profile=staging.StagingProfile.INFERENCEX)
    return cache


def test_the_subset_profile_stages_a_cache_without_the_client_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _stage_subset(tmp_path, monkeypatch)

    digest = staging.manifest_sha256(cache)

    assert digest is not None
    assert staging.verify(cache, digest, profile=staging.StagingProfile.INFERENCEX).missing == ()


def test_the_subset_profile_never_fetches_the_client_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_inputs: Path) -> None:
        raise AssertionError("the inferencex profile must not fetch the AgentX client build")

    monkeypatch.setattr(staging.client_env, "fetch_inputs", refuse)

    _stage_subset(tmp_path, monkeypatch)


def test_the_subset_cache_refuses_an_agentx_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _stage_subset(tmp_path, monkeypatch)
    digest = staging.manifest_sha256(cache)

    assert digest is not None
    report = staging.verify(cache, digest, profile=staging.StagingProfile.AGENTX)

    assert report.failed_step is FailedStep.STAGING_MISSING
    assert staging.AGENTX_PYTHON in report.detail


def test_the_full_profile_still_builds_the_client_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_inputs: Path) -> None:
        raise ClientEnvError("no client build on this host")

    monkeypatch.setattr(staging, "_extract_pinned_subset", lambda source, pin, partial: _subset_files(partial))
    monkeypatch.setattr(staging.client_env, "fetch_inputs", broken)
    cache = tmp_path / "tools"
    cache.mkdir()

    with pytest.raises(staging.StagingError, match="no client build"):
        staging.stage(tmp_path / "source", cache, profile=staging.StagingProfile.AGENTX)
