# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The delegated Ascend kernel and quantization turns: what they require of the agent's artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import ascend_kernel_agent as kernel_agent
from hyperloom.orchestrator.actions.executors import ascend_quantize_agent as quantize_agent


def _skills(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    for name in ("cannbot-triton-ascend", "cannbot-ascend-c"):
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return root


def _candidates(tmp_path: Path) -> Path:
    path = tmp_path / "kernel_candidates.json"
    path.write_text(json.dumps({"hot_kernels": [{"kernel_name": "x", "duration_ms": 1.0}]}), encoding="utf-8")
    return path


# --- kernel layer -------------------------------------------------------------------------------------------------


def test_the_kernel_prompt_names_the_skills_and_the_patch_contract(tmp_path: Path) -> None:
    prompt = kernel_agent.build_prompt(
        candidates=_candidates(tmp_path),
        repo=tmp_path / "repo",
        out_dir=tmp_path / "out",
        skills=_skills(tmp_path),
    )

    assert "cannbot-triton-ascend" in prompt
    assert kernel_agent.PATCH_FILENAME in prompt
    assert kernel_agent.REPORT_FILENAME in prompt


def test_a_kernel_turn_returns_the_patch_it_wrote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr(kernel_agent, "agent_argv", lambda *_a, **_kw: ["true"])

    def run(argv: list[str], **_kw: object) -> int:
        (out / kernel_agent.PATCH_FILENAME).write_text("diff --git a/x b/x\n", encoding="utf-8")
        (out / kernel_agent.REPORT_FILENAME).write_text(
            json.dumps({"status": "ok", "kernel": "AddRmsNormBias", "files": ["vllm_ascend/ops/layernorm.py"]}),
            encoding="utf-8",
        )
        return 0

    result = kernel_agent.run_kernel_agent(_candidates(tmp_path), tmp_path / "repo", out, skills=None, run_agent=run)

    assert result["status"] == "ok"
    assert result["patch_path"].endswith(kernel_agent.PATCH_FILENAME)
    assert result["files"] == ["vllm_ascend/ops/layernorm.py"]


def test_an_ok_kernel_turn_without_a_patch_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr(kernel_agent, "agent_argv", lambda *_a, **_kw: ["true"])

    def run(argv: list[str], **_kw: object) -> int:
        (out / kernel_agent.REPORT_FILENAME).write_text(
            json.dumps({"status": "ok", "files": ["x.py"]}), encoding="utf-8"
        )
        return 0

    with pytest.raises(kernel_agent.KernelAgentError):
        kernel_agent.run_kernel_agent(_candidates(tmp_path), tmp_path / "repo", out, skills=None, run_agent=run)


def test_a_skipped_kernel_turn_needs_no_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr(kernel_agent, "agent_argv", lambda *_a, **_kw: ["true"])

    def run(argv: list[str], **_kw: object) -> int:
        (out / kernel_agent.REPORT_FILENAME).write_text(
            json.dumps({"status": "skipped", "approach": "no skill covers this shape"}), encoding="utf-8"
        )
        return 0

    result = kernel_agent.run_kernel_agent(_candidates(tmp_path), tmp_path / "repo", out, skills=None, run_agent=run)

    assert result["status"] == "skipped"
    assert result["patch_path"] == ""


# --- quantization layer -------------------------------------------------------------------------------------------


def test_the_quantize_prompt_names_the_checkpoint_contract(tmp_path: Path) -> None:
    prompt = quantize_agent.build_prompt(model=tmp_path / "model", out_dir=tmp_path / "out", skills=_skills(tmp_path))

    assert "cannbot-triton-ascend" in prompt
    assert quantize_agent.CHECKPOINT_DIR in prompt
    assert "W8A8" in prompt


def test_a_quantize_turn_returns_a_serveable_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    out = tmp_path / "out"
    monkeypatch.setattr(quantize_agent, "agent_argv", lambda *_a, **_kw: ["true"])

    def run(argv: list[str], **_kw: object) -> int:
        checkpoint = out / quantize_agent.CHECKPOINT_DIR
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / quantize_agent.QUANT_DESCRIPTION).write_text("{}", encoding="utf-8")
        (out / quantize_agent.REPORT_FILENAME).write_text(
            json.dumps({"status": "ok", "checkpoint_path": str(checkpoint), "method": "w8a8 per-tensor"}), "utf-8"
        )
        return 0

    result = quantize_agent.run_quantize_agent(model, out, skills=None, run_agent=run)

    assert result["status"] == "ok"
    assert Path(result["checkpoint_path"]).name == quantize_agent.CHECKPOINT_DIR


def test_an_ok_quantize_turn_without_a_checkpoint_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    out = tmp_path / "out"
    monkeypatch.setattr(quantize_agent, "agent_argv", lambda *_a, **_kw: ["true"])

    def run(argv: list[str], **_kw: object) -> int:
        (out / quantize_agent.REPORT_FILENAME).write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        return 0

    with pytest.raises(quantize_agent.QuantizeAgentError):
        quantize_agent.run_quantize_agent(model, out, skills=None, run_agent=run)


# --- skill catalogs -----------------------------------------------------------------------------------------------


def test_the_catalog_reads_a_curated_root_of_symlinked_skill_dirs(tmp_path: Path) -> None:
    """An operator points at a flat directory of links into a vendor tree; the catalog must see through the links."""
    vendor = tmp_path / "vendor" / "skills" / "triton-npu-optimize"
    vendor.mkdir(parents=True)
    (vendor / "SKILL.md").write_text("# s\n", encoding="utf-8")
    curated = tmp_path / "curated"
    curated.mkdir()
    (curated / "triton-npu-optimize").symlink_to(vendor)

    assert kernel_agent._skill_catalog(curated) == ["triton-npu-optimize"]


def test_the_catalog_reads_a_nested_vendor_tree(tmp_path: Path) -> None:
    """A CANNBot checkout nests skills under plugins/<plugin>/skills/<name>; pointing at the checkout must work."""
    skill = tmp_path / "cannbot" / "plugins-community" / "triton-optimizer" / "skills" / "triton-npu-optimize"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# s\n", encoding="utf-8")

    catalog = kernel_agent._skill_catalog(tmp_path / "cannbot")

    assert catalog == ["plugins-community/triton-optimizer/skills/triton-npu-optimize"]


def test_an_ok_turn_whose_checkpoint_is_not_quantized_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint with a config but no quantized tensors is a copy, not a quantization."""
    model = tmp_path / "model"
    model.mkdir()
    out = tmp_path / "out"
    monkeypatch.setattr(quantize_agent, "agent_argv", lambda *_a, **_kw: ["true"])

    def run(argv: list[str], **_kw: object) -> int:
        checkpoint = out / quantize_agent.CHECKPOINT_DIR
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (out / quantize_agent.REPORT_FILENAME).write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        return 0

    with pytest.raises(quantize_agent.QuantizeAgentError):
        quantize_agent.run_quantize_agent(model, out, skills=None, run_agent=run)
