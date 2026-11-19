# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent-driven quantization on Ascend: msModelSlim through its packaged skills.

The optimizer's quantize step normally rewrites the model with a fixed recipe. On Ascend the recipe tool is
msModelSlim and its workflow is packaged as a skill tree, so the step is delegated to a headless agent that runs the
tool and writes the checkpoint and a report the session can bind to a round.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from hyperloom.orchestrator.trace_analysis.msagent_analysis import _run_agent, agent_argv

log = logging.getLogger(__name__)

#: The msModelSlim skill tree (the packaged ``agfsm`` skills): a directory of skill directories, each with a ``SKILL.md``.
SKILLS_ROOT_ENV = "HYPERLOOM_AGFSM_SKILLS"
#: The exported checkpoint directory the agent writes; a round then serves this path instead of the source model.
CHECKPOINT_DIR = "checkpoint"
REPORT_FILENAME = "quantize_report.json"
REPORT_SCHEMA = "hyperloom-ascend-quantize/1"
DEFAULT_TIMEOUT_SEC = 7200.0


class QuantizeAgentError(RuntimeError):
    """The delegated quantization turn produced no usable checkpoint."""


def skills_root() -> Path | None:
    """The msModelSlim skills directory named by the environment, or ``None`` when it is unset."""
    raw = (os.environ.get(SKILLS_ROOT_ENV) or "").strip()
    return Path(raw) if raw else None


def _skill_catalog(root: Path) -> list[str]:
    """The skill directories under ``root``, named relative to it; a skills tree nests them at any depth."""
    if not root.is_dir():
        return []
    # A curated root is a set of symlinks to skill directories, which rglob does not descend into; a vendor tree
    # nests them one or more levels down, which the glob does not reach. Try the flat layout first.
    flat = [str(marker.parent.relative_to(root)) for marker in root.glob("*/SKILL.md")]
    nested = [str(marker.parent.relative_to(root)) for marker in root.rglob("SKILL.md")]
    return sorted(set(flat) | set(nested))


def prompt_addendum(*, skills: Path | None = None) -> str:
    """The Ascend instruction a quantization turn needs: msModelSlim through its skill tree, exported for vLLM.

    The prelude folds this into the operator's ``--quantize`` text, so it stays a body of instructions rather than a
    full prompt: the workspace and the export contract belong to the prelude.
    """
    root = skills if skills is not None else skills_root()
    catalog = _skill_catalog(root) if root else []
    where = (
        f"Read the `SKILL.md` under {root} first (skills: {', '.join(catalog)}) and follow its workflow."
        if catalog
        else "Use the installed msModelSlim CLI directly."
    )
    return (
        "On this Ascend host, quantize with msModelSlim. "
        f"{where} "
        "Export a HuggingFace-format checkpoint that `vllm serve <dir>` loads on vLLM-Ascend (weights, config and "
        "tokenizer together), and keep the recipe you used in the report."
    )


def build_prompt(*, model: Path, out_dir: Path, skills: Path | None) -> str:
    """The one turn's prompt: the source model, the destination, the skills, and the exact output contract."""
    catalog = _skill_catalog(skills) if skills else []
    skills_block = (
        f"Skill directories (read the `SKILL.md` of each before running anything): {skills}\n"
        "Available skills: " + ", ".join(catalog)
        if catalog
        else (
            f"The msModelSlim skills root {skills!r} is set but holds no skill directories."
            if skills
            else "No msModelSlim skills root is configured; use the installed msModelSlim CLI directly."
        )
    )
    return f"""You are the quantization agent of an inference optimization session running on an Ascend NPU host.

Inputs
- Source model: {model}
- Output directory: {out_dir}
- Target: W8A8 (INT8 weights, INT8 activations), exported as a checkpoint the vLLM-Ascend server can load directly.

{skills_block}

Task
1. Follow the skill's workflow to quantize the source model with msModelSlim. Do not invent a recipe: use the one the
   skill prescribes for a Qwen-style dense model, and record it.
2. Export the checkpoint so that `vllm serve {out_dir / CHECKPOINT_DIR}` works: the export must carry the tokenizer
   and config files, not only the weights.
3. Write `{out_dir / REPORT_FILENAME}`:
   {{"schema": "{REPORT_SCHEMA}", "status": "ok"|"skipped", "checkpoint_path": "{out_dir / CHECKPOINT_DIR}",
     "method": "<recipe name and precision>", "source_model": "{model}",
     "notes": "<validation done, accuracy expectations, anything the session must know>"}}.

Constraints
- One checkpoint. If the model or the tool cannot be quantized here, write `"status": "skipped"` with the reason in
  `notes` and export nothing.
- Do not modify the source model directory.
"""


def validate_checkpoint(out_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Check the turn's output: an ``ok`` report names a checkpoint directory that holds a model config.

    Raises:
        QuantizeAgentError: When the report is malformed or an ``ok`` turn exported no usable checkpoint.
    """
    if not isinstance(report, dict):
        raise QuantizeAgentError("the report is not a JSON object")
    status = report.get("status")
    if status not in ("ok", "skipped"):
        raise QuantizeAgentError(f"the report status must be 'ok' or 'skipped', not {status!r}")
    if status == "ok":
        checkpoint = Path(str(report.get("checkpoint_path") or out_dir / CHECKPOINT_DIR))
        if not (checkpoint / "config.json").is_file():
            raise QuantizeAgentError(f"status ok but {checkpoint} holds no config.json")
        report["checkpoint_path"] = str(checkpoint)
    return report


def run_quantize_agent(
    model: Path,
    out_dir: Path,
    *,
    skills: Path | None = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    run_agent: Callable[..., int] | None = None,
) -> dict[str, Any]:
    """Run one agent turn that quantizes ``model`` and return its checkpoint and report.

    Raises:
        QuantizeAgentError: The turn failed, timed out, or exported something a round cannot serve.
    """
    model, out_dir = Path(model), Path(out_dir)
    if not model.is_dir():
        raise QuantizeAgentError(f"source model not found: {model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(model=model, out_dir=out_dir, skills=skills)
    (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    try:
        argv = agent_argv(prompt, out_dir, timeout_sec=timeout_sec, mcp_command=None)
    except Exception as exc:
        raise QuantizeAgentError(str(exc)) from exc
    exit_code = (run_agent or _run_agent)(argv, timeout_sec=timeout_sec, log_path=out_dir / "agent.log")
    report_path = out_dir / REPORT_FILENAME
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuantizeAgentError(f"the turn wrote no readable report at {report_path}: {exc}") from exc
    validate_checkpoint(out_dir, report)
    return {
        "status": report.get("status"),
        "checkpoint_path": report.get("checkpoint_path", ""),
        "report_path": str(report_path),
        "method": report.get("method", ""),
        "agent_exit": exit_code,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one quantization turn from the command line, printing the result document."""
    parser = argparse.ArgumentParser(prog="hyperloom-ascend-quantize-agent")
    parser.add_argument("--model", type=Path, required=True, help="the source model directory")
    parser.add_argument("--out", type=Path, required=True, help="where the checkpoint and report are written")
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    args = parser.parse_args(argv)
    try:
        result = run_quantize_agent(args.model, args.out, skills=skills_root(), timeout_sec=args.timeout_sec)
    except QuantizeAgentError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
