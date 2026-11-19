# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent-driven kernel work on Ascend: the capability comes from CANNBot skills, not from GEAK.

GEAK has no NPU path, so on an Ascend host the kernel layer is delegated to a headless agent that reads the analysis'
candidates and a CANNBot skill tree, edits the serving tree, and writes a unified diff. The diff is the artifact; the
existing integrate/measure machinery decides whether it stays.
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

#: The CANNBot skills root: a directory of skill directories, each with a ``SKILL.md``.
SKILLS_ROOT_ENV = "HYPERLOOM_CANNBOT_SKILLS"
#: The file the agent writes: a unified diff the integrate step can apply to the serving tree.
PATCH_FILENAME = "kernel_patch.diff"
REPORT_FILENAME = "kernel_agent_report.json"
REPORT_SCHEMA = "hyperloom-ascend-kernel-agent/1"
DEFAULT_TIMEOUT_SEC = 3600.0


class KernelAgentError(RuntimeError):
    """The delegated kernel turn could not produce a usable patch."""


def skills_root() -> Path | None:
    """The CANNBot skills directory named by the environment, or ``None`` when it is unset."""
    raw = (os.environ.get(SKILLS_ROOT_ENV) or "").strip()
    return Path(raw) if raw else None


def _skill_catalog(root: Path) -> list[str]:
    """The skill names under ``root`` that carry a ``SKILL.md``."""
    return sorted(p.parent.name for p in root.glob("*/SKILL.md")) if root.is_dir() else []


def build_prompt(*, candidates: Path, repo: Path, out_dir: Path, skills: Path | None) -> str:
    """The one turn's prompt: the candidate set, the tree it may edit, the skills, and the exact output contract."""
    catalog = _skill_catalog(skills) if skills else []
    skills_block = (
        f"Skill directories (read the `SKILL.md` of each before writing code): {skills}\n"
        "Available skills: " + ", ".join(catalog)
        if catalog
        else (
            f"The CANNBot skills root {skills!r} is set but holds no skill directories; work from the candidate "
            "document and the tree."
            if skills
            else "No CANNBot skills root is configured; work from the candidate document and the tree."
        )
    )
    return f"""You are the kernel agent of an inference optimization session running on an Ascend NPU host.

Inputs
- Kernel candidates (JSON, written by the trace analysis): {candidates}
- Serving tree you may edit: {repo}
- Output directory: {out_dir}

{skills_block}

Task
1. Read the candidates and pick the single best one to implement (highest device-time share that a skill can
   address; prefer a fused/combined operator over a new tiling when both are listed).
2. Implement it in the serving tree: replace or add the operator in the tree's own style, keeping the interface and
   numerics. Do not bump dependencies, do not reformat unrelated files, do not touch tests.
3. Write your change as a unified diff at `{out_dir / PATCH_FILENAME}` (`git -C {repo} diff` is the expected shape).
4. Write `{out_dir / REPORT_FILENAME}`: {{"schema": "{REPORT_SCHEMA}", "status": "ok"|"skipped",
   "kernel": "<name>", "files": ["<repo-relative path>", ...], "approach": "<one paragraph>",
   "expected_effect": "<why this should move time>"}}.

Constraints
- One operator, one patch. If nothing in the candidates is implementable with the available skills, write
  `"status": "skipped"` and say why in `approach`; write no patch.
- The patch must apply to the tree as it stands; do not commit, do not stash, do not revert anything else.
"""


def validate_patch(patch: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Check the turn's pair: an ``ok`` report names files and a non-empty diff exists alongside it.

    Raises:
        KernelAgentError: When the report is malformed or an ``ok`` turn left no diff.
    """
    if not isinstance(report, dict):
        raise KernelAgentError("the report is not a JSON object")
    status = report.get("status")
    if status not in ("ok", "skipped"):
        raise KernelAgentError(f"the report status must be 'ok' or 'skipped', not {status!r}")
    if status == "ok":
        if not patch.is_file() or patch.stat().st_size == 0:
            raise KernelAgentError(f"status ok but {patch} is missing or empty")
        files = report.get("files")
        if not isinstance(files, list) or not files:
            raise KernelAgentError("status ok but the report names no files")
    return report


def run_kernel_agent(
    candidates: Path,
    repo: Path,
    out_dir: Path,
    *,
    skills: Path | None = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    run_agent: Callable[..., int] | None = None,
) -> dict[str, Any]:
    """Run one agent turn over the candidates and return its patch and report.

    Raises:
        KernelAgentError: The turn failed, timed out, or wrote something the integrate step cannot use.
    """
    candidates, repo, out_dir = Path(candidates), Path(repo), Path(out_dir)
    if not candidates.is_file():
        raise KernelAgentError(f"candidate document not found: {candidates}")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(candidates=candidates, repo=repo, out_dir=out_dir, skills=skills)
    (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    try:
        argv = agent_argv(prompt, out_dir, timeout_sec=timeout_sec, mcp_command=None)
    except Exception as exc:
        raise KernelAgentError(str(exc)) from exc
    exit_code = (run_agent or _run_agent)(argv, timeout_sec=timeout_sec, log_path=out_dir / "agent.log")
    report_path = out_dir / REPORT_FILENAME
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KernelAgentError(f"the turn wrote no readable report at {report_path}: {exc}") from exc
    validate_patch(out_dir / PATCH_FILENAME, report)
    return {
        "status": report.get("status"),
        "patch_path": str(out_dir / PATCH_FILENAME) if report.get("status") == "ok" else "",
        "report_path": str(report_path),
        "kernel": report.get("kernel", ""),
        "files": report.get("files") or [],
        "agent_exit": exit_code,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one kernel turn from the command line, printing the result document."""
    parser = argparse.ArgumentParser(prog="hyperloom-ascend-kernel-agent")
    parser.add_argument("--candidates", type=Path, required=True, help="the analysis' kernel_candidates.json")
    parser.add_argument("--repo", type=Path, required=True, help="the serving tree the agent may edit")
    parser.add_argument("--out", type=Path, required=True, help="where the patch and report are written")
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    args = parser.parse_args(argv)
    try:
        result = run_kernel_agent(
            args.candidates, args.repo, args.out, skills=skills_root(), timeout_sec=args.timeout_sec
        )
    except KernelAgentError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
