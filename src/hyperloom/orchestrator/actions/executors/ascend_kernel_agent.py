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
import subprocess
import shutil
import sys
import tempfile
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
    """The skill directories under ``root``, named relative to it; a skills tree nests them at any depth."""
    if not root.is_dir():
        return []
    # A curated root is a set of symlinks to skill directories, which rglob does not descend into; a vendor tree
    # nests them one or more levels down, which the glob does not reach. Try the flat layout first.
    flat = [str(marker.parent.relative_to(root)) for marker in root.glob("*/SKILL.md")]
    nested = [str(marker.parent.relative_to(root)) for marker in root.rglob("SKILL.md")]
    return sorted(set(flat) | set(nested))


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
- Open only the files the candidate and the skill name — do not survey the tree.
- Leave the tree clean: after `git -C {repo} diff > {out_dir / PATCH_FILENAME}`, revert your edits
  (`git -C {repo} checkout -- .`). The patch must apply to the tree as it stands — verify with
  `git -C {repo} apply --check {out_dir / PATCH_FILENAME}` before writing the report.
- Verify the code you changed still imports: `PYTHONPATH={repo} python -c "import <the module you edited>"` (from any
  directory). A patch that breaks the import is worse than no patch. Do not commit, do not stash, do not touch
  anything else.
"""


def patch_applies(repo: Path, patch: Path) -> tuple[bool, str]:
    """Whether ``patch`` applies cleanly to ``repo``; the reason comes back when it does not."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "apply", "--check", str(patch)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip().splitlines()[-3:][-1:] or "git apply --check failed"
    return True, ""


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


def import_gate(repo: Path, patch: Path, modules: Sequence[str], *, python: str | None = None) -> tuple[bool, str]:
    """Apply ``patch`` to a throwaway copy of ``repo`` and import the edited modules there.

    A patch the platform cannot import must be refused before it costs a server boot and a measurement.
    """
    interpreter = python or os.environ.get("HYPERLOOM_FRAMEWORK_PYTHON") or sys.executable
    with tempfile.TemporaryDirectory(prefix="hl-kernel-gate-") as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(repo, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
        applied = subprocess.run(
            ["patch", "-p1", "-d", str(copy), "-i", str(patch)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if applied.returncode != 0:
            return False, "the patch does not apply to a copy of the tree: " + (applied.stderr or "").strip()[-200:]
        for module in modules:
            try:
                proc = subprocess.run(
                    [interpreter, "-c", f"import {module}"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                    env={**os.environ, "PYTHONPATH": os.pathsep.join([str(copy), os.environ.get("PYTHONPATH", "")])},
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return False, f"could not run the import gate for {module}: {exc}"
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
                return False, f"import {module} failed after the patch: " + " | ".join(tail)
    return True, ""


def serve_gate(repo: Path, model: Path, *, port: int, timeout_sec: float = 900.0) -> tuple[bool, str]:
    """Boot a server from a throwaway copy of ``repo`` with the patch applied and wait for ``/health``.

    The import gate cannot see a kernel that only fails when a graph is captured; booting is the cheapest check that
    the platform still serves.
    """
    import socket
    import time
    import urllib.error
    import urllib.request

    interpreter = os.environ.get("HYPERLOOM_FRAMEWORK_PYTHON") or sys.executable
    with tempfile.TemporaryDirectory(prefix="hl-kernel-serve-") as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(repo, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(copy), os.environ.get("PYTHONPATH", "")]),
            "ASCEND_RT_VISIBLE_DEVICES": "0",
        }
        log_path = Path(tmp) / "serve.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(  # nosec B603 - argv assembled here, no shell.
                [
                    interpreter,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                    "--model",
                    str(model),
                    "--trust-remote-code",
                    "--max-model-len",
                    "8192",
                    "--port",
                    str(port),
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            try:
                deadline = time.time() + timeout_sec
                while time.time() < deadline and proc.poll() is None:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                            if response.status == 200:
                                return True, ""
                    except (urllib.error.URLError, OSError, socket.timeout):
                        time.sleep(5)
                tail = log_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-5:]
                return False, "the patched tree does not serve: " + " | ".join(tail)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()


def _edited_modules(files: Sequence[str]) -> list[str]:
    """The dotted module names of the ``.py`` files a patch touches."""
    modules = []
    for name in files:
        if not name.endswith(".py"):
            continue
        path = Path(name)
        parts = list(path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        modules.append(".".join(parts))
    return modules


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
    patch = out_dir / PATCH_FILENAME
    validate_patch(patch, report)
    applies, why = (True, "") if report.get("status") != "ok" else patch_applies(repo, patch)
    if not applies:
        raise KernelAgentError(f"the patch does not apply to {repo}: {why}")
    if report.get("status") == "ok":
        modules = _edited_modules([str(f) for f in report.get("files") or []])
        if modules:
            imports, why = import_gate(repo, patch, modules)
            if not imports:
                raise KernelAgentError(why)
    return {
        "status": report.get("status"),
        "patch_path": str(patch) if report.get("status") == "ok" else "",
        "report_path": str(report_path),
        "kernel": report.get("kernel", ""),
        "files": report.get("files") or [],
        "patch_applies": applies,
        "agent_exit": exit_code,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one kernel turn from the command line, printing the result document."""
    parser = argparse.ArgumentParser(prog="hyperloom-ascend-kernel-agent")
    parser.add_argument("--candidates", type=Path, required=True, help="the analysis' kernel_candidates.json")
    parser.add_argument("--repo", type=Path, required=True, help="the serving tree the agent may edit")
    parser.add_argument("--out", type=Path, required=True, help="where the patch and report are written")
    parser.add_argument("--timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--port", type=int, default=18070, help="the port the serve gate boots on")
    parser.add_argument(
        "--serve-model",
        type=Path,
        default=None,
        help="boot a server from the patched tree on this model and refuse the patch when it does not serve",
    )
    args = parser.parse_args(argv)
    try:
        result = run_kernel_agent(
            args.candidates, args.repo, args.out, skills=skills_root(), timeout_sec=args.timeout_sec
        )
    except KernelAgentError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1
    if args.serve_model is not None and result.get("status") == "ok":
        serves, why = serve_gate(args.repo, args.serve_model, port=args.port)
        if not serves:
            print(json.dumps({**result, "status": "failed", "error": why}))
            return 1
        result["serve_gate"] = True
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
