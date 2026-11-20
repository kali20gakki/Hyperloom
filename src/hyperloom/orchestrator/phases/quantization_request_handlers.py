"""Adapter: drive the quantization-agent from hyperloom.inference_optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

# Distinct from SESSION_BUSY_EXIT_CODE (3) so a launcher that retries a busy session does not retry a failed quantization.
QUANTIZATION_FAILED_EXIT_CODE = 4


async def _quantize_on_ascend(*, source_model: str, workspace: Path) -> str:
    """Run the msModelSlim turn instead of the vendor-neutral skill runner, which drives AMD's Quark and refuses to
    start without it (``quark_root_missing``)."""
    import asyncio

    from hyperloom.orchestrator.actions.executors.ascend_quantize_agent import (
        QuantizeAgentError,
        run_quantize_agent,
        skills_root,
    )

    out_dir = workspace / "ascend"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = await asyncio.to_thread(run_quantize_agent, Path(source_model), out_dir, skills=skills_root())
    except QuantizeAgentError as exc:
        print(f"Quantization: FAILED on Ascend: {exc}", file=sys.stderr)
        raise SystemExit(QUANTIZATION_FAILED_EXIT_CODE) from exc
    if result.get("status") != "ok" or not result.get("checkpoint_path"):
        # Fail closed: the loop must not silently optimize the un-quantized source model.
        print(f"Quantization: FAILED on Ascend: status={result.get('status')!r}", file=sys.stderr)
        raise SystemExit(QUANTIZATION_FAILED_EXIT_CODE)
    print(f"Quantization: success on Ascend -> {result['checkpoint_path']} ({result.get('method', '')})")
    return str(result["checkpoint_path"])


async def run_quantization_prelude_async(
    *,
    prompt: str,
    source_model: str,
    workspace: Path,
) -> str:
    """Quantize ``source_model`` per ``prompt``; return the exported dir."""
    workspace = Path(workspace)

    from hyperloom_common.device_backend import is_ascend

    if is_ascend():
        return await _quantize_on_ascend(source_model=source_model, workspace=workspace)

    # Import lazily so this module loads without the quantization runtime deps.
    from hyperloom.agents.quantization import QuantSkillRunOptions, quantize_via_prompt

    # Fold the source model + export dir into the prompt so the user's --quantize text can be just the scheme. The
    # export dir is relative: every run gets its own directory under the workspace and must export inside it.
    effective_prompt = (
        f"Quantize the model at {source_model}. "
        f"Export the HuggingFace-format quantized model to `quantized/` in your workspace. "
        f"Run the COMPLETE PTQ phase chain end-to-end in this one session "
        f"(intake -> plan -> manifest -> exec -> export -> validate -> eval); "
        f"do not stop early or hand back to a parent agent. "
        f"interactive=off: accept all CRITICAL STOP defaults. "
        f"{prompt}"
    )

    result = await quantize_via_prompt(
        effective_prompt,
        workspace=workspace,
        options=QuantSkillRunOptions(interactive=False),
    )

    final = result.assessment.final
    qdir = result.quantized_model_dir
    details = result.run_dir or workspace

    if result.status == "success":
        print(f"Quantization: success (final={final}, eval_gap={result.assessment.eval_gap}) -> {qdir}")
        return str(qdir)

    if result.status == "partial" and qdir is not None:
        print(
            f"Quantization: PARTIAL (final={final}); quantized model is loadable "
            f"so continuing, but audit/eval was incomplete. Review {details}.",
            file=sys.stderr,
        )
        return str(qdir)

    print(
        f"ERROR: quantization {result.status} (final={final}). Refusing to "
        f"optimize the un-quantized source model. See {details} for details.",
        file=sys.stderr,
    )
    raise SystemExit(QUANTIZATION_FAILED_EXIT_CODE)


__all__ = ["run_quantization_prelude_async"]
