# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a warm-reuse baseline's accuracy is read from: the sibling warmup round that ran the eval."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.actions.executors import _accuracy_gate


def _warm_reuse_round(root: Path, accuracy: float) -> Path:
    """A round whose measure half ran no eval, and whose warmup half did."""
    (root / "measure_round").mkdir(parents=True)
    eval_dir = root / "warmup_round" / "suite_run_0" / "eval" / "__models__Qwen3-8B"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results_2026.json").write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": accuracy,
                        "alias": "gsm8k",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return root / "measure_round"


def test_a_measure_round_reads_the_eval_its_sibling_warmup_round_ran(tmp_path: Path) -> None:
    measure = _warm_reuse_round(tmp_path / "round", 0.90625)

    parsed = _accuracy_gate.parse_eval_results(measure)

    assert parsed["accuracy"] == 0.90625
    assert "warmup_round" in parsed["source_file"]


def test_a_measure_round_with_its_own_eval_still_wins(tmp_path: Path) -> None:
    measure = _warm_reuse_round(tmp_path / "round", 0.5)
    eval_dir = measure / "suite_run_0" / "eval" / "__models__Qwen3-8B"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results_2026.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.9375, "alias": "gsm8k"}}}),
        encoding="utf-8",
    )

    parsed = _accuracy_gate.parse_eval_results(measure)

    assert parsed["accuracy"] == 0.9375
    assert "warmup_round" not in parsed["source_file"]


def test_a_variant_round_reads_the_eval_its_sibling_warmup_round_ran(tmp_path: Path) -> None:
    """An explore variant nests one level deeper: v00_<name>/variant_<name>/ beside its warmup_round."""
    root = tmp_path / "v00_variant"
    measure = root / "variant_00_variant" / "suite_run_0"
    measure.mkdir(parents=True)
    eval_dir = root / "warmup_round" / "variant_00_variant" / "suite_run_0" / "eval" / "__models__Qwen3-8B"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results_2026.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.9219, "alias": "gsm8k"}}}),
        encoding="utf-8",
    )

    parsed = _accuracy_gate.parse_eval_results(measure)

    assert parsed["accuracy"] == 0.9219
