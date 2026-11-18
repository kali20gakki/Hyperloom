# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The routes a trace analysis can take, shared by the handler that validates one and the tool that runs it."""

from __future__ import annotations

from enum import Enum


class AnalysisRoute(Enum):
    """How a trace is analysed.

    ``agent`` runs the TraceLens skill under a model and ``non-llm`` runs TraceLens alone; both read an AMD trace.
    ``msagent`` reads an Ascend ``torch_npu`` capture under a model through msprof-mcp and the msAgent profiler skills.
    """

    AGENT = "agent"
    NON_LLM = "non-llm"
    MSAGENT = "msagent"
