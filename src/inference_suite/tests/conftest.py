# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""InferenceSuite's tests describe one host: the AMD GPU host they were written against.

``hyperloom_common.device_backend`` reads the host, and the tests run on whatever host CI uses. Pinning the backend
here keeps the AMD assertions (amd-smi card order, ROCr masks, holder VRAM) about the backend they state, and leaves
the NPU ones (``test_ascend_npu_gate.py``) to select Ascend themselves.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _amd_device_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_DEVICE_BACKEND", "amd")
