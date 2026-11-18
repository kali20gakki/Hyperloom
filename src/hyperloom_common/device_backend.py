# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which accelerator family this host's tooling must read: AMD GPUs (``rocm-smi``/``amd-smi``) or Ascend NPUs
(``npu-smi``).

The backend is a property of the host a process runs on, not of a session, so it is read from the host once per call
rather than carried through every layer. ``HYPERLOOM_DEVICE_BACKEND`` names it explicitly; otherwise a host whose
NPU tooling answers and whose GPU tooling does not is Ascend, and everything else keeps the AMD behaviour that was
the only one before this module existed.
"""

from __future__ import annotations

import os
import shutil

__all__ = ["AMD", "ASCEND", "DEVICE_BACKEND_ENV", "device_backend", "is_ascend"]

AMD = "amd"
ASCEND = "ascend"

#: Names the backend explicitly; an unknown value is a mistake the caller must see rather than a host to guess about.
DEVICE_BACKEND_ENV = "HYPERLOOM_DEVICE_BACKEND"


def _explicit() -> str | None:
    raw = (os.environ.get(DEVICE_BACKEND_ENV) or "").strip().lower()
    if not raw:
        return None
    if raw not in (AMD, ASCEND):
        raise ValueError(f"{DEVICE_BACKEND_ENV}={raw!r} is not {AMD!r} or {ASCEND!r}")
    return raw


def device_backend() -> str:
    """``"amd"`` or ``"ascend"`` for this host.

    Raises:
        ValueError: When ``HYPERLOOM_DEVICE_BACKEND`` names neither backend.
    """
    explicit = _explicit()
    if explicit is not None:
        return explicit
    if shutil.which("rocm-smi") or shutil.which("amd-smi"):
        return AMD
    if shutil.which("npu-smi"):
        return ASCEND
    return AMD


def is_ascend() -> bool:
    """Whether this host's accelerators are Ascend NPUs."""
    return device_backend() == ASCEND
