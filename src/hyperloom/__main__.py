# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom <command>``."""

from hyperloom_common.env import sanitize_no_proxy

from hyperloom.cli import main

# Before any HTTP client is built: a bracketed no_proxy entry breaks httpx's URL parse.
sanitize_no_proxy()
raise SystemExit(main())
