# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Science Guide hub sidebar: list of web ATBDs (not chapter outlines)."""

from __future__ import annotations

from typing import Any

SCIENCE_GUIDE_HUB = "science-guide/index"

# Top-level web ATBD conversions shown on the Science Guide hub sidebar.
SCIENCE_GUIDE_ATBDS: tuple[dict[str, str], ...] = (
    {
        "title": "L2 AGB ATBD",
        "docname": "science-guide/atbd-l2-agb/index",
        "reference": "BIO-BPS-AGB-ATBD-ARE-024912",
    },
)


def _inject_science_guide_nav(
    app: Any,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:
    if pagename != SCIENCE_GUIDE_HUB:
        return
    context["science_guide_atbds"] = SCIENCE_GUIDE_ATBDS


def setup(app: Any) -> dict[str, Any]:
    app.connect("html-page-context", _inject_science_guide_nav)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
