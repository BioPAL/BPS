# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Science Guide hub sidebar: list of web ATBDs (not chapter outlines)."""

from __future__ import annotations

from typing import Any

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

SCIENCE_GUIDE_HUB = "science-guide/index"

# Top-level web ATBD conversions shown on the Science Guide hub sidebar.
SCIENCE_GUIDE_ATBDS: tuple[dict[str, str], ...] = (
    {
        "title": "L2 AGB ATBD",
        "docname": "science-guide/atbd-l2-agb/index",
        "reference": "BIO-BPS-AGB-ATBD-ARE-024912",
    },
)


class atbd_logo_banner(nodes.General, nodes.Element):
    """Placeholder for the ESA / Aresys partner logo banner."""


class AtbdLogoBannerDirective(SphinxDirective):
    """ESA / Aresys partner logo banner with paths resolved per page."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        return [atbd_logo_banner()]


def _static_uri(docname: str, asset: str) -> str:
    """Relative path from a built HTML page to a file under _static/."""
    depth = len(docname.split("/")) - 1
    return f"{'../' * depth}{asset}"


def _visit_atbd_logo_banner_html(self: Any, node: atbd_logo_banner) -> None:
    docname = self.builder.current_docname

    def static(path: str) -> str:
        return _static_uri(docname, path)

    self.body.append(
        '<div class="atbd-logo-banner docutils">'
        '<div class="atbd-logo-banner__inner">'
        '<a class="atbd-logo-banner__link atbd-logo-banner__link--left" '
        'href="https://www.aresys.it/" rel="noopener noreferrer">'
        f'<img class="atbd-logo-banner__img atbd-logo-banner__img--light only-light" '
        f'src="{static("_static/logos/Aresys_light.png")}" alt="Aresys" />'
        f'<img class="atbd-logo-banner__img atbd-logo-banner__img--dark only-dark" '
        f'src="{static("_static/logos/Aresys_dark.svg")}" alt="Aresys" />'
        "</a>"
        '<a class="atbd-logo-banner__link atbd-logo-banner__link--right" '
        'href="https://www.esa.int/" rel="noopener noreferrer">'
        f'<img class="atbd-logo-banner__img atbd-logo-banner__img--light only-light" '
        f'src="{static("_static/logos/ESA_light.png")}" alt="European Space Agency" />'
        f'<img class="atbd-logo-banner__img atbd-logo-banner__img--dark only-dark" '
        f'src="{static("_static/logos/ESA_dark.png")}" alt="European Space Agency" />'
        "</a>"
        "</div>"
        "</div>"
    )
    raise nodes.SkipNode


def _skip_atbd_logo_banner(_self: Any, _node: atbd_logo_banner) -> None:
    raise nodes.SkipNode


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
    app.add_node(
        atbd_logo_banner,
        html=(_visit_atbd_logo_banner_html, None),
        latex=(_skip_atbd_logo_banner, None),
        text=(_skip_atbd_logo_banner, None),
        override=True,
    )
    app.add_directive("atbd-logo-banner", AtbdLogoBannerDirective)
    app.connect("html-page-context", _inject_science_guide_nav)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
