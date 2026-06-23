# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Collapsible PDF-style sidebar TOC and web-export PDF helpers for ATBD drafts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

ATBD_ROOT = "science-guide/atbd-l2-agb"
# Web export built by `make atbd-pdf` → docs/_static/pdf/<stem>.pdf
ATBD_PDF_FILENAME = "atbd-l2-agb.pdf"
ATBD_PDF_BUILD_CMD = "make atbd-pdf"
ATBD_CHAPTERS = (
    f"{ATBD_ROOT}/01-introduction",
    f"{ATBD_ROOT}/02-bps-agb-overview",
    f"{ATBD_ROOT}/03-ground-cancellation",
    f"{ATBD_ROOT}/04-agb-estimation",
    f"{ATBD_ROOT}/05-appendix",
    f"{ATBD_ROOT}/references",
)


def _section_tree(section: nodes.section, *, max_depth: int = 1) -> list[dict[str, Any]]:
    """Build section entries up to max_depth (1 = x.x only, no x.x.x)."""
    items: list[dict[str, Any]] = []
    for child in section.children:
        if not isinstance(child, nodes.section):
            continue
        title_node = child.next_node(nodes.title)
        title = title_node.astext() if title_node else ""
        anchor = child["ids"][0] if child.get("ids") else ""
        children: list[dict[str, Any]] = []
        if max_depth > 1:
            children = _section_tree(child, max_depth=max_depth - 1)
        items.append(
            {
                "title": title,
                "anchor": anchor,
                "children": children,
            }
        )
    return items


def _chapter_sections(doctree: nodes.document) -> list[dict[str, Any]]:
    """Return top-level subsections (1.1, 2.1, 3.1, …) for a chapter doctree."""
    for child in doctree.children:
        if not isinstance(child, nodes.section):
            continue
        return _section_tree(child)
    return []


def _build_atbd_sidebar_toc(env: Any) -> list[dict[str, Any]]:
    toc: list[dict[str, Any]] = []
    for docname in ATBD_CHAPTERS:
        if docname not in env.titles:
            continue
        doctree = env.get_doctree(docname)
        toc.append(
            {
                "docname": docname,
                "title": env.titles[docname].astext(),
                "sections": _chapter_sections(doctree),
            }
        )
    return toc


def _atbd_pdf_path(app: Any) -> Path:
    return Path(app.srcdir) / "_static" / "pdf" / ATBD_PDF_FILENAME


class atbd_pdf_export(nodes.General, nodes.Element):
    """Placeholder for the HTML-only PDF download block on the References page."""


class AtbdPdfExportDirective(SphinxDirective):
    """Render a download link when the web-export PDF has been built."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        return [atbd_pdf_export()]


def _visit_atbd_pdf_export_html(self: Any, node: atbd_pdf_export) -> None:
    pdf_path = _atbd_pdf_path(self.builder.app)
    if pdf_path.is_file():
        url = self.builder.get_relative_uri(
            self.builder.current_docname, f"_static/pdf/{ATBD_PDF_FILENAME}"
        )
        self.body.append(
            '<div class="admonition tip">'
            '<p class="admonition-title">Download web export</p>'
            f'<p><a class="reference download internal" href="{url}" download>'
            f'<i class="fa-solid fa-file-pdf"></i> '
            "Download this ATBD as PDF</a> "
            "(generated from the same MyST source as this website).</p>"
            "<p>The "
            '<a class="reference external" '
            'href="https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_AGB_ATBD_v3_1_4.pdf">'
            "official archival PDF</a> on biomass-disc.info remains authoritative "
            "until Aresys and ESA approve this web conversion.</p>"
            "</div>"
        )
    else:
        self.body.append(
            '<div class="admonition note">'
            '<p class="admonition-title">PDF export</p>'
            f"<p>Build the web-export PDF from the <code>docs/</code> directory with "
            f"<code>{ATBD_PDF_BUILD_CMD}</code> (requires TeX Live or MacTeX). "
            f"The file is written to <code>_static/pdf/{ATBD_PDF_FILENAME}</code> "
            "and linked here once generated.</p>"
            "</div>"
        )
    raise nodes.SkipNode


def _visit_atbd_pdf_export_latex(_self: Any, _node: atbd_pdf_export) -> None:
    raise nodes.SkipNode


def _inject_atbd_sidebar_toc(
    app: Any,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: nodes.document,
) -> None:
    if not pagename.startswith(ATBD_ROOT):
        return
    cache_key = "_atbd_sidebar_toc_cache"
    if not hasattr(app.env, cache_key):
        setattr(app.env, cache_key, _build_atbd_sidebar_toc(app.env))
    context["atbd_sidebar_toc"] = getattr(app.env, cache_key)
    context["atbd_sidebar_current"] = pagename
    index_doc = f"{ATBD_ROOT}/index"
    context["atbd_index_docname"] = index_doc
    context["atbd_index_title"] = (
        app.env.titles[index_doc].astext()
        if index_doc in app.env.titles
        else "Above-Ground Biomass Product ATBD"
    )
    pdf_path = _atbd_pdf_path(app)
    if pdf_path.is_file():
        context["atbd_pdf_url"] = app.builder.get_relative_uri(
            pagename, f"_static/pdf/{ATBD_PDF_FILENAME}"
        )
    else:
        context["atbd_pdf_url"] = None


def _silence_sidebar_wildcard_warnings(_app: Any, _config: Any) -> None:
    """ATBD pages intentionally match both ** and science-guide/atbd-l2-agb/*."""
    import logging

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not (
                "matches two" in msg
                and ("html_sidebars" in msg or "secondary_sidebar_items" in msg)
            )

    logging.getLogger("sphinx").addFilter(_Filter())


def setup(app: Any) -> dict[str, Any]:
    app.add_node(
        atbd_pdf_export,
        html=(_visit_atbd_pdf_export_html, None),
        latex=(_visit_atbd_pdf_export_latex, None),
        text=(_visit_atbd_pdf_export_latex, None),
        override=True,
    )
    app.add_directive("atbd-pdf-export", AtbdPdfExportDirective)
    app.connect("config-inited", _silence_sidebar_wildcard_warnings)
    app.connect("html-page-context", _inject_atbd_sidebar_toc)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
