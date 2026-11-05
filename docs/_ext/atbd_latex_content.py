# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Make the converted ATBD/PFD pages carry their content into LaTeX/PDF builds.

The web conversions write figures and tables as ``sphinx-design`` divs whose
captions are raw HTML (``<p class="atbd-figure-caption">``), which every
non-HTML builder drops. For non-HTML builders only, this post-transform:

* turns ``:::{div} atbd-mermaid-figure`` into a real ``figure`` node (images,
  Mermaid diagrams, caption ``Figure N: ...`` with the source number N, label
  kept) and drops the "Original figure" dropdown when a Mermaid diagram is shown;
* turns ``:::{div} atbd-table`` into a captioned ``longtable``
  (``Table N: ...``, source number N, label kept);
* gives tables explicit column widths so they wrap inside the margins, and a
  body to header-only tables (the LaTeX writer requires one);
* writes ``\\tag{N}`` equations as ``equation`` environments (``\\tag`` is not
  allowed inside the ``split`` Sphinx wraps them in) and labels their anchors;
* makes MathJax-tolerant math valid TeX (``^{'}`` primes), breaks long inline
  literals and keeps figures within the page height;
* turns ``<span id="rd8"></span>`` anchors into LaTeX labels and drops the other
  inline raw HTML.

HTML builds are not touched.
"""

from __future__ import annotations

import html
import re
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.texescape import escape as tex_escape

FIGURE_CLASS = "atbd-mermaid-figure"
TABLE_CLASS = "atbd-table"

_CAPTION_RE = re.compile(
    r'<span class="caption-number">\s*(?P<kind>[A-Za-z.]+)\s*(?P<num>[^<]*?)\s*</span>'
    r'\s*<span class="caption-text">(?P<text>.*)</span>\s*</p>\s*$',
    re.DOTALL,
)
_MATH_SPAN_RE = re.compile(
    r'<span class="math[^"]*">\\\((?P<tex>.*?)\\\)</span>', re.DOTALL
)
_SPAN_ID_RE = re.compile(r'^<span\s+id="(?P<id>[^"]+)"\s*>(?:</span>)?$')
_TAG_RE = re.compile(r"\\tag\*?\{(?P<num>[^{}]*)\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# ``x^{'}`` is fine for MathJax but TeX wants ``\prime``; and ``{{\widehat{s}}^{'}}^2``
# is a "Double superscript" for TeX, which unwraps a group holding a single
# accent atom: an empty group after the prime keeps the outer group intact.
_PRIME_SUP_RE = re.compile(r"\^\{('+)\}")
_PRIME_GROUP_RE = re.compile(r"(\^\{(?:\\prime)+\})\}(?=\s*[\^_])")
# Operators the sources use without a definition (amsmath has no \argmin).
_ARG_OP_RE = re.compile(r"\\arg(min|max)(?![A-Za-z])")


def _tex_math(text: str) -> str:
    text = _PRIME_SUP_RE.sub(lambda m: "^{%s}" % (r"\prime" * len(m.group(1))), text)
    text = _PRIME_GROUP_RE.sub(r"\1{}}", text)
    return _ARG_OP_RE.sub(r"\\operatorname*{arg\\,\1}", text)


# Break opportunities in long inline literals (file name patterns, XML types,
# enumeration values): after _ . / : > , and inside camelCase words.
_LITERAL_BREAK_RE = re.compile(r"(?<=[_./:>,])(?=.)|(?<=[a-z])(?=[A-Z])|(?=<)")
_LONG_LITERAL = 20


def _breakable_literal(text: str, engine: str) -> str | None:
    """LaTeX for an inline literal with break points, or None if it is short."""
    if max((len(w) for w in text.split()), default=0) <= _LONG_LITERAL:
        return None
    parts = [p for p in _LITERAL_BREAK_RE.split(text) if p]
    body = r"\allowbreak{}".join(tex_escape(p, engine) for p in parts)
    return r"\sphinxcode{\sphinxupquote{%s}}" % body


def _latex_label(docname: str, ident: str) -> str:
    return r"\phantomsection\label{\detokenize{%s:%s}}" % (docname, ident)


def _caption_inlines(fragment: str) -> list[nodes.Node]:
    """Inline nodes (text and math) for a raw-HTML caption fragment."""
    out: list[nodes.Node] = []
    pos = 0
    for match in _MATH_SPAN_RE.finditer(fragment):
        before = html.unescape(_HTML_TAG_RE.sub("", fragment[pos:match.start()]))
        if before:
            out.append(nodes.Text(before))
        tex = _tex_math(html.unescape(match.group("tex")))
        out.append(nodes.math(tex, tex))
        pos = match.end()
    rest = html.unescape(_HTML_TAG_RE.sub("", fragment[pos:]))
    if rest:
        out.append(nodes.Text(rest))
    return out


def _parse_caption(raw: nodes.raw) -> tuple[str, list[nodes.Node]] | None:
    """(source number, caption inlines) of an ``atbd-*-caption`` raw block."""
    match = _CAPTION_RE.search(raw.astext())
    if match is None:
        return None
    text = match.group("text")
    text = re.sub(r"^\s*(?:—|–|-|:)\s*", "", text)
    return match.group("num").strip(), _caption_inlines(text)


def _is_caption(node: nodes.Node, css: str) -> bool:
    return isinstance(node, nodes.raw) and f'class="{css}"' in node.astext()


def _header_only(table: nodes.table) -> None:
    """Turn the head of a header-only table into its body.

    The LaTeX writer needs a ``tbody`` (``depart_table`` fails without one);
    a GFM table without data rows has only a ``thead``.
    """
    for tgroup in table.findall(nodes.tgroup):
        if any(isinstance(c, nodes.tbody) for c in tgroup.children):
            continue
        thead = next((c for c in tgroup.children if isinstance(c, nodes.thead)), None)
        if thead is None:
            continue
        tbody = nodes.tbody("", *thead.children)
        tgroup.replace(thead, tbody)


def _column_widths(table: nodes.table) -> None:
    """Replace ``colwidths-auto`` by widths proportional to the cell contents."""
    classes = table["classes"]
    if "colwidths-given" in classes:
        return
    for tgroup in table.findall(nodes.tgroup):
        colspecs = [c for c in tgroup.children if isinstance(c, nodes.colspec)]
        ncols = len(colspecs)
        if not ncols:
            continue
        lengths: list[list[int]] = [[] for _ in range(ncols)]
        words = [1] * ncols
        for row in tgroup.findall(nodes.row):
            col = 0
            for entry in row.children:
                if col >= ncols:
                    break
                text = " ".join(entry.astext().split())
                span = 1 + entry.get("morecols", 0)
                if span == 1:
                    lengths[col].append(len(text))
                    longest = max((len(w) for w in text.split()), default=1)
                    words[col] = max(words[col], min(longest, 28))
                col += span
        minimum = [w + 1.0 for w in words]
        desired = []
        for col in range(ncols):
            vals = lengths[col] or [1]
            mean = sum(vals) / len(vals)
            top = min(max(vals), 80)
            desired.append(max(minimum[col], 0.6 * top + 0.4 * mean, 3.0))
        # About 75 characters fit in the text width; each column also pays
        # its \tabcolsep padding. Keep every column wide enough for its
        # longest word, then share what is left in proportion to the content.
        budget = 75.0 - 2.0 * ncols
        if sum(desired) <= budget or sum(minimum) >= budget:
            scores = desired if sum(desired) <= budget else minimum
        else:
            spare = (budget - sum(minimum)) / (sum(desired) - sum(minimum))
            scores = [m + (d - m) * spare for m, d in zip(minimum, desired)]
        total = sum(scores)
        widths = [max(1, round(1000 * s / total)) for s in scores]
        for spec, width in zip(colspecs, widths):
            spec["colwidth"] = width
        if "colwidths-auto" in classes:
            classes.remove("colwidths-auto")
        classes.append("colwidths-given")
        return


def _walk(node: nodes.Node, docname: str):
    """Yield ``(node, docname)`` for the whole tree.

    LaTeX builds inline every chapter in an ``addnodes.start_of_file`` whose
    child does not point back to it (``children`` is assigned directly), so
    the document is found top-down rather than through ``node.parent``.
    """
    stack = [(node, docname)]
    while stack:
        current, doc = stack.pop()
        if isinstance(current, addnodes.start_of_file):
            doc = current["docname"]
        yield current, doc
        if isinstance(current, nodes.Element):
            stack.extend((child, doc) for child in reversed(current.children))


def _preceding_targets(node: nodes.Element) -> list[nodes.target]:
    parent = node.parent
    index = parent.index(node)
    found = []
    while index > 0 and isinstance(parent[index - 1], nodes.target):
        index -= 1
        found.append(parent[index])
    return found


class AtbdLatexContent(SphinxPostTransform):
    """Rewrite the web-only constructs of the converted documents for print."""

    default_priority = 50

    def is_supported(self) -> bool:
        return self.app.builder.format != "html"

    def run(self, **kwargs: Any) -> None:
        found: dict[str, list[tuple[nodes.Node, str]]] = {
            "figure": [], "table": [], "math": [], "raw": [], "literal": []
        }
        for node, docname in _walk(self.document, self.env.docname):
            if isinstance(node, nodes.container):
                if FIGURE_CLASS in node.get("classes", []):
                    found["figure"].append((node, docname))
                elif TABLE_CLASS in node.get("classes", []):
                    found["table"].append((node, docname))
            elif isinstance(node, (nodes.math, nodes.math_block)):
                text = node.astext()
                fixed = _tex_math(text)
                if fixed != text:
                    node.children = []
                    node += nodes.Text(fixed)
                if isinstance(node, nodes.math_block):
                    found["math"].append((node, docname))
            elif isinstance(node, nodes.raw):
                found["raw"].append((node, docname))
            elif isinstance(node, nodes.literal):
                found["literal"].append((node, docname))
        for container, docname in found["figure"]:
            self._figure(container, docname)
        for container, _docname in found["table"]:
            self._table(container)
        for table in self.document.findall(nodes.table):
            _header_only(table)
            _column_widths(table)
        for math, docname in found["math"]:
            self._math(math, docname)
        for raw, docname in found["raw"]:
            if raw.parent is not None:
                self._raw_html(raw, docname)
        if self.app.builder.format == "latex":
            engine = self.config.latex_engine
            for literal, _docname in found["literal"]:
                if isinstance(literal.parent, (nodes.title, nodes.caption)):
                    continue
                latex = _breakable_literal(literal.astext(), engine)
                if latex is not None and literal.parent is not None:
                    literal.replace_self(nodes.raw("", latex, format="latex"))

    # -- figures ---------------------------------------------------------
    def _figure(self, container: nodes.container, docname: str) -> None:
        mermaid_cls = self._mermaid_class()
        has_mermaid = mermaid_cls is not None and any(
            True for _ in container.findall(mermaid_cls)
        )
        visuals: list[nodes.Node] = []
        extra: list[nodes.Node] = []
        caption = None
        for child in container.children:
            if _is_caption(child, "atbd-figure-caption"):
                caption = _parse_caption(child)
            elif isinstance(child, nodes.raw):
                continue  # <div class="atbd-figure-content"> wrappers, comments
            elif isinstance(child, nodes.paragraph) and child.children and all(
                isinstance(c, nodes.image)
                or (isinstance(c, nodes.Text) and not c.astext().strip())
                for c in child.children
            ):
                visuals.extend(c for c in child.children if isinstance(c, nodes.image))
            elif isinstance(child, nodes.image):
                visuals.append(child)
            elif mermaid_cls is not None and isinstance(child, mermaid_cls):
                visuals.append(child)
            elif mermaid_cls is not None and isinstance(child, nodes.figure) and any(
                True for _ in child.findall(mermaid_cls)
            ):
                visuals.extend(child.findall(mermaid_cls))
            elif has_mermaid and self._is_original_dropdown(child):
                continue  # the Mermaid rendering already shows this figure
            else:
                extra.append(child)

        figure = nodes.figure("", ids=list(container["ids"]),
                              names=list(container["names"]))
        # Stacked images share the page height, leaving room for the caption
        # (a float taller than the page overflows the bottom margin).
        share = 0.78 / max(1, len(visuals))
        setup = r"\csname spx@image@maxheight\endcsname=%.3f\textheight" % share
        if caption is not None:
            number, inlines = caption
            setup += r"\renewcommand{\figurename}{Figure\space}"
            if number:
                setup += r"\renewcommand{\thefigure}{%s}" % number
        else:
            setup += "".join(_latex_label(docname, i) for i in container["ids"])
        figure += nodes.raw("", setup, format="latex")
        for visual in visuals:
            visual.parent.remove(visual)
            figure += visual
        if caption is not None:
            figure += nodes.caption("", "", *inlines)
        container.replace_self([figure, *extra])

    def _mermaid_class(self) -> type[nodes.Element] | None:
        try:
            from sphinxcontrib.mermaid import mermaid
        except ImportError:
            return None
        return mermaid

    @staticmethod
    def _is_original_dropdown(node: nodes.Node) -> bool:
        if not isinstance(node, nodes.Element):
            return False
        if node.get("design_component") != "dropdown":
            return False
        title = next(iter(node.findall(nodes.rubric)), None)
        text = title.astext() if title is not None else node.astext()[:80]
        return "original figure" in text.lower()

    # -- tables ----------------------------------------------------------
    def _table(self, container: nodes.container) -> None:
        tables = [c for c in container.children if isinstance(c, nodes.table)]
        if len(tables) != 1:
            for raw in [c for c in container.children if isinstance(c, nodes.raw)]:
                container.remove(raw)
            return
        table = tables[0]
        caption = None
        rest: list[nodes.Node] = []
        for child in container.children:
            if child is table:
                continue
            if _is_caption(child, "atbd-table-caption"):
                caption = _parse_caption(child)
            elif not isinstance(child, nodes.raw):
                rest.append(child)
        container.remove(table)
        table["ids"] = list(container["ids"]) + [
            i for i in table["ids"] if i not in container["ids"]
        ]
        if "longtable" not in table["classes"]:
            table["classes"].append("longtable")
        before: list[nodes.Node] = []
        if caption is not None:
            number, inlines = caption
            table.insert(0, nodes.title("", "", *inlines))
            # longtable sets captions 4in wide by default: use the text width.
            setup = r"\setlength{\LTcapwidth}{\linewidth}"
            setup += r"\renewcommand{\tablename}{Table\space}"
            if number:
                setup += r"\renewcommand{\thetable}{%s}" % number
            before.append(nodes.raw("", setup, format="latex"))
        targets = _preceding_targets(container)
        if before and targets:
            first = targets[-1]
            parent = first.parent
            idx = parent.index(first)
            for node in reversed(before):
                parent.insert(idx, node)
            before = []
        container.replace_self([*before, table, *rest])

    # -- equations -------------------------------------------------------
    def _math(self, math: nodes.math_block, docname: str) -> None:
        ids = [i for i in math["ids"]]
        for target in _preceding_targets(math):
            refid = target.get("refid")
            if refid and refid not in ids:
                ids.append(refid)
        labels = "".join(_latex_label(docname, i) for i in ids)
        text = math.astext()
        tag = _TAG_RE.search(text)
        if math.get("nowrap", False) or tag is None:
            if labels:
                math.parent.insert(math.parent.index(math),
                                   nodes.raw("", labels, format="latex"))
            return
        body = _TAG_RE.sub("", text)
        body = "\n".join(line for line in body.splitlines() if line.strip())
        label = ""
        if math.get("label"):
            label = r"\label{equation:%s:%s}" % (math["docname"], math["label"])
        latex = (
            f"{labels}\n\\begin{{equation}}{label}\n\\begin{{split}}\n{body}\n"
            f"\\end{{split}}\\tag{{{tag.group('num')}}}\n\\end{{equation}}"
        )
        math.replace_self(nodes.raw("", latex, format="latex"))

    # -- raw HTML --------------------------------------------------------
    def _raw_html(self, raw: nodes.raw, docname: str) -> None:
        if "html" not in raw.get("format", "").split():
            return
        match = _SPAN_ID_RE.match(raw.astext().strip())
        if match is not None:
            raw.replace_self(
                nodes.raw("", _latex_label(docname, match.group("id")), format="latex")
            )
        elif isinstance(raw.parent, nodes.TextElement):
            raw.parent.remove(raw)


def _default_figure_align(app: Sphinx, config: Any) -> None:
    """Place figures where the source puts them, as the original PDFs do.

    Floating figures next to the many longtables of the PFDs can be pushed off
    the page (longtable's page breaks ignore pending floats). An explicit
    ``latex_elements["figure_align"]`` in conf.py still wins.
    """
    elements = dict(config.latex_elements or {})
    elements.setdefault("figure_align", "H")
    config.latex_elements = elements


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_post_transform(AtbdLatexContent)
    app.connect("config-inited", _default_figure_align)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
