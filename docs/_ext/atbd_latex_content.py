# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Make the converted ATBD/PFD pages carry their content into LaTeX/PDF builds.

The web conversions write figures and tables as ``sphinx-design`` divs whose
captions are raw HTML (``<p class="atbd-figure-caption">``), which every
non-HTML builder drops. For non-HTML builders only, this post-transform:

* turns ``:::{div} atbd-mermaid-figure`` into a real ``figure`` node (images,
  Mermaid diagrams, caption ``Figure N: ...`` with the source number N, label
  kept) and drops the "Original figure" dropdown when a Mermaid diagram is shown;
  a left-to-right diagram too wide for a readable text size is laid out top to
  bottom when that is larger, and one that stays too small shows the original
  figure of the dropdown instead;
* turns ``:::{div} atbd-table`` into a captioned table (``Tab.N ...``, source
  number N, label kept);
* makes every table a ``longtable`` (page breaks, repeated header) as wide as
  the headings, with explicit column widths so it wraps inside the margins,
  and gives a body to header-only tables (the LaTeX writer requires one);
* sets a table row taller than most of a page as several rows, so that
  the page can break inside it (LaTeX cannot break a longtable row);
* drops the "Acronyms" titles of the web dropdowns and the empty References
  chapter (its bibliography has no entry: the documents list their
  references in chapter 1);
* writes ``\\tag{N}`` equations as ``equation`` environments (``\\tag`` is not
  allowed inside the ``split`` Sphinx wraps them in) and labels their anchors;
* makes MathJax-tolerant math valid TeX (``^{'}`` primes), breaks long inline
  literals and keeps figures within the page height;
* turns ``<span id="rd8"></span>`` anchors into LaTeX labels and drops the other
  inline raw HTML.

HTML and the other builders (text, man, ...) are not touched.
"""

from __future__ import annotations

import html
import math
import os
import re
import struct
import zlib
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util import logging
from sphinx.util.texescape import escape as tex_escape

logger = logging.getLogger(__name__)

FIGURE_CLASS = "atbd-mermaid-figure"
# Size of the Mermaid PDFs in the page (see AtbdLatexContent._figure).
_MERMAID_SCALE = 0.75
# Text of the Mermaid PDFs (mmdc --pdfFit, 16 px = 12 pt) and the page of the
# export (A4, margins of atbd_pdf_layout): a diagram scaled down to fit the
# text width prints its text at 12 pt x scale.
_MERMAID_FONT_PT = 12.0
_TEXT_WIDTH_PT = 448.0
_TEXT_HEIGHT_PT = 683.0
# Below _MERMAID_MIN_PT, a left-to-right diagram is also rendered top to
# bottom and the larger of the two is kept. Below _MERMAID_FALLBACK_PT, clearly
# smaller than the text of the original XMLSpy figures in the Word PDFs (about
# 4.5 to 5 pt), the original figure is printed instead, at the full text width.
_MERMAID_MIN_PT = 5.0
_MERMAID_FALLBACK_PT = 4.0
# Share of the text height a Mermaid figure may take; a tall diagram (a
# top-to-bottom flowchart of the Science Guide) under _MERMAID_MIN_PT may take
# a page of its own, its caption included.
_MERMAID_SHARE = 0.8
_MERMAID_TALL_SHARE = 0.95
# "direction LR" (XSD class diagrams) and "flowchart LR" / "graph LR" (the
# redrawn Science Guide figures).
_LR_RE = re.compile(r"^(\s*(?:flowchart|graph|direction)\s+)LR\s*$", re.MULTILINE)
_MEDIABOX_RE = re.compile(
    rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]"
)
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
# Body of a raw ``\begin{equation} ... \tag{N} \end{equation}`` block.
_EQUATION_BODY_RE = re.compile(
    r"(\\begin\{equation\}(?:\\label\{[^{}]*\})?)(.*?)(\\tag\*?\{[^{}]*\}\s*\\end\{equation\})",
    re.DOTALL,
)
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
    # Greek capital iota (from the DOCX equations) is a Latin I in TeX math.
    text = text.replace("\u0399", "I")
    text = _PRIME_SUP_RE.sub(lambda m: "^{%s}" % (r"\prime" * len(m.group(1))), text)
    text = _PRIME_GROUP_RE.sub(r"\1{}}", text)
    return _ARG_OP_RE.sub(r"\\operatorname*{arg\\,\1}", text)


# Break opportunities in long inline literals (file name patterns, XML types,
# enumeration values): after _ . / : > , and inside camelCase words.
_LITERAL_BREAK_RE = re.compile(r"(?<=[_./:>,])(?=.)|(?<=[a-z])(?=[A-Z])|(?=<)")
_LONG_LITERAL = 20
_LONG_TABLE_LITERAL = 8  # table columns are narrow
# Break opportunities in long words of running text (WKT strings, numbers):
# after punctuation, and inside long digit runs (high-precision constants).
_WORD_BREAK_RE = re.compile(r"(?<=[,;:/_.\[\](=&])(?=\S)|(?<=\d{4})(?=\d)")
_LONG_WORD = 40
_LONG_TABLE_WORD = 15
# C0 control characters left by the DOCX conversion (no glyph in any font).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _breakable_literal(text: str, engine: str, longest: int = _LONG_LITERAL) -> str | None:
    """LaTeX for an inline literal with break points, or None if it is short."""
    if max((len(w) for w in text.split()), default=0) <= longest:
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


# Inputs / Outputs tables of the ATBDs (Symbol | Name | Origin | Size |
# Variability | Default | Description): column widths of the originals
# (equal columns; a wider Description when the table uses the indent).
_IO_TABLE_WIDTHS = {False: (1, 1, 1, 1, 1, 1, 1), True: (13, 14, 13, 13, 13, 13, 19)}


def _header_texts(tgroup: nodes.tgroup) -> list[str]:
    thead = next((c for c in tgroup.children if isinstance(c, nodes.thead)), None)
    row = next(iter(thead.findall(nodes.row)), None) if thead is not None else None
    if row is None:
        return []
    return [" ".join(entry.astext().split()) for entry in row.children]


def _cell_blocks(entry: nodes.Element) -> list[list[int]]:
    """Word lengths of each paragraph (or line block) of a table cell."""
    blocks = [p for p in entry.findall(nodes.TextElement)
              if isinstance(p, (nodes.paragraph, nodes.line, nodes.literal_block))]
    texts = [p.astext() for p in blocks] or [entry.astext()]
    return [[len(w) for w in t.split()] or [0] for t in texts]


def _cell_atom(entry: nodes.Element) -> int:
    """Longest piece of a table cell that cannot wrap.

    Long words and inline literals get break points (``_text`` and
    ``_breakable_literal``), and formulas are scaled down to the column.
    """
    longest = 1
    for text in entry.findall(nodes.Text):
        parent = text.parent
        literal = isinstance(parent, nodes.literal)
        for word in str(text).split():
            if isinstance(parent, nodes.math):
                pieces = [word[:12]]
            elif literal and len(word) > _LONG_TABLE_LITERAL:
                pieces = _LITERAL_BREAK_RE.split(word)
            elif not literal and len(word) > _LONG_TABLE_WORD:
                pieces = _WORD_BREAK_RE.split(word)
            else:
                pieces = [word]
            longest = max(longest, *(len(p) for p in pieces))
    return longest


def _wrapped_lines(blocks: list[list[int]], width: float) -> tuple[int, int]:
    """Lines of a cell ``width`` characters wide (greedy word wrap), and the
    number of words broken across lines."""
    width = max(1.0, 0.95 * width)
    lines = broken = 0
    for block in blocks:
        lines += 1
        used = -1.0
        for word in block:
            if used >= 0 and used + 1 + word <= width:
                used += 1 + word
            elif word <= width:
                lines += used >= 0
                used = word
            else:  # broken at its break points (identifiers, numbers)
                lines += (used >= 0) + int((word - 1) // width)
                used = word % width or width
                broken += 1
    return lines, broken


# Cost of a word broken across lines, in table lines.
_BROKEN_WORD_COST = 2.0


def _fit_columns(rows: list[list[tuple[int, list[list[int]]]]], minimum: list[float],
                 natural: list[int], budget: float) -> list[float]:
    """Column widths (characters) that keep the table short (fewest lines)
    without breaking words when a few characters avoid it."""
    ncols = len(minimum)
    nrows = len(rows)
    widths = [math.ceil(m) for m in minimum]
    spare = int(budget) - sum(widths)
    by_col: list[list[tuple[int, list[list[int]]]]] = [[] for _ in range(ncols)]
    for r, cells in enumerate(rows):
        for col, blocks in cells:
            by_col[col].append((r, blocks))

    def layout(col: int, width: int) -> tuple[list[int], int]:
        out = [0] * nrows
        broken = 0
        for r, blocks in by_col[col]:
            out[r], b = _wrapped_lines(blocks, width)
            broken += b
        return out, broken

    def cost(lines: list[list[int]], broken: list[int]) -> float:
        height = sum(max(lines[c][r] for c in range(ncols)) for r in range(nrows))
        return height + _BROKEN_WORD_COST * sum(broken)

    cols = [layout(c, widths[c]) for c in range(ncols)]
    lines = [c[0] for c in cols]
    broken = [c[1] for c in cols]
    current = cost(lines, broken)
    while spare > 0:
        best = None
        for c in range(ncols):
            for step in (1, 2, 3, 5, 8, 13, 21, 34):
                if step > spare:
                    break
                trial, trial_broken = layout(c, widths[c] + step)
                gain = current - cost(lines[:c] + [trial] + lines[c + 1:],
                                      broken[:c] + [trial_broken] + broken[c + 1:])
                if gain > 0 and (best is None or gain / step > best[0]):
                    best = (gain / step, c, step, trial, trial_broken)
        if best is None:
            break
        _, c, step, lines[c], broken[c] = best
        widths[c] += step
        spare -= step
        current = cost(lines, broken)
    scores = [float(w) for w in widths]
    if spare > 0:
        need = [max(0.0, n - w) for n, w in zip(natural, scores)]
        if sum(need) > 0:
            give = min(float(spare), sum(need))
            scores = [w + give * n / sum(need) for w, n in zip(scores, need)]
            spare -= give
        if spare > 0:
            total = sum(scores)
            scores = [w + spare * w / total for w in scores]
    return scores


def _column_widths(table: nodes.table, budget: float = 75.0, wide: bool = False) -> None:
    """Replace ``colwidths-auto`` by widths proportional to the cell contents.

    ``budget`` is the number of characters of the table font that fit in the
    table width; ``wide`` tables also use the heading indent.
    """
    classes = table["classes"]
    if "colwidths-given" in classes:
        return
    for tgroup in table.findall(nodes.tgroup):
        colspecs = [c for c in tgroup.children if isinstance(c, nodes.colspec)]
        ncols = len(colspecs)
        if not ncols:
            continue
        header = _header_texts(tgroup)
        io_widths = _IO_TABLE_WIDTHS[wide]
        if (ncols == len(io_widths) and header
                and header[0] == "Symbol" and header[-1] == "Description"):
            for spec, width in zip(colspecs, io_widths):
                spec["colwidth"] = width
            if "colwidths-auto" in classes:
                classes.remove("colwidths-auto")
            classes.append("colwidths-given")
            return
        # Cells of each row: blocks (paragraphs, list items) of word lengths.
        rows: list[list[tuple[int, list[list[int]]]]] = []
        words = [1.0] * ncols
        atoms = [1.0] * ncols
        natural = [3] * ncols
        for row in tgroup.findall(nodes.row):
            # Bold header: wider characters.
            scale = 1.2 if isinstance(row.parent, nodes.thead) else 1.0
            cells = []
            col = 0
            for entry in row.children:
                if col >= ncols:
                    break
                span = 1 + entry.get("morecols", 0)
                if span == 1:
                    blocks = _cell_blocks(entry)
                    cells.append((col, blocks))
                    longest = max((w for b in blocks for w in b), default=1)
                    words[col] = max(words[col], scale * min(longest, 28))
                    atoms[col] = max(atoms[col], scale * min(_cell_atom(entry), 28))
                    for b in blocks:
                        natural[col] = max(natural[col], min(sum(b) + len(b) - 1, 80))
                col += span
            rows.append(cells)
        # Each column also pays its \tabcolsep padding. Keep every column
        # wide enough for its longest word (or, when these do not fit, for
        # its longest unbreakable piece), then give the rest, a few
        # characters at a time, to the column that saves the most wrapped
        # lines and broken words; what no column can use goes to the columns
        # not yet at their one-line width, then in proportion to the widths.
        budget = budget - 2.0 * ncols
        minimum = [w + 1.0 for w in words]
        if sum(minimum) >= budget:
            minimum = [a + 1.0 for a in atoms]
        if sum(minimum) >= budget:
            scores = minimum
        else:
            scores = _fit_columns(rows, minimum, natural, budget)
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


def _in_table(node: nodes.Node) -> bool:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, nodes.table):
            return True
        parent = parent.parent
    return False


# A longtable cannot sit in a box (admonition frame, footnote) nor in a table.
_NO_LONGTABLE_IN = (nodes.Admonition, nodes.footnote, nodes.table, nodes.sidebar,
                    nodes.topic)


def _can_break(table: nodes.table) -> bool:
    if any(True for _ in table.findall(nodes.table, include_self=False)):
        return False
    parent = table.parent
    while parent is not None:
        if isinstance(parent, _NO_LONGTABLE_IN):
            return False
        parent = parent.parent
    return True


def _lead_paragraph(table: nodes.table) -> nodes.paragraph | None:
    """Short paragraph right before ``table`` (targets and raw skipped)."""
    parent = table.parent
    if parent is None or "longtable" not in table["classes"]:
        return None
    for sibling in reversed(parent.children[:parent.index(table)]):
        if isinstance(sibling, (nodes.target, nodes.raw, nodes.comment)):
            continue
        if isinstance(sibling, nodes.paragraph) and len(sibling.astext()) <= 120:
            return sibling
        return None
    return None


def _followed_by_longtable(node: nodes.Element) -> bool:
    parent = node.parent
    if parent is None:
        return False
    for sibling in parent.children[parent.index(node) + 1:]:
        if isinstance(sibling, (nodes.target, nodes.raw, nodes.comment)):
            continue
        return isinstance(sibling, nodes.table) and "longtable" in sibling["classes"]
    return False


def _drop_dropdown_title(dropdown: nodes.container) -> None:
    """Web dropdowns ("Acronyms") print their title as a rubric: drop it.

    The section heading above already names the content (front-matter
    dropdowns are handled by ``atbd_pdf_layout``).
    """
    for rubric in list(dropdown.findall(nodes.rubric)):
        if rubric.parent is not None:
            rubric.parent.remove(rubric)
        break


def _unescape_uri(reference: nodes.reference) -> None:
    r"""``http://host/a\_b`` (Markdown escape kept in a bare URL) -> ``a_b``."""
    uri = reference.get("refuri", "")
    if "\\_" not in uri:
        return
    reference["refuri"] = uri.replace("\\_", "_")
    for text in list(reference.findall(nodes.Text)):
        if "\\_" in text:
            text.parent.replace(text, nodes.Text(text.replace("\\_", "_")))


def _drop_empty_chapters(document: nodes.document) -> None:
    """Remove included chapters with a title only (empty References page)."""
    for sof in list(document.findall(addnodes.start_of_file)):
        sections = [c for c in sof.children if isinstance(c, nodes.section)]
        if len(sections) != 1 or sof.parent is None:
            continue
        section = sections[0]
        title = section.next_node(nodes.title)
        text = section.astext().strip()
        if title is not None and text == title.astext().strip():
            sof.parent.remove(sof)


def _preceding_targets(node: nodes.Element) -> list[nodes.target]:
    parent = node.parent
    index = parent.index(node)
    found = []
    while index > 0 and isinstance(parent[index - 1], nodes.target):
        index -= 1
        found.append(parent[index])
    return found


# ---------------------------------------------------------------------------
# Rows taller than a page
# ---------------------------------------------------------------------------
# LaTeX cannot break a page inside a longtable row: a cell taller than the
# page (a long enumeration) runs over the footer and its end is lost. Such a
# row is set as several rows, each holding the next lines of its cells, and
# without rule between them (``\atbdrowcontinues``), so that the page can
# break between them.

_ROW_SPLIT_LINES = 60  # estimated lines of the tallest cell before splitting
_ROW_CHUNK_LINES = 24  # estimated lines of each row the cell is split into


def _column_chars(tgroup: nodes.tgroup, budget: float) -> list[float]:
    """Width in characters of each column (``budget``: table width)."""
    specs = [c for c in tgroup.children if isinstance(c, nodes.colspec)]
    weights = [float(s.get("colwidth", 1) or 1) for s in specs]
    total = sum(weights) or 1.0
    usable = budget - 2.0 * len(specs)
    return [max(1.0, usable * w / total) for w in weights]


def _tokens(paragraph: nodes.paragraph) -> list[tuple[nodes.Node, int]]:
    """Words of a paragraph (Text split at spaces; inline elements whole)."""
    out: list[tuple[nodes.Node, int]] = []
    for child in paragraph.children:
        if isinstance(child, nodes.Text):
            words = re.findall(r"\S+\s*", str(child))
            if words and str(child)[:1].isspace() and out:
                # Keep the space between the previous element and this word.
                words[0] = " " + words[0]
            out.extend((nodes.Text(w), len(w.strip())) for w in words)
        else:
            out.append((child, max(1, len(child.astext()))))
    return out


def _split_paragraph(paragraph: nodes.paragraph, width: float,
                     chunk: int) -> list[nodes.paragraph]:
    """``paragraph`` cut at word boundaries into pieces of ``chunk`` lines."""
    pieces: list[nodes.paragraph] = []
    current: list[nodes.Node] = []
    lengths: list[int] = []
    for node, length in _tokens(paragraph):
        if current and _wrapped_lines([lengths + [length]], width)[0] > chunk:
            pieces.append(nodes.paragraph("", "", *current))
            current, lengths = [], []
        if isinstance(node, nodes.Text) and not current:
            node = nodes.Text(str(node).lstrip())
        if node.parent is not None:
            node.parent.remove(node)
        current.append(node)
        lengths.append(length)
    if current:
        pieces.append(nodes.paragraph("", "", *current))
    return pieces


def _block_lines(block: nodes.Node, width: float) -> int:
    return _wrapped_lines(_cell_blocks(block) if isinstance(block, nodes.Element)
                          else [[len(w) for w in block.astext().split()] or [0]],
                          width)[0]


def _cell_chunks(entry: nodes.entry, width: float, chunk: int) -> list[list[nodes.Node]]:
    """Children of ``entry`` grouped in pieces of at most ``chunk`` lines
    (a paragraph longer than that is cut; other blocks are kept whole)."""
    blocks: list[nodes.Node] = []
    for child in list(entry.children):
        entry.remove(child)
        if (isinstance(child, nodes.paragraph)
                and _block_lines(child, width) > chunk):
            blocks.extend(_split_paragraph(child, width, chunk))
        else:
            blocks.append(child)
    chunks: list[list[nodes.Node]] = []
    used = chunk + 1
    for block in blocks:
        lines = _block_lines(block, width)
        if chunks and used + lines <= chunk:
            chunks[-1].append(block)
            used += lines
        else:
            chunks.append([block])
            used = lines
    return chunks


def _split_tall_rows(table: nodes.table, budget: float) -> int:
    """Set each row taller than most of a page as several rows; return
    the number of rows split."""
    split = 0
    for tgroup in table.findall(nodes.tgroup):
        widths = _column_chars(tgroup, budget)
        for tbody in [c for c in tgroup.children if isinstance(c, nodes.tbody)]:
            for row in list(tbody.children):
                entries = [e for e in row.children if isinstance(e, nodes.entry)]
                if (len(entries) != len(widths)
                        or any(e.get("morerows") or e.get("morecols") for e in entries)):
                    continue  # spanning cells: kept as they are
                lines = [_wrapped_lines(_cell_blocks(e), w)[0]
                         for e, w in zip(entries, widths)]
                if max(lines, default=0) <= _ROW_SPLIT_LINES:
                    continue
                cells = [_cell_chunks(e, w, _ROW_CHUNK_LINES)
                         for e, w in zip(entries, widths)]
                count = max(len(c) for c in cells)
                rows = [nodes.row("", *(nodes.entry() for _ in entries))
                        for _ in range(count)]
                rows[0] = row
                for col, chunks in enumerate(cells):
                    for i, children in enumerate(chunks):
                        rows[i].children[col].extend(children)
                for continued in rows[:-1]:
                    # No rule under this row: the next one continues it.
                    continued.children[0].append(nodes.raw(
                        "", r"\atbdrowcontinues", format="latex"))
                index = tbody.index(row)
                for offset, extra in enumerate(rows[1:], start=1):
                    tbody.insert(index + offset, extra)
                split += 1
    return split


class AtbdLatexContent(SphinxPostTransform):
    """Rewrite the web-only constructs of the converted documents for print."""

    default_priority = 50

    def is_supported(self) -> bool:
        return self.app.builder.format == "latex"

    def run(self, **kwargs: Any) -> None:
        found: dict[str, list[tuple[nodes.Node, str]]] = {
            "figure": [], "table": [], "math": [], "raw": [], "literal": [],
            "tables": [], "dropdown": [], "reference": [], "text": [],
            "inline_math": [],
        }
        for node, docname in _walk(self.document, self.env.docname):
            if isinstance(node, nodes.container):
                if FIGURE_CLASS in node.get("classes", []):
                    found["figure"].append((node, docname))
                elif TABLE_CLASS in node.get("classes", []):
                    found["table"].append((node, docname))
                elif node.get("design_component") == "dropdown":
                    found["dropdown"].append((node, docname))
            elif isinstance(node, (nodes.math, nodes.math_block)):
                text = node.astext()
                fixed = _tex_math(text)
                if fixed != text:
                    node.children = []
                    node += nodes.Text(fixed)
                if isinstance(node, nodes.math_block):
                    found["math"].append((node, docname))
                else:
                    found["inline_math"].append((node, docname))
            elif isinstance(node, nodes.raw):
                found["raw"].append((node, docname))
            elif isinstance(node, nodes.literal):
                found["literal"].append((node, docname))
            elif isinstance(node, nodes.table):
                found["tables"].append((node, docname))
            elif isinstance(node, nodes.reference):
                found["reference"].append((node, docname))
            elif isinstance(node, nodes.Text):
                found["text"].append((node, docname))
        figures = [self._figure(container, docname) for container, docname in found["figure"]]
        for container, _docname in found["table"]:
            self._table(container)
        for dropdown, _docname in found["dropdown"]:
            _drop_dropdown_title(dropdown)
        pfd = "/pfd-" in f"/{self.env.docname}"
        wide = bool(self.config.atbd_wide_tables)
        # Characters of the table font in the table width: ATBD 9 pt,
        # PFD 10 pt; wide tables also use the heading indent.
        budget = (85.0 if pfd else 94.0) * (1.0 if wide else 450 / 485)
        for table, _docname in found["tables"]:
            _header_only(table)
            _column_widths(table, budget, wide)
            if _can_break(table):
                for cls in ("longtable", "atbd-wide") if wide else ("longtable",):
                    if cls not in table["classes"]:
                        table["classes"].append(cls)
                if pfd and "atbd-caption-below" not in table["classes"]:
                    table["classes"].append("atbd-caption-below")
        for table, _docname in found["tables"]:
            lead = _lead_paragraph(table)
            if lead is not None:
                # Keep a short lead-in ("Inputs", "... as follows:") with the
                # start of its table.
                lead.parent.insert(lead.parent.index(lead), nodes.raw(
                    "", r"\needspace{4\baselineskip}", format="latex"))
        for figure in figures:
            if _followed_by_longtable(figure):
                # A float pending when a longtable starts can be set off the
                # page (with the table start): keep this one in place.
                figure.parent.insert(figure.parent.index(figure),
                                     nodes.raw("", r"\atbdfigurehere", format="latex"))
        for math, docname in found["math"]:
            self._math(math, docname)
        for raw, docname in found["raw"]:
            if raw.parent is not None:
                self._raw_html(raw, docname)
        for reference, _docname in found["reference"]:
            _unescape_uri(reference)
        for text, _docname in found["text"]:
            self._text(text)
        for math, _docname in found["inline_math"]:
            if math.parent is not None and _in_table(math):
                # Formulas wider than their column are scaled down to it.
                math.replace_self(nodes.raw(
                    "", r"\atbdfitmath{\(%s\)}" % math.astext(), format="latex"))
        engine = self.config.latex_engine
        for literal, _docname in found["literal"]:
            if isinstance(literal.parent, (nodes.title, nodes.caption)):
                continue
            in_table = _in_table(literal)
            latex = _breakable_literal(
                literal.astext(), engine,
                _LONG_TABLE_LITERAL if in_table else _LONG_LITERAL)
            if latex is not None and literal.parent is not None:
                literal.replace_self(nodes.raw("", latex, format="latex"))
        for table, _docname in found["tables"]:
            if "longtable" in table["classes"]:
                _split_tall_rows(table, budget)
        _drop_empty_chapters(self.document)

    # -- text ----------------------------------------------------------
    def _text(self, text: nodes.Text) -> None:
        """Drop control characters; let long unbreakable tokens wrap."""
        parent = text.parent
        if parent is None or isinstance(parent, (nodes.literal, nodes.raw, nodes.title,
                                                 nodes.math, nodes.literal_block,
                                                 nodes.FixedTextElement)):
            return
        clean = _CONTROL_RE.sub("", text)
        longest = _LONG_TABLE_WORD if _in_table(text) else _LONG_WORD
        if max((len(w) for w in clean.split()), default=0) <= longest:
            if clean != text:
                parent.replace(text, nodes.Text(clean))
            return
        engine = self.config.latex_engine
        parts = [p for p in _WORD_BREAK_RE.split(clean) if p]
        latex = r"\allowbreak{}".join(tex_escape(p, engine) for p in parts)
        parent.replace(text, nodes.raw("", latex, format="latex"))

    # -- figures ---------------------------------------------------------
    def _figure(self, container: nodes.container, docname: str) -> nodes.figure:
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
            elif (has_mermaid or "atbd-drawn-figure" in container.get("classes", [])) \
                    and self._is_original_dropdown(child):
                continue  # the Mermaid rendering or the redrawn figure already shows it
            else:
                extra.append(child)

        original = False
        single_share = _MERMAID_SHARE
        if has_mermaid and len(visuals) == 1 and isinstance(visuals[0], mermaid_cls):
            fallback, single_share = self._fit_mermaid(visuals[0], container)
            if fallback:
                visuals, original = fallback, True

        figure = nodes.figure("", ids=list(container["ids"]),
                              names=list(container["names"]))
        # At most half a page per figure, as in the originals; stacked images
        # share the page height, leaving room for the caption (a float taller
        # than the page overflows the bottom margin).
        share = 0.5 if len(visuals) <= 1 else 0.75 / len(visuals)
        mermaid_only = has_mermaid and all(isinstance(v, mermaid_cls) for v in visuals)
        if mermaid_only or original:
            # Mermaid diagrams (XSD class diagrams) are vector PDFs the size
            # of the drawing (mmdc --pdfFit) in 12 pt text: at 75 % (9 pt
            # text), up to the full text width and most of the page height,
            # so that wide and tall diagrams keep a readable text size.
            share = single_share if len(visuals) <= 1 else _MERMAID_SHARE / len(visuals)
        setup = r"\csname spx@image@maxheight\endcsname=%.3f\textheight" % share
        if mermaid_only:
            setup += r"\def\atbdfigscale{%s}\def\atbdfigmaxwidth{\linewidth}" % _MERMAID_SCALE
        elif original:
            # the original figure of a diagram too wide for the page: as large
            # as the page allows (its text is small in the Word PDFs too)
            setup += r"\def\atbdfigscale{1}\def\atbdfigmaxwidth{\linewidth}"
        if caption is not None:
            number, inlines = caption
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
        return figure

    def _mermaid_font(self, code: str, options: Any, height: float) -> float | None:
        """Text size (pt) of a Mermaid diagram once fitted in the page.

        Renders it with sphinxcontrib.mermaid (same file and cache as the
        LaTeX writer, which then reuses it); None when it cannot be rendered.
        """
        from types import SimpleNamespace

        from sphinxcontrib.mermaid import MermaidError, render_mm

        try:
            _relfn, outfn = render_mm(SimpleNamespace(builder=self.app.builder),
                                      code, options, "pdf")
            if outfn is None:  # mmdc not found: the writer reports it
                return None
            match = _MEDIABOX_RE.search(Path(outfn).read_bytes())
        except MermaidError:
            return None  # the writer reports the same error
        except Exception as exc:  # noqa: BLE001 - keep the default layout
            logger.warning("cannot size a Mermaid diagram for the PDF layout: %s", exc,
                           location=self.env.docname)
            return None
        if not match:
            return None
        x0, y0, x1, y1 = (float(v) for v in match.groups())
        width, tall = x1 - x0, y1 - y0
        if width <= 0 or tall <= 0:
            return None
        return _MERMAID_FONT_PT * min(_MERMAID_SCALE, _TEXT_WIDTH_PT / width,
                                      height * _TEXT_HEIGHT_PT / tall)

    def _fit_mermaid(self, diagram: nodes.Element, container: nodes.container
                     ) -> tuple[list[nodes.Node] | None, float]:
        """Keep a large diagram readable in the PDF.

        Returns (images, share): the images of the "Original figure" dropdown
        to print instead when the text stays under _MERMAID_FALLBACK_PT (None:
        print the diagram), and the share of the text height the figure may
        take. A left-to-right diagram is laid out top to bottom when that gives
        larger text; a tall one may take a page of its own (_MERMAID_TALL_SHARE).
        """
        code, options = diagram["code"], diagram.get("options", {})
        best = self._mermaid_font(code, options, _MERMAID_SHARE)
        if best is None or best >= _MERMAID_MIN_PT:
            return None, _MERMAID_SHARE
        if _LR_RE.search(code):
            top_down = _LR_RE.sub(r"\1TB", code)
            font = self._mermaid_font(top_down, options, _MERMAID_SHARE)
            if font is not None and font > best:
                diagram["code"], best = top_down, font
        share = _MERMAID_SHARE
        if best < _MERMAID_MIN_PT:
            tall = self._mermaid_font(diagram["code"], options, _MERMAID_TALL_SHARE)
            if tall is not None and tall > best:
                best, share = tall, _MERMAID_TALL_SHARE
        if best >= _MERMAID_FALLBACK_PT:
            return None, share
        for child in container.children:
            if self._is_original_dropdown(child):
                images = list(child.findall(nodes.image))
                if images:
                    return images, _MERMAID_SHARE
        logger.warning("Mermaid diagram printed at %.1f pt: no original figure to "
                       "print instead", best, location=container)
        return None, share

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
        after: list[nodes.Node] = []
        if caption is not None:
            number, inlines = caption
            table.insert(0, nodes.title("", "", *inlines))
            if number:
                # Source number, for this table only.
                before.append(nodes.raw(
                    "", r"\begingroup\renewcommand{\thetable}{%s}" % number,
                    format="latex"))
                after.append(nodes.raw("", r"\endgroup", format="latex"))
        targets = _preceding_targets(container)
        if before and targets:
            first = targets[-1]
            parent = first.parent
            idx = parent.index(first)
            for node in reversed(before):
                parent.insert(idx, node)
            before = []
        container.replace_self([*before, table, *after, *rest])

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
            # Same width limit as the numbered equations below.
            fitted = text
            if text.count(r"\begin{aligned}") == 1 and text.count(r"\end{aligned}") == 1:
                fitted = text.replace(r"\begin{aligned}", r"\atbdfitdisplay{").replace(
                    r"\end{aligned}", "}")
            elif r"\begin{aligned}" not in text:
                fitted = _EQUATION_BODY_RE.sub(r"\1\\atbdfitdisplay{\2}\3", text, count=1)
            if fitted != text:
                math.children = []
                math += nodes.Text(fitted)
            if labels:
                math.parent.insert(math.parent.index(math),
                                   nodes.raw("", labels, format="latex"))
            return
        body = _TAG_RE.sub("", text)
        body = "\n".join(line for line in body.splitlines() if line.strip())
        label = ""
        if math.get("label"):
            label = r"\label{equation:%s:%s}" % (math["docname"], math["label"])
        # \atbdfitdisplay (atbdlayout.sty) sets the body like ``split`` and
        # scales it down when it is wider than the text.
        latex = (
            f"{labels}\n\\begin{{equation}}{label}\n\\atbdfitdisplay{{\n{body}\n"
            f"}}\\tag{{{tag.group('num')}}}\n\\end{{equation}}"
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
    """Let figures float (``[!htp]``) as Word places them in the originals.

    Forced ``[H]`` placement leaves half-empty pages before tall figures. An
    explicit ``latex_elements["figure_align"]`` in conf.py still wins.
    """
    elements = dict(config.latex_elements or {})
    # No bottom floats: longtable does not leave room for them and a figure
    # set at the bottom of a page broken by a longtable overflows it.
    elements.setdefault("figure_align", "!htp")
    config.latex_elements = elements


# ---------------------------------------------------------------------------
# White margins of raster figures
# ---------------------------------------------------------------------------
# Many converted figures are an A4 page rendering with the drawing in the
# middle. The LaTeX build trims those margins (``\includegraphics[trim=...,
# clip]``) with the values written to ``atbdtrim.tex`` in the output
# directory; the source images are not modified.

_TRIM_PAD = 8  # pixels kept around the drawing
_TRIM_MIN_GAIN = 0.15  # only trim when the margins are at least 15 % of the area


def _png_content_box(data: bytes) -> tuple[int, int, int, int, int, int, float] | None:
    """``(width, height, left, top, right, bottom, dpi)`` of the drawing.

    Pure Python, without decoding the pixels: in rows filtered with "Up" or
    "Paeth" (what PNG encoders use for flat areas), a pixel equal to the
    uniform background around it is stored as zero bytes. The drawing is the
    box of the non-zero filtered bytes. Returns None for PNGs this cannot
    tell (other filters, interlacing, bit depth other than 8).
    """
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos, idat, header, dpi = 8, [], None, 72.0
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"pHYs" and body[8] == 1:  # pixels per metre
            dpi = struct.unpack(">I", body[:4])[0] * 0.0254
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None:
        return None
    width, height, depth, colour, _c, _f, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(colour)
    if depth != 8 or interlace or channels is None:
        return None
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    if len(raw) < height * (stride + 1):
        return None
    top = bottom = None
    left, right = width, -1
    for y in range(height):
        start = y * (stride + 1)
        kind, row = raw[start], raw[start + 1:start + 1 + stride]
        if y == 0:
            # No row above: Up/Paeth predict from the left neighbour only.
            if kind not in (1, 4) or row[channels:].strip(b"\0"):
                return None  # the first row is not a uniform background
            continue
        if kind not in (2, 4):
            return None
        stripped = row.lstrip(b"\0")
        if not stripped:
            continue
        first = (stride - len(stripped)) // channels
        last = (len(row.rstrip(b"\0")) - 1) // channels
        top = y if top is None else top
        bottom = y
        left, right = min(left, first), max(right, last)
    if top is None:
        return None
    return width, height, left, top, right, bottom, dpi or 72.0


def _write_trims(app: Sphinx, exception: Exception | None) -> None:
    if exception is not None or app.builder.format != "latex":
        return
    lines = []
    for source, dest in sorted(getattr(app.builder, "images", {}).items()):
        path = Path(app.srcdir, source)
        if path.suffix.lower() != ".png" or not path.is_file():
            continue
        box = _png_content_box(path.read_bytes())
        if box is None:
            continue
        width, height, left, top, right, bottom, dpi = box
        left, top = max(0, left - _TRIM_PAD), max(0, top - _TRIM_PAD)
        right, bottom = min(width - 1, right + _TRIM_PAD), min(height - 1, bottom + _TRIM_PAD)
        kept = (right - left + 1) * (bottom - top + 1)
        if kept > (1 - _TRIM_MIN_GAIN) * width * height:
            continue
        unit = 72.0 / dpi  # bp per pixel, as \includegraphics sizes the PNG
        trims = (left, height - 1 - bottom, width - 1 - right, top)
        base, ext = os.path.splitext(dest)
        lines.append(
            "\\atbdsettrim{{%s}%s}{%s}"
            % (base, ext, " ".join("%.2fbp" % (t * unit) for t in trims))
        )
    Path(app.outdir, "atbdtrim.tex").write_text(
        "% White margins of the raster figures (generated)\n"
        + "".join(line + "\n" for line in lines),
        encoding="utf-8",
    )


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_post_transform(AtbdLatexContent)
    # Tables also use the left indent of the headings (as in the original
    # PFDs and L1 SAR ATBD; the other ATBDs keep them in the text width).
    app.add_config_value("atbd_wide_tables", False, "env", bool)
    app.connect("config-inited", _default_figure_align)
    app.connect("build-finished", _write_trims)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
