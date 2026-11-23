# SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS
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
import json
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
# A diagram whose text prints below _LANDSCAPE_BELOW_PT (and gains 15 % or more)
# goes on a landscape page, at most _LANDSCAPE_SHARE of its height.
_LANDSCAPE_BELOW_PT = 6.0
_LANDSCAPE_SHARE = 0.8
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
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# ``x^{'}`` is fine for MathJax but TeX wants ``\prime``; and ``{{\widehat{s}}^{'}}^2``
# is a "Double superscript" for TeX, which unwraps a group holding a single
# accent atom: an empty group after the prime keeps the outer group intact.
_PRIME_SUP_RE = re.compile(r"\^\{('+)\}")
_PRIME_GROUP_RE = re.compile(r"(\^\{(?:\\prime)+\})\}(?=\s*[\^_])")
# Operators the sources use without a definition (amsmath has no \argmin).
_ARG_OP_RE = re.compile(r"\\arg(min|max)(?![A-Za-z])")


# Content that a fixed-size delimiter would not fit (stacked material).
_TALL_RE = re.compile(r"\\(?:d|t)?frac\b|\\(?:sum|prod|int|sqrt|binom|begin|over)\b")
_LEFT_ANY_RE = re.compile(r"\\left(?![A-Za-z])")
_RIGHT_ANY_RE = re.compile(r"\\right(?![A-Za-z])")
_LEFT_BRACKET_RE = re.compile(r"\\left\s*(?=[(\[]|\\\{)")
_RIGHT_BRACKET_RE = re.compile(r"\\right\s*(?=[)\]]|\\\})")


def _breakable_delimiters(tex: str) -> str:
    """``tex`` with its ``\\left( ... \\right)`` pairs as fixed-size ``\\bigl( ... \\bigr)``,
    which TeX can break inside (after \\times, +, ...), or "" when that does not
    apply (no pair, other delimiters such as ``\\left.``, stacked content)."""
    lefts, rights = len(_LEFT_ANY_RE.findall(tex)), len(_RIGHT_ANY_RE.findall(tex))
    if not lefts or _TALL_RE.search(tex):
        return ""
    if (len(_LEFT_BRACKET_RE.findall(tex)), len(_RIGHT_BRACKET_RE.findall(tex))) != (lefts, rights):
        return ""
    tex = _LEFT_BRACKET_RE.sub(r"\\bigl", tex)
    return _RIGHT_BRACKET_RE.sub(r"\\bigr", tex)


_ENV_RE = re.compile(r"\\(begin|end)\{")


def _rows(tex: str) -> list[str]:
    """Rows of an aligned body: split at the ``\\\\`` of the top level (not inside
    a group or an environment), alignment marks (``&``) of the top level dropped."""
    rows: list[str] = []
    depth = envs = start = 0
    out: list[str] = []
    i = 0
    while i < len(tex):
        ch = tex[i]
        m = _ENV_RE.match(tex, i)
        if m:
            envs += 1 if m.group(1) == "begin" else -1
            close = tex.find("}", m.end())
            i = len(tex) if close < 0 else close + 1
            continue
        if ch == "\\" and tex.startswith("\\\\", i):
            if depth == 0 and envs == 0:
                out.append(tex[start:i])
                rows.append("".join(out))
                out = []
                i += 2
                start = i
                continue
            i += 2
            continue
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "&" and depth == 0 and envs == 0:
            out.append(tex[start:i])
            start = i + 1
        i += 1
    out.append(tex[start:])
    rows.append("".join(out))
    return [r.strip() for r in rows if r.strip()]


def _rows_alternative(tex: str) -> str:
    """Optional argument of ``\\atbdfitdisplay``: the rows of an aligned body, each
    broken after its operators (a body too wide to be scaled is set that way)."""
    rows = _rows(tex)
    if len(rows) < 2:
        return ""
    return "[%s]" % "".join(
        "\\atbdrow{%s}" % (_breakable_delimiters(r) or r) for r in rows)


def _alternative(tex: str) -> str:
    """Optional argument of the atbdmath.sty macros: the breakable variant
    (fixed-size brackets, or the rows of a multi-line body)."""
    if "&" in tex or "\\\\" in tex:
        return _rows_alternative(tex)
    alt = _breakable_delimiters(tex)
    return "[{%s}]" % alt if alt else ""


def _display_macro(body: str) -> str:
    """Macro of atbdmath.sty that fits a displayed formula to the text width:
    one line breaks after its operators, an alignment (``&``, ``\\\\``) is scaled."""
    return "atbdfitdisplay" if ("&" in body or "\\\\" in body) else "atbdfitline"


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
# In table cells, long words also break between the words of a camelCase identifier.
# and after the hyphens of a long compound.
_TABLE_WORD_BREAK_RE = re.compile(_WORD_BREAK_RE.pattern + r"|(?<=[a-z])(?=[A-Z])|(?<=[a-z]-)(?=[a-z])")
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


def _set_number(kind: str, number: str) -> str:
    """LaTeX that makes the next caption of ``kind`` ("figure", "table") carry the
    number of the original document. The counter itself is set (and not only
    the printed number), so that a caption without a source number goes on
    from the numbered ones instead of from the count of the export."""
    if number.isdigit():
        return r"\setcounter{%s}{%d}" % (kind, int(number) - 1)
    return r"\renewcommand{\the%s}{%s}" % (kind, number)


def _latex_label(docname: str, ident: str) -> str:
    return r"\phantomsection\label{\detokenize{%s:%s}}" % (docname, ident)


_FORMAT_TAGS = {"sub": nodes.subscript, "sup": nodes.superscript, "em": nodes.emphasis,
                "i": nodes.emphasis, "strong": nodes.strong, "b": nodes.strong}
_FMT_TAG_RE = re.compile(r"<(/?)(\w+)[^>]*>")
"""An HTML tag of a caption (not to be confused with ``_TAG_RE``, the \\tag of a formula)."""


def _formatted(text: str) -> list[nodes.Node]:
    """Text of a caption with its <sub>, <sup>, <em>, <strong> as docutils nodes
    (other tags are dropped)."""
    root: nodes.Element = nodes.inline()
    stack = [root]
    pos = 0
    for m in _FMT_TAG_RE.finditer(text):
        if m.start() > pos:
            stack[-1] += nodes.Text(html.unescape(text[pos:m.start()]))
        pos = m.end()
        closing, name = m.group(1), m.group(2).lower()
        cls = _FORMAT_TAGS.get(name)
        if cls is None:
            continue
        if not closing:
            node = cls()
            stack[-1] += node
            stack.append(node)
        elif len(stack) > 1 and isinstance(stack[-1], cls):
            stack.pop()
    if pos < len(text):
        stack[-1] += nodes.Text(html.unescape(text[pos:]))
    return list(root.children)


_TALL = re.compile(r"\\(?:d?frac|sum|prod|int|begin|binom|sqrt)(?![A-Za-z])")


_LONG_INLINE_MATH = 30  # characters of TeX: shorter formulas keep their delimiters


def _comma_breaks(tex: str) -> str:
    """``tex`` with a break allowed after each comma of its top level (a list
    ``yaw, pitch, roll`` wraps in a narrow cell; TeX does not break at the glue
    of a formula)."""
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(tex):
        ch = tex[i]
        if ch == "\\":
            out.append(tex[i:i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        out.append(ch)
        if ch == "," and depth == 0 and not tex[i + 1:i + 2].isdigit():
            out.append("\\allowbreak{}")
        i += 1
    return "".join(out)


def _breakable(tex: str) -> str:
    """A formula of a table cell without growing delimiters when nothing tall
    needs them: TeX cannot break a line inside ``\\left[ … \\right]``, so a
    long ``[N_pol × N_pol × …]`` overflowed its cell (FH p. 30)."""
    if not _TALL.search(tex):
        tex = re.sub(r"\\(?:left|right)(?![A-Za-z])\s*(\\[{}|]|\\[A-Za-z]+|[.\[\]()|])",
                     lambda m: "" if m.group(1) == "." else m.group(1), tex)
    return _comma_breaks(tex)


_MATH_BREAK_RE = re.compile(r"(?<=,)|(?<=[=+\-])|(?<=\\times)|(?<=\\cdot)|(?<=\\ )")


def _math_atom(tex: str) -> float:
    """Width, in characters (half an em), of the longest piece of a formula that
    cannot break (between its commas and operators): letters and digits of the
    first level 1.1, of the scripts 0.75, of the second level 0.55."""
    best = width = 0.0
    depth = 0
    i = 0
    while i < len(tex):
        ch = tex[i]
        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", tex[i:])
            name = m.group(1)
            i += len(m.group(0))
            if name in ("times", "cdot", " "):
                width += 0.9 * (0.75 ** min(depth, 2))
                if depth == 0:
                    best, width = max(best, width), 0.0
                continue
            if name in ("left", "right", "bigl", "bigr", "mathrm", "mathit",
                        "text", "operatorname", "frac", "mathbf", "allowbreak"):
                continue
            width += 1.1 * (0.75 ** min(depth, 2))
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch in "^_":
            pass
        elif ch == " ":
            pass
        else:
            width += (0.55 if ch in "[](),.;" else 1.1) * (0.75 ** min(depth, 2))
            if ch in ",=+-" and depth == 0:
                best, width = max(best, width), 0.0
        i += 1
    return min(13.0, 1.1 * max(best, width))


def _caption_inlines(fragment: str) -> list[nodes.Node]:
    """Inline nodes (text, sub/superscripts, emphasis and math) for a raw-HTML
    caption fragment."""
    out: list[nodes.Node] = []
    pos = 0
    for match in _MATH_SPAN_RE.finditer(fragment):
        out.extend(_formatted(fragment[pos:match.start()]))
        tex = _tex_math(html.unescape(match.group("tex")))
        out.append(nodes.math(tex, tex))
        pos = match.end()
    out.extend(_formatted(fragment[pos:]))
    for n in out:
        n.parent = None
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


# Advance widths (1/1000 em) of Liberation Sans Regular and Bold for ASCII 32..126
# (metrics of Arial, the font of the table text of the originals).
_SANS_WIDTHS = [278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,556,556,556,278,278,584,584,584,556,1015,667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,667,778,722,667,611,722,667,944,667,667,611,278,278,278,469,556,333,556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,556,556,333,500,278,556,500,722,500,500,500,334,260,334,584]
_SANS_BOLD_WIDTHS = [278,333,474,556,556,889,722,238,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,556,556,556,333,333,584,584,584,611,975,722,722,722,722,667,611,778,722,278,556,722,611,833,722,778,667,778,722,667,611,722,667,944,667,667,611,333,278,333,584,556,333,556,611,556,611,556,333,611,611,278,278,556,278,889,611,611,611,611,389,556,333,556,556,778,556,556,500,389,280,389,584]
# Padding of the table cells (\tabcolsep, atbdfloats.sty).
_TABCOLSEP_PT = 5.4
# A "character" of the column budgets is half an em of the table font.
_CHAR_EM = 0.5


def _wlen(word: str, bold: bool = False) -> float:
    """Width of ``word`` in the table font, in characters (half an em)."""
    widths = _SANS_BOLD_WIDTHS if bold else _SANS_WIDTHS
    total = sum(widths[ord(c) - 32] if 32 <= ord(c) < 127 else 600 for c in word)
    return total / 1000.0 / _CHAR_EM


def _header_texts(tgroup: nodes.tgroup) -> list[str]:
    thead = next((c for c in tgroup.children if isinstance(c, nodes.thead)), None)
    row = next(iter(thead.findall(nodes.row)), None) if thead is not None else None
    if row is None:
        return []
    return [" ".join(entry.astext().split()) for entry in row.children]


def _cell_blocks(entry: nodes.Element, bold: bool = False) -> list[list[float]]:
    """Word widths of each paragraph (or line block) of a table cell."""
    blocks = [p for p in entry.findall(nodes.TextElement)
              if isinstance(p, (nodes.paragraph, nodes.line, nodes.literal_block))]
    texts = [p.astext() for p in blocks] or [entry.astext()]
    return [[_wlen(w, bold) for w in t.split()] or [0] for t in texts]


def _cell_atom(entry: nodes.Element, bold: bool = False) -> float:
    """Longest piece of a table cell that cannot wrap.

    Long words and inline literals get break points (``_text`` and
    ``_breakable_literal``), and formulas are scaled down to the column.
    """
    longest = 1.0
    for text in entry.findall(nodes.Text):
        parent = text.parent
        literal = isinstance(parent, nodes.literal)
        for word in str(text).split():
            if isinstance(parent, nodes.math):
                continue
            elif literal and len(word) > _LONG_TABLE_LITERAL:
                pieces = _LITERAL_BREAK_RE.split(word)
            elif not literal and len(word) > _LONG_TABLE_WORD:
                pieces = _TABLE_WORD_BREAK_RE.split(word)
            else:
                pieces = [word]
            longest = max(longest, *(_wlen(p, bold) for p in pieces))
    for math in entry.findall(nodes.math):
        longest = max(longest, _math_atom(_breakable(math.astext())))
    # A short word cut in several text nodes (``N`` + subscript + ``-N``...)
    # does not break between them.
    for block in entry.findall(nodes.TextElement):
        if isinstance(block, (nodes.paragraph, nodes.line)) and not any(
                True for _ in block.findall(nodes.math)):
            for word in block.astext().split():
                if len(word) <= _LONG_TABLE_WORD:
                    longest = max(longest, _wlen(word, bold))
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


def _column_widths(table: nodes.table, budget: float = 100.0, pad: float = 2.5,
                   centered: bool = False) -> None:
    """Replace ``colwidths-auto`` by widths proportional to the cell contents.

    ``budget`` is the number of characters (half an em) of the table font
    that fit in the table width, ``pad`` the characters taken by the padding
    and rule of each column. Every column is wide enough for its longest word
    (``_wlen``: widths of the Arial metrics), or when these do not fit, for
    its longest unbreakable piece. ``centered`` tables (the input and output
    tables of the ATBDs) also get ``atbd-centered`` (centred cells).
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
        if centered and header and header[0] == "Symbol" and header[-1] == "Description":
            classes.append("atbd-centered")
        # Cells of each row: blocks (paragraphs, list items) of word widths.
        rows: list[list[tuple[int, list[list[float]]]]] = []
        words = [1.0] * ncols
        atoms = [1.0] * ncols
        natural = [3.0] * ncols
        for row in tgroup.findall(nodes.row):
            bold = isinstance(row.parent, nodes.thead)
            cells = []
            col = 0
            for entry in row.children:
                if col >= ncols:
                    break
                span = 1 + entry.get("morecols", 0)
                if span == 1:
                    blocks = _cell_blocks(entry, bold)
                    cells.append((col, blocks))
                    longest = max((w for b in blocks for w in b), default=1)
                    words[col] = max(words[col], min(longest, 30.0))
                    atoms[col] = max(atoms[col], min(_cell_atom(entry, bold), 30.0))
                    for b in blocks:
                        natural[col] = max(natural[col], min(sum(b) + len(b) - 1, 80.0))
                col += span
            rows.append(cells)
        # Keep every column wide enough for its longest word (or, when these
        # do not fit, for its longest unbreakable piece), then give the rest
        # to the columns that save the most wrapped lines and broken words;
        # what no column can use goes to the columns not yet at their
        # one-line width, then in proportion to the widths. The weights of
        # the columns include their padding, so that the text of each gets
        # exactly its share of characters.
        text_budget = budget - pad * ncols
        minimum = [w + 0.6 for w in words]
        if sum(minimum) >= text_budget:
            minimum = [a + 0.6 for a in atoms]
        if sum(minimum) >= text_budget:
            scores = minimum
        else:
            scores = _fit_columns(rows, minimum, natural, text_budget)
        weights = [s + pad for s in scores]
        total = sum(weights)
        widths = [max(1, round(1000 * w / total)) for w in weights]
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
    if "atbd-in-minipage" in table.get("classes", []):
        return False  # a figure redrawn as a table, kept with its caption
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


_CELL_FILL = re.compile(r'<span class="atbd-cell-fill" style="background-color:#([0-9A-Fa-f]{6});'
                        r'color:#([0-9A-Fa-f]{6})">')


def _cell_fills(document: nodes.document) -> None:
    """``<span class="atbd-cell-fill" …>`` of a converted table (raw HTML, lost in
    LaTeX) -> ``\cellcolor`` and the text colour of the cell."""
    for raw in list(document.findall(nodes.raw)):
        if "html" not in raw.get("format", ""):
            continue
        text = raw.astext().strip()
        m = _CELL_FILL.fullmatch(text)
        if m:
            raw.replace_self(nodes.raw("", r"\cellcolor[HTML]{%s}\color[HTML]{%s}"
                                       % (m.group(1).upper(), m.group(2).upper()),
                                       format="latex"))
        elif text == "</span>" and isinstance(raw.parent, nodes.paragraph) \
                and any(isinstance(n, nodes.raw) and "cellcolor" in n.astext()
                        for n in raw.parent.children):
            raw.parent.remove(raw)


def _underlines(document: nodes.document) -> None:
    r"""``<u>…</u>`` of a converted page (raw HTML: the PFDs underline type
    names) -> ``\underline{…}``, when both tags sit in the same paragraph."""
    for para in list(document.findall(nodes.TextElement)):
        kids = list(para.children)
        opens = [k for k in kids if isinstance(k, nodes.raw)
                 and "html" in k.get("format", "") and k.astext().strip() == "<u>"]
        closes = [k for k in kids if isinstance(k, nodes.raw)
                  and "html" in k.get("format", "") and k.astext().strip() == "</u>"]
        if not opens or len(opens) != len(closes):
            continue
        for k in opens:
            k.replace_self(nodes.raw("", r"\underline{", format="latex"))
        for k in closes:
            k.replace_self(nodes.raw("", "}", format="latex"))


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


def _column_chars(tgroup: nodes.tgroup, budget: float, pad: float = 2.0) -> list[float]:
    """Width in characters of each column (``budget``: table width; the
    column weights include ``pad`` characters of padding)."""
    specs = [c for c in tgroup.children if isinstance(c, nodes.colspec)]
    weights = [float(s.get("colwidth", 1) or 1) for s in specs]
    total = sum(weights) or 1.0
    return [max(1.0, budget * w / total - pad) for w in weights]


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


def _split_tall_rows(table: nodes.table, budget: float, pad: float = 2.0) -> int:
    """Set each row taller than most of a page as several rows; return
    the number of rows split."""
    split = 0
    for tgroup in table.findall(nodes.tgroup):
        widths = _column_chars(tgroup, budget, pad)
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


# Redrawn figures in parts: rows up to _PARTS_ROW_PT wide (a little over the
# text width, centred), parts at a text size of 12 pt x scale between
# _PARTS_SCALES; floats of at most _PARTS_FLOAT_PT of height, the caption
# of the last one (_PARTS_CAPTION_PT) included.
_PARTS_ROW_PT = 490.0
_PART_GAP_PT = 8
_PARTS_SCALES = (0.6, 0.58, 0.56, 0.54, 0.52, 0.51, 0.505)  # text of 6 pt or more
_PARTS_LABEL_PT = 22.0
_PARTS_FLOAT_PT = 0.93 * _TEXT_HEIGHT_PT  # 0.95 let a part and its caption reach the page foot (L1 SAR Fig. 4)
_PARTS_CAPTION_PT = 40.0  # a caption of up to two lines and its skip


def _caption_pt(text: str) -> float:
    """Height kept for a figure caption under its parts: the label and text at
    11 pt (15.8 pt leading, ~80 characters a line), plus the caption skip. A
    long caption (L1 SAR Fig. 16: five lines) would otherwise run over the
    page foot."""
    lines = -(-(len(text) + 12) // 80)
    return max(_PARTS_CAPTION_PT, 12.0 + 15.8 * lines)


# Subscripts, superscripts and small text of the labels of a Mermaid flowchart are drawn
# at 83 % of the text (the browser default), which puts them under 6 pt when
# the diagram is scaled down to the page: they are set at the size of the text.
_SCRIPT_CSS = "sub,sup,small{font-size:100%}"
_INIT_RE = re.compile(r"^%%\{init:\s*(\{.*\})\s*\}%%[ \t]*$", re.MULTILINE)


def _readable_scripts(code: str) -> str:
    """Mermaid code whose <sub> and <sup> labels print at the size of the text."""
    if not any(tag in code for tag in ("<sub>", "<sup>", "<small>")):
        return code
    match = _INIT_RE.search(code)
    config: dict[str, Any] = {}
    if match:
        try:
            config = json.loads(match.group(1))
        except ValueError:
            return code
    if "themeCSS" in config:
        return code
    config["themeCSS"] = _SCRIPT_CSS
    init = "%%{init: " + json.dumps(config) + "}%%"
    if match:
        return code[:match.start()] + init + code[match.end():]
    return init + "\n" + code


# XSD class diagrams (``classDiagram``, "direction LR") whose text would print
# under 6 pt are cut by sub-trees: a first part holds the root and the levels
# below it that fit, the types at the cut are drawn by their name only, and
# each of them heads a part of its own with its descendants.
_CLASS_RE = re.compile(r'^  class (\S+?)(?:\["(.*?)"\])? \{\n(.*?)^  \}[ \t]*\n?',
                       re.DOTALL | re.MULTILINE)
_EDGE_RE = re.compile(r"^  (\S+) --> (\S+) : (.*)$", re.MULTILINE)
# What a part may take at the smallest scale of _PARTS_SCALES (text of 6 pt).
_CLASS_PART_WIDTH_PT = 1000.0  # estimate; the real size (about 5 % less) is checked
_CLASS_PART_HEIGHT_PT = 1000.0


def _parse_classes(code: str):
    """(preamble, ids in order, {id: (label, member lines)}, edges) of a class
    diagram written by the XSD converter; None when it is some other shape."""
    if "classDiagram" not in code or "--|>" in code or "..>" in code:
        return None
    classes: dict[str, tuple[str, list[str]]] = {}
    order: list[str] = []
    first = None
    for match in _CLASS_RE.finditer(code):
        first = match.start() if first is None else first
        cid = match.group(1)
        classes[cid] = (match.group(2) or cid, match.group(3).rstrip("\n").split("\n"))
        order.append(cid)
    if len(order) < 2 or first is None:
        return None
    edges = [(a, b, label) for a, b, label in _EDGE_RE.findall(code)
             if a in classes and b in classes]
    return code[:first], order, classes, edges


def _class_box(label: str, lines: list[str]) -> tuple[float, float]:
    """Estimated (width, height) in pt of one class box (calibrated on mmdc)."""
    longest = max([len(label) * 1.1] + [len(x.strip()) for x in lines])
    return 7.0 * longest + 30, 18.0 * max(len(lines), 1) + 96


def _class_extent(classes: dict, edges: list, root: str) -> tuple[float, float]:
    """Estimated size of a left-to-right class diagram rooted at ``root``."""
    depth = {root: 0}
    for _ in range(len(classes)):
        for a, b, _label in edges:
            if a in depth and depth.get(b, -1) < depth[a] + 1:
                depth[b] = depth[a] + 1
    ranks: dict[int, list[str]] = {}
    for cid in classes:
        if cid in depth:
            ranks.setdefault(depth[cid], []).append(cid)
    width = height = 0.0
    for d, ids in sorted(ranks.items()):
        boxes = [_class_box(*classes[i]) for i in ids]
        width += max(w for w, _h in boxes)
        height = max(height, sum(h - 60 for _w, h in boxes) + 50 * (len(ids) - 1) + 60)
        if d:
            width += 60 + 7.0 * max([len(lab) for _a, b, lab in edges if b in ids] or [0])
    return width, height


def _class_part(preamble: str, classes: dict, edges: list, root: str,
                levels: int | None, leaf_cut: tuple[str, ...] = ()
                ) -> tuple[str, list[str], dict, list]:
    """Mermaid code of the part rooted at ``root``: the types of the first
    ``levels`` levels in full, the next one by name only when it has a sub-tree
    of its own (``None``: everything). Returns the code, the types at the cut,
    and the (classes, edges) drawn, for the estimate."""
    children: dict[str, list[tuple[str, str]]] = {}
    for a, b, label in edges:
        children.setdefault(a, []).append((b, label))
    drawn: dict[str, tuple[str, list[str]]] = {}
    shown_edges: list[tuple[str, str, str]] = []
    cut: list[str] = []
    stack = [(root, 0)]
    seen: set[str] = set()
    while stack:
        cid, depth = stack.pop(0)
        if cid in seen:
            continue
        seen.add(cid)
        label, lines = classes[cid]
        if levels is not None and depth == levels and (children.get(cid) or cid in leaf_cut):
            drawn[cid] = (label, [])
            if cid != root:
                cut.append(cid)
            continue
        drawn[cid] = (label, lines)
        for child, edge_label in children.get(cid, []):
            shown_edges.append((cid, child, edge_label))
            stack.append((child, depth + 1))
    out = [preamble.rstrip("\n")]
    for cid, (label, lines) in drawn.items():
        title = cid if label == cid else '%s["%s"]' % (cid, label)
        if lines:
            out.append("  class %s {\n%s\n  }" % (title, "\n".join(lines)))
        else:
            out.append("  class %s" % title)
    out.extend("  %s --> %s : %s" % e for e in dict.fromkeys(shown_edges))
    return "\n".join(out) + "\n", cut, drawn, shown_edges


def _split_class_diagram(code: str, width: float = _CLASS_PART_WIDTH_PT,
                         height: float = _CLASS_PART_HEIGHT_PT) -> list[str] | None:
    """The sub-tree parts of a class diagram (see above); None if it is not a
    tree-shaped XSD diagram or fits as it is."""
    parsed = _parse_classes(code)
    if parsed is None:
        return None
    preamble, order, classes, edges = parsed
    levels_max = len(order)
    parts: list[str] = []
    todo = [order[0]]
    done: set[str] = set()
    while todo:
        root = todo.pop(0)
        if root in done:
            continue
        done.add(root)
        chosen = None
        for levels in [None, *range(levels_max, 0, -1)]:
            candidate = _class_part(preamble, classes, edges, root, levels)
            w, h = _class_extent(candidate[2], candidate[3], root)
            chosen = candidate
            if w <= width and h <= height:
                break
        else:
            # the root and its direct children are still too big: the largest
            # childless ones are drawn apart as well, until the rest fits
            leaves = sorted((c for a, c, _l in edges if a == root and not any(
                x == c for x, _y, _z in edges)),
                key=lambda c: -len(classes[c][1]) * max(len(x) for x in classes[c][1] or [""]))
            for count in range(1, len(leaves) + 1):
                candidate = _class_part(preamble, classes, edges, root, 1, tuple(leaves[:count]))
                w, h = _class_extent(candidate[2], candidate[3], root)
                chosen = candidate
                if w <= width and h <= height:
                    break
        parts.append(chosen[0])
        todo[:0] = chosen[1]
    return parts if len(parts) > 1 else None


# A flowchart drawn in lanes (``subgraph``) whose text would print under 6 pt
# is cut between lanes: each part keeps some lanes, and a lane left out that is
# linked to a kept one stays as a box of that kind at the end of the link.
_SUBGRAPH_RE = re.compile(r"^(\s*)subgraph\s+([A-Za-z0-9_]+)")
_FLOW_NODE_RE = re.compile(r'^\s*([A-Za-z0-9_]+)(\[.*\])(?::::([A-Za-z0-9_]+))?\s*$')
_FLOW_EDGE_RE = re.compile(
    r'^(\s*)([A-Za-z0-9_]+)(\s+(?:-->|==>|~~~|---|--\s+".*"\s+-->|--\s+".*"\s+==>)\s+)'
    r'([A-Za-z0-9_]+)\s*$')
_STYLE_RE = re.compile(r"^\s*style\s+([A-Za-z0-9_]+)\s")
_LINKSTYLE_RE = re.compile(r"^\s*linkStyle\s+(\d+)(\s.*)$")


class _Lane:
    def __init__(self, ident: str, header: str) -> None:
        self.ident, self.header = ident, header
        self.items: list[Any] = []  # lines (str) and nested _Lane


def _parse_flow(code: str):
    """(preamble, top-level lanes, loose nodes, edges, styles, link styles) of a
    flowchart of the converter's layout; None when it has no lane."""
    lines = code.split("\n")
    first = next((i for i, x in enumerate(lines) if _SUBGRAPH_RE.match(x)), None)
    if first is None or not re.search(r"^flowchart\s", code, re.MULTILINE):
        return None
    preamble: list[str] = []
    top: list[Any] = []
    loose: dict[str, str] = {}
    stack: list[_Lane] = []
    edges: list[tuple[str, str, str, str]] = []
    styles: list[str] = []
    links: dict[int, str] = {}
    for line in lines:
        match = _SUBGRAPH_RE.match(line)
        if match:
            lane = _Lane(match.group(2), line)
            (stack[-1].items if stack else top).append(lane)
            stack.append(lane)
        elif line.strip() == "end" and stack:
            stack.pop()
        elif _FLOW_NODE_RE.match(line):
            if stack:
                stack[-1].items.append(line)
            else:
                loose[_FLOW_NODE_RE.match(line).group(1)] = line
        elif _FLOW_EDGE_RE.match(line):
            m = _FLOW_EDGE_RE.match(line)
            edges.append((m.group(2), m.group(4), line, ""))
        elif _LINKSTYLE_RE.match(line):
            m = _LINKSTYLE_RE.match(line)
            links[int(m.group(1))] = m.group(2)
        elif _STYLE_RE.match(line):
            styles.append(line)
        elif not top and not loose and not stack:
            preamble.append(line)
    return preamble, top, loose, edges, styles, links


def _flow_lanes(items: list[Any]) -> list[_Lane]:
    """Lanes in reading order: a subgraph holding nodes, or no other subgraph
    (a wrapper of lanes only is looked through)."""
    out: list[_Lane] = []
    for item in items:
        if isinstance(item, _Lane):
            if any(isinstance(x, str) for x in item.items) or not any(
                    isinstance(x, _Lane) for x in item.items):
                out.append(item)
            else:
                out.extend(_flow_lanes(item.items))
    return out


def _lane_nodes(lane: _Lane) -> dict[str, str]:
    nodes_: dict[str, str] = {}
    for item in lane.items:
        if isinstance(item, _Lane):
            nodes_.update(_lane_nodes(item))
        else:
            nodes_[_FLOW_NODE_RE.match(item).group(1)] = item
    return nodes_


def _emit_lane(lane: _Lane, keep: set[str]) -> list[str]:
    """Lines of a lane kept whole, or of the wrapper around the kept lanes."""
    if lane.ident in keep:
        return _lane_lines(lane)
    inner: list[str] = []
    for item in lane.items:
        if isinstance(item, _Lane):
            inner.extend(_emit_lane(item, keep))
    return [lane.header, *inner, "  end"] if inner else []


def _lane_lines(lane: _Lane) -> list[str]:
    out = [lane.header]
    for item in lane.items:
        out.extend(_lane_lines(item) if isinstance(item, _Lane) else [item])
    return [*out, "  end"]


def _flow_owners(parsed, lanes: list[_Lane]) -> dict[str, int]:
    """Index of the lane each node belongs to; a loose node goes with the first
    lane it is linked to."""
    _pre, _top, loose, edges, _styles, _links = parsed
    owner = {n: i for i, lane in enumerate(lanes) for n in _lane_nodes(lane)}
    pending = set(loose) - set(owner)
    for _round in range(len(pending) + 1):
        for node in sorted(pending):
            near = [owner[o] for a, b, _l, _x in edges
                    for n, o in ((a, b), (b, a)) if n == node and o in owner]
            if near:
                owner[node] = min(near)
        pending -= set(owner)
    for node in pending:
        owner[node] = 0
    return owner


def _flow_part(parsed, lanes: list[_Lane], owner: dict[str, int], keep: set[int]) -> str:
    """Mermaid code of the flowchart restricted to the lanes ``keep`` (indexes)
    and the loose nodes that go with them."""
    preamble, top, loose, edges, styles, links = parsed
    names = {lanes[i].ident for i in keep}
    inside = {n for n, i in owner.items() if i in keep}
    everything = dict(loose)
    for lane in lanes:
        everything.update(_lane_nodes(lane))
    body: list[str] = [loose[n] for n in loose if n in inside]
    for item in top:
        if isinstance(item, _Lane):
            body.extend(_emit_lane(item, names))
    stubs: dict[str, str] = {}
    kept_edges: list[str] = []
    kept_index: dict[int, int] = {}
    for index, (src, dst, line, _x) in enumerate(edges):
        ends = [e in inside for e in (src, dst)]
        if not any(ends):
            continue
        for end, ok in ((src, ends[0]), (dst, ends[1])):
            if not ok and end in everything and end not in stubs:
                m = _FLOW_NODE_RE.match(everything[end])
                stubs[end] = "  stub_%s%s%s" % (
                    end, m.group(2), ":::" + m.group(3) if m.group(3) else "")
        m = _FLOW_EDGE_RE.match(line)
        a = src if ends[0] else "stub_" + src
        b = dst if ends[1] else "stub_" + dst
        kept_index[index] = len(kept_edges)
        kept_edges.append("%s%s%s%s" % (m.group(1), a, m.group(3), b))
    present = set(inside) | {m.group(2) for m in map(_SUBGRAPH_RE.match, body) if m}
    out = [*preamble, *body, *stubs.values(), *kept_edges]
    out += [x for x in styles if _STYLE_RE.match(x).group(1) in present]
    out += ["  linkStyle %d%s" % (kept_index[i], rest)
            for i, rest in sorted(links.items()) if i in kept_index]
    return "\n".join(out) + "\n"


def _split_flow_lanes(code: str, count: int) -> list[str] | None:
    """The flowchart in ``count`` parts of consecutive lanes (with the nodes
    between them), the parts holding about as many nodes as one another; None
    if it has no such lanes."""
    parsed = _parse_flow(code)
    if parsed is None:
        return None
    lanes = _flow_lanes(parsed[1])
    if len(lanes) < count or count < 2:
        return None
    owner = _flow_owners(parsed, lanes)
    weights = [sum(1 for i in owner.values() if i == k) for k in range(len(lanes))]
    from itertools import combinations
    best, cuts = None, None
    for cut in combinations(range(1, len(lanes)), count - 1):
        edges_ = [0, *cut, len(lanes)]
        sums = [sum(weights[a:b]) for a, b in zip(edges_, edges_[1:])]
        if best is None or max(sums) < best:
            best, cuts = max(sums), edges_
    return [_flow_part(parsed, lanes, owner, set(range(a, b)))
            for a, b in zip(cuts, cuts[1:])]


def _pack_parts(sizes: list[tuple[float, float]], scale: float,
                each_caption: bool = False, caption_pt: float = _PARTS_CAPTION_PT
                ) -> tuple[list[float], list[list[list[int]]]]:
    """Parts (width, height) in pt at a common scale: per-part scales (a part
    wider than a row is scaled down to it), rows of one or two neighbours,
    rows grouped in floats."""
    scales = [min(scale, _PARTS_ROW_PT / w) for w, _h in sizes]
    # a part is not taller than a page with its label and caption (every
    # float carries the caption)
    room = _PARTS_FLOAT_PT - _PARTS_LABEL_PT - caption_pt
    scales = [min(sc, room / h) for sc, (_w, h) in zip(scales, sizes)]
    rows: list[list[int]] = []
    i = 0
    while i < len(sizes):
        if i + 1 < len(sizes):
            pair = (sizes[i][0] * scales[i] + sizes[i + 1][0] * scales[i + 1]
                    + _PART_GAP_PT)
            if pair <= _PARTS_ROW_PT:
                rows.append([i, i + 1])
                i += 2
                continue
        rows.append([i])
        i += 1
    heights = [max(sizes[k][1] * scales[k] for k in row) + _PARTS_LABEL_PT for row in rows]
    groups: list[list[list[int]]] = []
    used = 0.0
    room = _PARTS_FLOAT_PT - caption_pt  # every float carries a caption
    for row, height in zip(rows, heights):
        if groups and used + height <= room:
            groups[-1].append(row)
            used += height
        else:
            groups.append([row])
            used = height
    return scales, groups


def _plan_parts(sizes: list[tuple[float, float]], each_caption: bool = False,
                caption_pt: float = _PARTS_CAPTION_PT
                ) -> tuple[float, list[float], list[list[list[int]]]]:
    """The largest scale of _PARTS_SCALES giving the fewest floats."""
    plans = [(s, *_pack_parts(sizes, s, each_caption, caption_pt)) for s in _PARTS_SCALES]
    fewest = min(len(groups) for _s, _sc, groups in plans)
    return next(p for p in plans if len(p[2]) == fewest)



class AtbdLatexContent(SphinxPostTransform):
    """Rewrite the web-only constructs of the converted documents for print."""

    default_priority = 50

    def is_supported(self) -> bool:
        return self.app.builder.format == "latex"

    def run(self, **kwargs: Any) -> None:
        _cell_fills(self.document)  # before the raw HTML is dropped
        _underlines(self.document)
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
        # Characters (half an em) of the table font in the table width; wide
        # tables also use the heading indent (35 pt; L1 SAR: 22.7 pt more on
        # the right, see atbdwideextra). Each column pays 2 \tabcolsep and a rule.
        size = float(self.config.atbd_table_font_pt)
        width_pt = 449.9 + (35.1 if wide else 0.0)
        if wide and not pfd and "atbd-l1-sar" in self.env.docname:
            width_pt += 22.7
        budget = width_pt / (_CHAR_EM * size)
        pad = (2 * _TABCOLSEP_PT + 0.5) / (_CHAR_EM * size)
        for table, _docname in found["tables"]:
            _header_only(table)
            _column_widths(table, budget, pad, centered=not pfd)
            if _can_break(table):
                for cls in ("longtable", "atbd-wide") if wide else ("longtable",):
                    if cls not in table["classes"]:
                        table["classes"].append(cls)
                if pfd and "atbd-caption-below" not in table["classes"]:
                    table["classes"].append("atbd-caption-below")
        for table, _docname in found["tables"]:
            lead = _lead_paragraph(table)
            if lead is None:
                continue
            if not any(isinstance(c, nodes.title) for c in table.children):
                # Keep a short lead-in ("Inputs", "... as follows:") with the
                # table: it becomes the first line of the table (head of the
                # first page), as the page cannot then break between them.
                # (Left apart, the page break before it was taken while the
                # table output routine was already active, and the repeated
                # header of the table was left alone at the top of the page.)
                inlines = list(lead.children)
                for child in inlines:
                    lead.remove(child)
                table.insert(0, nodes.title("", "", *inlines))
                table["classes"].append("atbd-lead")
                lead.parent.remove(lead)
            else:
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
                    "", r"\atbdfitmath{\(%s\)}" % _breakable(math.astext()),
                    format="latex"))
            elif (math.parent is not None and isinstance(math.parent, nodes.paragraph)
                  and len(math.astext()) > _LONG_INLINE_MATH
                  and _LEFT_ANY_RE.search(math.astext())
                  and r"\middle" not in math.astext()):
                # A long formula of running text with \left...\right cannot
                # break: it stuck out of the column (AGB p. 27, x1 = 583).
                tex = _breakable(math.astext())
                if tex != math.astext():
                    math.replace_self(nodes.math(tex, tex))
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
                _split_tall_rows(table, budget, pad)
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
        in_table = _in_table(text)
        parts = [p for p in (_TABLE_WORD_BREAK_RE if in_table else _WORD_BREAK_RE).split(clean) if p]
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
                if "latex" in child.get("format", "").split():
                    # a coloured text tree (fancyvrb Verbatim) is the figure itself
                    # (L2 PFD Fig. 2, relecture W3-14)
                    visuals.append(child)
                continue  # <div class="atbd-figure-content"> wrappers, comments
            elif isinstance(child, nodes.paragraph) and child.children and all(
                isinstance(c, nodes.image)
                or (isinstance(c, nodes.Text) and not c.astext().strip())
                for c in child.children
            ):
                visuals.extend(c for c in child.children if isinstance(c, nodes.image))
            elif isinstance(child, nodes.image):
                visuals.append(child)
            elif isinstance(child, nodes.literal_block) and (
                    "atbd-drawn-figure" in container.get("classes", [])
                    or "atbd-text-figure" in container.get("classes", [])):
                visuals.append(child)  # a redrawn folder tree: inside the float, captioned
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

        for visual in visuals:
            if mermaid_cls is not None and isinstance(visual, mermaid_cls):
                visual["code"] = _readable_scripts(visual["code"])
        if has_mermaid and len(visuals) == 1 and isinstance(visuals[0], mermaid_cls):
            parts = self._class_parts(visuals[0])
            if parts:
                return self._figure_parts(container, docname, parts, caption, extra,
                                          each_caption=True)
            parts = self._lane_parts(visuals[0])
            if parts:
                return self._figure_parts(container, docname, parts, caption, extra,
                                          each_caption=True)
        original = False
        single_share = _MERMAID_SHARE
        if has_mermaid and len(visuals) == 1 and isinstance(visuals[0], mermaid_cls):
            fallback, single_share = self._fit_mermaid(visuals[0], container)
            if fallback:
                visuals, original = fallback, True
            if single_share > _MERMAID_SHARE and caption is not None:
                # a figure on a page of its own leaves room for its caption
                # (about 95 characters per line, 3 % of the text height each)
                text = "".join(n.astext() for n in caption[1])
                lines = -(-len(text) // 95)
                single_share = min(single_share, 0.98 - 0.03 * max(lines, 1))

        if ("atbd-text-figure" in container.get("classes", []) and caption is not None
                and visuals and all(isinstance(v, (nodes.raw, nodes.literal_block))
                                    for v in visuals)):
            return self._text_figure(container, docname, visuals, caption, extra)

        tables = [x for x in extra if isinstance(x, nodes.table)]
        if (not visuals and len(tables) == 1 and caption is not None
                and "atbd-drawn-figure" in container.get("classes", [])
                and "longtable" not in tables[0].get("classes", [])):
            return self._table_figure(container, docname, tables[0], caption, extra)

        if (has_mermaid and len(visuals) > 1 and "atbd-drawn-figure" in container.get("classes", [])
                and all(isinstance(v, mermaid_cls) for v in visuals)):
            return self._figure_parts(container, docname, visuals, caption, extra)

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
                setup += _set_number("figure", number)
        else:
            setup += "".join(_latex_label(docname, i) for i in container["ids"])
        figure += nodes.raw("", setup, format="latex")
        for visual in visuals:
            visual.parent.remove(visual)
            figure += visual
        if caption is not None:
            figure += nodes.caption("", "", *inlines)
        landscape = False
        if (mermaid_only and not original and len(visuals) == 1
                and "atbd-drawn-figure" not in container.get("classes", [])):
            landscape = self._wants_landscape(visuals[0], single_share)
        if landscape:
            # a diagram too wide for its text to stay readable on a portrait
            # page: a landscape page of its own (pdflscape, atbdfloats.sty)
            setup += r"\csname spx@image@maxheight\endcsname=%.3f\textheight" % _LANDSCAPE_SHARE
            figure.children[0] = nodes.raw("", setup, format="latex")
            figure["classes"].append("atbd-landscape")
            container.replace_self([
                nodes.raw("", r"\begin{landscape}\atbdfigurehere", format="latex"),
                figure,
                nodes.raw("", r"\end{landscape}", format="latex"), *extra])
            return figure
        if mermaid_only and not original and len(visuals) == 1:
            box = self._mermaid_box(visuals[0]["code"], visuals[0].get("options", {}))
            if box is not None and _MERMAID_FONT_PT * min(
                    _MERMAID_SCALE, _TEXT_WIDTH_PT / box[0],
                    single_share * _TEXT_HEIGHT_PT / box[1]) < _LANDSCAPE_BELOW_PT:
                # text under 6 pt: the diagram may extend into the margins, as
                # wide tables do (centred, up to _PARTS_ROW_PT)
                figure.children[0] = nodes.raw("", setup.replace(
                    r"\def\atbdfigmaxwidth{\linewidth}",
                    r"\def\atbdfigmaxwidth{%dpt}" % _PARTS_ROW_PT), format="latex")
        container.replace_self([figure, *extra])
        return figure

    def _wants_landscape(self, diagram: nodes.Element, share: float) -> bool:
        """True when a landscape page gives a Mermaid diagram readably larger text."""
        box = self._mermaid_box(diagram["code"], diagram.get("options", {}))
        if box is None:
            return False
        width, tall = box
        portrait = min(_MERMAID_SCALE, _TEXT_WIDTH_PT / width, share * _TEXT_HEIGHT_PT / tall)
        land = min(_MERMAID_SCALE, _TEXT_HEIGHT_PT / width, _LANDSCAPE_SHARE * _TEXT_WIDTH_PT / tall)
        return _MERMAID_FONT_PT * portrait < _LANDSCAPE_BELOW_PT and land > 1.15 * portrait

    def _text_figure(self, container: nodes.container, docname: str,
                     visuals: list[nodes.Node], caption: Any,
                     extra: list[nodes.Node]) -> nodes.Node:
        """A figure typed as text (the product header tree of the PFDs, 80 to
        120 lines): no float, which cannot break across pages and overflowed the
        page foot; the block then its caption below it, as in the original."""
        number, inlines = caption
        label = "".join(_latex_label(docname, i) for i in container["ids"])
        cap = nodes.paragraph("", "")
        cap += nodes.raw("", (_set_number("figure", number) if number else "")
                         + r"\captionof{figure}{", format="latex")
        cap.extend(inlines)
        cap += nodes.raw("", "}" + label, format="latex")
        for v in visuals:
            v.parent.remove(v)
        out = [*visuals, cap, *extra]
        container.replace_self(out)
        return cap

    def _table_figure(self, container: nodes.container, docname: str, table: nodes.table,
                      caption: Any, extra: list[nodes.Node]) -> nodes.Node:
        """A figure redrawn as a table (AGB/FD Fig. 4 and 5): no float, the table
        then its caption below it, as in the original, on the same page (a float
        holding only the caption left the table behind)."""
        number, inlines = caption
        label = "".join(_latex_label(docname, i) for i in container["ids"])
        head = nodes.raw("", r"\par\noindent\begin{minipage}{\linewidth}\centering ",
                         format="latex")
        cap = nodes.paragraph("", "")
        cap += nodes.raw("", (_set_number("figure", number) if number else "")
                         + r"\captionof{figure}{", format="latex")
        cap.extend(inlines)
        cap += nodes.raw("", "}" + label, format="latex")
        tail = nodes.raw("", r"\end{minipage}\par\medskip", format="latex")
        others = [x for x in extra if x is not table]
        table["classes"].append("atbd-in-minipage")
        table.parent.remove(table)
        container.replace_self([head, table, cap, tail, *others])
        return table

    def _figure_parts(self, container: nodes.container, docname: str,
                      visuals: list[nodes.Node], caption: Any,
                      extra: list[nodes.Node], each_caption: bool = False) -> nodes.figure:
        """A complex figure redrawn in parts (a), (b)...: the parts are set side
        by side where two fit in a row, in as few floats (pages) as the text
        size allows, the caption under the last one (see _pack_parts).
        ``each_caption``: every float carries the caption, with the letters of
        its parts, under the same figure number (sub-trees of an XSD diagram)."""
        sizes = []
        for visual in visuals:
            if not each_caption:
                self._fit_mermaid(visual, container)  # may lay the part out top to bottom
            box = self._mermaid_box(visual["code"], visual.get("options", {}))
            if box is None:
                sizes = []
                break
            sizes.append(box)
        if not sizes:
            return self._figure_parts_stacked(container, docname, visuals, caption, extra)
        caption_text = ("".join(n.astext() for n in caption[1])
                        if caption is not None else "")
        scale, scales, groups = _plan_parts(sizes, each_caption, _caption_pt(caption_text))
        figures = []
        for g, rows in enumerate(groups):
            figure = nodes.figure("")
            last_group = g == len(groups) - 1
            setup = ""
            if caption is not None and caption[0]:
                setup += _set_number("figure", caption[0])
            figure += nodes.raw("", setup, format="latex")
            for r, row in enumerate(rows):
                row_tex = r"\par\noindent\makebox[\linewidth][c]{" if r == 0 else (
                    r"\par\medskip\noindent\makebox[\linewidth][c]{")
                figure += nodes.raw("", row_tex, format="latex")
                for k, i in enumerate(row):
                    w, h = sizes[i]
                    s = scales[i]
                    head = (r"\hspace{%dpt}" % _PART_GAP_PT if k else "") + (
                        r"\begin{minipage}[b]{%.1fpt}\centering{\small (%s)}\par\smallskip "
                        r"\csname spx@image@maxheight\endcsname=%.1fpt "
                        r"\def\atbdfigscale{%.4f}\def\atbdfigmaxwidth{\linewidth}"
                        % (w * s * 1.004 + 1, chr(97 + i), h * s * 1.01 + 2, s))
                    figure += nodes.raw("", head, format="latex")
                    if visuals[i].parent is not None:
                        visuals[i].parent.remove(visuals[i])
                    figure += visuals[i]
                    figure += nodes.raw("", r"\end{minipage}", format="latex")
                figure += nodes.raw("", "}", format="latex")
            if caption is not None:
                inlines = [n.deepcopy() for n in caption[1]]
                if each_caption:
                    letters = sorted(i for row in rows for i in row)
                    inlines.append(nodes.Text(" (%s)" % (
                        chr(97 + letters[0]) if len(letters) == 1 else
                        "%s%s%s" % (chr(97 + letters[0]),
                                    "–" if letters[-1] - letters[0] == len(letters) - 1 else ", ",
                                    chr(97 + letters[-1])))))
                elif g > 0:
                    inlines.append(nodes.Text(" (continued)"))
                figure += nodes.caption("", "", *inlines)
                if g == 0:
                    figure += nodes.raw("", "".join(_latex_label(docname, x)
                                                    for x in container["ids"]), format="latex")
            figures.append(figure)
        barrier = nodes.raw("", r"\FloatBarrier", format="latex")
        container.replace_self([barrier, *figures, *extra])
        return figures[0]

    def _class_parts(self, diagram: nodes.Element) -> list[nodes.Element] | None:
        """An XSD class diagram whose text would print under 6 pt, cut by
        sub-trees into mermaid nodes (None: print it as it is)."""
        code, options = diagram["code"], diagram.get("options", {})
        if "classDiagram" not in code:
            return None
        font = self._mermaid_font(code, options, _MERMAID_SHARE)
        if font is None or font >= _LANDSCAPE_BELOW_PT:
            return None
        width = _CLASS_PART_WIDTH_PT
        for _attempt in range(4):
            codes = _split_class_diagram(code, width)
            if not codes:
                return None
            boxes = [self._mermaid_box(c, options) for c in codes]
            if any(b is None for b in boxes):
                return None
            worst = max(b[0] for b in boxes)
            if worst <= _PARTS_ROW_PT / 0.508 and all(
                    b[1] <= _CLASS_PART_HEIGHT_PT * 1.05 for b in boxes):
                break
            width *= 0.88  # the estimate was short: cut finer
        parts = []
        for part_code in codes:
            part = diagram.deepcopy()
            part["code"] = part_code
            parts.append(part)
        return parts

    def _lane_parts(self, diagram: nodes.Element) -> list[nodes.Element] | None:
        """A flowchart in lanes whose text would print under 6 pt even on a page
        of its own, cut between lanes (None: print it as it is)."""
        code, options = diagram["code"], diagram.get("options", {})
        if not code.lstrip().startswith("%%{init") and "flowchart" not in code:
            return None
        if _parse_flow(code) is None:
            return None
        font = self._mermaid_font(code, options, _MERMAID_TALL_SHARE)
        if font is None or font >= _LANDSCAPE_BELOW_PT:
            return None
        best = None
        for count in (2, 3, 4):
            candidate = _split_flow_lanes(code, count)
            if not candidate:
                break
            boxes = [self._mermaid_box(c, options) for c in candidate]
            if any(b is None for b in boxes):
                return None
            _scale, scales, groups = _plan_parts(boxes, True)
            if _MERMAID_FONT_PT * min(scales) < _LANDSCAPE_BELOW_PT:
                continue
            if best is None or len(groups) < best[0]:
                best = (len(groups), candidate)
        if best is None:
            return None
        codes = best[1]
        parts = []
        for part_code in codes:
            part = diagram.deepcopy()
            part["code"] = part_code
            parts.append(part)
        return parts

    def _figure_parts_stacked(self, container: nodes.container, docname: str,
                              visuals: list[nodes.Node], caption: Any,
                              extra: list[nodes.Node]) -> nodes.figure:
        """_figure_parts when a part cannot be measured: one float per part."""
        figures = []
        last = len(visuals) - 1
        for i, visual in enumerate(visuals):
            _images, share = self._fit_mermaid(visual, container)
            figure = nodes.figure("")
            setup = (r"\csname spx@image@maxheight\endcsname=%.3f\textheight" % share
                     + r"\def\atbdfigscale{%s}\def\atbdfigmaxwidth{\linewidth}" % _MERMAID_SCALE)
            if i == 0:
                setup += "".join(_latex_label(docname, x) for x in container["ids"])
            if i == last and caption is not None and caption[0]:
                setup += _set_number("figure", caption[0])
            setup += r"{\small (%s)}\par\smallskip " % chr(97 + i)
            figure += nodes.raw("", setup, format="latex")
            visual.parent.remove(visual)
            figure += visual
            if i == last and caption is not None:
                figure += nodes.caption("", "", *caption[1])
            figures.append(figure)
        barrier = nodes.raw("", r"\FloatBarrier", format="latex")
        container.replace_self([barrier, *figures, *extra])
        return figures[0]

    def _mermaid_box(self, code: str, options: Any) -> tuple[float, float] | None:
        """(width, height) in pt of the PDF of a Mermaid diagram; None if unknown."""
        from types import SimpleNamespace

        from sphinxcontrib.mermaid import MermaidError, render_mm

        try:
            _relfn, outfn = render_mm(SimpleNamespace(builder=self.app.builder),
                                      code, options, "pdf")
            if outfn is None:
                return None
            match = _MEDIABOX_RE.search(Path(outfn).read_bytes())
        except MermaidError:
            return None
        except Exception:  # noqa: BLE001 - keep the stacked layout
            return None
        if not match:
            return None
        x0, y0, x1, y1 = (float(v) for v in match.groups())
        return (x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None

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
                    "", r"\begingroup" + _set_number("table", number),
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
                inner = text.split(r"\begin{aligned}", 1)[1].split(r"\end{aligned}", 1)[0]
                fitted = text.replace(
                    r"\begin{aligned}", r"\atbdfitdisplay%s{" % _rows_alternative(inner)
                ).replace(r"\end{aligned}", "}")
            elif r"\begin{aligned}" not in text:
                fitted = _EQUATION_BODY_RE.sub(
                    lambda m: "%s\\%s%s{%s}%s" % (
                        m.group(1), _display_macro(m.group(2)),
                        _alternative(m.group(2)), m.group(2), m.group(3)),
                    text, count=1)
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
        # \atbdfitline / \atbdfitdisplay (atbdmath.sty) fit the body to the text
        # width (broken after its operators, or scaled when aligned).
        latex = (
            f"{labels}\n\\begin{{equation}}{label}\n\\{_display_macro(body)}{_alternative(body)}"
            f"{{\n{body}\n"
            f"}}\\tag{{{tag.group('num')}}}\n\\end{{equation}}"
        )
        math.replace_self(nodes.raw("", latex, format="latex"))

    # -- raw HTML --------------------------------------------------------
    def _raw_html(self, raw: nodes.raw, docname: str) -> None:
        if "html" not in raw.get("format", "").split():
            return
        if _BR_RE.fullmatch(raw.astext().strip()):
            # <br> of a converted table cell: a line break, as in the original
            # (dropped, it glued the words: "ones.The")
            raw.replace_self(nodes.raw("", r"\newline{}", format="latex"))
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
    app.add_latex_package("placeins")
    app.add_latex_package("colortbl")  # \cellcolor of a table colour key  # \FloatBarrier around a figure in parts
    # Tables also use the left indent of the headings (as in the original
    # PFDs and L1 SAR ATBD; the other ATBDs keep them in the text width).
    app.add_config_value("atbd_wide_tables", False, "env", bool)
    app.add_config_value("atbd_table_font_pt", 9.0, "env", float)
    app.connect("config-inited", _default_figure_align)
    app.connect("build-finished", _write_trims)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
