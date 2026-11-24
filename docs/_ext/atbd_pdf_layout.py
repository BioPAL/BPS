# SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS
# SPDX-License-Identifier: Apache-2.0
"""PDF (LaTeX) layout of a single Science Guide document (ATBD / PFD).

``conf.py`` calls :func:`latex_config` when ``SPHINX_ATBD_PDF=<folder>`` is set;
the extension part (:func:`setup`) reshapes the document front matter for the
LaTeX builder only:

* removes web-only elements of ``index.md`` (back link to the Science Guide,
  "Draft" / "Authoritative document" admonitions, horizontal rules);
* turns the "Document information" and "Change history" dropdowns into
  front-matter pages, followed by the table of contents.

The page style itself (cover, header, footer, fonts, headings) lives in
``_templates/latex/atbdlayout.sty`` and the packages it loads.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx.transforms.post_transforms import SphinxPostTransform

# Logos as printed in the approved documents (images taken from their PDFs).
_LOGOS = ("Aresys_doc_cover.png", "Aresys_doc_header.png", "ESA_doc_cover.png", "BIOMASS_mission.png")
# Layout packages (_templates/latex): atbdlayout loads the others.
_STY_FILES = ("atbdlayout", "atbdbody", "atbdfloats", "atbdmath", "atbdcover", "atbdmeta")
# Front-matter headings (document information, change history, contents) as
# printed by the originals; the L1 Stack ATBD uses other labels.
_FRONT_LABELS = ("Document information", "Change history", "Table of contents")
_FRONT_LABELS_BY_DOC = {
    "atbd-l1-stack": ("Document Information", "Change Log", "Table of Contents"),
}
_FRONT_HEADINGS = ("document information", "change history")
# Cover titles of the originals, one entry per printed line (48 pt, as broken
# by Word in the originals).
_COVER_TITLES = {
    "atbd-l2-agb": ("BIOMASS Above", "Ground Biomass", "Product ATBD"),
    "atbd-l2-fd": ("BIOMASS Forest", "Disturbance Product", "ATBD"),
    "atbd-l2-fh": ("BIOMASS Forest", "Height Product ATBD"),
    "atbd-l1-sar": ("BIOMASS L1 SAR", "Product ATBD"),
    "atbd-l1-stack": ("BIOMASS L1c Stack", "Product ATBD"),
    "pfd-l1": ("BIOMASS", "L1a/b/c Products", "Format Specification"),
    "pfd-l2-agb": ("BIOMASS Above", "Ground Biomass", "Products Format", "Specification"),
    "pfd-l2-fd": ("BIOMASS Forest", "Disturbance Products", "Format Specification"),
    "pfd-l2-fh": ("BIOMASS Forest", "Height Products", "Format Specification"),
}
# Page header title of the originals where it differs from the title of the
# Document information (the L1 Stack ATBD page header is "BPS L1 STACK ATBD").
_HEADER_TITLES = {
    "atbd-l1-stack": "BPS L1 STACK ATBD - Algorithms Theoretical Baseline Document",
}
# Cover geometry measured on the originals (pt from the top-left corner of
# the page): top of the Aresys logo, top of the title's first line, top of
# the BIOMASS mission logo, left edge and top of the "Ref:" line of the
# reference box.
_COVER_LAYOUT = {
    "atbd-l2-agb": dict(logo_y=86.7, title_y=242.1, mission=(110.0, 448.0), ref=(337.6, 734.5)),
    "atbd-l2-fd": dict(logo_y=86.7, title_y=242.1, mission=(110.0, 448.0), ref=(337.6, 734.5)),
    "atbd-l2-fh": dict(logo_y=86.7, title_y=269.7, mission=(110.0, 444.2), ref=(340.6, 733.8)),
    "atbd-l1-sar": dict(logo_y=86.6, title_y=269.7, mission=(110.1, 445.7), ref=(317.2, 733.9)),
    # L1 Stack: the Aresys logo is 36 pt lower, the ESA logo only 28 pt
    "atbd-l1-stack": dict(logo_y=122.8, esa_bottom=179.0, title_y=269.7, mission=(91.9, 486.4),
                          ref=(294.7, 742.9)),
    "pfd-l1": dict(logo_y=86.7, title_y=242.1, mission=(103.8, 445.7), ref=(124.5, 734.2)),
    "pfd-l2-agb": dict(logo_y=86.7, title_y=214.5, mission=(102.5, 464.0), ref=(321.9, 729.8)),
    "pfd-l2-fd": dict(logo_y=86.6, title_y=242.1, mission=(110.1, 445.7), ref=(354.2, 734.2)),
    "pfd-l2-fh": dict(logo_y=86.7, title_y=242.1, mission=(97.1, 434.2), ref=(124.5, 734.2)),
}
# Numbers of the table of contents are printed "1.1." except in the L1 SAR ATBD.
_NO_TOC_DOT_DOCS = ("atbd-l1-sar",)
_COVER_LAYOUT_DEFAULT = _COVER_LAYOUT["atbd-l2-agb"]
_COVER_TITLE_ASCENT = 0.905 * 48  # baseline below the top of a 48 pt Arial line
_COVER_TITLE_LEADING = 55.2
_COVER_RULE_GAP = 3.8  # grey rule below the bottom of the last title line
_COVER_REF_ASCENT = 0.905 * 11
# ATBDs whose tables use the left indent of the headings (all PFDs do).
_WIDE_TABLE_DOCS = ("atbd-l1-sar",)
# Table font of the originals: size, leading and \arraystretch such that the
# rows have the height of the originals (a row of n lines is (stretch + n - 1)
# leadings high in LaTeX): ATBD 9 pt, 12.9 pt per line, 15.5 pt for one line
# (measured on AGB ATBD p. 27); PFD 10 pt, 11.5 pt per line, 26 pt for one
# line (AGB PFD p. 31); L1 Stack 7 pt (17 400 characters of its tables, 8.05 pt
# per line, 42.7 pt for 5 lines).
_TABLE_FONTS = {"atbd": (9, 12.9, 1.2), "pfd": (10, 11.5, 2.26),
                "atbd-l1-stack": (7, 8.1, 1.3)}
# Rules of the table grid: grey 0.48 pt (ATBD), black 0.72 pt and a grey
# header row (PFD), measured on the originals.
_TABLE_RULES = {"atbd": ("A6A6A6", 0.48, "F2F2F2"), "pfd": ("000000", 0.72, "D9D9D9")}
_WEB_ONLY_ADMONITIONS = ("draft", "authoritative document")
_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIALS.get(char, char) for char in text)


def document_folder(value: str) -> str:
    """``science-guide/atbd-l2-agb/index`` -> ``atbd-l2-agb``."""
    folder = value.strip().removesuffix("/index").strip("/")
    return folder.removeprefix("science-guide/")


def _metadata_comment(index_text: str) -> dict[str, str]:
    match = re.search(r"<!--\s*Pipeline metadata.*?-->", index_text, re.DOTALL)
    meta: dict[str, str] = {}
    if not match:
        return meta
    for line in match.group(0).splitlines():
        key, sep, value = line.partition(":")
        if sep and re.fullmatch(r"[a-z_]+", key.strip()):
            meta[key.strip()] = value.strip().strip("'\"")
    return meta


def _info_table(index_text: str) -> dict[str, str]:
    """Rows of the ``| Field | Value |`` Document information table."""
    rows: dict[str, str] = {}
    for line in index_text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] and not set(cells[0]) <= {"-", ":"}:
            rows.setdefault(cells[0].lower(), cells[1].strip("`"))
    return rows


_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")


def _format_date(raw: str) -> str:
    """Dates as printed in the originals: dd/mm/yyyy.

    English month names are parsed without ``%B`` (locale dependent: Sphinx
    sets the user locale).
    """
    value = raw.strip()
    match = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+) (\d{4})", value)
    if match and match.group(2).lower() in _MONTHS:
        value = f"{match.group(1)}/{_MONTHS.index(match.group(2).lower()) + 1}/{match.group(3)}"
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return value


_MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def header_date(value: str) -> str:
    """``02/04/2026`` -> ``Apr 02\\atbdsup{nd}, 2026`` (header of the originals)."""
    match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", value.strip())
    if not match:
        return latex_escape(value)
    day, month, year = int(match.group(1)), int(match.group(2)), match.group(3)
    if not 1 <= month <= 12:
        return latex_escape(value)
    suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return rf"{_MONTH_ABBR[month - 1]} {day:02d}\atbdsup{{{suffix}}}, {year}"


def read_document_meta(docs_dir: Path, folder: str) -> dict[str, str]:
    """Title, code, issue and date of ``science-guide/<folder>``.

    Code from ``targets.json`` (then the Document information table and the
    pipeline metadata comment of ``index.md``); title from the metadata
    comment, ``targets.json`` or the ``index.md`` heading; issue and date
    from the metadata comment or the Document information table.
    """
    doc_dir = docs_dir / "science-guide" / folder
    targets: dict[str, Any] = {}
    targets_path = doc_dir / "targets.json"
    if targets_path.is_file():
        targets = json.loads(targets_path.read_text(encoding="utf-8"))
    index_path = doc_dir / "index.md"
    index_text = index_path.read_text(encoding="utf-8") if index_path.is_file() else ""
    comment = _metadata_comment(index_text)
    table = _info_table(index_text)
    heading = re.search(r"^# (.+)$", index_text, re.MULTILINE)
    title = (comment.get("title") or targets.get("title") or table.get("title")
             or (heading.group(1).strip() if heading else folder))
    return {
        "title": title,
        "cover_title": "".join(
            rf"\atbdcoverline{{{latex_escape(line)}}}"
            for line in _COVER_TITLES.get(folder, (title,))
        ),
        "code": targets.get("doc_code") or table.get("external reference") or comment.get("doc_code", ""),
        "issue": comment.get("version") or table.get("issue", ""),
        "date": _format_date(comment.get("date") or table.get("date", "")),
    }


def _cover_definitions(folder: str) -> str:
    """Numeric cover geometry (pt, y downwards) for ``atbdcover.sty``."""
    layout = _COVER_LAYOUT.get(folder, _COVER_LAYOUT_DEFAULT)
    lines = len(_COVER_TITLES.get(folder, ("",)))
    title_base = layout["title_y"] + _COVER_TITLE_ASCENT
    last_bottom = layout["title_y"] + (lines - 1) * _COVER_TITLE_LEADING + 1.117 * 48
    mission_x, mission_y = layout["mission"]
    ref_x, ref_y = layout["ref"]
    values = {
        "aresysy": layout["logo_y"] + 76.2,  # bottom of the logos
        # visible ESA logo of the originals: y 95.0-151.3
        "esay": layout.get("esa_bottom", layout["logo_y"] - 86.7 + 151.3),
        "titley": title_base,
        "ruley": last_bottom + _COVER_RULE_GAP,
        "missionx": mission_x,
        "missiony": mission_y + 118.3,
        "refx": ref_x,
        "refy": ref_y + _COVER_REF_ASCENT,
    }
    return "\n".join(
        rf"\newcommand\atbdcover{name}{{{value:.2f}}}" for name, value in values.items()
    )


def latex_config(docs_dir: Path, folder: str, bps_version: str) -> dict[str, Any]:
    """Sphinx ``latex_*`` settings for the PDF export of one document."""
    folder = document_folder(folder)
    meta = read_document_meta(docs_dir, folder)
    version = "" if bps_version in ("", "unknown") else bps_version
    definitions = "\n".join(
        rf"\newcommand\atbd{name}{{{value}}}"
        for name, value in (
            ("title", latex_escape(_HEADER_TITLES.get(folder, meta["title"]))),
            ("code", latex_escape(meta["code"])),
            ("issue", latex_escape(meta["issue"])),
            ("date", latex_escape(meta["date"])),
            ("bpsversion", latex_escape(version)),
            ("covertitle", meta["cover_title"]),
        )
    )
    definitions += f"\n\\newcommand\\atbdheaderdate{{{header_date(meta['date'])}}}"
    definitions += "\n" + _cover_definitions(folder)
    if folder in _NO_TOC_DOT_DOCS:
        definitions += "\n\\newcommand\\atbdtocnodot{}"
    kind = "pfd" if folder.startswith("pfd-") else "atbd"
    size, leading, stretch = _TABLE_FONTS.get(folder, _TABLE_FONTS[kind])
    rule, rule_width, head_fill = _TABLE_RULES[kind]
    definitions += (
        f"\n\\definecolor{{atbdtablerule}}{{HTML}}{{{rule}}}"
        f"\n\\definecolor{{atbdtablehead}}{{HTML}}{{{head_fill}}}"
        f"\n\\newcommand\\atbdtableruledth{{{rule_width}bp}}"
    )
    definitions += (
        "\n\\newcommand\\atbdtablefont{\\linespread{1}"
        f"\\fontsize{{{size}}}{{{leading}}}\\selectfont"
        f"\\renewcommand\\arraystretch{{{stretch}}}}}"
    )
    # L1 SAR: the wide tables of the original run to x = 564 pt (headings
    # start at 56 pt), i.e. 22.7 pt beyond the text width on the right.
    extra = "22.7pt" if folder in _WIDE_TABLE_DOCS else "0pt"
    definitions += f"\n\\newcommand\\atbdwideextra{{{extra}}}"
    logos = [
        str(docs_dir / "_static" / "logos" / name)
        for name in _LOGOS
        if (docs_dir / "_static" / "logos" / name).is_file()
    ]
    return {
        # Table template of the exports (longtable.tex.jinja) first.
        "templates_path": ["_templates/latex", "_templates"],
        "latex_engine": "xelatex",
        "latex_documents": [
            (
                f"science-guide/{folder}/index",
                f"{folder}.tex",
                meta["title"],
                "ESA / Aresys",
                "manual",
            )
        ],
        "latex_additional_files": [
            *logos,
            *(str(docs_dir / "_templates" / "latex" / f"{name}.sty") for name in _STY_FILES),
        ],
        "atbd_wide_tables": folder.startswith("pfd-") or folder in _WIDE_TABLE_DOCS,
        "atbd_table_font_pt": float(size),
        "latex_domain_indices": False,
        "latex_show_urls": "no",
        "latex_show_pagerefs": False,
        # Grid tables with a grey header row, as in the original documents.
        "latex_table_style": ["standard", "colorrows"],
        "latex_elements": {
            "papersize": "a4paper",
            "pointsize": "11pt",
            "extraclassoptions": "openany,oneside",
            # Original: text 3.25 cm from the left edge (headings at 2 cm),
            # 1.95 cm on the right; logo header and footer rules.
            "geometry": (
                r"\usepackage[a4paper,left=92.1bp,right=53.3bp,top=76.1bp,"
                r"height=684.7bp,headheight=0bp,headsep=0bp,footskip=0bp]{geometry}"
            ),
            "fontpkg": "",
            "fncychap": "",
            "sphinxsetup": (
                "TitleColor={HTML}{2F5496},"
                "InnerLinkColor={HTML}{2F5496},"
                "OuterLinkColor={HTML}{2F5496},"
                "TableRowColorHeader={HTML}{D9D9D9},"
                "TableRowColorOdd={HTML}{FFFFFF},"
                "TableRowColorEven={HTML}{FFFFFF},"
                "VerbatimColor={HTML}{F5F5F5}"
            ),
            "preamble": definitions + "\n" + r"\usepackage{amsmath,amssymb}" + "\n" + r"\usepackage{atbdlayout}" + "\n",
            "maketitle": r"\atbdmaketitle",
            # The table of contents follows the Change history (see below).
            "tableofcontents": "",
            "makeindex": "",
            "printindex": "",
        },
        # Mermaid diagrams need mermaid-cli (mmdc) and a headless browser.
        "mermaid_output_format": "png",
    }


# ---------------------------------------------------------------------------
# LaTeX-only reshaping of the front matter
# ---------------------------------------------------------------------------


def _raw_latex(text: str) -> nodes.raw:
    return nodes.raw("", text, format="latex")


def _is_back_link(node: nodes.paragraph) -> bool:
    return node.astext().strip().startswith("←")


def _is_web_admonition(node: nodes.Element) -> bool:
    titles = [child for child in node.children if isinstance(child, nodes.title)]
    if not titles:
        return False
    title = titles[0].astext().strip().lower()
    return title.startswith(_WEB_ONLY_ADMONITIONS)


def _front_nodes(top: nodes.Node):
    """Nodes of the index document, without the included chapters."""
    from sphinx import addnodes

    stack = [top]
    while stack:
        node = stack.pop()
        if isinstance(node, addnodes.start_of_file):
            continue
        yield node
        if isinstance(node, nodes.Element):
            stack.extend(reversed(node.children))


# Start of an item of a Change history entry: a ticket key ("BPSPDR3-61:",
# "BPS-393 / BPS-392:", "BPS-FTO-BATCH4:") or an "Updated ..." line. The
# conversion joins the bullet list of the original into one paragraph, so the
# list is rebuilt from the text.
_TICKET = r"[A-Z][A-Za-z0-9]*-(?:[A-Za-z0-9]+-)*[A-Za-z0-9]*\d+"
_TICKET_KEYS = rf"{_TICKET}(?:\s*[,/]\s*{_TICKET})*\s*:"
_CHANGE_VERBS = (
    r"(?:Updated|Added|Renamed|Removed|Expanded|Separated|Fixed|Refactored|Revised|Finalized|Explained|"
    r"Rewritten|Clarified|Corrected|Changed|Moved|Replaced|Improved|Reformulated|Reviewed|Modified|"
    r"Specified|Addressed|Restructured|Major)\b"
)
_CHANGE_START_RE = re.compile(rf"(?<=[^\s,/])\s+(?=(?:{_TICKET_KEYS})|(?:{_CHANGE_VERBS}))")
_TICKET_START_RE = re.compile(rf"\s*{_TICKET_KEYS}")
# End of a sentence or of a "(Sec. 4.1)" reference followed by a capital letter.
_SENTENCE_RE = re.compile(r"(?<=[.)])\s+(?=[A-Z][a-z])")
# A colon after a few words ("... updates: Reference included ...") opens
# a list of sub-items.
_COLON_RE = re.compile(r"(?<=[a-z\)]:)\s+(?=(?!Section\b)[A-Z][a-z])")
# "Stacking: Section 4.4 IOB: Section 5.2.5 ...": one sub-item per label.
_LABEL_RE = re.compile(r"(?<=\d)\s+(?=[A-Z][A-Za-z]*:\s+Section\b)")
_ABBREVIATIONS = ("Sec.", "Secs.", "Fig.", "Figs.", "Eq.", "Eqs.", "No.", "Tab.", "Ref.", "Refs.", "e.g.", "i.e.", "vs.", "w.r.t.", "Sect.")
# Title of an entry, printed on its own line above the bullets.
_CHANGE_TITLE_RE = re.compile(
    r"^((?:[A-Za-z0-9_\-]+ ){1,3}(?:issue|version) of the document(?: \([^)]*\))?[:.]?)(?:\s+|$)"
)


def _split_text(text: str) -> list[str]:
    """Cut a flattened bullet list at the starts of its items."""
    cuts: set[int] = set()
    for regex in (_CHANGE_START_RE, _SENTENCE_RE, _COLON_RE, _LABEL_RE):
        for match in regex.finditer(text):
            before = text[:match.start()]
            if before.endswith(_ABBREVIATIONS) or re.search(r"\b[A-Z]\.$", before):
                continue
            if before.endswith(("-", "\u2013")):
                continue  # "BPS-42: PDR1-2 - Updated ...": one item
            if re.search(_TICKET_KEYS + r"$", before):
                continue  # "PDR1-3-57:" is followed by its text
            if regex is _COLON_RE and len(before.split()) < 3:
                continue
            cuts.add(match.end())
    parts, pos = [], 0
    for cut in sorted(cuts):
        parts.append(text[pos:cut].strip())
        pos = cut
    parts.append(text[pos:].strip())
    return [part for part in parts if part]


def _change_items(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Title line and ``(level, text)`` bullets of a Change history cell.

    An item ending with a colon (not a ticket key alone) opens a second
    level, which lasts up to the next ticket key.
    """
    text = " ".join(text.split())
    title = ""
    match = _CHANGE_TITLE_RE.match(text)
    if match and len(text) > match.end():
        title, text = match.group(1), text[match.end():]
    elif match:
        return match.group(1), []
    items: list[tuple[int, str]] = []
    nested = False
    labels = False  # the sub-items are "Label: Section x.y" lines
    for part in _split_text(text):
        label = bool(re.match(r"[A-Z][A-Za-z]*:\s+Section\b", part))
        if _TICKET_START_RE.match(part) or (nested and labels and not label):
            nested = False
        level = 2 if nested else 1
        items.append((level, part))
        if part.endswith(":") and not _TICKET_START_RE.fullmatch(part):
            nested, labels = True, False
        elif label and nested:
            labels = True
    return title, items


# Geometry (pt) of the grid tables of the originals; every cell is
# ``_ROW_PITCH`` high (rule included), the first column starts at the left
# margin of the body text.
_INFO_COLUMNS = (127.3, 318.7)  # Title ... Pages
_INFO_COLUMNS_BY_DOC = {"atbd-l1-stack": (108.2, 337.8)}  # narrower label column
_BLOCK_COLUMNS = (254.9, 191.1)  # Recipients ... Approved by
# Date | Issue | Change | Author widths and shift to the left (pt), measured
# on the vertical rules of each original (Word fits the columns to the text).
_CHANGE_COLUMNS = (60.8, 36.5, 337.4, 58.1, 0.0)
_CHANGE_COLUMNS_BY_DOC = {
    "atbd-l2-agb": (63.6, 49.6, 256.4, 123.2, 0.0),
    "atbd-l2-fh": (60.8, 36.5, 312.2, 83.3, 0.0),
    "atbd-l1-sar": (60.9, 36.4, 337.5, 58.1, 5.7),
    "pfd-l2-fd": (62.2, 46.8, 336.0, 68.0, 5.7),
    "pfd-l2-fh": (62.4, 71.2, 262.5, 95.8, 0.0),
}
# Rows of the "Name | Organisation" blocks of the originals (blank rows are
# kept: the originals are filled in by hand).
_BLOCKS = ("Recipients", "Prepared by", "Checked by", "Approved by")
_BLOCK_ROWS = {"Recipients": 4, "Prepared by": 9, "Checked by": 1, "Approved by": 1}


def _cell_text(node: nodes.Node) -> str:
    return " ".join(node.astext().split())


def _cell_lines(node: nodes.Node) -> list[tuple[int, str]]:
    """(depth, text) of each line of a cell whose lines are separated by
    ``<br>`` (raw HTML); the depth counts the leading no-break spaces by 4."""
    parts: list[str] = [""]
    for n in node.findall(lambda x: isinstance(x, (nodes.Text, nodes.raw))):
        if isinstance(n, nodes.raw):
            if re.fullmatch(r"\s*<br\s*/?>\s*", n.astext()):
                parts.append("")
            continue
        if isinstance(n.parent, nodes.raw):
            continue  # the text of a raw node, handled above
        parts[-1] += n.astext()
    out = []
    for p in parts:
        nb = len(p) - len(p.lstrip("\u00a0 "))
        text = " ".join(p.split())
        if text:
            out.append((p[:nb].count("\u00a0") // 4, text))
    return out


def _tex(text: str) -> str:
    return latex_escape(text)


def _table_rows(table: nodes.table) -> list[list[str]]:
    """Body rows of a docutils table as lists of cell texts."""
    return [
        [_cell_text(entry) for entry in row.findall(nodes.entry)]
        for tbody in table.findall(nodes.tbody)
        for row in tbody.findall(nodes.row)
    ]


def _front_blocks(container: nodes.Element) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
    """Fields and Name | Organisation blocks of the Document information."""
    fields: dict[str, str] = {}
    blocks: dict[str, list[list[str]]] = {}
    current = None
    for node in container.findall(lambda n: isinstance(n, (nodes.table, nodes.paragraph))):
        if isinstance(node, nodes.paragraph):
            if not isinstance(node.parent, nodes.entry) and _cell_text(node) in _BLOCKS:
                current = _cell_text(node)
            continue
        rows = _table_rows(node)
        if current is None:
            fields = {row[0].lower(): row[1] for row in rows if len(row) == 2}
        else:
            blocks[current] = [(row + [""])[:2] for row in rows]
            current = None
    return fields, blocks


def _split_authors(text: str, known: set[str]) -> list[str]:
    """One author per line: the conversion joins the names of a cell.

    Known names (Prepared by, ...) are taken first; the other words go by
    pairs (first name, family name); ``Review: Name Name`` stays together.
    """
    words = text.split()
    names: list[str] = []
    i = 0
    while i < len(words):
        for size in (3, 2):
            if " ".join(words[i:i + size]) in known:
                names.append(" ".join(words[i:i + size]))
                i += size
                break
        else:
            size = 3 if words[i].endswith(":") else 2
            chunk = words[i:i + size]
            if len(words) - i < size or not all(w[:1].isupper() for w in chunk):
                names.append(" ".join(words[i:]))
                break
            names.append(" ".join(chunk))
            i += size
    return names


def _change_rows(container: nodes.Element, known: set[str]) -> list[dict[str, Any]]:
    """Rows of the Change history: date, issue, intro lines, bullets, authors."""
    rows: list[dict[str, Any]] = []
    for table in container.findall(nodes.table):
        for tbody in table.findall(nodes.tbody):
            for row in tbody.findall(nodes.row):
                entries = list(row.findall(nodes.entry))
                if len(entries) != 4:
                    continue
                lines = _cell_lines(entries[2])
                if len(lines) > 1:
                    # the conversion keeps the lines of the cell (<br>) and the
                    # bullets with their depth (relecture D-rev-18, W1-7)
                    title, bullets = lines[0][1], [
                        (min(level + 1, 2), re.sub(r"^[•\-]\s*", "", text))
                        for level, text in lines[1:]]
                else:
                    # an older conversion joins the whole cell into one paragraph
                    # (the bullet lists of the original are lost): rebuild them.
                    title, bullets = _change_items(_cell_text(entries[2]))
                author_lines = [t for _l, t in _cell_lines(entries[3])]
                rows.append({
                    "date": _format_date(_cell_text(entries[0])),
                    "issue": _cell_text(entries[1]),
                    "title": title,
                    "bullets": bullets,
                    "authors": author_lines if len(author_lines) > 1
                    else _split_authors(_cell_text(entries[3]), known),
                })
    return rows


def _info_latex(heading: str, folder: str, fields: dict[str, str], blocks: dict[str, list[list[str]]]) -> str:
    """Single bordered Document information table, then the blocks."""
    pages = re.match(r"\d+", fields.get("pages", ""))
    external = fields.get("external reference") or fields.get("external ref.", "")
    # Pages: the count of the approved document, then that of this export.
    pages_tex = (
        _tex(pages.group(0)) + r" (approved document); this export: \the\numexpr\getpagerefnumber{LastPage}+1\relax"
        if pages else _tex(fields.get("pages", ""))
    )
    values = (
        ("Title", _tex(fields.get("title", "")), True),
        ("Internal Ref.", "", False),
        ("External Ref.", _tex(external), False),
        ("Date", _tex(_format_date(fields.get("date", ""))), False),
        ("Issue", _tex(fields.get("issue", "")), False),
        ("Pages", pages_tex, False),
    )
    out = [f"\\clearpage\n\\atbdfrontheading[24bp]{{{_tex(heading)}}}\n\\atbdexportnotice"]
    body = []
    for label, value, bold in values:
        shade = "\\atbdshade" if label == "Title" else ""
        value_tex = f"\\bfseries {value}" if bold else value
        body.append(f"{shade}\\atbdcell{{\\bfseries {label}}} & {shade}\\atbdcell{{{value_tex}}} \\\\ \\hline")
    columns = _INFO_COLUMNS_BY_DOC.get(folder, _INFO_COLUMNS)
    out.append(f"\\atbdgrid{{{columns[0]}bp}}{{{columns[1]}bp}}{{{' '.join(body)}}}")
    for name in _BLOCKS:
        rows = blocks.get(name)
        if rows is None:
            continue
        rows = [row for row in rows if any(row)]
        rows += [["", ""]] * max(0, _BLOCK_ROWS[name] - len(rows))
        body = [f"\\atbdshade\\atbdcell{{\\bfseries {name}}} & \\atbdshade\\atbdcell{{}} \\\\ \\hline"]
        body += [f"\\atbdcell{{{_tex(a)}}} & \\atbdcell{{{_tex(b)}}} \\\\ \\hline" for a, b in rows]
        out.append(f"\\vspace{{16bp}}\n\\atbdgrid{{{_BLOCK_COLUMNS[0]}bp}}{{{_BLOCK_COLUMNS[1]}bp}}{{{' '.join(body)}}}")
    return "\n".join(out) + "\n"


_CHANGE_SINGLE_ROW = 100.0  # estimated height (pt) up to which an entry is one row


def _author_tex(name: str, width: float) -> str:
    """Name of an author; words wider than the column break anywhere (as Word does)."""
    words = []
    for word in name.split():
        if len(word) * 4.6 > width:
            words.append("\\penalty0 ".join(_tex(char) for char in word))
        else:
            words.append(_tex(word))
    return " ".join(words)


def _change_line_chars(folder: str) -> int:
    """Characters of a bullet line of the Change column (10 pt, 5.2 pt each)."""
    return int((_CHANGE_COLUMNS_BY_DOC.get(folder, _CHANGE_COLUMNS)[2] - 10.8 - 18) / 5.6)


def _change_latex(heading: str, folder: str, rows: list[dict[str, Any]]) -> str:
    """Change history: Date | Issue | Change | Author.

    The title of an entry is a line of its own, then one row per bullet
    (levels 1 and 2); date, issue and authors sit in the row that holds the
    middle of the entry, centred vertically as in the originals.
    """
    body = []
    for row in rows:
        cells: list[tuple[float, str]] = []  # (height above the baseline, text)
        if row["title"]:
            cells.append((12.85, f"\\atbdchg{{{_tex(row['title'])}}}"))
        for level, text in row["bullets"]:
            top = 12.6 if len(cells) == 1 and row["title"] else (12.85 if not cells else 10.1)
            cells.append((top, f"\\atbdbullet[{level}]{{{_tex(text)}}}"))
        if not cells:
            cells.append((12.85, "\\mbox{}"))
        # estimated height of every cell (title line, bullets)
        heights = [max(1, -(-len(text) // _change_line_chars(folder))) * 14.6 + (top - 10.1) for top, text in cells]
        # authors: one per line, 15 pt a line
        width = _CHANGE_COLUMNS_BY_DOC.get(folder, _CHANGE_COLUMNS)[3] - 10.8
        names = row["authors"]
        authors = "\\newline ".join(_author_tex(name, width) for name in names)
        author_height = 15.0 * sum(max(1, -(-int(len(name) * 4.6) // int(width))) for name in names)
        date, issue = _tex(row["date"]), _tex(row["issue"])

        def joined(chunk: list[tuple[float, str]]) -> str:
            return "\\par ".join(
                ("\\vspace{2.4bp}" if i == 1 and row["title"] and chunk[0] is cells[0] else "") + text
                for i, (_, text) in enumerate(chunk)
            )

        # A short entry is one row: all cells are centred on it, as in the
        # originals. A long entry is one row per bullet (an entry can break
        # between two pages, a row cannot); its first row holds as many lines
        # as the authors need, and the date and issue hang from the middle one.
        first = len(cells) if sum(heights) <= _CHANGE_SINGLE_ROW else 1
        while first < len(cells) and sum(heights[:first]) < author_height * 1.25 + 10:
            first += 1
        groups = [cells[:first]] + [[cell] for cell in cells[first:]]
        group_heights = [sum(heights[:first])] + heights[first:]
        total = sum(group_heights)
        tops = [sum(group_heights[:i]) for i in range(len(groups))]
        middle = min(range(len(groups)), key=lambda i: abs(tops[i] + group_heights[i] / 2 - total / 2))
        for index, chunk in enumerate(groups):
            end_row = "\\\\" + (" \\hline" if index == len(groups) - 1 else "")
            if len(groups) == 1:
                date_tex, issue_tex = date, issue
            elif index == middle:
                date_tex, issue_tex = "\\atbdflat{" + date + "}", "\\atbdflat{" + issue + "}"
            else:
                date_tex = issue_tex = ""
            body.append(
                f"\\atbdchangerow{{{date_tex}}}{{{issue_tex}}}{{{chunk[0][0]}}}{{{joined(chunk)}}}"
                f"{{{authors if index == 0 else ''}}}{{{end_row}}}"
            )
    widths = "".join(f"{{{value}}}" for value in _CHANGE_COLUMNS_BY_DOC.get(folder, _CHANGE_COLUMNS))
    return (
        f"\\clearpage\n\\atbdfrontheading{{{_tex(heading)}}}\n"
        f"\\begin{{atbdchange}}{widths}\n{chr(10).join(body)}\n\\end{{atbdchange}}\n"
    )


class AtbdLatexFrontMatter(SphinxPostTransform):
    """Front matter of the PDF: info pages, then the table of contents."""

    default_priority = 5  # before reference resolution (back link xref)
    builders = ("latex",)

    def run(self, **kwargs: Any) -> None:
        docname = self.env.docname
        if not (docname.startswith("science-guide/") and docname.endswith("/index")):
            return
        # Front matter of index.md: everything before the first included
        # chapter (Sphinx wraps each included document in ``start_of_file``).
        from sphinx import addnodes

        top = self.document
        first_chapter = next(iter(top.findall(addnodes.start_of_file)), None)
        # Nodes of index.md itself. The included chapters are not visited:
        # ``node.parent`` cannot tell them apart (Sphinx moves their children
        # under ``start_of_file`` without updating the parents).
        front = list(_front_nodes(top))

        for para in [n for n in front if isinstance(n, nodes.paragraph)]:
            if _is_back_link(para) and para.parent is not None:
                para.parent.remove(para)
        for adm in [n for n in front if isinstance(n, nodes.Admonition)]:
            if _is_web_admonition(adm) and adm.parent is not None:
                adm.parent.remove(adm)
        for rule in [n for n in front if isinstance(n, nodes.transition)]:
            if rule.parent is not None:
                rule.parent.remove(rule)

        folder = document_folder(docname)
        info_label, change_label, toc_label = _FRONT_LABELS_BY_DOC.get(folder, _FRONT_LABELS)
        last_front: nodes.Element | None = None
        known: set[str] = set()
        rubrics = {}
        for rubric in [n for n in front if isinstance(n, nodes.rubric)]:
            key = rubric.astext().strip().lower()
            if key in _FRONT_HEADINGS and key not in rubrics:
                rubrics[key] = rubric
        info = rubrics.get("document information")
        if info is not None and info.parent is not None:
            container = info.parent
            fields, blocks = _front_blocks(container)
            known = {row[0] for rows in blocks.values() for row in rows if row[0]}
            self._replace(container, _info_latex(info_label, folder, fields, blocks))
            last_front = container
        change = rubrics.get("change history")
        if change is not None and change.parent is not None:
            container = change.parent
            self._replace(container, _change_latex(change_label, folder, _change_rows(container, known)))
            last_front = container

        toc = _raw_latex(f"\n\\atbdtableofcontents[{latex_escape(toc_label)}]\n")
        if last_front is not None and last_front.parent is not None:
            parent = last_front.parent
            parent.insert(parent.index(last_front) + 1, toc)
        elif first_chapter is not None and first_chapter.parent is not None:
            parent = first_chapter.parent
            parent.insert(parent.index(first_chapter), toc)
        else:
            top.insert(0, toc)

    @staticmethod
    def _replace(container: nodes.Element, latex: str) -> None:
        """The whole front-matter block becomes one raw LaTeX node."""
        raw = _raw_latex(latex)
        container[:] = [raw]


def setup(app: Any) -> dict[str, Any]:
    app.add_post_transform(AtbdLatexFrontMatter)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
