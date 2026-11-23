# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
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

_LOGOS = ("Aresys_light.png", "ESA_light.png")
# Layout packages (_templates/latex): atbdlayout loads the others.
_STY_FILES = ("atbdlayout", "atbdbody", "atbdfloats", "atbdmath", "atbdcover")
_FRONT_HEADINGS = ("Document information", "Change history")
# Full-width grid tables in the front matter (column specs as Sphinx
# ``tabularcolumns``): Field | Value, Name | Organisation, Change history.
_FRONT_TABLE_SPECS = {
    2: r"|\X{1}{3}|\X{2}{3}|",
    4: r"|\X{8}{48}|\X{4}{48}|\X{26}{48}|\X{10}{48}|",
}
# Cover titles of the originals when they differ from the document title
# (the converted metadata has no cover title yet).
_COVER_TITLES = {
    "atbd-l2-agb": "BIOMASS Above Ground Biomass Product ATBD",
    "atbd-l2-fd": "BIOMASS Forest Disturbance Product ATBD",
    "atbd-l2-fh": "BIOMASS Forest Height Product ATBD",
    "atbd-l1-sar": "BIOMASS L1 SAR Product ATBD",
    "atbd-l1-stack": "BIOMASS L1c Stack Product ATBD",
}
# ATBDs whose tables use the left indent of the headings (all PFDs do).
_WIDE_TABLE_DOCS = ("atbd-l1-sar",)
# Table font of the originals: ATBD 9 pt, PFD 10 pt (single line spacing),
# and the row padding of their Word tables (\arraystretch).
_TABLE_FONTS = {"atbd": (9, 10.5, 2.0), "pfd": (10, 11.5, 2.0)}
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
        "cover_title": _COVER_TITLES.get(folder, title),
        "code": targets.get("doc_code") or table.get("external reference") or comment.get("doc_code", ""),
        "issue": comment.get("version") or table.get("issue", ""),
        "date": _format_date(comment.get("date") or table.get("date", "")),
    }


def latex_config(docs_dir: Path, folder: str, bps_version: str) -> dict[str, Any]:
    """Sphinx ``latex_*`` settings for the PDF export of one document."""
    folder = document_folder(folder)
    meta = read_document_meta(docs_dir, folder)
    version = "" if bps_version in ("", "unknown") else bps_version
    definitions = "\n".join(
        rf"\newcommand\atbd{name}{{{latex_escape(value)}}}"
        for name, value in (
            ("title", meta["title"]),
            ("code", meta["code"]),
            ("issue", meta["issue"]),
            ("date", meta["date"]),
            ("bpsversion", version),
            ("covertitle", meta["cover_title"]),
        )
    )
    size, leading, stretch = _TABLE_FONTS["pfd" if folder.startswith("pfd-") else "atbd"]
    definitions += (
        "\n\\newcommand\\atbdtablefont{\\linespread{1}"
        f"\\fontsize{{{size}}}{{{leading}}}\\selectfont"
        f"\\renewcommand\\arraystretch{{{stretch}}}}}"
    )
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
                r"\usepackage[a4paper,left=3.25cm,right=1.95cm,top=2.7cm,"
                r"bottom=2.9cm,headheight=1.55cm,headsep=0.6cm,footskip=1.2cm]{geometry}"
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


def _set_front_table_spec(table: nodes.table) -> None:
    """Full-width columns; the Change history may run over several pages."""
    from sphinx import addnodes

    groups = list(table.findall(nodes.tgroup))
    cols = groups[0].get("cols", 0) if groups else 0
    table["classes"].append("atbd-front")  # compact rows (longtable.tex.jinja)
    spec = _FRONT_TABLE_SPECS.get(cols)
    parent = table.parent
    if spec is None or parent is None:
        return
    previous = parent.index(table) - 1
    if previous >= 0 and isinstance(parent[previous], addnodes.tabular_col_spec):
        return  # an explicit ``tabularcolumns`` wins
    colspec = addnodes.tabular_col_spec()
    colspec["spec"] = spec
    parent.insert(parent.index(table), colspec)
    if cols == 4 and "longtable" not in table["classes"]:
        table["classes"].append("longtable")


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
# conversion joins the bullet list of the original into one paragraph.
_CHANGE_ITEM_RE = re.compile(
    r"(?<=[^\s,/:])\s+(?=(?:[A-Z][A-Za-z0-9]*-(?:[A-Za-z0-9]+-)*[A-Za-z0-9]*\d+"
    r"(?:\s*[,/]\s*[A-Z][A-Za-z0-9]*-(?:[A-Za-z0-9]+-)*[A-Za-z0-9]*\d+)*\s*:)"
    r"|(?:Updated|Added|Renamed|Removed|Expanded|Separated|Fixed)\b)"
)


def _split_change_items(para: nodes.paragraph) -> None:
    """Rebuild the bullet list of a Change history cell."""
    segments: list[list[nodes.Node]] = [[]]
    for child in list(para.children):
        if not isinstance(child, nodes.Text):
            segments[-1].append(child.deepcopy())
            continue
        text, pos = str(child), 0
        for match in _CHANGE_ITEM_RE.finditer(text):
            if text[pos:match.start()]:
                segments[-1].append(nodes.Text(text[pos:match.start()]))
            segments.append([])
            pos = match.end()
        if text[pos:]:
            segments[-1].append(nodes.Text(text[pos:]))
    if len(segments) < 3 or para.parent is None:
        return  # a single change: keep the paragraph
    intro, items = segments[0], segments[1:]
    bullets = nodes.bullet_list(bullet="-")
    for item in items:
        bullets += nodes.list_item("", nodes.paragraph("", "", *item))
    new: list[nodes.Node] = [nodes.paragraph("", "", *intro)] if intro else []
    para.replace_self([*new, bullets])


_ROW_ITEMS = 6  # bullets per table row (a longtable row cannot break)


def _split_long_row(row: nodes.row) -> None:
    """Continue a Change history entry longer than a page on more rows."""
    cell = row.children[2]
    bullets = next(iter(cell.findall(nodes.bullet_list)), None)
    if bullets is None or len(bullets.children) <= _ROW_ITEMS:
        return
    items = list(bullets.children)
    bullets.children = []
    for item in items[:_ROW_ITEMS]:
        bullets += item
    parent, index = row.parent, row.parent.index(row)
    for start in range(_ROW_ITEMS, len(items), _ROW_ITEMS):
        chunk = nodes.bullet_list(bullet=bullets.get("bullet", "-"))
        for item in items[start:start + _ROW_ITEMS]:
            chunk += item
        new = nodes.row("", nodes.entry(), nodes.entry(), nodes.entry("", chunk), nodes.entry())
        index += 1
        parent.insert(index, new)


def _front_table_cells(table: nodes.table) -> None:
    """Dates as dd/mm/yyyy; no "Field | Value" header (not in the originals);
    bullet lists in the Change history."""
    for tgroup in table.findall(nodes.tgroup):
        if tgroup.get("cols") == 4:
            for row in list(tgroup.findall(nodes.row)):
                if len(row.children) == 4:
                    for para in list(row.children[2].findall(nodes.paragraph)):
                        _split_change_items(para)
                    _split_long_row(row)
        thead = next((c for c in tgroup.children if isinstance(c, nodes.thead)), None)
        if thead is not None and [e.astext().strip() for e in thead.findall(nodes.entry)] == ["Field", "Value"]:
            tgroup.remove(thead)
    for entry in table.findall(nodes.entry):
        for text in list(entry.findall(nodes.Text)):
            if re.fullmatch(r"\s*(\d{4}-\d{2}-\d{2}|\d{1,2} [A-Z][a-z]+ \d{4})\s*", text):
                text.parent.replace(text, nodes.Text(_format_date(text)))


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

        last_front: nodes.Element | None = None
        for rubric in [n for n in front if isinstance(n, nodes.rubric)]:
            title = rubric.astext().strip()
            if title not in _FRONT_HEADINGS:
                continue
            last_front = rubric.parent
            for table in list(last_front.findall(nodes.table)):
                _set_front_table_spec(table)
                _front_table_cells(table)
            rubric.replace_self(_raw_latex(f"\\clearpage\n\\atbdfrontheading{{{latex_escape(title)}}}\n"))

        toc = _raw_latex("\n\\atbdtableofcontents\n")
        if last_front is not None and last_front.parent is not None:
            parent = last_front.parent
            parent.insert(parent.index(last_front) + 1, toc)
        elif first_chapter is not None and first_chapter.parent is not None:
            parent = first_chapter.parent
            parent.insert(parent.index(first_chapter), toc)
        else:
            top.insert(0, toc)


def setup(app: Any) -> dict[str, Any]:
    app.add_post_transform(AtbdLatexFrontMatter)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
