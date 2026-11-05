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
``_templates/latex/atbdlayout.sty``.
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
_FRONT_HEADINGS = ("Document information", "Change history")
# Full-width grid tables in the front matter (column specs as Sphinx
# ``tabularcolumns``): Field | Value, Name | Organisation, Change history.
_FRONT_TABLE_SPECS = {
    2: r"|\X{1}{3}|\X{2}{3}|",
    4: r"|\X{6}{48}|\X{4}{48}|\X{26}{48}|\X{12}{48}|",
}
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


def _format_date(raw: str) -> str:
    """Dates as printed in the originals: dd/mm/yyyy."""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d %B %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return raw.strip()


def read_document_meta(docs_dir: Path, folder: str) -> dict[str, str]:
    """Title, code, issue and date of ``science-guide/<folder>``.

    Title and code come from ``targets.json``; issue and date from the
    pipeline metadata comment of ``index.md`` (or its Document information
    table when the comment is missing).
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
    return {
        "title": targets.get("title") or comment.get("title") or table.get("title", folder),
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
        )
    )
    logos = [
        str(docs_dir / "_static" / "logos" / name)
        for name in _LOGOS
        if (docs_dir / "_static" / "logos" / name).is_file()
    ]
    return {
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
            str(docs_dir / "_templates" / "latex" / "atbdlayout.sty"),
        ],
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

        def in_front(node: nodes.Node) -> bool:
            parent = node.parent
            while parent is not None:
                if isinstance(parent, addnodes.start_of_file):
                    return False
                parent = parent.parent
            return True

        for para in list(top.findall(nodes.paragraph)):
            if in_front(para) and _is_back_link(para):
                para.parent.remove(para)
        for adm in list(top.findall(nodes.Admonition)):
            if in_front(adm) and _is_web_admonition(adm):
                adm.parent.remove(adm)
        for rule in list(top.findall(nodes.transition)):
            if in_front(rule):
                rule.parent.remove(rule)

        last_front: nodes.Element | None = None
        for rubric in list(top.findall(nodes.rubric)):
            title = rubric.astext().strip()
            if not in_front(rubric) or title not in _FRONT_HEADINGS:
                continue
            last_front = rubric.parent
            for table in list(last_front.findall(nodes.table)):
                _set_front_table_spec(table)
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
