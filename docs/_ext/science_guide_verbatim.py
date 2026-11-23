# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Keep the characters of the converted Science Guide documents as written.

The pages of ``science-guide/<folder>/`` that esa-doc2open converts (the
folder ships a ``targets.json``) reproduce an approved document character by
character. Two typographic passes of the site would change them:

- MyST ``smartquotes`` and ``replacements`` (markdown-it typographer):
  ``"`` and ``'`` become curly quotes, ``--`` an en dash, ``...`` an
  ellipsis, ``(c)`` a copyright sign;
- Sphinx ``smartquotes`` (docutils): the same for quotes, dashes and ellipses.

For these pages only, the MyST rules are disabled through the file-level
configuration (``myst: disable_syntax``, injected at ``source-read``) and the
docutils pass through ``settings.smart_quotes``. Every other page of the site
keeps the global configuration.

In the HTML of these pages, a long identifier in a table cell
(``doubleWithUnitType``, ``rangeOrbitCoregistrationShifts``, an XPath) gets
``<wbr>`` break opportunities at its word boundaries (camelCase, ``_``, ``.``,
``:``, ``/``): a narrow column wraps it between words instead of cutting a
word or scrolling the table (L1 PFD review W5-1, W5-9). ``<wbr>`` is no
character: a copied identifier stays whole. The PDF is untouched.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import SphinxPostTransform

FRONT_MATTER = "---\nmyst:\n  disable_syntax: [smartquotes, replacements]\n---\n"


def is_converted(app: Sphinx, docname: str) -> bool:
    parts = docname.split("/")
    return (len(parts) >= 3 and parts[0] == "science-guide"
            and (Path(app.srcdir) / parts[0] / parts[1] / "targets.json").is_file())


def _source_read(app: Sphinx, docname: str, source: list[str]) -> None:
    if is_converted(app, docname) and not source[0].lstrip().startswith("---"):
        source[0] = FRONT_MATTER + source[0]


class NoSmartQuotes(SphinxTransform):
    """Runs before ``SphinxSmartQuotes`` (priority 750)."""

    default_priority = 740

    def apply(self, **kwargs: Any) -> None:
        if is_converted(self.app, self.env.docname):
            self.document.settings.smart_quotes = False


_BREAK_AT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[_.:/])(?=\w)")
MIN_LENGTH = 16
"""Shorter identifiers fit any column whole."""


def break_points(text: str) -> list[str]:
    """``text`` cut at its word boundaries (no cut: one piece)."""
    if len(text) < MIN_LENGTH:
        return [text]
    return [p for p in _BREAK_AT.split(text) if p]


class IdentifierBreaks(SphinxPostTransform):
    """``<wbr>`` inside long identifiers of table cells (HTML only)."""

    default_priority = 900
    formats = ("html",)

    def run(self, **kwargs: Any) -> None:
        if not is_converted(self.app, self.env.docname):
            return
        for lit in list(self.document.findall(nodes.literal)):
            if not any(isinstance(a, nodes.entry) for a in _ancestors(lit)):
                continue
            if len(lit.children) != 1 or not isinstance(lit[0], nodes.Text):
                continue
            pieces = break_points(lit.astext())
            if len(pieces) < 2:
                continue
            lit.children = []
            for i, piece in enumerate(pieces):
                if i:
                    lit += nodes.raw("", "<wbr>", format="html")
                lit += nodes.Text(piece)


def _ancestors(node: nodes.Node):
    while node.parent is not None:
        node = node.parent
        yield node


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("source-read", _source_read)
    app.add_transform(NoSmartQuotes)
    app.add_post_transform(IdentifierBreaks)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
