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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.transforms import SphinxTransform

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


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("source-read", _source_read)
    app.add_transform(NoSmartQuotes)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
