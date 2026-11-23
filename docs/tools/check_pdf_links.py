#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Check the "Download PDF" buttons of the built Science Guide pages.

Walks ``<html>/science-guide/*/**.html``, extracts the link of each page's
PDF button and fails when it points to a PDF absent from the built site.

With ``--expect-all`` (or ``ATBD_PDF_EXPECT_ALL=1``), also fails when a page of
a document has no button (every PDF is expected to have been built).

Usage (from docs/):  python3 tools/check_pdf_links.py [--html _build/html]
                     [--expect-all]   # or: make check-pdf-links EXPECT_ALL=1
Tests:               python3 -m pytest tools
"""

from __future__ import annotations

import argparse
import os
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class _ButtonParser(HTMLParser):
    """Collect the href of the link inside ``<div class="... atbd-download-pdf">``."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self._depth = 0  # >0 while inside the button's div

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "div" and "atbd-download-pdf" in (a.get("class") or "").split():
            self._depth = 1
        elif self._depth and tag == "div":
            self._depth += 1
        elif self._depth and tag == "a" and a.get("href"):
            self.hrefs.append(a["href"] or "")

    def handle_endtag(self, tag: str) -> None:
        if self._depth and tag == "div":
            self._depth -= 1


def button_links(page: Path) -> list[str]:
    parser = _ButtonParser()
    parser.feed(page.read_text(encoding="utf-8"))
    return parser.hrefs


def check(html_root: Path, expect_all: bool) -> tuple[list[str], int, int]:
    """Return (errors, pages checked, buttons found)."""
    errors: list[str] = []
    pages = buttons = 0
    sg = html_root / "science-guide"
    for page in sorted(sg.glob("*/**/*.html")):
        pages += 1
        links = button_links(page)
        rel = page.relative_to(html_root)
        if not links:
            if expect_all:
                errors.append(f"{rel}: no PDF button")
            continue
        for href in links:
            buttons += 1
            target = (page.parent / unquote(urlsplit(href).path)).resolve()
            if not target.is_file():
                errors.append(f"{rel}: button links to missing PDF {href}")
    return errors, pages, buttons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--html", default="_build/html", type=Path)
    ap.add_argument(
        "--expect-all",
        action="store_true",
        default=os.environ.get("ATBD_PDF_EXPECT_ALL", "") not in ("", "0"),
        help="fail when a page of a document has no PDF button",
    )
    args = ap.parse_args()
    if not (args.html / "science-guide").is_dir():
        print(f"No built site in {args.html}: run `make html` first", file=sys.stderr)
        return 2
    errors, pages, buttons = check(args.html, args.expect_all)
    for e in errors:
        print(f"ERROR {e}")
    print(f"{pages} document page(s), {buttons} PDF button(s), {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
