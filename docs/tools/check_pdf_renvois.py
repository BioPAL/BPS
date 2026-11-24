#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Check the links of exported PDFs: table references land on the page of the
table caption, and no external URL carries a backslash.

Usage: check_pdf_renvois.py FILE.pdf [FILE.pdf ...]   (needs PyMuPDF)
"""
from __future__ import annotations

import re
import sys

import pymupdf

TABLE = re.compile(r"^\s*(?:Table|Tab\.)\s*(\d+)\b")


def check(path: str) -> tuple[int, int, int]:
    doc = pymupdf.open(path)
    captions = {}
    for number, page in enumerate(doc):
        for m in re.finditer(r"^(?:Table|Tab\.)\s*(\d+)\s", page.get_text(), re.M):
            captions.setdefault(m.group(1), set()).add(number)
    refs = bad = 0
    for number, page in enumerate(doc):
        for link in page.get_links():
            if link["kind"] not in (pymupdf.LINK_GOTO, pymupdf.LINK_NAMED):
                continue
            text = page.get_textbox(link["from"])
            m = TABLE.match(text)
            if not m or m.group(1) not in captions:
                continue
            refs += 1
            if link["page"] not in captions[m.group(1)]:
                bad += 1
                print(f"{path}: p.{number + 1} {text.strip()!r} -> p.{link['page'] + 1}, "
                      f"caption on p.{sorted(p + 1 for p in captions[m.group(1)])}")
    urls = [lk["uri"] for page in doc for lk in page.get_links()
            if lk["kind"] == pymupdf.LINK_URI and ("\\" in lk["uri"] or "%5C" in lk["uri"])]
    for u in urls:
        print(f"{path}: URL with a backslash: {u}")
    return refs, bad, len(urls)


if __name__ == "__main__":
    total = [0, 0, 0]
    for f in sys.argv[1:]:
        for i, v in enumerate(check(f)):
            total[i] += v
    print(f"table references {total[0]}, mis-targeted {total[1]}, backslash URLs {total[2]}")
    sys.exit(1 if total[1] or total[2] else 0)
