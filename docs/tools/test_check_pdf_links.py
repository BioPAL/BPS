"""Tests of check_pdf_links (``python3 -m pytest docs/tools``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from check_pdf_links import check  # noqa: E402

BUTTON = (
    '<div class="tocsection editthispage atbd-download-pdf">'
    '<a href="{href}" download>Download PDF (1.0 MB)</a></div>'
)


def _site(tmp_path, pages):
    for name, body in pages.items():
        f = tmp_path / "science-guide" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
    return tmp_path


def test_existing_pdf_passes(tmp_path):
    pdf = tmp_path / "_static" / "pdf" / "d.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF")
    site = _site(tmp_path, {"d/index.html": BUTTON.format(href="../../_static/pdf/d.pdf")})
    assert check(site, True) == ([], 1, 1)


def test_missing_pdf_fails(tmp_path):
    site = _site(tmp_path, {"d/index.html": BUTTON.format(href="../../_static/pdf/d.pdf")})
    errors, _, _ = check(site, False)
    assert len(errors) == 1 and "missing PDF" in errors[0]


def test_missing_button_fails_only_when_expected(tmp_path):
    site = _site(tmp_path, {"d/ch1.html": "<p>no button</p>"})
    assert check(site, False)[0] == []
    assert "no PDF button" in check(site, True)[0][0]
