"""Unit test of the PDF layout guard of atbd_latex_content (``python3 -m pytest docs/_ext``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from atbd_latex_content import _LR_RE, _SCRIPT_CSS, _readable_scripts  # noqa: E402


def test_left_to_right_diagrams_are_recognised_and_turned_top_down():
    xsd = "classDiagram\n  direction LR\n  class A"
    flow = "flowchart LR\n  a --> b\n  subgraph s\n    direction LR\n  end"
    graph = "graph LR\n  a --> b"
    assert _LR_RE.sub(r"\1TB", xsd) == "classDiagram\n  direction TB\n  class A"
    assert _LR_RE.sub(r"\1TB", flow) == (
        "flowchart TB\n  a --> b\n  subgraph s\n    direction TB\n  end")
    assert _LR_RE.sub(r"\1TB", graph).startswith("graph TB")
    for other in ("flowchart TB\n  a --> b", "flowchart RL\n  a --> b",
                  "flowchart LR2\n", "  a[flowchart LR] --> b"):
        assert not _LR_RE.search(other), other


def test_equation_tag_pattern_is_not_the_html_tag_pattern():
    # A second ``_TAG_RE`` (HTML tags of captions) once shadowed the ``\tag``
    # one: numbered equations stayed in Sphinx's ``split`` (``\tag not allowed here``).
    from atbd_latex_content import _TAG_RE
    match = _TAG_RE.search("a=b\n\\tag{3.1}")
    assert match is not None and match.group("num") == "3.1"
    assert _TAG_RE.sub("", "a=b\n\\tag*{(A)}").strip() == "a=b"


def test_display_formulas_pick_the_fitting_macro():
    from atbd_latex_content import _alternative, _display_macro
    assert _display_macro("a=b+c") == "atbdfitline"
    assert _display_macro("a&=b\\\\ c&=d") == "atbdfitdisplay"
    assert _alternative("a=b") == ""
    assert _alternative(r"\left[ N\times{}M \right]") == r"[{\bigl[ N\times{}M \bigr]}]"
    assert _alternative(r"\left\{ a \right\}") == r"[{\bigl\{ a \bigr\}}]"
    # stacked content or mixed delimiters keep their automatic size
    assert _alternative(r"\left( \frac{a}{b} \right)") == ""
    assert _alternative(r"\left. f \right\vert_{x}") == ""
    assert _alternative(r"\left( a \right.") == ""


def test_mermaid_scripts_are_set_at_the_size_of_the_text():
    with_init = '%%{init: {"theme": "default"}}%%\nflowchart TB\n  a["A<sub>i</sub>"]'
    fixed = _readable_scripts(with_init)
    assert '"themeCSS": "%s"' % _SCRIPT_CSS in fixed and '"theme": "default"' in fixed
    assert fixed.endswith("\nflowchart TB\n  a[\"A<sub>i</sub>\"]")
    bare = _readable_scripts("flowchart TB\n  a[\"x<sup>2</sup>\"]")
    assert bare.startswith("%%{init: ") and "themeCSS" in bare
    assert _readable_scripts("classDiagram\n  class A") == "classDiagram\n  class A"
    assert _readable_scripts(fixed) == fixed


def test_table_columns_are_as_wide_as_their_longest_word():
    from docutils import nodes
    from atbd_latex_content import _column_widths, _wlen

    # Arial metrics: "i" is narrower and "W" wider than the half em of a character.
    assert _wlen("iiii") < 4 < _wlen("WWWW") and _wlen("Name", bold=True) > _wlen("Name")

    def table(rows):
        tgroup = nodes.tgroup(cols=len(rows[0]))
        for _ in rows[0]:
            tgroup += nodes.colspec(colwidth=1)
        tbody = nodes.tbody()
        for cells in rows:
            tbody += nodes.row("", *(nodes.entry("", nodes.paragraph("", text)) for text in cells))
        tgroup += tbody
        return nodes.table("", tgroup, classes=["colwidths-auto"])

    word = "meanAGBLimitsAUXPP2Name"
    tab = table([["x", "y y y y y y y y y y y y y y y y y y y y y y y y y y y y y y y y"],
                 [word, "z"]])
    _column_widths(tab, budget=60.0, pad=2.5)
    first, second = [int(c["colwidth"]) for c in tab.findall(nodes.colspec)]
    assert "colwidths-given" in tab["classes"]
    # the first column holds the whole identifier: its share is at least its width
    assert first / (first + second) * 60.0 >= _wlen(word) + 2.5


def test_captions_carry_the_number_of_the_original():
    from atbd_latex_content import _set_number
    assert _set_number("table", "71") == r"\setcounter{table}{70}"
    assert _set_number("figure", "A.1") == r"\renewcommand{\thefigure}{A.1}"
