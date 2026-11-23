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


def _chain(depth, members=3):
    body = "".join("    m%d : string [1]\n" % i for i in range(members))
    code = "%%{init: {}}%%\nclassDiagram\n  direction LR\n"
    for d in range(depth):
        code += "  class T%d {\n%s  }\n" % (d, body)
    for d in range(depth - 1):
        code += "  T%d --> T%d : child%d\n" % (d, d + 1, d)
    return code


def test_class_diagram_is_cut_by_subtrees():
    from atbd_latex_content import _class_extent, _parse_classes, _split_class_diagram
    code = _chain(8)
    _pre, order, classes, edges = _parse_classes(code)
    assert _class_extent(classes, edges, order[0])[0] > 1000
    parts = _split_class_diagram(code)
    assert parts and len(parts) > 1
    # the first part draws its last type by name only, the next one heads on it
    assert "  class T1\n" in _split_class_diagram(code, width=300.0)[0]
    for part in parts:
        pre, ids, cls, edg = _parse_classes(part) or ("", [], {}, [])
        assert part.startswith("%%{init: {}}%%\nclassDiagram\n  direction LR\n")
    # a diagram that fits stays whole
    assert _split_class_diagram(_chain(2)) is None


_FLOW = """%%{init: {}}%%
flowchart TB
  accTitle: t
  classDef data fill:#fff
  src[/"in"/]:::data
  subgraph A["lane A"]
    a1["a1"]
    a2["a2"]
  end
  mid[/"mid"/]:::data
  subgraph B["lane B"]
    b1["b1"]
  end
  out[/"out"/]:::data
  src --> a1
  a1 --> a2
  a2 -- "x" --> mid
  mid --> b1
  b1 --> out
  linkStyle 3 stroke:#f00
  style A fill:#eee
  style B fill:#ddd
"""


def test_flowchart_is_cut_between_lanes():
    from atbd_latex_content import _split_flow_lanes
    first, second = _split_flow_lanes(_FLOW, 2)
    # loose nodes go with the first lane they are linked to
    assert 'src[/"in"/]:::data' in first and 'mid[/"mid"/]:::data' in first
    assert 'out[/"out"/]:::data' in second and "subgraph B" not in first
    # a link to a lane left out ends on a copy of its node, and link styles follow
    assert 'stub_b1["b1"]' in first and "mid --> stub_b1" in first
    assert 'stub_mid[/"mid"/]:::data' in second and "stub_mid --> b1" in second
    assert "linkStyle 0 stroke:#f00" in second and "linkStyle 3 stroke:#f00" in first
    assert "style B" in second and "style A" not in second
    assert _split_flow_lanes("flowchart TB\n  a --> b\n", 2) is None
