"""Unit test of the PDF layout guard of atbd_latex_content (``python3 -m pytest docs/_ext``)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from atbd_latex_content import _LR_RE  # noqa: E402


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
