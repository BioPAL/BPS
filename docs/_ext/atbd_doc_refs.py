# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Link [ADn] / [RDn] codes, Fig.N, sec. X.Y.Z, and eq. X.Y references in the ATBDs.

Targets come from ``science-guide/<folder>/targets.json`` when the folder ships one
(documents converted by esa-doc2open), otherwise from the hard-coded AGB tables
below. ``targets.json``::

    {"doc": "science-guide/atbd-l2-fh",
     "targets": {"fig:3": {"file": "02-bps-fh-overview", "anchor": "fig-atbd-fh-3"},
                 "rd:8": {"file": "01-introduction", "anchor": "rd8"}}}

Every role also accepts an optional display prefix, ``{fig}`Figure|3```.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxRole

INTRO_DOC = "science-guide/atbd-l2-agb/01-introduction"

# PDF figure number -> (source document, HTML anchor id on the figure wrapper).
FIGURE_TARGETS: dict[str, tuple[str, str]] = {
    "1": ("science-guide/atbd-l2-agb/02-bps-agb-overview", "fig-atbd-agb-bps-scheme-tom"),
    "2": ("science-guide/atbd-l2-agb/02-bps-agb-overview", "fig-atbd-agb-bps-scheme-int"),
    "3": ("science-guide/atbd-l2-agb/02-bps-agb-overview", "fig-atbd-agb-workflow"),
    "4": ("science-guide/atbd-l2-agb/03-ground-cancellation", "fig-atbd-agb-int-subset-7"),
    "5": ("science-guide/atbd-l2-agb/03-ground-cancellation", "fig-atbd-agb-int-subset-8"),
    "6": ("science-guide/atbd-l2-agb/03-ground-cancellation", "fig-atbd-agb-calibration"),
    "7": ("science-guide/atbd-l2-agb/03-ground-cancellation", "fig-atbd-agb-gc-two"),
    "8": ("science-guide/atbd-l2-agb/03-ground-cancellation", "fig-atbd-agb-gc-multi"),
    "9": ("science-guide/atbd-l2-agb/03-ground-cancellation", "fig-atbd-agb-sigma-naught-normalisation"),
    "10": ("science-guide/atbd-l2-agb/03-ground-cancellation", "fig-atbd-agb-geocoding"),
    "11": ("science-guide/atbd-l2-agb/03-ground-cancellation", "fig-atbd-agb-geocoding-forest-point"),
    "12": ("science-guide/atbd-l2-agb/04-agb-estimation", "fig-atbd-agb-processing-blocks"),
    "13": ("science-guide/atbd-l2-agb/04-agb-estimation", "fig-atbd-agb-l2b-workflow"),
    "14": ("science-guide/atbd-l2-agb/04-agb-estimation", "fig-atbd-agb-model-parameter-estimation"),
    "15": ("science-guide/atbd-l2-agb/04-agb-estimation", "fig-atbd-agb-agb-mean-std-estimation"),
}

# PDF section number -> (source document, HTML anchor id).
SECTION_TARGETS: dict[str, tuple[str, str]] = {
    "3.3.4": ("science-guide/atbd-l2-agb/03-ground-cancellation", "sec-calibration-math"),
    "3.4.3": ("science-guide/atbd-l2-agb/03-ground-cancellation", "sec-ground-cancellation-outputs"),
    "3.4.4.1": ("science-guide/atbd-l2-agb/03-ground-cancellation", "sec-ground-cancellation-two-images"),
    "3.4.4.2": (
        "science-guide/atbd-l2-agb/03-ground-cancellation",
        "interpolated-ground-cancellation-single-reference",
    ),
    "3.4.4.2.1": (
        "science-guide/atbd-l2-agb/03-ground-cancellation",
        "reference-acquisition-selection",
    ),
    "3.4.4.2.2": (
        "science-guide/atbd-l2-agb/03-ground-cancellation",
        "multi-reference-selection",
    ),
    "4.1.4.2": ("science-guide/atbd-l2-agb/04-agb-estimation", "sec-training-data-consolidation"),
    "4.1.4.3": (
        "science-guide/atbd-l2-agb/04-agb-estimation",
        "regression-based-parameter-estimation",
    ),
}

# PDF equation number (\\tag{X.Y}) -> (source document, HTML anchor id on the equation block).
EQUATION_TARGETS: dict[str, tuple[str, str]] = {
    "3.1": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-forest-coverage-check-1"),
    "3.2": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-calibration-1"),
    "3.3": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-calibration-2"),
    "3.4": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-ground-cancellation-1"),
    "3.5": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-ground-cancellation-2"),
    "3.6": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-ground-cancellation-3"),
    "3.7": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-ground-cancellation-4"),
    "3.8": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-ground-cancellation-5"),
    "3.9": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-ground-cancellation-6"),
    "3.10": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-ground-cancellation-7"),
    "3.11": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-ground-cancellation-8"),
    "3.12": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-ground-cancellation-9"),
    "3.13": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-sigma-naught-normalisation-1"),
    "3.14": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-sigma-naught-normalisation-2"),
    "3.15": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-sigma-naught-normalisation-3"),
    "3.16": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-geocoding-1"),
    "3.17": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-geocoding-2"),
    "3.18": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-geocoding-3"),
    "3.19": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-fnf-annotation-1"),
    "3.20": ("science-guide/atbd-l2-agb/03-ground-cancellation", "equation-eq-fnf-annotation-2"),
    "4.1": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-1"),
    "4.2": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-2"),
    "4.3": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-3"),
    "4.4": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-4"),
    "4.5": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-5"),
    "4.6": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-6"),
    "4.7": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-7"),
    "4.8": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-8"),
    "4.9": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-9"),
    "4.10": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-10"),
    "4.11": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-11"),
    "4.12": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-12"),
    "4.13": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-13"),
    "4.14": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-14"),
    "4.15": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-15"),
    "4.16": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-16"),
    "4.17": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-17"),
    "4.18": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-18"),
    "4.19": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-19"),
    "4.20": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-20"),
    "4.21": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-21"),
    "4.22": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-22"),
    "4.23": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-23"),
    "4.24": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-24"),
    "4.25": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-25"),
    "4.26": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-26"),
    "4.27": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-27"),
    "4.28": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-28"),
    "4.29": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-29"),
    "4.30": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-30"),
    "4.31": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-31"),
    "4.32": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-32"),
    "4.33": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-33"),
    "4.34": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-34"),
    "4.35": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-35"),
    "4.36": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-36"),
    "4.37": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-37"),
    "4.38": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-38"),
    "4.39": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-39"),
    "4.40": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-40"),
    "4.41": ("science-guide/atbd-l2-agb/04-agb-estimation", "equation-eq-model-parameter-estimation-41"),
    "5.1": ("science-guide/atbd-l2-agb/05-appendix", "equation-eq-appendix-gn-1"),
}


def _doc_link_uri(from_docname: str, target_docname: str, anchor: str) -> str:
    if from_docname == target_docname:
        return f"#{anchor}"
    rel = PurePosixPath(
        os.path.relpath(
            PurePosixPath(target_docname).with_suffix(".html"),
            start=PurePosixPath(from_docname).parent,
        )
    )
    return f"{rel.as_posix()}#{anchor}"


def _intro_link_uri(from_docname: str, anchor: str) -> str:
    return _doc_link_uri(from_docname, INTRO_DOC, anchor)


def _folder_of(docname: str) -> str | None:
    parts = docname.split("/")
    if len(parts) >= 3 and parts[0] == "science-guide":
        return "/".join(parts[:2])
    return None


_TARGETS_CACHE: dict[str, tuple[float, dict[str, dict[str, str]] | None]] = {}


def folder_targets(env: Any, docname: str) -> dict[str, dict[str, str]] | None:
    """``targets`` of the document folder holding ``docname``, or None (AGB tables).

    Cached per file modification time in a module dict (not in the pickled build
    environment), so an incremental build always sees a regenerated targets.json.
    """
    folder = _folder_of(docname)
    if folder is None:
        return None
    path = Path(env.srcdir) / folder / "targets.json"
    if not path.is_file():
        return None
    env.note_dependency(str(path))
    mtime = path.stat().st_mtime
    hit = _TARGETS_CACHE.get(str(path))
    if hit is None or hit[0] != mtime:
        data = json.loads(path.read_text(encoding="utf-8")).get("targets")
        _TARGETS_CACHE[str(path)] = hit = (mtime, data)
    return hit[1]


def _resolve(role: SphinxRole, kind: str, num: str,
             table: dict[str, tuple[str, str]] | None) -> tuple[str, str] | None:
    """(target docname, anchor) for ``kind:num``, from targets.json or ``table``."""
    targets = folder_targets(role.env, role.env.docname)
    if targets is not None:
        hit = targets.get(f"{kind}:{num}")
        if hit is None:
            return None
        return f"{_folder_of(role.env.docname)}/{hit['file']}", hit["anchor"]
    if table is None:  # RD/AD of the AGB: introduction page
        return INTRO_DOC, f"{kind}{num}"
    return table.get(num)


def _split_prefix(raw: str) -> tuple[str | None, str]:
    if "|" in raw:
        prefix, num = (part.strip() for part in raw.split("|", 1))
        return prefix, num
    return None, raw.strip()


class _DocCodeRole(SphinxRole):
    prefix: str

    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        num = self.text.strip()
        if not num.isdigit():
            msg = self.inliner.reporter.error(
                f"Invalid {self.prefix.upper()} number: {self.text!r}",
                line=self.lineno,
            )
            return [], [msg]
        label = f"[{self.prefix.upper()}{num}]"
        hit = _resolve(self, self.prefix, num, None)
        if hit is None:
            msg = self.inliner.reporter.error(
                f"Unknown {self.prefix.upper()} number: {self.text!r}",
                line=self.lineno,
            )
            return [], [msg]
        refuri = _doc_link_uri(self.env.docname, *hit)
        node = nodes.reference(self.rawtext, label, refuri=refuri, internal=True)
        return [node], []


class ADRole(_DocCodeRole):
    prefix = "ad"


class RDRole(_DocCodeRole):
    prefix = "rd"


class FigRole(SphinxRole):
    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        prefix, num = _split_prefix(self.text)
        hit = _resolve(self, "fig", num, FIGURE_TARGETS)
        if hit is None:
            msg = self.inliner.reporter.error(
                f"Unknown figure number: {self.text!r}",
                line=self.lineno,
            )
            return [], [msg]
        target_doc, anchor = hit
        label = f"{prefix} {num}" if prefix else f"Fig.{num}"
        refuri = _doc_link_uri(self.env.docname, target_doc, anchor)
        node = nodes.reference(self.rawtext, label, refuri=refuri, internal=True)
        return [node], []


class SecRole(SphinxRole):
    """Link PDF-style section references, e.g. {sec}`3.4.4.1` or {sec}`Section|3.3.4`."""

    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        raw = self.text.strip()
        if "|" in raw:
            prefix, num = (part.strip() for part in raw.split("|", 1))
        else:
            prefix, num = "sec.", raw

        prefix_lower = prefix.lower().rstrip(".")
        if prefix_lower == "section":
            label = f"Section {num}"
        elif prefix_lower == "sec":
            label = f"sec. {num}"
        elif prefix.endswith("."):
            label = f"{prefix} {num}"
        else:
            label = f"{prefix} {num}"

        hit = _resolve(self, "sec", num, SECTION_TARGETS)
        if hit is None:
            msg = self.inliner.reporter.error(
                f"Unknown section number: {num!r}",
                line=self.lineno,
            )
            return [], [msg]

        target_doc, anchor = hit
        refuri = _doc_link_uri(self.env.docname, target_doc, anchor)
        node = nodes.reference(self.rawtext, label, refuri=refuri, internal=True)
        return [node], []


class EqRole(SphinxRole):
    """Link PDF-style equation references, e.g. {eq}`3.17` or {eq}`eq|3.17` (no dot)."""

    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        raw = self.text.strip()
        if "|" in raw:
            prefix, num = (part.strip() for part in raw.split("|", 1))
        else:
            prefix, num = "eq.", raw

        prefix_lower = prefix.lower().rstrip(".")
        if prefix_lower == "eq":
            label = f"eq {num}" if prefix == "eq" else f"eq. {num}"
        elif prefix.endswith("."):
            label = f"{prefix} {num}"
        else:
            label = f"{prefix} {num}"

        hit = _resolve(self, "eq", num, EQUATION_TARGETS)
        if hit is None:
            msg = self.inliner.reporter.error(
                f"Unknown equation number: {num!r}",
                line=self.lineno,
            )
            return [], [msg]

        target_doc, anchor = hit
        refuri = _doc_link_uri(self.env.docname, target_doc, anchor)
        node = nodes.reference(self.rawtext, label, refuri=refuri, internal=True)
        return [node], []


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_role("ad", ADRole())
    app.add_role("rd", RDRole())
    app.add_role("fig", FigRole())
    app.add_role("sec", SecRole())
    app.add_role("eq", EqRole(), override=True)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
