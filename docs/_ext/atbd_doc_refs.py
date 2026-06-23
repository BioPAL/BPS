# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Link [ADn] / [RDn] codes, Fig.N, sec. X.Y.Z, and eq. X.Y references in the ATBD."""

from __future__ import annotations

import os
from pathlib import PurePosixPath
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
        anchor = f"{self.prefix}{num}"
        refuri = _intro_link_uri(self.env.docname, anchor)
        node = nodes.reference(self.rawtext, label, refuri=refuri, internal=True)
        return [node], []


class ADRole(_DocCodeRole):
    prefix = "ad"


class RDRole(_DocCodeRole):
    prefix = "rd"


class FigRole(SphinxRole):
    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        num = self.text.strip()
        if num not in FIGURE_TARGETS:
            msg = self.inliner.reporter.error(
                f"Unknown figure number: {self.text!r}",
                line=self.lineno,
            )
            return [], [msg]
        target_doc, anchor = FIGURE_TARGETS[num]
        label = f"Fig.{num}"
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

        if num not in SECTION_TARGETS:
            msg = self.inliner.reporter.error(
                f"Unknown section number: {num!r}",
                line=self.lineno,
            )
            return [], [msg]

        target_doc, anchor = SECTION_TARGETS[num]
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

        if num not in EQUATION_TARGETS:
            msg = self.inliner.reporter.error(
                f"Unknown equation number: {num!r}",
                line=self.lineno,
            )
            return [], [msg]

        target_doc, anchor = EQUATION_TARGETS[num]
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
