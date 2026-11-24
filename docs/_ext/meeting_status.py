# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Render meeting status badges from presentations/<deck>/meeting.yaml."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

_PLACEHOLDER_RE = re.compile(
    r"<!--\s*meeting-status:([a-z0-9-]+)(?::(inline))?\s*-->",
    re.IGNORECASE,
)

STATUSES: dict[str, dict[str, str]] = {
    "planned": {
        "label": "Planned",
        "modifier": "planned",
        "icon": "fa-regular fa-calendar-check",
    },
    "held": {
        "label": "Held",
        "modifier": "held",
        "icon": "fa-solid fa-circle-check",
    },
    "cancelled": {
        "label": "Cancelled",
        "modifier": "cancelled",
        "icon": "fa-solid fa-circle-xmark",
    },
}


def _load_status(confdir: Path, deck_id: str) -> str:
    path = confdir / "presentations" / deck_id / "meeting.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing meeting config: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    status = str(data.get("status", "")).strip().lower()
    if status not in STATUSES:
        allowed = ", ".join(sorted(STATUSES))
        raise ValueError(f"Invalid status {status!r} in {path.name} — use one of: {allowed}")
    return status


def render_badge(confdir: Path, deck_id: str, *, inline: bool = False) -> str:
    status = _load_status(confdir, deck_id)
    meta = STATUSES[status]
    modifier = meta["modifier"]
    size_class = " bps-meeting-status--inline" if inline else ""
    return (
        f'<span class="bps-meeting-status bps-meeting-status--{modifier}{size_class}">'
        f'<i class="{meta["icon"]}" aria-hidden="true"></i>'
        f'<span>{meta["label"]}</span>'
        f"</span>"
    )


class MeetingStatusDirective(SphinxDirective):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False
    option_spec = {"inline": directives.flag}

    def run(self) -> list[nodes.Node]:
        deck_id = self.arguments[0].strip()
        inline = "inline" in self.options
        try:
            html = render_badge(Path(self.env.srcdir), deck_id, inline=inline)
        except (FileNotFoundError, ValueError) as exc:
            raise self.error(str(exc)) from exc
        raw = nodes.raw("", html, format="html")
        return [raw]


def _inject_meeting_status_placeholders(app: Any, docname: str, source: list[str]) -> None:
    confdir = Path(app.srcdir)

    def repl(match: re.Match[str]) -> str:
        deck_id = match.group(1)
        inline = match.group(2) is not None
        try:
            return render_badge(confdir, deck_id, inline=inline)
        except (FileNotFoundError, ValueError) as exc:
            app.warn(str(exc))
            return match.group(0)

    source[0] = _PLACEHOLDER_RE.sub(repl, source[0])


def setup(app: Any) -> dict[str, Any]:
    app.add_directive("meeting-status", MeetingStatusDirective)
    app.connect("source-read", _inject_meeting_status_placeholders)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
