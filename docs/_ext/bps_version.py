# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
"""Resolve BPS release version from git tags or GitHub."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

_BPS_TAG_RE = re.compile(r"^bps-v(?P<version>.+)$", re.IGNORECASE)
_GITHUB_TAGS_API = "https://api.github.com/repos/BioPAL/BPS/tags?per_page=100"


def _parse_bps_tag(tag: str) -> tuple[str, str] | None:
    match = _BPS_TAG_RE.match(tag.strip())
    if not match:
        return None
    version = match.group("version")
    return version, f"bps-v{version}"


def _from_env() -> tuple[str, str] | None:
    raw = os.environ.get("BPS_RELEASE_VERSION") or os.environ.get("DOCS_BPS_VERSION")
    if not raw:
        return None
    raw = raw.strip()
    if raw.lower().startswith("bps-v"):
        parsed = _parse_bps_tag(raw)
        return parsed
    version = raw.lstrip("v")
    return version, f"bps-v{version}"


def _from_git(repo_root: Path) -> tuple[str, str] | None:
    try:
        proc = subprocess.run(
            ["git", "tag", "-l", "bps-v*", "--sort=-v:refname"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=repo_root,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    for line in proc.stdout.splitlines():
        parsed = _parse_bps_tag(line)
        if parsed:
            return parsed
    return None


def _from_github_api() -> tuple[str, str] | None:
    try:
        request = urllib.request.Request(
            _GITHUB_TAGS_API,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "BioPAL-Sphinx-Docs",
            },
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return None

    candidates: list[tuple[tuple[int, ...], str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        parsed = _parse_bps_tag(str(item.get("name", "")))
        if not parsed:
            continue
        version, tag = parsed
        parts = tuple(int(part) for part in version.split("."))
        candidates.append((parts, version, tag))

    if not candidates:
        return None

    _, version, tag = max(candidates, key=lambda item: item[0])
    return version, tag


def resolve_bps_release(
    *,
    docs_dir: Path,
    fallback_version: str = "unknown",
) -> tuple[str, str]:
    """Return (display_version, tag_name) e.g. ('4.4.5', 'bps-v4.4.5')."""
    env = _from_env()
    if env:
        return env

    repo_root = docs_dir.parent
    git = _from_git(repo_root)
    if git:
        return git

    remote = _from_github_api()
    if remote:
        return remote

    return fallback_version, f"bps-v{fallback_version}"


class IndexHeroMetaDirective(SphinxDirective):
    """Render the homepage BPS version badge from config.bps_version."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        return _hero_meta_nodes(self.config)


def _hero_meta_html(config: Any) -> str:
    version = config.bps_version
    url = config.bps_releases_url
    return (
        f'<p class="index-hero__meta">'
        f'<a class="index-version-badge" href="{url}">'
        f"BPS v{version}</a>"
        f'<span class="index-version-note">Processing Suite release</span>'
        f"</p>"
    )


def _hero_meta_nodes(config: Any) -> list[nodes.Node]:
    return [nodes.raw("", _hero_meta_html(config), format="html")]


_BADGE_PLACEHOLDER = "<!-- bps-version-badge -->"


def _inject_bps_version_badge(app: Any, docname: str, source: list[str]) -> None:
    if docname != "index" or _BADGE_PLACEHOLDER not in source[0]:
        return
    source[0] = source[0].replace(_BADGE_PLACEHOLDER, _hero_meta_html(app.config))


def setup(app: Any) -> dict[str, Any]:
    docs_dir = Path(app.confdir)
    version, tag = resolve_bps_release(docs_dir=docs_dir)
    releases_url = f"https://github.com/BioPAL/BPS/releases/tag/{tag}"

    app.add_config_value("bps_version", version, "html")
    app.add_config_value("bps_tag", tag, "html")
    app.add_config_value("bps_releases_url", releases_url, "html")
    app.add_directive("index-hero-meta", IndexHeroMetaDirective)
    app.connect("source-read", _inject_bps_version_badge)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
