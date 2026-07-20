#!/usr/bin/env python3
"""Assemble source/slides + deck.yaml into dist/index.html."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
DIST = ROOT / "dist"
DECK_YAML = SOURCE / "deck.yaml"

FONT_LINKS = """\
<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,700;1,400&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css" crossorigin="anonymous">"""

CSS_FILES = [
    "css/tokens.css",
    "css/chrome.css",
    "css/components.css",
    "css/interactions.css",
]


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def build() -> None:
    deck = yaml.safe_load(DECK_YAML.read_text(encoding="utf-8"))
    title = deck.get("title", "BPS Presentation")
    width = deck.get("width", 1920)
    height = deck.get("height", 1080)

    slide_parts: list[str] = []
    for entry in deck["slides"]:
        slide_path = SOURCE / entry["file"]
        if not slide_path.is_file():
            raise SystemExit(f"Missing slide file: {slide_path}")
        slide_parts.append(slide_path.read_text(encoding="utf-8").strip())

    css_links = "\n".join(f'<link rel="stylesheet" href="./{f}">' for f in CSS_FILES)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{FONT_LINKS}
{css_links}
<style>deck-stage:not(:defined){{visibility:hidden}}</style>
</head>
<body>
<deck-stage width="{width}" height="{height}">
{chr(10).join(slide_parts)}
</deck-stage>
<script src="./js/deck-stage.js"></script>
<script src="./js/deck-interactions.js"></script>
</body>
</html>
"""

    DIST.mkdir(parents=True, exist_ok=True)
    (DIST / "index.html").write_text(html, encoding="utf-8")

    copy_tree(SOURCE / "css", DIST / "css")
    copy_tree(SOURCE / "js", DIST / "js")
    for sub in ("screenshots", "diagrams"):
        src = SOURCE / "assets" / sub
        if src.is_dir():
            copy_tree(src, DIST / "assets" / sub)

    print(f"Wrote {DIST / 'index.html'} ({len(slide_parts)} slides)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build presentation dist/")
    parser.parse_args()
    if not DECK_YAML.is_file():
        raise SystemExit(f"Missing {DECK_YAML}")
    build()


if __name__ == "__main__":
    main()
