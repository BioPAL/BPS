#!/usr/bin/env python3
"""Copy dist/ and shared assets into docs/_extra_static for the Sphinx site."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESENTATIONS = ROOT.parent
DOCS = PRESENTATIONS.parent
DIST = ROOT / "dist"
DECK_ID = ROOT.name
SHARED_SRC = PRESENTATIONS / "_shared"
STATIC_PRESENTATIONS = DOCS / "_extra_static" / "presentations"
DECK_TARGET = STATIC_PRESENTATIONS / DECK_ID
SHARED_TARGET = STATIC_PRESENTATIONS / "_shared"


def publish_shared() -> None:
    if not SHARED_SRC.is_dir():
        raise SystemExit(f"Missing shared assets folder: {SHARED_SRC}")
    if SHARED_TARGET.exists():
        shutil.rmtree(SHARED_TARGET)
    shutil.copytree(SHARED_SRC, SHARED_TARGET)
    print(f"Published shared assets → {SHARED_TARGET}")


def publish_deck() -> None:
    if not (DIST / "index.html").is_file():
        raise SystemExit(f"Missing {DIST / 'index.html'} — run scripts/build.py first")
    if DECK_TARGET.exists():
        shutil.rmtree(DECK_TARGET)
    shutil.copytree(DIST, DECK_TARGET)
    print(f"Published deck → {DECK_TARGET}")
    print(f"Live URL (after docs deploy): /docs/presentations/{DECK_ID}/index.html")


def publish() -> None:
    STATIC_PRESENTATIONS.mkdir(parents=True, exist_ok=True)
    publish_shared()
    publish_deck()


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish deck to docs/_extra_static/")
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Sync docs/presentations/_shared/ only (e.g. after editing a logo)",
    )
    args = parser.parse_args()
    if args.shared_only:
        STATIC_PRESENTATIONS.mkdir(parents=True, exist_ok=True)
        publish_shared()
        return
    publish()


if __name__ == "__main__":
    main()
