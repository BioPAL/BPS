# First Developer Meeting — 2026-06-30

Slide deck for the **BPS First Developer Meeting** (Open Source & Open Science). Published on the docs site at `/docs/presentations/2026-06-30-first-dev-meeting/`.

Logos are shared across decks: [`../_shared/logos/`](../_shared/logos/).

## Quick start

```bash
cd docs/presentations/2026-06-30-first-dev-meeting
pip install -r requirements.txt   # once: PyYAML

make publish                      # build dist/ + copy to docs/_extra_static/
make serve                        # http://localhost:8765/2026-06-30-first-dev-meeting/index.html
```

From the docs tree:

```bash
cd docs
make html                         # syncs shared logos, publishes deck, builds site
```

Navigation in the deck: **← / →**, **Space**, thumbnail rail, **R** to reset.

## Source layout

| Path | Purpose |
|------|---------|
| `source/slides/` | One HTML file per slide |
| `source/deck.yaml` | Slide order, deck title, dimensions |
| `meeting.yaml` | Docs status badge (`planned`, `held`, `cancelled`) |
| `source/css/`, `source/js/` | Shared chrome and interactions |
| `source/assets/` | Session screenshots and diagrams |
| `../_shared/logos/` | Logos used by all meeting decks |
| `dist/` | Generated output (gitignored) |
| `scripts/` | `build`, `publish_docs`, `serve` |

Edit slides in `source/slides/` and update `source/deck.yaml` if you add, remove, or reorder slides. Then run `make publish` (or `make html` from `docs/`).

**Session status on the docs site:** edit `meeting.yaml` at the deck root and set `status` to one of `planned`, `held`, or `cancelled`. Rebuild docs with `make html` from `docs/`.

Published copy lands in `docs/_extra_static/presentations/2026-06-30-first-dev-meeting/`.

## Docs page

Community entry point: [`docs/communication/developer-meeting.md`](../../communication/developer-meeting.md)
