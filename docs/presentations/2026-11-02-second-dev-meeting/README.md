# Second Developer Meeting, 2026-11-02

Slide deck presented by Yoann Rey-Ricord (ACRI-ST) at the **BPS Second Developer Meeting** (Open Source & Open Science), Monday 2 November 2026. Published on the docs site at `/docs/presentations/2026-11-02-second-dev-meeting/`.

40 slides in four acts: I Introduction & intent, II What's live today, III How you contribute (fork and draft PR from the GitHub UI, contributor path, branch and release governance, CI/CD tiers, branch protection), IV Join us (channels, live demo, status update). The deck follows the [First Developer Meeting](../2026-06-30-first-dev-meeting/README.md) structure. The slides are kept as presented on 2 November 2026 (record of the session); only the private demo guide link was removed before publication.

Logos are shared across decks: [`../_shared/logos/`](../_shared/logos/).

## Quick start

```bash
cd docs/presentations/2026-11-02-second-dev-meeting
pip install -r requirements.txt   # once: PyYAML

make publish                      # build dist/ + copy to docs/_extra_static/
make serve                        # http://localhost:8766/2026-11-02-second-dev-meeting/index.html
```

From the docs tree:

```bash
cd docs
make html                         # syncs shared logos, publishes decks, builds site
```

Navigation in the deck: **← / →**, **Space**, thumbnail rail, **R** to reset.

## Source layout

| Path | Purpose |
|------|---------|
| `source/slides/` | One HTML file per slide (`NN-slug.html`, `NN` = slide 01–40) |
| `source/deck.yaml` | Slide order, deck title, dimensions |
| `meeting.yaml` | Docs status badge (`planned`, `held`, `cancelled`) |
| `source/css/`, `source/js/` | Shared chrome and interactions (incl. fork/draft PR components added for this session) |
| `acronyms.md` | Acronym definitions and tooltip reference |
| `source/assets/` | Session screenshots and diagrams |
| `../_shared/logos/` | Logos used by all meeting decks |
| `dist/` | Generated output (gitignored) |
| `scripts/` | `build`, `publish_docs`, `serve` |

Edit slides in `source/slides/` and update `source/deck.yaml` if you add, remove, or reorder slides. Then run `make publish` (or `make html` from `docs/`).

**Session status on the docs site:** edit `meeting.yaml` at the deck root and set `status` to one of `planned`, `held`, or `cancelled`. Rebuild docs with `make html` from `docs/`.

Published copy lands in `docs/_extra_static/presentations/2026-11-02-second-dev-meeting/`.

## Docs page

Community entry point (slides + session recording): [`docs/communication/developer-meeting-2.md`](../../communication/developer-meeting-2.md)
