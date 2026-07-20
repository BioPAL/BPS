<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Open Source readiness — QSM#3 (2026-07-17)

**Audience:** whole BPS project team (scientists, engineers, operators, partners)  
**Language:** English · **~30 min / 25 slides**  
**Tone:** Statement of work · status today · remaining tasks

Published on the docs site at `/docs/presentations/2026-07-17-QSM-3/`.  
Logos: [`../_shared/logos/`](../_shared/logos/).

## Quick start

```bash
cd docs/presentations/2026-07-17-QSM-3
pip install -r requirements.txt   # once: PyYAML

make publish                      # build dist/ + copy to docs/_extra_static/
make serve                        # http://localhost:8765/2026-07-17-QSM-3/index.html
```

From the docs tree:

```bash
cd docs
make html                         # syncs shared logos, publishes deck, builds site
```

Navigation: **← / →**, **Space**, thumbnail rail, **R** to reset.

## Source layout

| Path | Purpose |
|------|---------|
| `source/slides/` | One HTML file per slide (`NN-slug.html`, 01–25) |
| `source/deck.yaml` | Slide order, deck title, dimensions |
| `meeting.yaml` | Docs status badge (`planned`, `held`, `cancelled`) |
| `source/css/`, `source/js/` | Shared chrome and interactions |
| `source/assets/` | Session screenshots |
| `../_shared/logos/` | Logos used by all meeting decks |
| `dist/` | Generated output (gitignored) |
| `scripts/generate_slides.py` | Regenerates HTML from embedded content |
| `scripts/` | `build`, `publish_docs`, `serve`, `generate` |

Edit content in `scripts/generate_slides.py`, then `make generate && make publish`. Or edit slides under `source/slides/` and update `source/deck.yaml` if you add, remove, or reorder slides.

**Session status on the docs site:** edit `meeting.yaml` and set `status` to one of `planned`, `held`, or `cancelled`. Rebuild with `make html` from `docs/`.

Published copy lands in `docs/_extra_static/presentations/2026-07-17-QSM-3/`.

## Acts

1. **Context and foundations** (01–07) — where BPS stands today  
2. **Public materials** (08–16) — website, contribution path; walkthrough on slide 15  
3. **Statement of work** (17–25) — task tables, status today, summary close  
