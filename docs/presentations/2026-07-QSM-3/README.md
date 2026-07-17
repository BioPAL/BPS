<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Open Source readiness - Statement of work

**Audience:** whole BPS project team (scientists, engineers, operators, partners)  
**Language:** English · **~30 min / ~26 slides**  
**Tone:** Statement of work · status today · remaining tasks

Logos: [`../_shared/logos/`](../_shared/logos/).

## Quick start

```bash
cd docs/presentations/2026-07-QSM-3
pip install -r requirements.txt   # once: PyYAML

make publish                      # build dist/ + copy to docs/_extra_static/
make serve                        # http://localhost:8765/2026-07-QSM-3/index.html
```

Navigation: **← / →**, **Space**, thumbnail rail, **R** to reset.

## Files

| File | Role |
|------|------|
| `source/slides/` | 26 HTML slides |
| `source/deck.yaml` | Slide order |
| [speaker-notes.md](speaker-notes.md) | Oral script + jargon crib sheet |
| [analyse-open-source-readiness.md](analyse-open-source-readiness.md) | Full diagnostic |
| [backlog-actions.md](backlog-actions.md) | Full backlog (slides show curated P0/P1) |
| `scripts/generate_slides.py` | Regenerates HTML from embedded content |

## Acts

1. **Context and foundations** (01-07) - where BPS stands today
2. **Public materials** (08-16) - website, contribution path; walkthrough on slide 15
3. **Statement of work** - task tables, status today, summary close

After editing `scripts/generate_slides.py`: `make generate && make publish`.
