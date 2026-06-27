# Meeting slide decks

HTML slide decks for BioPAL community meetings and events. Each session lives in a **date-prefixed folder**; shared logos live under `_shared/`.

## Layout

```
docs/presentations/
  _shared/
    logos/                 ← ESA, BioPAL, ACRI-ST, BIOMASS disc (all decks)
  2026-06-30-first-dev-meeting/
    source/                ← slides, CSS, JS, session screenshots/diagrams
    scripts/               ← build, publish
    dist/                  ← generated (gitignored)
```

Published output (via `make html` from `docs/`) lands in `docs/_extra_static/presentations/` and is served at `/docs/presentations/<folder>/index.html`.

**Editing shared logos:** replace files in `_shared/logos/`, then from `docs/` run `make html` (or `make presentation-shared` for a quick logo-only sync).

## Add a new meeting deck

1. Copy `2026-06-30-first-dev-meeting/` to `YYYY-MM-DD-short-title/`
2. Update `source/deck.yaml` (`id`, `date`, `title`, slides)
3. Reference logos as `../_shared/logos/…` in slide HTML
4. Run `make publish` inside the new folder, then `make html` from `docs/`

See [`2026-06-30-first-dev-meeting/README.md`](2026-06-30-first-dev-meeting/README.md).
