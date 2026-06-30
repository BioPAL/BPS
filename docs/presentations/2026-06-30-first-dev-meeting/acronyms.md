# Acronyms reference — First Developer Meeting deck

Hover tooltips use the `.bps-acronym` class in slide HTML. Only the **first occurrence** of each acronym in deck order shows a tooltip (`source/js/deck-interactions.js`).

**Tooltip styling:** `source/css/interactions.css` → `.bps-acronym::after` (`font-size`, `padding`, `max-width`).

**Markup:**

```html
<span class="bps-acronym" data-acronym="PR" data-def="Pull Request">PR</span>
```

Rebuild after edits: `make publish` (or `cd docs && make html`).

---

## Implemented in the deck

| Acronym | Definition | First slide (`data-label`) | Source file |
|---------|------------|----------------------------|-------------|
| BPS | BioMASS Processing Suite | 04 | `04-biomass-bps-…html` |
| SAR | Synthetic Aperture Radar | 04 | `04-biomass-bps-…html` |
| ESA | European Space Agency | 04 | `04-biomass-bps-…html` |
| BioPAL | BioMASS open-source programme on GitHub | 04 | `04-biomass-bps-…html` |
| L1 | Level-1 product | 04 | `04-biomass-bps-…html` |
| L2A | Level-2A product | 04 | `04-biomass-bps-…html` |
| L2B | Level-2B product | 04 | `04-biomass-bps-…html` |
| AGB | Above-Ground Biomass | 04 | `04-biomass-bps-…html` |
| PR | Pull Request | 07 | `07-five-pillars-one-open-project.html` |
| ATBD | Algorithm Theoretical Basis Document | 08 | `08-document-the-project.html` |
| DPM | Detailed Processing Model | 08 | `08-document-the-project.html` |
| FAIR | Findable, Accessible, Interoperable, Reusable | 08 | `08-document-the-project.html` |
| REUSE | REUSE specification (SPDX licence compliance) | 08 | `08-document-the-project.html` |
| SUM | Software User Manual | 11 | `11-find-your-path.html` |
| L1F | Level-1 Framed product | 13 | `13-run-bps-on-your-own-computer.html` |
| STA | Stack / statistics processing step | 13 | `13-run-bps-on-your-own-computer.html` |
| MAAP | Multi-Mission Algorithm and Analysis Platform (ESA) | 13 | `13-run-bps-on-your-own-computer.html` |
| AFD | Algorithm Functional Description | 14 | `14-atbds-afd-pfds-catalog.html` |
| PFD | Product Format Document | 14 | `14-atbds-afd-pfds-catalog.html` |
| FH | Forest Height | 14 | `14-atbds-afd-pfds-catalog.html` |
| FD | Forest Disturbance | 14 | `14-atbds-afd-pfds-catalog.html` |
| CI/CD | Continuous Integration / Continuous Delivery | 19 | `19-what-is-github.html` |
| DCO | Developer Certificate of Origin | 20 | `20-how-biomass-runs-on-github.html` |
| CODEOWNERS | GitHub path-based reviewer routing | 20 | `20-how-biomass-runs-on-github.html` |
| SME | Scientific Module Expert | 20 | `20-how-biomass-runs-on-github.html` |
| SPDX | Software Package Data Exchange | 28 | `28-the-ci-cd-pipeline.html` |
| GPG | GNU Privacy Guard (signed commits) | 28 | `28-the-ci-cd-pipeline.html` |
| DOI | Digital Object Identifier | 32 | `32-after-the-pr.html` |
| Zenodo | Zenodo open research archive | 32 | `32-after-the-pr.html` |

---

## Candidate acronyms (not yet wired)

Mark any row in a slide with `.bps-acronym`; later occurrences in the deck will not show a tooltip.

### Mission and organisations

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| BIOMASS | ESA BIOMASS Earth Explorer mission (P-band SAR) | 04 |
| ACRI-ST | ACRI-ST (project maintainer) | 04 |
| Aresys | Aresys (original BPS developer for ESA) | 04 |

### Documents and science

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| RST | reStructuredText | 17 |
| YAML | YAML configuration format | 17 |

### Products and processing chain

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| JobOrder | BPS job configuration file (XML) | 13 |

### Governance and contribution

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|

### Licences and compliance

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| MIT | MIT open-source licence | 17 |
| Apache 2.0 | Apache Licence 2.0 | 04 |

### Infrastructure and publication

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| Sphinx | Sphinx documentation generator | 07 or 09 |
| GitLab | GitLab (secondary CI trigger) | 17 |

### PR risk tiers (optional)

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| T0 | PR risk tier 0 (low) | 27 |
| T1 | PR risk tier 1 (medium) | 27 |
| T2 | PR risk tier 2 (high) | 27 |

---

## Deck order (for first-occurrence planning)

Slides are ordered in `source/deck.yaml`. Tooltip precedence follows that file, not footer numbers.

| Order | Diapo / `data-label` | Title (short) |
|------:|----------------------|---------------|
| 1 | 01 | Title |
| 2 | 02 | Today's plan |
| 3 | 04 | ACT I |
| 4 | 05 | What is BPS |
| 4 | 05 | Open Source / Open Science |
| 5 | 06 | Why open source |
| 6 | 07 | Five pillars |
| 7 | 08 | Document the project |
| 8 | 09 | ACT II |
| 9 | 13 | Documentation portal |
| 10 | 11 | Find your path |
| 11 | 12 | SUM |
| 12 | 13 | Tutorial |
| 16 | 14 | Science guide catalogue |
| 14 | 15 | Contributing |
| 15 | 16 | Governance |
| 16 | 17 | One repo, one site |
| 17 | 18 | ACT III |
| 18 | 19 | What is GitHub |
| 19 | 20 | How BIOMASS runs on GitHub |
| … | … | … |

See `source/deck.yaml` for the full list (38 slides). File names use the same numeric prefix as diapo order.