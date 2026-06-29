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
| BPS | BioMASS Processing Suite | 02 | `02-biomass-bps-…html` |
| SAR | Synthetic Aperture Radar | 02 | `02-biomass-bps-…html` |
| ESA | European Space Agency | 02 | `02-biomass-bps-…html` |
| BioPAL | BioMASS open-source programme on GitHub | 02 | `02-biomass-bps-…html` |
| L1 | Level-1 product | 02 | `02-biomass-bps-…html` |
| L2A | Level-2A product | 02 | `02-biomass-bps-…html` |
| L2B | Level-2B product | 02 | `02-biomass-bps-…html` |
| AGB | Above-Ground Biomass | 02 | `02-biomass-bps-…html` |
| PR | Pull Request | 04 | `04-five-pillars-one-open-project.html` |
| ATBD | Algorithm Theoretical Basis Document | 04b | `04b-document-the-project.html` |
| DPM | Detailed Processing Model | 04b | `04b-document-the-project.html` |
| FAIR | Findable, Accessible, Interoperable, Reusable | 04b | `04b-document-the-project.html` |
| REUSE | REUSE specification (SPDX licence compliance) | 04b | `04b-document-the-project.html` |
| SUM | Software User Manual | 07 | `07-find-your-path.html` |
| L1F | Level-1 Framed product | 09 | `09-run-bps-on-your-own-computer.html` |
| STA | Stack / statistics processing step | 09 | `09-run-bps-on-your-own-computer.html` |
| MAAP | Multi-Mission Algorithm and Analysis Platform (ESA) | 09 | `09-run-bps-on-your-own-computer.html` |
| AFD | Algorithm Functional Description | 10 | `10-atbds-afd-pfds-catalog.html` |
| PFD | Product Format Document | 10 | `10-atbds-afd-pfds-catalog.html` |
| FH | Forest Height | 10 | `10-atbds-afd-pfds-catalog.html` |
| FD | Forest Disturbance | 10 | `10-atbds-afd-pfds-catalog.html` |
| CI/CD | Continuous Integration / Continuous Delivery | GH-1 | `GH-1-what-is-github.html` |
| DCO | Developer Certificate of Origin | 14 | `14-how-biomass-runs-on-github.html` |
| CODEOWNERS | GitHub path-based reviewer routing | 14 | `14-how-biomass-runs-on-github.html` |
| SME | Scientific Module Expert | 14 | `14-how-biomass-runs-on-github.html` |
| SPDX | Software Package Data Exchange | 21 | `21-the-ci-cd-pipeline.html` |
| GPG | GNU Privacy Guard (signed commits) | 21 | `21-the-ci-cd-pipeline.html` |
| DOI | Digital Object Identifier | 25 | `25-after-the-pr.html` |
| Zenodo | Zenodo open research archive | 25 | `25-after-the-pr.html` |

---

## Candidate acronyms (not yet wired)

Mark any row in a slide with `.bps-acronym`; later occurrences in the deck will not show a tooltip.

### Mission and organisations

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| BIOMASS | ESA BIOMASS Earth Explorer mission (P-band SAR) | 02 |
| ACRI-ST | ACRI-ST (project maintainer) | 02 |
| Aresys | Aresys (original BPS developer for ESA) | 02 |

### Documents and science

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| RST | reStructuredText | 13 |
| YAML | YAML configuration format | 04b |

### Products and processing chain

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| JobOrder | BPS job configuration file (XML) | 09 |

### Governance and contribution

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|

### Licences and compliance

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| MIT | MIT open-source licence | 04b |
| Apache 2.0 | Apache Licence 2.0 | 02 |

### Infrastructure and publication

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| Sphinx | Sphinx documentation generator | 04b or 06 |
| GitLab | GitLab (secondary CI trigger) | 13 |

### PR risk tiers (optional)

| Acronym | Definition | Suggested first slide |
|---------|------------|----------------------|
| T0 | PR risk tier 0 (low) | 20 |
| T1 | PR risk tier 1 (medium) | 20 |
| T2 | PR risk tier 2 (high) | 20 |

---

## Deck order (for first-occurrence planning)

Slides are ordered in `source/deck.yaml`. Tooltip precedence follows that file, not footer numbers.

| Order | `data-label` | Title (short) |
|------|--------------|---------------|
| 1 | 01 | Title |
| 2 | CH-I | ACT I |
| 3 | 02 | What is BPS |
| 4 | 03 | Open Source / Open Science |
| 5 | 05 | Why open source |
| 6 | 04 | Five pillars |
| 7 | 04b | Document the project |
| 8 | CH-II | ACT II |
| 9 | 06 | Documentation portal |
| 10 | 07 | Find your path |
| 11 | 08 | SUM |
| 12 | 09 | Tutorial |
| 13 | 10 | Science guide catalogue |
| 14 | 11 | Contributing |
| 15 | 12 | Governance |
| 16 | 13 | One repo, one site |
| 17 | CH-III | ACT III |
| 18 | GH-1 | What is GitHub |
| 19 | 14 | How BIOMASS runs on GitHub |
| … | … | … |

See `source/deck.yaml` for the full list (37 slides).
