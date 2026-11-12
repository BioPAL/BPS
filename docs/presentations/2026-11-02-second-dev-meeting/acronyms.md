# Acronyms reference, Second Developer Meeting deck

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
| BPS | BioMASS Processing Suite | 04 | `04-biomass-bps-transforming-p-band-sar-data-into-global-forest-biomass-maps.html` |
| SAR | Synthetic Aperture Radar | 04 | `04-biomass-bps-transforming-p-band-sar-data-into-global-forest-biomass-maps.html` |
| AGB | Above-Ground Biomass | 04 | `04-biomass-bps-transforming-p-band-sar-data-into-global-forest-biomass-maps.html` |
| ESA | European Space Agency | 04 | `04-biomass-bps-transforming-p-band-sar-data-into-global-forest-biomass-maps.html` |
| BioPAL | BioMASS open-source programme on GitHub | 04 | `04-biomass-bps-transforming-p-band-sar-data-into-global-forest-biomass-maps.html` |
| L1 | Level-1 product | 04 | `04-biomass-bps-transforming-p-band-sar-data-into-global-forest-biomass-maps.html` |
| L2A | Level-2A product | 04 | `04-biomass-bps-transforming-p-band-sar-data-into-global-forest-biomass-maps.html` |
| L2B | Level-2B product | 04 | `04-biomass-bps-transforming-p-band-sar-data-into-global-forest-biomass-maps.html` |
| PR | Pull Request | 07 | `07-five-pillars-one-open-project.html` |
| ATBD | Algorithm Theoretical Basis Document | 08 | `08-document-the-project.html` |
| DPM | Detailed Processing Model | 08 | `08-document-the-project.html` |
| FAIR | Findable, Accessible, Interoperable, Reusable | 08 | `08-document-the-project.html` |
| REUSE | REUSE specification (SPDX licence compliance) | 08 | `08-document-the-project.html` |
| SUM | Software User Manual | 11 | `11-find-your-path.html` |
| L1F | Level-1 Framed product | 13 | `13-run-bps-on-your-own-computer.html` |
| STA | Stack / statistics processing step | 13 | `13-run-bps-on-your-own-computer.html` |
| MAAP | Multi-Mission Algorithm and Analysis Platform (ESA) | 13 | `13-run-bps-on-your-own-computer.html` |
| PAT | Personal Access Token | 13 | `13-run-bps-on-your-own-computer.html` |
| AFD | Algorithm Functional Description | 14 | `14-atbds-afd-pfds-catalog.html` |
| PFD | Product Format Document | 14 | `14-atbds-afd-pfds-catalog.html` |
| FH | Forest Height | 14 | `14-atbds-afd-pfds-catalog.html` |
| FD | Forest Disturbance | 14 | `14-atbds-afd-pfds-catalog.html` |
| CI/CD | Continuous Integration / Continuous Delivery | 19 | `19-what-is-github.html` |
| DCO | Developer Certificate of Origin | 20 | `20-how-biomass-runs-on-github.html` |
| CODEOWNERS | GitHub path-based reviewer routing | 20 | `20-how-biomass-runs-on-github.html` |
| SME | Scientific Module Expert | 20 | `20-how-biomass-runs-on-github.html` |
| SPDX | Software Package Data Exchange | 30 | `30-the-ci-cd-pipeline.html` |
| GPG | GNU Privacy Guard (signed commits) | 30 | `30-the-ci-cd-pipeline.html` |
| DOI | Digital Object Identifier | 34 | `34-after-the-pr.html` |
| Zenodo | Zenodo open research archive | 34 | `34-after-the-pr.html` |

Order follows `source/deck.yaml` (40 slides). File names use the same numeric prefix as the slide order.
