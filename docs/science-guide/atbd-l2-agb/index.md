<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

```{include} ../_includes/atbd-logo-banner.md
```


<!--
Pipeline metadata (not rendered): {{ reference }} issue {{ version }},
{{ date }}, status draft.
-->

# Above-Ground Biomass Product ATBD

[← Science Guide](../index.md)

:::{admonition} Draft, pending Aresys and ESA approval
:class: warning

This is a **draft web conversion** of the official ATBD. The
[PDF on biomass-disc.info](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_AGB_ATBD_v3_1_4.pdf)
remains the authoritative reference until Aresys and ESA approve this version.
:::

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-AGB-ATBD-ARE-024912` issue
**3.1.4** dated **02 April 2026**. The official PDF remains the archival
reference and is linked in
[About → Applicable documents](../../about/applicable-documents.md).

[Download the PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_AGB_ATBD_v3_1_4.pdf)
:::

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BPS AGB ATBD — Algorithms Theoretical Baseline Document |
| External reference | `BIO-BPS-AGB-ATBD-ARE-024912` |
| Issue | 3.1.4 |
| Date | 02 April 2026 |
| Pages | 54 (authoritative PDF) |

**Recipients**

| Name | Organisation |
|---|---|
| Cristiano Lopes | ESA – ESRIN |
| Michele Caccia | ESA – ESRIN |
| Muriel Pinheiro | ESA – ESRIN |

**Prepared by**

| Name | Affiliation |
|---|---|
| Maciej Soja | Wageningen University & Research |
| Lars Ulander | Chalmers University of Technology |
| Shaun Quegan | University of Sheffield |
| Paolo Mazzucchelli | Aresys |
| Francesco Banda | Aresys |

**Checked by**

| Name | Affiliation |
|---|---|
| Davide D'Aria | Aresys |

**Approved by**

| Name | Affiliation |
|---|---|
| Davide D'Aria | Aresys |
:::

:::{dropdown} Change history
:icon: history

| Date | Issue | Change summary | Author |
|---|---|---|---|
| 2023-06-16 | 3.0 | First issue of the document | M. Soja, L. Ulander, S. Quegan, P. Mazzucchelli, F. Banda |
| 2023-08-04 | 3.0.1 | Post-PDR3: added no-selection criterion (§3.2); updated §4.1.4; clarified §3.4.4.2.2; updated §4; added second-iteration notes (§4.1.4.5); editorial corrections | P. Mazzucchelli |
| 2023-11-17 | 3.1 | CDR3: editorial corrections; updated §3.2.1, §4, §4.1.4.3, §4.1.4.5 | P. Mazzucchelli |
| 2024-09-06 | 3.1.1 | FAT3: updated real-elevation description; updated §4.1.2 input table; added reference-data selection at second iteration | P. Mazzucchelli |
| 2024-12-13 | 3.1.2 | Delta_FAT3: updated {eq}`3.14`–{eq}`3.15`, §3.5.5; updated §4.1.4.3 (power law model) and §4.1.4.6; renamed `decimationFactor` to `upsamplingFactor` | P. Mazzucchelli |
| 2025-04-30 | 3.1.3 | BPS-FTO-BATCH4: split SKP and flattening calibration screens (§3.3.2, §3.3.4) | P. Mazzucchelli, E. Giorgi |
| 2026-04-02 | 3.1.4 | Post-launch (BPS V4.4.1): updated {eq}`3.13` (sigma naught normalisation — removed spatial resolution normalisation factor to preserve dynamic range) | F. Banda, E. Giorgi |
:::

---

```{toctree}
:maxdepth: 2
:hidden:

1. Introduction <01-introduction>
2. BPS and AGB Processing Overview <02-bps-agb-overview>
3. Ground cancellation (L2A_P) <03-ground-cancellation>
4. AGB estimation (L2B_AGB_P) <04-agb-estimation>
5. Appendix: effect of GN <05-appendix>
References <references>
```
