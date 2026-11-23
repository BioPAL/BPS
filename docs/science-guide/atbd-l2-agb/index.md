<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BPS AGB ATBD - Algorithms Theoretical Baseline Document
doc_code: BIO-BPS-AGB-ATBD-ARE-024912
version: 3.1.4
date: 02/04/2026
family: atbd
license: Apache-2.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-23'
source_sha256: d9055900c12833b4dfc7d81e28088dc169f2bbd1e084b802d1a2969ee7f4039c
-->

# BPS AGB ATBD - Algorithms Theoretical Baseline Document

[← Science Guide](../index.md)

::::{only} html

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-AGB-ATBD-ARE-024912` issue **3.1.4** dated **02 April 2026**.
[Download the official ESA PDF (BPS v4.5.0)](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_AGB_ATBD_v3_1_4.pdf)
:::

::::

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BPS AGB ATBD - Algorithms Theoretical Baseline Document |
| External reference | `BIO-BPS-AGB-ATBD-ARE-024912` |
| Issue | 3.1.4 |
| Date | 02 April 2026 |
| Pages | 54 (authoritative PDF) |

**Recipients**

| Name | Organisation |
|---|---|
| Cristiano Lopes | ESA – ESRIN |
| Michele Caccia | ESA – ESRIN |
| Muriel Pinheiro | ESA - ESRIN |

**Prepared by**

| Name | Organisation |
|---|---|
| Maciej Soja | Wageningen University & Research |
| Lars Ulander | Chalmers University of Technology |
| Shaun Quegan | University of Sheffield |
| Paolo Mazzucchelli | Aresys |
| Francesco Banda | Aresys |

**Checked by**

| Name | Organisation |
|---|---|
| Davide D’Aria | Aresys |

**Approved by**

| Name | Organisation |
|---|---|
| Davide D’Aria | Aresys |
:::

:::{dropdown} Change history
:icon: history

| Date | Issue | Change summary | Author |
|---|---|---|---|
| 2023-06-16 | 3.0 | First issue of the document | Maciej Soja Lars Ulander Shaun Quegan Paolo Mazzucchelli Francesco Banda |
| 2023-08-04 | 3.0.1 | Post-PDR3 issue of the document BPSPDR3-61: added “no-selection” criterion in section 3.2 BPSPDR3-29: updated section 4.1.4 BPSPDR3-24, BPSPDR3-25: clarified section 3.4.4.2.2 BPSPDR3-27: updated section 4 BPSPDR3-32: added insights on second iteration BPSPDR3-22: editorials corrected | Paolo Mazzucchelli |
| 2023-11-17 | 3.1 | CDR3 issue of the document BPSPDR3-22: remaining editorial corrected BPSPDR3-23: updated section 3.2.1 BPSPDR3-27: updated section 4 BPSPDR3-28: updated section 4.1.4.3 BPSPDR3-31, BPSPDR3-32: updated section 4.1.4.5 | Paolo Mazzucchelli |
| 2024-09-06 | 3.1.1 | FAT3 issue of the document: BPSCDR2-10: updated description of “real elevation” BPSPDR3-63: updated input table section 4.1.2 Added selection for combination of reference data at second iteration | Paolo Mazzucchelli |
| 2024-12-13 | 3.1.2 | Delta\_FAT3 issue of the document: Updated eq. 3.14, 3.15 Updated section 3.5.5 Updated section 4.1.4.3 (power law model) Updated section 4.1.4.6 Renamed “decimationFactor” to “upsamplingFactor” | Paolo Mazzucchelli |
| 2025-04-30 | 3.1.3 | BPS-FTO-BATCH4: Separated calibration screen table in skp and flattening (section 3.3.2); expanded description in how to use them (section 3.3.4) | Paolo Mazzucchelli Emanuele Giorgi |
| 2026-04-02 | 3.1.4 | Post-launch issue of the document (associated to BPS V4.4.1) Updated Eq. 3.13 (Sigma naught normalization): removed the spatial resolution normalization factor to preserve the dynamic range of the estimation | Francesco Banda Emanuele Giorgi |
:::

---

```{toctree}
:maxdepth: 2
:hidden:

1. Introduction <01-introduction>
2. BPS and AGB Processing Overview <02-bps-agb-overview>
3. Ground cancellation <03-ground-cancellation>
4. AGB estimation <04-agb-estimation>
5. Appendix: effect of GN <05-appendix>
References <references>
```
