<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BPS FH ATBD - Algorithms Theoretical Baseline Document
doc_code: BIO-BPS-FH-ATBD-ARE-10343
version: 2.2.0
date: 13/03/2026
family: atbd
license: Apache-2.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-05'
source_sha256: 67efaa2dea5518bc3149e6b17f51ddd0206999dbe43cf92aa8c5e9b189544dac
-->

# BPS FH ATBD - Algorithms Theoretical Baseline Document

[← Science Guide](../index.md)

:::{admonition} Draft, automatic web conversion
:class: warning

This page is an **automatic web conversion** of the official document, pending review. The official PDF remains the authoritative reference.
[Download the PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_FH_ATBD_v2_2_0.pdf)
:::

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-FH-ATBD-ARE-10343` issue **2.2.0** dated **13 March 2026**.
:::

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BPS FH ATBD - Algorithms Theoretical Baseline Document |
| External reference | `BIO-BPS-FH-ATBD-ARE-10343` |
| Issue | 2.2.0 |
| Date | 13 March 2026 |
| Pages | 45 (authoritative PDF) |

**Recipients**

| Name | Organisation |
|---|---|
| Cristiano Lopes | ESA – ESRIN |
| Michele Caccia | ESA – ESRIN |
| Muriel Pinheiro | ESA - ESRIN |

**Prepared by**

| Name | Organisation |
|---|---|
| Francesco Banda | Aresys |
| Kostas Papathanassiou | DLR |
| Paolo Mazzucchelli | Aresys |
| Riccardo Piantanida | Aresys |
| Laurent Ferro-Famil | ISAE-SUPAERO, CESBIO |

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
| 2022-06-09 | 2.0 | First issue of the document. | Francesco Banda Review: Kostas Papathanassiou |
| 2022-10-28 | 2.1 | CDR2 issue of the document BPSPDR2-14: Processing Parameters clarified ({sec}`4.1.4`) BPSPDR2-18, BPSPDR2-27, BPSPDR2-28, BPSPDR2-30: ATBD revised according to guidelines BPSPDR2-23: RVOG Reflectivity profile clarified ({sec}`3.4.2`) BPSPDR2-24: editorials amended BPSPDR2-32: solved TBD/TBC in delivery pack BPSPDR2-54: uncertainty threshold clarified ({sec}`3.4.4`) BPSPDR2-72: Hectare amended BPSPDR2-74: Inputs and outputs dimensions clarified BPSPDR2-75: number clarified ({sec}`4.1.4`) BPSPDR2-78: Vertical reflectivity profiles clarified ({sec}`3.4.2`) BPSPDR2-83: documents review to enforce consistency BPSPDR2-84: Revision of re-use of backgeocoding results at L2 ({sec}`3.2`) BPSPDR2-86: Computation of spatial average window size clarified ({sec}`3.3.4`) BPSPDR2-87: Cross-pol terminology aligned BPSPDR2-88: Incidence angle clarified ({sec}`3.3.4`) BPSPDR2-89: FH quality index revised ({sec}`4.1.4`) | Francesco Banda Review: Kostas Papathanassiou Paolo Mazzucchelli |
| 2022-03-13 | 2.1.1 | Post CDR2 issue of the document: BPSCDR2-3: added docs acronyms in i/o tables BPSCDR2-4: added missing details for Forest Cov Check BPSCDR2-8: updated {sec}`3.4.2` BPSCDR2-10: vertical datum/reference added {sec}`3.6.4.1.1` BPSCDR2-13: updated {sec}`2.3.1` BPSCDR2-14: note added {sec}`3.3.2` BPSCDR2-16: note added {sec}`3.4.4.3` BPSCDR2-18: {sec}`3.5.2` rephrased to be crystal clear BPSCDR2-19: {sec}`3.5.4.1.2` note added BPSCDR2-20: {sec}`3.5.4.1.3` note added BPSCDR2-21: updated {sec}`3.5.4.1.5` BPSCDR2-22: updated {sec}`3.6.4.1.2` BPSCDR2-23: note added {sec}`3.4.4` BPSCDR2-30: Heat Map added {sec}`4.1.4` BPSCDR2-43: minimum coverage percentage added {sec}`4.1.4` BPSCDR2-45: configuration added {sec}`3.5.2` Wavenumber LUT dimension decoupled from height dimension: updated {sec}`3.5.2` Feathering added at L2B: updated {sec}`4` Formulas corrected: updated {sec}`3.4.4` Ground to volume ratio configuration updated: updated {sec}`3.5` | Paolo Mazzucchelli Kostas Papathanassiou Francesco Banda Laurent Ferro-Famil |
| 2023-09-15 | 2.1.2 | BPS-393 / BPS-392: added notes on FH inversion algorithm efficient implementation. | Paolo Mazzucchelli |
| 2024-01-19 | 2.1.3 | DeltaFAT2 version of the document: BPSCDR2-10: updated description of “real elevation” | Paolo Mazzucchelli |
| 17/520/24 | 2.1.4 | Post DeltaFAT2 version of the document: updated default values to avoid data masking modified relations 3.12, 3.13 updated {eq}`3.2` added TBD/TBC table | Paolo Mazzucchelli |
| 2024-09-06 | 2.1.5 | FAT3 version of the document | Paolo Mazzucchelli |
| 2024-12-13 | 2.1.6 | Delta\_FAT3 version of the document: updated {sec}`3.4.5` renamed “decimationFactor” to “upsamplingFactor” | Paolo Mazzucchelli |
| 2025-03-07 | 2.1.7 | Post-FAT3 issue of the document Fixed Wavenumbers (3.4.2), IncidenceAngle and Terrain Slope (3.5.2) descriptions unit of measurement | Paolo Mazzucchelli |
| 2025-04-30 | 2.1.8 | BPS-FTO-BATCH4: Separated calibration screen table in skp and flattening ({sec}`3.3.2`); expanded description in how to use them ({sec}`3.3.4`) | Paolo Mazzucchelli Emanuele Giorgi |
| 2026-03-13 | 2.2.0 | Post-launch issue of the document (associated to BPS V4.4.0) 3.4.4.3. Coherence calibration: removed cos(incidence angle) term | Paolo Mazzucchelli Emanuele Giorgi |
:::

---

```{toctree}
:maxdepth: 2
:hidden:

1. Introduction <01-introduction>
2. BPS and FH Processing Overview <02-bps-and-fh-processing-overview>
3. Forest height <03-forest-height>
4. Forest height aggregation <04-forest-height-aggregation>
References <references>
```
