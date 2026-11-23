<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BPS FH ATBD - Algorithms Theoretical Baseline Document
doc_code: BIO-BPS-FH-ATBD-ARE-10343
version: 2.2.0
date: 13/03/2026
family: atbd
license: CC-BY-4.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-23'
source_sha256: 67efaa2dea5518bc3149e6b17f51ddd0206999dbe43cf92aa8c5e9b189544dac
-->

# BPS FH ATBD - Algorithms Theoretical Baseline Document

[← Science Guide](../index.md)

::::{only} html

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-FH-ATBD-ARE-10343` issue **2.2.0** dated **13 March 2026**.
[Download the official ESA PDF (BPS v4.5.0)](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_FH_ATBD_v2_2_0.pdf)
:::

::::

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
| 2022-10-28 | 2.1 | CDR2 issue of the document<br>• BPSPDR2-14: Processing Parameters clarified (Sec. 4.1.4)<br>• BPSPDR2-18, BPSPDR2-27, BPSPDR2-28, BPSPDR2-30: ATBD revised according to guidelines<br>• BPSPDR2-23: RVOG Reflectivity profile clarified (Sec. 3.4.2)<br>• BPSPDR2-24: editorials amended<br>• BPSPDR2-32: solved TBD/TBC in delivery pack<br>• BPSPDR2-54: uncertainty threshold clarified (Sec. 3.4.4)<br>• BPSPDR2-72: Hectare amended<br>• BPSPDR2-74: Inputs and outputs dimensions clarified<br>• BPSPDR2-75: number clarified (Sec. 4.1.4)<br>• BPSPDR2-78: Vertical reflectivity profiles clarified (Sec. 3.4.2)<br>• BPSPDR2-83: documents review to enforce consistency<br>• BPSPDR2-84: Revision of re-use of backgeocoding results at L2 (Sec. 3.2)<br>• BPSPDR2-86: Computation of spatial average window size clarified (Sec. 3.3.4)<br>• BPSPDR2-87: Cross-pol terminology aligned<br>• BPSPDR2-88: Incidence angle clarified (Sec. 3.3.4)<br>• BPSPDR2-89: FH quality index revised (Sec. 4.1.4) | Francesco Banda Review: Kostas Papathanassiou Paolo Mazzucchelli |
| 2022-03-13 | 2.1.1 | Post CDR2 issue of the document:<br>• BPSCDR2-3: added docs acronyms in i/o tables<br>• BPSCDR2-4: added missing details for Forest Cov Check<br>• BPSCDR2-8: updated section 3.4.2<br>• BPSCDR2-10: vertical datum/reference added section 3.6.4.1.1<br>• BPSCDR2-13: updated section 2.3.1<br>• BPSCDR2-14: note added section 3.3.2<br>• BPSCDR2-16: note added section 3.4.4.3<br>• BPSCDR2-18: section 3.5.2 rephrased to be crystal clear<br>• BPSCDR2-19: section 3.5.4.1.2 note added<br>• BPSCDR2-20: section 3.5.4.1.3 note added<br>• BPSCDR2-21: updated section 3.5.4.1.5<br>• BPSCDR2-22: updated section 3.6.4.1.2<br>• BPSCDR2-23: note added section 3.4.4<br>• BPSCDR2-30: Heat Map added section 4.1.4<br>• BPSCDR2-43: minimum coverage percentage added section 4.1.4<br>• BPSCDR2-45: configuration added section 3.5.2<br>• Wavenumber LUT dimension decoupled from height dimension: updated section 3.5.2<br>• Feathering added at L2B: updated section 4<br>• Formulas corrected: updated section 3.4.4<br>• Ground to volume ratio configuration updated: updated section 3.5 | Paolo Mazzucchelli Kostas Papathanassiou Francesco Banda Laurent Ferro-Famil |
| 2023-09-15 | 2.1.2 | BPS-393 / BPS-392: added notes on FH inversion algorithm efficient implementation. | Paolo Mazzucchelli |
| 2024-01-19 | 2.1.3 | DeltaFAT2 version of the document:<br>• BPSCDR2-10: updated description of “real elevation” | Paolo Mazzucchelli |
| 17/520/24 | 2.1.4 | Post DeltaFAT2 version of the document:<br>• updated default values to avoid data masking<br>• modified relations 3.12, 3.13<br>• updated equation 3.2<br>• added TBD/TBC table | Paolo Mazzucchelli |
| 2024-09-06 | 2.1.5 | FAT3 version of the document | Paolo Mazzucchelli |
| 2024-12-13 | 2.1.6 | Delta\_FAT3 version of the document:<br>• updated section 3.4.5<br>• renamed “decimationFactor” to “upsamplingFactor” | Paolo Mazzucchelli |
| 2025-03-07 | 2.1.7 | Post-FAT3 issue of the document<br>Fixed Wavenumbers (3.4.2), IncidenceAngle and Terrain Slope (3.5.2) descriptions unit of measurement | Paolo Mazzucchelli |
| 2025-04-30 | 2.1.8 | BPS-FTO-BATCH4:<br>Separated calibration screen table in skp and flattening (section 3.3.2); expanded description in how to use them (section 3.3.4) | Paolo Mazzucchelli Emanuele Giorgi |
| 2026-03-13 | 2.2.0 | Post-launch issue of the document (associated to BPS V4.4.0)<br>• 3.4.4.3. Coherence calibration: removed cos(incidence angle) term | Paolo Mazzucchelli Emanuele Giorgi |
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
