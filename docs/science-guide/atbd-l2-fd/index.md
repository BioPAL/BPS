<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BPS FD ATBD - Algorithms Theoretical Baseline Document
doc_code: BIO-BPS-FD-ATBD-ARE-10344
version: 2.1.8
date: 30/04/2025
family: atbd
license: Apache-2.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-05'
source_sha256: 0201ac373fc95e9fd03ba980496534653b71a289e7f0b410ea5b554df0341118
-->

# BPS FD ATBD - Algorithms Theoretical Baseline Document

[← Science Guide](../index.md)

:::{admonition} Draft, automatic web conversion
:class: warning

This page is an **automatic web conversion** of the official document, pending review. The official PDF remains the authoritative reference.
[Download the PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_FD_ATBD_v2.1.8.pdf)
:::

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-FD-ATBD-ARE-10344` issue **2.1.8** dated **30 April 2025**.
:::

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BPS FD ATBD - Algorithms Theoretical Baseline Document |
| External reference | `BIO-BPS-FD-ATBD-ARE-10344` |
| Issue | 2.1.8 |
| Date | 30 April 2025 |
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
| Paolo Mazzucchelli | Aresys |
| Shaun Quegan | University of Sheffield |
| Riccardo Piantanida | Aresys |

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
| 2022-06-09 | 2.0 | First issue of the document | Francesco Banda Review: Shaun Quegan |
| 2022-10-28 | 2.1 | CDR2 issue of the document BPSPDR2-17: editorials clarified ({sec}`4.1.4`) BPSPDR2-18, BPSPDR2-20, BPSPDR2-26: formula moved to appendix ({sec}`5`), ATBD revised according to guidelines. BPSPDR2-19: FD first cycle clarified ({sec}`3.7.4.4`) BPSPDR2-22: modulus clarified ({sec}`4.1.4`) BPSPDR2-32: solved TBD/TBC in delivery pack BPSPDR2-52: scenario clarified ({sec}`3.3`) BPSPDR2-70, BPSPDR2-71: geo-coding clarified ({sec}`3.5`) BPSPDR2-73: subsampling rule clarified ({sec}`3.4`) BPSPDR2-74: Dimensions specified ({sec}`3.4`) BPSPDR2-83: documents review to enforce consistency BPSPDR2-84: Revision of re-use of backgeocoding results at L2 ({sec}`3.2`) BPSPDR2-85: Multi-looking in FD clarified ({sec}`3.4`) BPSPDR2-86: Computation of spatial average window size clarified ({sec}`3.4`) BPSPDR2-87: Cross-pol terminology aligned BPSPDR2-90: Ground cancellation with two images clarified ({sec}`3.3`) BPSPDR2-91: FD emphasized forest height clarified ({sec}`3.3`) | Francesco Banda Paolo Mazzucchelli |
| 2023-03-13 | 2.1.1 | Post CDR2 issue of the document: BPSCDR2-1: added acronym BPSCDR2-3: added docs acronyms in i/o tables BPSCDR2-4: added missing details for Forest Cov Check BPSCDR2-6: fixed wavenumber description BPSCDR2-7: insarPair added in {sec}`3.4.4.1` BPSCDR2-8: updated {sec}`3.5.2` BPSCDR2-9: updated {sec}`3.6.2` BPSCDR2-10: vertical datum/reference added {sec}`3.6.2` BPSCDR2-11: updated {sec}`3.7.2` BPSCDR2-23: note added {sec}`3.5.4` BPSCDR2-30: Heat Map added {sec}`4.1.4` BPSCDR2-43: minimum coverage percentage added {sec}`4.1.4` Formulas corrected: updated {sec}`3.5.4` Formulas corrected: updated {sec}`3.4.4` (desired elevation to wavenumber correction and demodulation height) | Paolo Mazzucchelli Francesco Banda |
| 2023-09-15 | 2.1.2 | BPS-393 / BPS-390: updated FD computation to make robust to numerical problems | Paolo Mazzucchelli |
| 2024-01-19 | 2.1.3 | DeltaFAT2 version of the document: BPSCDR2-10: updated description of “real elevation” | Paolo Mazzucchelli |
| 2024-05-17 | 2.1.4 | Post-DeltaFAT2 version of the document: Updated {eq}`3.2` Updated description in {sec}`3.7.4.5` Added TBD/TBC table | Paolo Mazzucchelli |
| 2024-09-06 | 2.1.5 | FAT3 version of the document | Paolo Mazzucchelli |
| 2024-12-13 | 2.1.6 | Delta\_FAT3 version of the document: Updated {sec}`3.5.5` Renamed “decimationFactor” to “upsamplingFactor” | Paolo Mazzucchelli |
| 2025-03-07 | 2.1.7 | Post-FAT3 issue of the document Fixed Wavenumbers (3.4.2) and IncidenceAngle (3.5.2) descriptions unit of measurement | Paolo Mazzucchelli |
| 2025-04-30 | 2.1.8 | BPS-FTO-BATCH4: Separated calibration screen table in skp and flattening ({sec}`3.3.2`); expanded description in how to use them ({sec}`3.3.4`) | Paolo Mazzucchelli Emanuele Giorgi |
:::

---

```{toctree}
:maxdepth: 2
:hidden:

1. Introduction <01-introduction>
2. BPS and FD Processing Overview <02-bps-and-fd-processing-overview>
3. Change detection <03-change-detection>
4. Change detection aggregation <04-change-detection-aggregation>
5. Appendix: effect of GN <05-appendix-effect-of-gn>
References <references>
```
