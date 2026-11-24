<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BPS FD ATBD - Algorithms Theoretical Baseline Document
doc_code: BIO-BPS-FD-ATBD-ARE-10344
version: 2.1.8
date: 30/04/2025
family: atbd
license: CC-BY-4.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-23'
source_sha256: 0201ac373fc95e9fd03ba980496534653b71a289e7f0b410ea5b554df0341118
-->

# BPS FD ATBD - Algorithms Theoretical Baseline Document

[← Science Guide](../index.md)

::::{only} html

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-FD-ATBD-ARE-10344` issue **2.1.8** dated **30 April 2025**.
[Download the official ESA PDF (BPS v4.5.0)](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_FD_ATBD_v2.1.8.pdf)
:::

::::

<p class="atbd-licence">Documentation © European Space Agency (ESA) and ARESYS, licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC-BY-4.0</a>.</p>

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
| 2022-06-09 | 2.0 | First issue of the document | Francesco Banda<br>Review: Shaun Quegan |
| 2022-10-28 | 2.1 | CDR2 issue of the document<br>• BPSPDR2-17: editorials clarified (Sec. 4.1.4)<br>• BPSPDR2-18, BPSPDR2-20, BPSPDR2-26: formula moved to appendix (Sec. 5), ATBD revised according to guidelines.<br>• BPSPDR2-19: FD first cycle clarified (Sec. 3.7.4.4)<br>• BPSPDR2-22: modulus clarified (Sec. 4.1.4)<br>• BPSPDR2-32: solved TBD/TBC in delivery pack<br>• BPSPDR2-52: scenario clarified (Sec. 3.3)<br>• BPSPDR2-70, BPSPDR2-71: geo-coding clarified (Sec. 3.5)<br>• BPSPDR2-73: subsampling rule clarified (Sec. 3.4)<br>• BPSPDR2-74: Dimensions specified (Sec. 3.4)<br>• BPSPDR2-83: documents review to enforce consistency<br>• BPSPDR2-84: Revision of re-use of backgeocoding results at L2 (Sec. 3.2)<br>• BPSPDR2-85: Multi-looking in FD clarified (Sec. 3.4)<br>• BPSPDR2-86: Computation of spatial average window size clarified (Sec. 3.4)<br>• BPSPDR2-87: Cross-pol terminology aligned<br>• BPSPDR2-90: Ground cancellation with two images clarified (Sec. 3.3)<br>• BPSPDR2-91: FD emphasized forest height clarified (Sec. 3.3) | Francesco Banda<br>Paolo Mazzucchelli |
| 2023-03-13 | 2.1.1 | Post CDR2 issue of the document:<br>• BPSCDR2-1: added acronym<br>• BPSCDR2-3: added docs acronyms in i/o tables<br>• BPSCDR2-4: added missing details for Forest Cov Check<br>• BPSCDR2-6: fixed wavenumber description<br>• BPSCDR2-7: insarPair added in section 3.4.4.1<br>• BPSCDR2-8: updated section 3.5.2<br>• BPSCDR2-9: updated section 3.6.2<br>• BPSCDR2-10: vertical datum/reference added section 3.6.2<br>• BPSCDR2-11: updated section 3.7.2<br>• BPSCDR2-23: note added section 3.5.4<br>• BPSCDR2-30: Heat Map added section 4.1.4<br>• BPSCDR2-43: minimum coverage percentage added section 4.1.4<br>• Formulas corrected: updated section 3.5.4<br>• Formulas corrected: updated section 3.4.4 (desired elevation to wavenumber correction and demodulation height) | Paolo Mazzucchelli Francesco Banda |
| 2023-09-15 | 2.1.2 | • BPS-393 / BPS-390: updated FD computation to make robust to numerical problems | Paolo Mazzucchelli |
| 2024-01-19 | 2.1.3 | DeltaFAT2 version of the document:<br>• BPSCDR2-10: updated description of “real elevation” | Paolo Mazzucchelli |
| 2024-05-17 | 2.1.4 | Post-DeltaFAT2 version of the document:<br>• Updated equation 3.2<br>• Updated description in section 3.7.4.5<br>• Added TBD/TBC table | Paolo Mazzucchelli |
| 2024-09-06 | 2.1.5 | FAT3 version of the document | Paolo Mazzucchelli |
| 2024-12-13 | 2.1.6 | Delta\_FAT3 version of the document:<br>• Updated section 3.5.5<br>• Renamed “decimationFactor” to “upsamplingFactor” | Paolo Mazzucchelli |
| 2025-03-07 | 2.1.7 | Post-FAT3 issue of the document<br>Fixed Wavenumbers (3.4.2) and IncidenceAngle (3.5.2) descriptions unit of measurement | Paolo Mazzucchelli |
| 2025-04-30 | 2.1.8 | BPS-FTO-BATCH4:<br>Separated calibration screen table in skp and flattening (section 3.3.2); expanded description in how to use them (section 3.3.4) | Paolo Mazzucchelli<br>Emanuele Giorgi |
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
