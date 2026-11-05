<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BIOMASS Forest Disturbance Products Format Specification
doc_code: BIO-BPS- FDPFD-ARE-010258
version: 3.4.0
date: 13/03/2026
family: pfd
license: Apache-2.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-05'
source_sha256: 8287bdd97138aa52c761d13795f69ca90d559608913c9dde9de4e74070c6d95d
-->

# BIOMASS Forest Disturbance Products Format Specification

[← Science Guide](../index.md)

:::{admonition} Draft, automatic web conversion
:class: warning

This page is an **automatic web conversion** of the official document, pending review. The official PDF remains the authoritative reference.
[Download the PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_FD_PFD_v3_4_0.pdf)
:::

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS- FDPFD-ARE-010258` issue **3.4.0** dated **13 March 2026**.
:::

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BIOMASS Forest Disturbance Products Format Specification |
| External reference | `BIO-BPS- FDPFD-ARE-010258` |
| Issue | 3.4.0 |
| Date | 13 March 2026 |
| Pages | 77 (authoritative PDF) |

**Recipients**

| Name | Organisation |
|---|---|
| Cristiano Lopes | ESA - ESRIN |
| Michele Caccia | ESA - ESRIN |
| Muriel Pinheiro | ESA - ESRIN |

**Prepared by**

| Name | Organisation |
|---|---|
| Francesco Banda | Aresys |
| Emanuele Giorgi | Aresys |
| Riccardo Piantanida | Aresys |
| Davide D’Aria | Aresys |
| Paolo Mazzucchelli | Aresys |

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
| 2022-03-23 | 1.0 | First issue of the document | Francesco Banda |
| a | 2.0 | Added L2a and L2b/L3 Products Format Specification ({sec}`4`, 6) Product schemas and descriptions updated ({sec}`3`, 5) CDR1-007: generic referencing to processing environment ({sec}`2`) CDR1-008: explained L2a/L2b/L3 geocoding grids ({sec}`2`) CDR1-014: added probability of change as a quality layer ({sec}`2`, 3, 5) CDR1-015: L2a product naming convention updated ({sec}`3.2`) CDR1-016 : L2b/L3 product naming convention updated (Sec. 5.3) CDR1-167: added STAC metadata placeholder ({sec}`3`, 5) CDR1-187: forest mask put in measurement for L2b/L3 and in annotation for L2a ({sec}`3`, 4, 5, 6) | Emanuele Giorgi Francesco Banda |
| 2022-10-28 | 2.1 | BPSPDR2-1, BPSPDR2-10: fixed reference AD7 ({sec}`2`) BPSPDR2-4: Removed optional and not needed elements from MPH ({sec}`4.1`) BPSPDR2-5: GeoCellID substituted with tileID (all the document) BPSPDR2-6: updated name and description of the ground cancellation flag ({sec}`4.3.1`) BPSPDR2-9: reordered acronyms in alphabetical order ({sec}`1.3`) BPSPDR2-11: Added ZSTD compression EXIF code to GeoTIFF tag table ({sec}`4.2`) BPSPDR2-12: specialized the processing parameters related to ground cancellation, for the FD algorithm ({sec}`4.3.1`) BPSPDR2-29: removed repetitions already described in Sec 1.2 ({sec}`2`); added missing quotes ({sec}`3.1`) BPSPDR2-32: solved TBD/TBC from document (all) BPSPDR2-33: clarification added to the covariance matrix images and its storage on NetCDF layers ({sec}`4.2`, {sec}`4.3`) BPSPDR2-54: specify that “no data values” are not used to apply further modifications of the FNF ({sec}`3`, 5) BPSPDR2-63: compression strategy (description and processing parameters) updated for COG and NetCDF ({sec}`4.2`, {sec}`4.3`, 6.3) BPSPDR2-64: specified TAG numbers for custom GeoTIFF private tags ({sec}`4.2`) BPSPDR2-66 removed redundant NetCDF parameters ({sec}`4.3.2`, 6.3.2) BPSPDR2-79: updated L2b/L3 product structure, consequent removal of unneeded ADS LUT section ({sec}`5`, 6.2, 6.3) BPSPDR2-80: added to L2b/L3 input information, the tracking of L1c acquisitions used, in xml and with a heat map ({sec}`5`, 6.2, 6.3.1.3) BPSPDR2-81: removed start and stop times from L2b/L3 file name convention (Sec. 5.3) BPSPDR2-82: added description in the text for the pixels marked as “no data value” and for its approach for each case ({sec}`3`, 4.2, 4.3.1.2, 5) BPSPDR2-83: improved consistency between documents (all the document) Updated LUT ADS NetCDF format description for consistency with NetCDF4 ({sec}`4.3.2`, 6.3.2) Other processing parameters updates out of BPSPDR2 ({sec}`4.3`, 6.3) Introduced CFM (Computed Forest Mask), and updated initial FNF description ({sec}`2`, 3, 4, 4.2, 5, 6, 6.2) Removed Quick look and overlay ADS sections, not foreseen for L2 (all the document). Inserted RFI decorrelation factor in ADS ({sec}`4.3.1`) | Emanuele Giorgi Paolo Mazzucchelli Francesco Banda |
| 2023-03-13 | 2.2 | BPS-219: updated naming convention, sections 3.2 and 5.3 BPSCDR2-44: Quicklook reinserted | Paolo Mazzucchelli |
| 2023-07-31 | 2.2.1 | Post-FAT2 issue of the document | Paolo Mazzucchelli |
| 2023-11-17 | 3.0 | CDR3 issue of the document Added Overviews ({sec}`4.2`, 6.2) Basin ID converted to list of strings ({sec}`4.2`, 4.3.1.1, 4.3.2.1, 6.3.1.1) Units of measurement fixed where needed ({sec}`4.3`, 6.3) Input Information DSR updated (sections 4.3.1, 6.3.1) Removed L3 references across all the document BPS-491: Acquisition ID image COG layers order specified | Paolo Mazzucchelli Emanuele Giorgi |
| 2023-11-30 | 3.0.1 | Release v2.1.4: new issue of the document: Updated productType in Sec 4.3.1, 6.3.1 and 7.1.1 Updated pixelRepresentation, pixelType, noDataValue types in Sec 4.3.1, 6.3.1 and 7.1.1 BPS-376: STAC position updated in doc Updated least\_significant\_digit in sec Sec 4.3.1, Sec 4.3.2 Updated L2b Product Format overview (Sec 5) Updated acquisitionFolderName definition ({ref}`Tab 12 <tab:pfd-fd-12>`, {ref}`Tab 26 <tab:pfd-fd-26>`) Added numericalDeterminantLimit Sec 4.3.1 BPSCDR2-79: MPH: reinserted bio:rds (Sec 4.1) BPSCDR2-25: added new ref. BPSCDR2-26: underline of common checked BPS-484: Updated Primitive and common types ({ref}`Tab 25 <tab:pfd-fd-25>`) also replaced “uint8” with “ubyte” (all document) | Paolo Mazzucchelli Emanuele Giorgi |
| 2023-12-12 | 3.0.2 | Release v2.1.4 updated issue of the document LUT attributes (list of strings), specified as json encoded BPS-435: L2b product structure updated Table with primitive types with attributes extended | Paolo Mazzucchelli Emanuele Giorgi |
| 2024-01-19 | 3.0.3 | DeltaFAT2 issue of the document: BPSCDR2-29: removed references from LUT content | Paolo Mazzucchelli |
| 2024-05-17 | 3.0.4 | Post DeltaFAT2 issue of the document Added Acquistion Id Image to pixelRepresentationType | Paolo Mazzucchelli Emanuele Giorgi |
| 2024-09-06 | 3.0.5 | FAT3 issue of the document | Paolo Mazzucchelli Emanuele Giorgi |
| 2024-12-13 | 3.1.0 | Post-FAT3 issue of the document | Paolo Mazzucchelli Emanuele Giorgi |
| 2025-03-07 | 3.2.0 | Post-FAT3 issue of the document BPS-668: applyCalibrationScreen type updated least\_significant\_digit type updated in ACM layers lut annotation rfiDecorrelation removed from main annotation inputInformation/polarisationList/polarization BPS-FTO-BATCH4.2, L2 format changes added forestCoveragePercentage and selectedReferenceImage in 4.3.1.1 product L2a DSR added verticalWavenumbers and heightOfAmbiguity in 4.3.1.3 Input Information DSR added forest\_coverage\_percentage, selected\_reference\_image and acquisition\_id\_reference\_image in NetCDF, in 4.3.2.1 BPS-685: updated naming convention | Paolo Mazzucchelli Emanuele Giorgi |
| 2025-04-30 | 3.2.1 | BPS-680: {sec}`4.3.2`, added units to LUT ACM phase variables and fixed units typo in LUT latitude longitude | Paolo Mazzucchelli Emanuele Giorgi |
| 2025-11-28 | 3.3.0 | Post-launch issue of the document (associated to BPS V4.2.0) Added Overlay ADS | Paolo Mazzucchelli Emanuele Giorgi |
| 2026-03-13 | 3.4.0 | Post-launch issue of the document (associated to BPS V4.4.0) Updated staQualityParametersType Added polarisationCombinationMethod to generalConfigurationParametersType Removed old, solved comments | Paolo Mazzucchelli Emanuele Giorgi |
:::

---

```{toctree}
:maxdepth: 2
:hidden:

1. Introduction <01-introduction>
2. BPS and Forest Disturbance Products Overview <02-bps-and-forest-disturbance-products-over>
3. L2a Products Format Overview <03-l2a-products-format-overview>
4. L2a Products Format Specification <04-l2a-products-format-specification>
5. L2b Products Format Overview <05-l2b-products-format-overview>
6. L2b Products Format Specification <06-l2b-products-format-specification>
7. Appendix: Common Products Format Specification <07-appendix-common-products-format-specific>
References <references>
```
