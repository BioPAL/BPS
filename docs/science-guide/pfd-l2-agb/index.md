<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BIOMASS Above Ground Biomass Products Format Specification
doc_code: BIO-BPS-AGBPFD-ARE-010257
version: 3.4.0
date: 13/03/2026
family: pfd
license: Apache-2.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-05'
source_sha256: 6d7997d1ff636524acc3c6f641b480c69a430bd3242adcbf55195ac9c95993ca
-->

# BIOMASS Above Ground Biomass Products Format Specification

[← Science Guide](../index.md)

:::{admonition} Draft, automatic web conversion
:class: warning

This page is an **automatic web conversion** of the official document, pending review. The official PDF remains the authoritative reference.
[Download the PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_AGB_PFD_v3_4_0.pdf)
:::

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-AGBPFD-ARE-010257` issue **3.4.0** dated **13 March 2026**.
:::

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BIOMASS Above Ground Biomass Products Format Specification |
| External reference | `BIO-BPS-AGBPFD-ARE-010257` |
| Issue | 3.4.0 |
| Date | 13 March 2026 |
| Pages | 80 (authoritative PDF) |

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
| 2022-06-09 | 2.0 | CDR1-007: generic referencing to processing environment ({sec}`2`) CDR1-008: explained L2a/L2b/L3 geocoding grids ({sec}`2`) CDR1-009: clarification of GN image polarizations layering in COG ({sec}`3`, 4.2) CDR1-010: added L2a and L2b/L3 Products Format Specification ({sec}`4`, 6) CDR1-019: L2a product naming convention updated ({sec}`3.2`) CDR1-020: L2b/L3 product naming convention updated (Sec. 5.3) CDR1-163: added STAC metadata placeholder ({sec}`3`, 5) | Emanuele Giorgi Francesco Banda |
| 2022-10-28 | 2.1 | BPSPDR2-1, BPSPDR2-10: fixed reference AD7 ({sec}`2`) BPSPDR2-3: Local Incidence Angle MDS moved to LUT ADS ({sec}`2`, 3, 4.2) BPSPDR2-4: Removed optional and not needed elements from MPH ({sec}`4.1`) BPSPDR2-5: GeoCellID substituted with tileID (all the document) BPSPDR2-6: updated name and description of the ground cancellation flag ({sec}`4.3.1`) BPSPDR2-9: reordered acronyms in alphabetical order ({sec}`1.3`) BPSPDR2-11: Added ZSTD compression EXIF code to GeoTIFF tag table ({sec}`4.2`) BPSPDR2-12: specialized the processing parameters related to ground cancellation, for the AGB algorithm ({sec}`4.3.1`) BPSPDR2-29: removed repetitions already described in Sec 1.2 ({sec}`2`); added missing quotes ({sec}`3.1`) BPSPDR2-32: solved TBD/TBC from document (all) BPSPDR2-34: added to processing parameters of L2b, the forest Masking Flag ({sec}`6.3.1.4`) BPSPDR2-46: multiLookingWindowSize renamed as product resolution for L2a AGB ({sec}`4.3.1`) BPSPDR2-54: specify that “no data values” are not used to apply modifications of the FNF ({sec}`3`, 5) BPSPDR2-63: compression strategy (description and processing parameters) updated for COG and NetCDF ({sec}`4.2`, {sec}`4.3`, 6.3) BPSPDR2-64: specified TAG numbers for custom GeoTIFF private tags ({sec}`4.2`) BPSPDR2-66 removed redundant NetCDF parameters ({sec}`4.3.2`, 6.3.2) BPSPDR2-79: updated L2b/L3 product structure, consequent removal of unneeded ADS LUT section ({sec}`5`, 6.2, 6.3) BPSPDR2-80: added to L2b/L3 input information, the tracking of L1c acquisitions used, in xml and with a heat map ({sec}`5`, 6.2, 6.3.1.3) BPSPDR2-81: removed start and stop times from L2b/L3 file name convention (Sec. 5.3) BPSPDR2-82: added description in the text for the pixels marked as “no data value” and for its approach for each case ({sec}`3`, 4.2, 4.3.1.2, 5) BPSPDR2-83: improved consistency between documents (all the document) Updated LUT ADS NetCDF format description for consistency with NetCDF4 ({sec}`4.3.2`, 6.3.2) Other processing parameters updates out of BPSPDR2 ({sec}`4.3`, 6.3) Introduced CFM (Computed Forest Mask), and updated initial FNF description ({sec}`2`, 5, 6, 6.2) Removed Quick look and overlay ADS sections, not foreseen for L2 (all the document). Inserted RFI decorrelation factor in ADS ({sec}`4.3.1`) | Emanuele Giorgi Paolo Mazzucchelli Francesco Banda |
| 2023-03-13 | 2.2 | BPS-219: updated naming convention, sections 3.2 and 5.3 BPSCDR2-44: Quicklook reinserted | Paolo Mazzucchelli Emanuele Giorgi |
| 2023-07-31 | 2.2.1 | Post-FAT2 issue of the document | P. Mazzucchelli |
| 2023-11-17 | 3.0 | CDR3 issue of the document Added Overviews ({sec}`4.2`, 6.2) Basin ID converted to list of strings ({sec}`4.2`, 4.3.1.1, 4.3.2.1, 6.3.1.1) Units of measurement fixed where needed ({sec}`4.3`, 6.3) AGB processing parameters section filled ({sec}`6.3.1.4`) Input Information DSR updated (sections 4.3.1, 6.3.1) Inserted agbTileStatus in MPH (sections 4.1, 6.1) Removed L3 references across all the document | Paolo Mazzucchelli Emanuele Giorgi |
| 2023-11-30 | 3.0.1 | Release v2.1.4: new issue of the document Updated productType in Sec 4.3.1, 6.3.1 and 7.1.1 Updated pixelRepresentation, pixelType, noDataValue types in Sec 4.3.1, 6.3.1 and 7.1.1 Updated least\_significant\_digit in sec Sec 4.3.1, Sec 4.3.2 Updated acquisitionFolderName definition ({ref}`Tab 12 <tab:pfd-agb-12>`, {ref}`Tab 32 <tab:pfd-agb-32>`) Updated L2b Product Format overview (Sec 5) BPSCDR2-25: added new ref. BPSCDR2-26: underline of common checked BPSCDR2-79: added bio:rds in MPH BPS-484: Updated Primitive and common types ({ref}`Tab 31 <tab:pfd-agb-31>`) also replaced “uint8” with “ubyte” (all document) | Paolo Mazzucchelli Emanuele Giorgi |
| 2023-12-12 | 3.0.2 | Release v2.1.4 updated issue of the document LUT attributes (list of strings), specified as json encoded BPS-435: L2b product structure updated Table with primitive types with attributes extended | Paolo Mazzucchelli Emanuele Giorgi |
| 2024-02-05 | 3.0.3 | Post-CDR3 issue of the document BPSCDR3-1: added bio:agbTileIteration Added referenceSelection in {sec}`6.3.1.4` | Paolo Mazzucchelli |
| 2024-09-06 | 3.0.4 | FAT3 issue of the document | Paolo Mazzucchelli Emanuele Giorgi |
| 2024-12-13 | 3.1.0 | Post-FAT3 issue of the document | Paolo Mazzucchelli Emanuele Giorgi |
| 2025-03-07 | 3.2.0 | Post-FAT3 issue of the document BPS-668: applyCalibrationScreen type updated least\_significant\_digit type updated in localIncidenceAngle lut annotation BPS-FTO-BATCH4.2, L2 format changes added forestCoveragePercentage and selectedReferenceImage in 4.3.1.1 product L2a DSR added verticalWavenumbers and heightOfAmbiguity in 4.3.1.3 Input Information DSR added forest\_coverage\_percentage, selected\_reference\_image and acquisition\_id\_reference\_image in NetCDF, in 4.3.2.1 BPS-685: updated naming convention | Paolo Mazzucchelli Emanuele Giorgi |
| 2025-04-30 | 3.2.1 | Post-FAT3 issue of the document BPS-693: 4.3.2.1 Local Incidence Angle variable name renamed to localIncidenceAngle BPS-FTO-BATCH4: updated L2B AGB, {sec}`6.3.1`, main annotation xml with new \<product\> element fields (contributingTiles, calABcoveragePerTile, GNcoveragePerTile) BPS-680: {sec}`4.3.2`, added units to local incidence angle LUT variable and fixed units typo in LUT latitude longitude | Paolo Mazzucchelli Emanuele Giorgi |
| 2025-11-28 | 3.3.0 | Post-launch issue of the document (associated to BPS V4.2.0) Added Overlay ADS | Paolo Mazzucchelli Emanuele Giorgi |
| 2026-03-13 | 3.4.0 | Post-launch issue of the document (associated to BPS V4.4.0) Updated staQualityParametersType Added polarisationCombinationMethod to generalConfigurationParametersType Removed old, solved comments | Paolo Mazzucchelli Emanuele Giorgi |
:::

---

```{toctree}
:maxdepth: 2
:hidden:

1. Introduction <01-introduction>
2. BPS and Above Ground Biomass Products Overview <02-bps-and-above-ground-biomass-products-ov>
3. L2a Products Format Overview <03-l2a-products-format-overview>
4. L2a Products Format Specification <04-l2a-products-format-specification>
5. L2b Products Format Overview <05-l2b-products-format-overview>
6. L2b Products Format Specification <06-l2b-products-format-specification>
7. Appendix: Common Products Format Specification <07-appendix-common-products-format-specific>
References <references>
```
