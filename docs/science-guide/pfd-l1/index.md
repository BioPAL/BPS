<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BIOMASS L1a/b/c Products Format Specification
doc_code: BIO-BPS-L1PFD-ARE-010076
version: 1.7.0
date: 06/11/2026
family: pfd
license: Apache-2.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-18'
source_sha256: 0b1514b320fbf1523ddd4be9eaaa80680f0b4bac9969bfc0c90d41fc6592eed9
-->

# BIOMASS L1a/b/c Products Format Specification

[← Science Guide](../index.md)

::::{only} html

:::{admonition} Draft, automatic web conversion
:class: warning

This page is an **automatic web conversion** of the official document, pending review. The official PDF remains the authoritative reference.
:::

:::{admonition} Source document
:class: tip

This page reproduces `BIO-BPS-L1PFD-ARE-010076` issue **1.7.0** dated **06 November 2026**.
:::

::::

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BIOMASS L1a/b/c Products Format Specification |
| External reference | `BIO-BPS-L1PFD-ARE-010076` |
| Issue | 1.7.0 |
| Date | 06 November 2026 |
| Pages | 140 (authoritative PDF) |

**Recipients**

| Name | Organisation |
|---|---|
| Cristiano Lopes | ESA – ESRIN |
| Michele Caccia | ESA – ESRIN |
| Muriel Pinheiro | ESA – ESRIN |

**Prepared by**

| Name | Organisation |
|---|---|
| Riccardo Piantanida |  |
| Davide D’Aria |  |
| Paolo Mazzucchelli |  |
| Federico Boniardi |  |

**Checked by**

| Name | Organisation |
|---|---|
| Davide D’Aria |  |

**Approved by**

| Name | Organisation |
|---|---|
| Davide D’Aria |  |
:::

:::{dropdown} Change history
:icon: history

| Date | Issue | Change summary | Author |
|---|---|---|---|
| 2021-05-14 | 1.0 | First issue of the document | Riccardo Piantanida |
| 2021-06-28 | 1.0.1 | Post-SRR issue of the document: SRR-034: Figure with product graphical representation added (Sec. 3) SRR-035: Guideline section added (Sec. 3.1, 5.1) SRR-036: Section renamed and expanded (Sec. 3.3, 5.2) SRR-037: Product naming convention (high-level) added (Sec. 3.4, 5.3) SRR-038: Fields added to SAR Image MDS (Sec. 4.2.1) SRR-039: SCS-CAL products description added (Sec. 2, 3.2, 3.4) SRR-040, SRR-158: References to GDAL libraries removed where not needed (all) SRR-043: Tables formatting reviewed and improved (all) SRR-044: List cardinalities justified (all) SRR-045, SRR-162: References for datum description added (Sec. 4.3.1.16) SRR-046: Polarimetric calibration section renamed (Sec. 4.3.1, 4.3.1.11.) SRR-047: Polynomials update frequency clarified (all) SRR-048: Height computation reference added (Sec. 4.3.2.3) SRR-137: References to Google Earth removed (Sec. 3, 4.3.5) SRR-150: References to launch date removed (Sec. 1.1) SRR-151: Geolocation grids cardinality and accuracy described (Sec. 4.2.1) SRR-152: SAR image MDS components ordering described (Sec. 4.2.1) SRR-153: KML overlay replaced with KMZ one (Sec. 3, 4.3.5) SRR-156: Typo in Dask library name corrected (Sec. 4.2.1) SRR-157: SAR image MDS strip-related tags removed (Sec. 4.2.1) SRR-159: Frame field description improved, productComposition moved to Acquisition information DSR (Sec. 4.3.1.1) SRR-160: RangeTimeInterval field description improved (Sec. 4.3.1.2) SRR-161: ValidityStartLine/ValidityStopLine fields removed, azimuthTime field description improved in polynomial annotations (all) SRR-164: UTC time fields description improved (Sec. 4.3.1.16) SRR-165: txPulseList field description improved (Sec. 4.3.1.3) SRR-166: Range compression method field added (Sec. 4.3.1.5, 4.3.1.16) SRR-167: Absolute calibration constant application clarified (Sec. 4.2.1, 4.3.1.10) SRR-168: Incidence and elevation angles computation clarified (Sec. 4.3.2.3) SRR-169: Formulas in LUTs description reviewed, cardinality better detailed (Sec. 4.3.2.3) | Riccardo Piantanida |
| 2021-10-15 | 1.1 | PDR1 issue of the document L1c Products Format Specification section completed (Sec. 5, 6) Primitive and common types description section moved to Appendix (Sec. 7) Solved several TBDs on L1a/b Products Format Specification: Added SAR image MDS overviews details (Sec. 4.2.1) Added LUT ADS compression details (Sec. 4.3.2) Added LUT ADS sampling grid details (Sec. 4.3.2) Added Quick-look ADS colour coding details (Sec. 4.3.4) Added Overlay ADS KML description (Sec. 4.3.5) Remaining SRR RIDs addressed: SRR-038: noPixelValue field in Main and LUT ADS renamed noDataValue, to be better aligned with GDAL\_NODATA tag in SAR image MDS (Sec. 4.3.1, 4.3.2) SRR-044: Updated cardinality of internalCalibrationSequence (Sec. 4.3.1.6) SRR-135: SAR image MDS compression strategy updated (Sec. 4.2.1) SRR-155: MPH high-level summary added (Sec. 4.1) | Riccardo Piantanida |
| 2021-12-17 | 1.1.1 | Post-PDR1 issue of the document: PDR1-106: SAR image MDS overviews levels updated (Sec.4.2.1) PDR1-107: Colour coding scheme updated (Sec. 4.3.4) PDR1-108: Various editorials fixed (ALL) PDR1-109: Clarification about LUT down-sampling factors added (Sec. 4.3.2.3) PDR1-110: Footprint coordinates convention clarified (Sec. 4.1, 4.3.1.2) PDR1-112: dcValue type updated to double (Sec. 4.3.1.5) PDR1-113: noDataValue type updated (Sec. 4.3.2.2) PDR1-114: Master/slave images terminology replaced with primary/secondary one (Sec. 5, 6, 7) PDR1-115: insarPreProcessingFlag meaning clarified (Sec. 6.3.3.1) PDR1-116: Baseline errors annotations replaced with more generic ones (Sec. 6.3.3.3) PDR1-118: meanBitRate meaning clarified (Sec. 4.3.1.3) PDR1-120: missionPhaseID values updated, driftPhaseFlag added (Sec. 4.3.1.1) PDR1-122: Figure and tables labels harmonized (ALL) | Riccardo Piantanida |
| 2022-03-23 | 1.2 | CDR1 issue of the document DEM version specified as heightModelType attribute (Sec. 7.1.1) PDR1-115: InSAR Pre-Processing step replaced by Range Spectral Filtering step (Sec. 6.3.3.1) PDR1-116: Figure updated (Sec. 6.3.3.3) PDR1-119: L1a/b Antenna pattern DSR section removed as currently empty; L1a/b and L1c Quality DSR sections filled (Sec. 4.3.1.13, 6.3.3.4) | Riccardo Piantanida |
| 2022-06-29 | 1.2.1 | Post-CDR1 issue of the document Various (minor) updates to align document to latest issues of L1 and Stack ATBDs (ALL) CDR1-191, CDR1-158: Parameters added to ionosphere correction DSR (Sec. 4.3.1.11) CDR1-217, CDR1-175: Framing parameters definition updated to reflect latest ICD changes (Sec. 4.3.1.1, 7.1.1) CDR1-176: RFI mitigation-related processing parameters aligned to AUX\_PP1 ones (Sec. 4.3.1.5, 7.1.1) CDR1-198, CDR1-159: Added the possibility to have 3 or 4 polarizations in the L1c product (Sec. 5, 4.2.1, 6.3.3.1, 7.1.1) CDR1-161: Name and meaning of rangeSpectralFilteringFlag clarified (Sec. 6.3.3.1) CDR1-162, CDR1-173: Quality thresholds taken from AUX\_PP\* files added to quality DSRs (Sec. 4.3.1.13, 6.3.3.4) CDR1-177: L1c product RFI-related parameters reviewed and updated (Sec. 6.3.3, 6.3.4) CDR1-178: Clarified meaning of shifts included in LUT ADS (Sec. 5) CDR1-180: LUT ADS groundPhasesScreenQuality map kept as float to support potential future evolution (currently it can assume only 0 or 1 values) (Sec. 6.3.4) CDR1-181: BIOMASS DTM added to height models list (Sec. 7.1.1) CDR1-183: Added note on L1c LUT ADS sampling (Sec. 6.3.4) CDR1-184: Added ionospheric corrections parameters useful for L1c processing (Sec. 4.3.1.11) CDR1-185: Parameter slowIonosphereQuality added (Sec. 6.3.3.3) CDR1-171: Broken links fixed (ALL) CDR1-066: Quicklook generation procedure updated to have more homogeneous results (Sec. 4.3.4) | Riccardo Piantanida |
| 2023-03-13 | 1.2.2 | Post-FAT1 issue of the document CDR1-249, BPSPDR2-83, BPSCDR2-27: Overall update of the document to ensure consistency with ATBDs and PFDs documents (ALL) CDR1-158, CDR1-184: Added required quality flags (Sec. 4.3.1.11, 4.3.1.13, 4.3.2) CDR1-172: Added up-to-date details on SAR image MDS and LUT ADS compression (Sec. 4.2.1, 4.3.2) CDR1-174: Added details about SAR image MDS VRT file (Sec. 4.2.1) CDR1-178: Added clarification on azimuthCoregistrationShifts (Sec. 6.3.4) CDR1-228, BPSPDR2-31, BPSPDR2-84: List of geometric layers in LUT ADS reviewed and updated (Sec. 4.3.2, 6.3.4) BPSPDR2-87: Polarizations enumerative extended (Sec. 7.1.1) | Riccardo Piantanida |
| 2023-07-31 | 1.2.3 | Post-FAT2 issue of the document BPSCDR1-1: Added tag for MAX\_Z\_ERROR (Sec. 4.2.1) BPSFAT2-19 | Riccardo Piantanida |
| 2023-10-06 | 1.2.4 | Post-FAT2 issue of the document (associated to BPS V2.1.3) Updated type of dataTakeID field (Sec. 4.3.1.1) | Riccardo Piantanida |
| 2023-11-17 | 1.2.5 | Post-FAT2 issue of the document (associated to BPS V2.1.4, 1st review) Review of the document to specify units of measure where missing (ALL) Reviewed description of LUT ADS (Sec. 4.3.2, 6.3.4) | Riccardo Piantanida Federico Boniardi |
| 2023-11-30 | 1.2.6 | Post-FAT2 issue of the document (associated to BPS V2.1.4, 2nd review) Added clarifications to some coregistered image main ADS fields (Sec. 7.3.3) Updated product types of some coregistered image main ADS fields (Sec. 7.3.3) Update layerType and uomType enumerative (Sec. 7.1.1) | Riccardo Piantanida Federico Boniardi |
| 2023-12-10 | 1.2.7 | Post-FAT2 issue of the document (associated to BPS V2.1.4, 3rd review) Corrected cardinality in Annotation LUT DSR (Sec. 4.3.1.14) Removed noDataValueFloat and noDataValueBool in L1c LUTs (in favor of noDataValue) (Sec. 6.3.4.2) | Riccardo Piantanida Federico Boniardi |
| 2024-05-17 | 1.2.8 | Post-DeltaFAT2 issue of the document BPS-221: PGA removed from autofocus methods (Sec. 4.3.1.5, 7.1.1) BPS-437: VRT file removed for L1b products (Sec. 4, 4.2.1) BPS-494: heightModel field description updated, heightModelUsedFlag field added (Sec. 4.3.1.12) BPS-505: Updated L1 LUT description (Sec. 4.3.2) BPS-541, BPSFAT2-144: extendedSwathProcessingFlag field added (Sec. 4.3.1.5) New grids for of the stack LUTs (no longer configurable but instead defined by the subsampling step of the LUTs of the primary L1a frame) Use 64-bit LUTs for Lat/Lon/Height and axes Added missing “Temporal Baseline” to primaryImageSelectionInformationType and primaryImageSelectionMethodType Added skpDecompositionIndex to staQualityParametersType Added skpEstimationWindowSize to staProcessingParametersType Added rangeSpectralFilteringFlag and polarizationUsed to staCoregistrationParametersType Section 4.3.1.5 Tab.23 new annotation to flag extended swath processing (BPS-541) | Riccardo Piantanida Federico Boniardi |
| 2024-09-06 | 1.2.9 | FAT3 issue of the document Updated L1a/b RFI mitigation section in main ADS, added new layers for RFI Frequency-Domain masks in LUT ADS (Sec. 4.3.1.7, 4.3.2) Changed polarizationUsedForCalibration to polarisationUsedForCalibration (consistent use of US/UK English). Added Sec. 4.3.1.13.1 and Sec. 6.3.3.4.1 (overallProductQualityIndex for L1 and Stack respectively) | Riccardo Piantanida Federico Boniardi |
| 2024-12-13 | 1.3.0 | Post-FAT3 issue of the document Updated description and tables of the L1 and Stack overallProductQualityIndex (Sec. 4.3.1.13.1, 6.3.3.4.1) Overall update of data types (ALL) | Riccardo Piantanida Federico Boniardi |
| 2025-03-07 | 1.4.0 | Post-FAT3 issue of the document Overall review and update of L1a/b annotations, to remove unused fields (Sec. 4.3.1): aocsMode / aocsSubMode: removed rxGain: removed temperatureList: removed errorCounters: only two counters kept (numIspHeaderErrors, numIspMissing) rollAngleList: replaced by rollBias incidenceAngleList / elevationAngleList: removed qualityParameters: maxTimeGap / maxTimeGapThreshold removed Fixed typo in L1a/b channel imbalance annotation (Sec. 4.3.1.10) Added default L1a/b LUT compression level (Sec. 4.3.2) Updated L1c annotations (DSRs and LUTs) to comply with new specs of SKP, IOB, and BAE Phase-like parameters now expressed in radians instead of degrees (ALL) | Riccardo Piantanida Federico Boniardi |
| 2025-04-30 | 1.4.1 | Post-FAT3 issue of the document Fixing of residual issues on phase-like parameters conversion to radians instead of degrees (ALL) rxGainList field added back to L1a/b annotations (Sec. 4.3.1.3) Channel imbalance and transmit power tracking field updated (Sec. 4.3.1.6) Updated computation of L1a/b overall product quality index (Sec. 4.3.1.13.1) | Riccardo Piantanida Federico Boniardi |
| 2025-05-13 | 1.4.2 | Launch issue of the document KMZ overlay replaced with KML one (Sec. 3, 4.3.5, 5, 6.3.7) | Riccardo Piantanida Federico Boniardi |
| 2025-05-30 | 1.4.3 | Post-launch issue of the document Updated conventions used for L1a, L1b and L1c quick-looks generation (Sec. 4.3.4, 6.3.6) | Riccardo Piantanida Federico Boniardi |
| 2025-07-23 | 1.4.4 | Post-launch issue of the document (associated to BPS V4.0.0) Overlay ADS integrated with additional metadata (Sec. 4.3.5) Updated L1c product format (Sec. 6): Updated LUT definition Updated bitset of the overallProductQualityIndex Updated staInSARParametersType Fixed extra “s” in “PhasesScreen” | Riccardo Piantanida Federico Boniardi |
| 2025-11-28 | 1.5.0 | Post-launch issue of the document (associated to BPS V4.2.0) Updated L1a/b main ADS RFI related annotations (Sec. 4.3.1.5, 4.3.1.7, 7) Updated L1a/b overall product quality index to add DC fallback activation flag (Sec. 4.3.1.13.1) Updated L1c product format (Sec. 6, 7): Deprecate BIC annotations Add annotations for InSAR calibration (PPR) | Riccardo Piantanida Federico Boniardi |
| 2026-01-23 | 1.5.1 | Post-launch issue of the document (associated to BPS V4.3.0) Updated L1a/b overall product quality index to add geomagnetic equator crossing fallback activation flag (Sec. 4.3.1.13.1) | Riccardo Piantanida |
| 2026-03-13 | 1.6.0 | Post-launch issue of the document (associated to BPS V4.4.0) Updated L1a/b product format (Sec. 4): Foreseen the possibility to have 2 quick-looks for L1a product (Sec. 3) and updated naming convention (Sec. 3.4) and description (Sec. 4.3.4) Added inOutBandPowerRatioList section (Sec. 4.3.1.4) terrainSlope LUT replaced by rangeTerrainSlope and azimuthTerrainSlope ones (Sec. 4.3.2) Updated L1c product format (Sec. 6): Foreseen the possibility to have 2 quick-looks for L1c product (Sec. 5) and updated naming convention (Sec. 5.3) Update L1c annotations with Multi-Squint Calibration annotations Deprecate PPR-related annotations from L1c products Deprecate rangeSpectralFilteringFlag from L1c product annotations terrainSlope LUT replaced by rangeTerrainSlope and azimuthTerrainSlope ones (Sec. 4.3.2) | Riccardo Piantanida Federico Boniardi |
| 2026-04-02 | 1.6.1 | Post-launch issue of the document (associated to BPS V4.4.1) Updated L1a/b product format (Sec. 4): Fixed cardinality of dcEstimate and fmRateEstimate (Sec. 4.3.1.8) Updated L1c product format (Sec. 5, 6): Fixed list of quick-look files in L1c products (Sec. 5, 6.3.6) Added coherence and interferogram LUTs (Sec. 5, 6.3.4) Fixed residual issue in Tab. 80 and 82 (Sec. 6.3.3.1, 6.3.3.4) | Riccardo Piantanida Federico Boniardi |
| 2026-11-06 | 1.7.0 | Post-launch issue of the document (associated to BPS V4.5.0) Format version information added to main ADS and LUT ADS (Sec. 4.3.1, 4.3.2, 6.3.3, 6.3.4) Fixed typo in denoising LUT description (LUT are expressed as amplitudes, not power) (Sec. 4.3.2.3) | Riccardo Piantanida Federico Boniardi |
:::

---

```{toctree}
:maxdepth: 2
:hidden:

1. Introduction <01-introduction>
2. BPS and L1 Products Overview <02-bps-and-l1-products-overview>
3. L1a/b Products Format Overview <03-l1a-b-products-format-overview>
4. L1a/b Products Format Specification <04-l1a-b-products-format-specification>
5. L1c Products Format Overview <05-l1c-products-format-overview>
6. L1c Products Format Specification <06-l1c-products-format-specification>
7. Appendix: Common Products Format Specification <07-appendix-common-products-format-specific>
References <references>
```
