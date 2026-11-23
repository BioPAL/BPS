<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BPS L1c Stack ATBD - Algorithms Theoretical Baseline Document
doc_code: BIO-BPS-L1-STACK-ATBD-ARE-010166
version: 1.4.0
date: 02/04/2026
family: atbd
license: CC-BY-4.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-23'
source_sha256: 9d87e31a8a0f7cf0a6838d25db2683b1ff55d37dfcc02eaee129c1ba07d6dfe4
-->

# BPS L1c Stack ATBD - Algorithms Theoretical Baseline Document

[← Science Guide](../index.md)

::::{only} html

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-L1-STACK-ATBD-ARE-010166` issue **1.4.0** dated **02 April 2026**.
[Download the official ESA PDF (BPS v4.5.0)](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_L1_STACK_ATBD_v1_4_0.pdf)
:::

::::

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BPS L1c Stack ATBD - Algorithms Theoretical Baseline Document |
| External reference | `BIO-BPS-L1-STACK-ATBD-ARE-010166` |
| Issue | 1.4.0 |
| Date | 02 April 2026 |
| Pages | 87 (authoritative PDF) |

**Recipients**

| Name | Organisation |
|---|---|
| Nuno Miranda | ESA – ESRIN |
| Michele Caccia | ESA – ESRIN |
| Muriel Pinheiro | ESA - ESRIN |

**Prepared by**

| Name | Organisation |
|---|---|
| Riccardo Piantanida |  |
| Simone Mancon |  |
| Davide D’Aria |  |
| Paolo Mazzucchelli |  |
| Francesco Banda |  |
| Federico Boniardi |  |
| Giovanni Amoroso |  |

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
| 2021-11-15 | 1.0 | First issue of the document | Riccardo Piantanida |
| 2022-03-18 | 1.1 | CDR1 issue of the document<br>• PDR1-3-57, PDR1-3-80, PDR1-3-99: Added optional SKP phase removal.<br>• PDR1-3-58: Corrected the time span of TOM stack.<br>• PDR1-3-59: Selection of reference image updated.<br>• PDR1-3-60: Removed master and slave terminology.<br>• PDR1-3-61, PDR1-3-62: Back-geocoding sections moved to the SAR ATBD.<br>• PDR1-3-63: Added missing description of: coregistration shifts computation, geometric phase computation, vertical wavenumbers, and image interpolation.<br>• PDR1-3-65: Updated formula in “Model-based inversion of residual shifts”.<br>• PDR1-3-66: Corrected the reference to the equation.<br>• PDR1-3-67: Added the XYZ product as input.<br>• PDR1-3-68, PDR1-3-72: Removed the RFI quality measure for TOM stack.<br>• PDR1-3-70: Removed references to DEM processor.<br>• PDR1-3-72: Moved the “Data quality measure according to RFI pattern” into the Stacking section.<br>• PDR1-3-73: Updated the “Azimuth spectral filtering” section.<br>• PDR1-3-74: Added in the I/O sections if one or multiple polarization will used.<br>• PDR1-3-75: Clarified in “Slow changing ionosphere estimation” which pre-processing will be applied.<br>• PDR1-3-76, PDR1-3-77, PDR1-3-83: Removed the “Multi-baseline/coherence optimization”<br>• PDR1-3-77, PDR1-3-84: Added geometric phase removal to in the “Range spectral filtering” and in the “Slow changing ionosphere estimation”.<br>• PDR1-3-78: Updated the “Range spectral filtering” section.<br>• PDR1-3-79: Updated the Stack processor workflow figure.<br>• PDR1-3-81: Updated the inputs of the “Reference image selection”.<br>• PDR1-3-82: corrected the correlation formula.<br>• PDR1-3-85: Added missing steps description in “Slow changing ionosphere estimation”.<br>• PDR1-3-86: Added Weighted Least Square inversion in the Multi-Baseline using the weights computed in the single baseline steps.<br>• PDR1-3-87: Removed the autofocus inputs<br>• PDR1-3-87, PDR1-3-89: added the formulation of the shifts correction in the “Residual coregistration from data” and in the “Baseline error estimation”<br>• PDR1-3-87: The sub-looks computation applied before the coregistration in the “Slow changing ionosphere estimation”.<br>• PDR1-3-87: “Residual coregistration from data” added to the calibration block diagram.<br>• PDR1-3-90: Added the update of the coregistration shifts annotated in the product in the “Residual coregistration from data” and in the “Baseline error estimation”.<br>• PDR1-3-91: Rephrased the sentence.<br>• PDR1-3-92: Removed the duplicated computation of the MS InSAR phase<br>• PDR1-3-92: Provide reference to the computation of the bandwidth of the azimuth sub-looks in the “Residual ionospheric scintillation estimation”.<br>• PDR1-3-94: Updated the I/O section and flow-chart of the “Phase calibration by SKP”<br>• PDR1-3-96: Removed the “Residual ionospheric scintillation estimation” algorithm using SVD.<br>• PDR1-3-97: Added the scintillation DC component removal in the “Residual ionospheric scintillation estimation”<br>• PDR1-3-98: Updated “Stacking” section according to the implementation. | Simone Mancon |
| 2022-05-23 | 1.1.1 | • CDR1-224, CDR1-223: New algorithm to compute the RFI impact to the InSAR coherence.<br>• CDR1-152, CDR1-153, CDR1-221: Updated Fig.3 and Section 2.3 with proper naming conventions and last algorithmic updates.<br>• CDR1-251: Split-band parameters (central frequency and bandwidth of sub-looks) stated as configurable.<br>• CDR1-252: The processing chain of the slow ionosphere removal has been restored to PDR1 version (Split-spectrum applied to the coregistered data).<br>• CDR1-231: the “Residual range shift estimation” has been removed.<br>• CDR1-232: Synthetic phase removal moved before the range spectral filtering.<br>• CDR1-233: Corrected the formulation of the central frequency in the “Range spectral filtering”.<br>• CDR1-234: Removed from the ”Azimuth spectral filtering” the dependence from the desired resolution.<br>• CDR1-235: Specified the type of low pass-filtering used in the “Slow ionosphere estimation”.<br>• CDR1-219: Added placemarks for two missing steps: Cross-pol merging and LUT interpolation.<br>• CDR1-226: Added list of geometric parameters generated by the back-geocoding.<br>• CDR1-236, CDR1-237: Clarified the dimensions of the matrixes in the single-baseline and multi-baseline of the slow ionosphere.<br>• CDR1-238: Added the formulation of the quality indicator of the multi-baseline results.<br>• CDR1-239: Removed the redundant synthetic phase correction in the “Baseline error estimation”.<br>• CDR1-240: Added the coregistration shift among the output of the “Baseline error correction”.<br>• CDR1-244: Azimuth filtering removed from the configurable parameters of “Fast ionosphere estimation”.<br>• CDR1-246: Clarified the single-baseline “Fast ionosphere estimation” is performed block-wise in range and the block in the same for ionospheric height and ionospheric phase estimation.<br>• CDR1-247: Clarified that “Fast ionosphere estimation” need look angle in input if the ionospheric height is passed as external parameter.<br>• CDR1-248: Stack calibrated by SKP phases defined as optional output.<br>• CDR1-229: Clarified in the formulation that speckle tracking is performed on the intensity of the images.<br>• CDR1-230: Modified the terminology: from scintillation to fast ionosphere. | Simone Mancon |
| 2023-02-08 | 1.1.2 draft | Pre-FAT1 issue of the document<br>• Updated the methodology to select the reference image for the coregistration by geometry.<br>• BPS-42: PDR1-2 - Updated the algorithm of the single-baseline estimation of the residual fast-varying ionosphere.<br>• Clarified that the SKP ground phase is composed by both the synthetic phase from geometry plus the residual phase computed by SKP.<br>• BPS-58: PDR1-21 – Added the SKP ground phase quality and the related threshold. | Simone Mancon |
| 2023-03-24 | 1.1.2 | • Explained the method to perform the cross-pol merging.<br>• Updated the methodology to select the reference image for the coregistration using the temporal baseline.<br>• Explained the method to perform the oversampling of geometrical quantity.<br>• Explained the method to perform the shift and interpolation of the LUTs<br>• Add input/output tables to the calibration chapter and other minor refactoring to that chapter.<br>• Add the IOS description of the approach used for the ionosphere height estimation.<br>• Added the revert of the ionospheric phase screen estimated by L1 processor in the slow varying ionosphere estimation.<br>• Added the correction of bias in the slow varying ionosphere estimation due to the combined coregistration mode.<br>• Added the formulation of the estimated accuracy for baseline error estimation and the quality of the step has been modified accordingly.<br>• Removed the direct geocoding step and added the exploitation of the geometric LUT for coregistration and calibration.<br>• The Stack is now split in three sub-processor (according to the code structure of the code): Pre-processing, Stacking and Calibration. | Simone Mancon Federico Boniardi Giovanni Amoroso |
| 2023-01-09 | 1.1.3 | Delta-FAT2 issue of the document<br>• Major refactoring of the sections related to calibration:<br>• Restructured th AZF sections with major corrections.<br>• Rewritten IOB section.<br>• Refactored BAE section.<br>• Updated IOS section.<br>• Changed = with ≈ in equation of SKP phase screen.<br>• Added tables in the stacking section.<br>• Added description of Automatic coregistration modality. | Federico Boniardi |
| 2024-05-17 | 1.1.4 | Post-Delta-FAT2 issue of the document<br>• Fixed typos in calibration tables (IOB, IOS, BAE).<br>• Added note on possibility to disable weights in MB estimation of IOS.<br>• Updated fitting quality metric for coregistration.<br>• Removed “scintillation” in favor of “fast-varying ionosphere”.<br>• Added sections that describe how contingency cases are handled:<br>• Stacking: Section 4.4<br>• IOB: Section 5.2.5<br>• BAE: Section 5.3.5<br>• IOS: Section 5.4.6<br>• SKP: Section 5.5.3<br>• Refactored Section 4.1, including references to DEM.<br>• Finalized incomplete cross-pol merging (formula) and input table<br>• Added input/output table in Section 3.4.2.<br>• Update quality for BAE’s Single-baseline estimation.<br>• Revised references and formulas (new numbering)<br>• Revised table of acronyms, including removal of unused entries<br>• Major refactoring of the LUT shifting section.<br>• Include note on the output resolution of LUTs. | Federico Boniardi |
| 2024-09-06 | 1.1.5 | FAT3 issue of the document<br>• Fixed section numbering for AZF.<br>• Updated I/O tables in Sections 5.2.1, Section 5.3.1, and Section 5.4.1.<br>• Addressed BPSCDR1-25 (multi-look size in azimuth and range direction)<br>• Addressed CDR1-245 (how to establish that there’s no peak in the ionosphere profiles)<br>• Substitute “Sequestrating” with “suppressing” | Federico Boniardi |
| 2024-12-13 | 1.1.6 | Post-FAT3 issue of the document | Federico Boniardi |
| 2025-03-07 | 1.2.0 | Post-FAT3 issue of the document<br>• Fixed contingency handling of Multi-baseline method (IOB, BAE, and IOS). Now a check on factor-graph connectivity is used instead of the previous laxer (and wrong) approach.<br>• IOB includes now recommendations from [RD25] (BIO-TOMEX). This includes updated I/O tables consistent with those.<br>• Updated I/O table for BAE with new parameter associated to MB<br>• Added reference to median filtering in SKP and updated I/O table<br>• Fixed notations for coregistrationShiftsQuality<br>• Fixed numbering of equations and associated references<br>• Rewritten Section 3.3.3 (same content, better wording) and added missing contingency case in Section 3.3.4<br>• Fixed typos and sentences here and there | Federico Boniardi |
| 2025-04-30 | 1.2.1 | Post-FAT3 issue of the document<br>• Added more details on selection of interferometric pairs in Section 5.2.3 and Section 5.3.3. | Federico Boniardi |
| 2025-07-23 | 1.3.0 | Post-launch issue of the document (associated to BPS V4.0.0)<br>• Add notes to explain a few details about the calculation of the vertical wavenumbers<br>• Deprecate BAE (former Section 5.3)<br>• Deprecate IOS (former Section 5.4)<br>• Substitute former Section 5.3 with newly added BIC, consistently with [RD25] (from BIO-TOMEX) | Federico Boniardi |
| 2025-07-30 | 1.3.1 | Fixes to the post-launch issue of the document<br>• Fix stale skpPhaseCorrectionFlag<br>• Fix stale “PhasesScreen”<br>• Add excludeMPMBPolarizationCrossCovarianceFlag | Federico Boniardi |
| 2025-11-28 | 1.3.2 | Fixes associated to BPS V4.2.0<br>• Deprecated BIC<br>• Added Phase Plane Removal (PPR) module<br>• Updated coregistration outputs (coreg from orbit LUTs)<br>• Added possibility to disable model-based fitting of coreg shifts<br>• Added warning to user to avoid using median-filter in SKP<br>• Updated figures and diagrams | Federico Boniardi |
| 2026-04-02 | 1.4.0 | Post-launch issue of the document (associated to BPS V4.4.1)<br>• Deprecate the range spectral filtering in coregistration<br>• Deprecate the PPR<br>• Added MSC<br>• Added boxcar and Goldstein filter to SKP<br>• Removed warning on filtering the SKP phase with median filter | Federico Boniardi |
:::

---

```{toctree}
:maxdepth: 2
:hidden:

1. Introduction <01-introduction>
2. BPS and Stack Processor Overview <02-bps-and-stack-processor-overview>
3. Pre-processing <03-pre-processing>
4. Stacking <04-stacking>
5. Calibration <05-calibration>
6. Configurable Input Parameters <06-configurable-input-parameters>
References <references>
```
