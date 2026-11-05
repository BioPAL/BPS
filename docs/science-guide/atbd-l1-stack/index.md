<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BPS L1c Stack ATBD - Algorithms Theoretical Baseline Document
doc_code: BIO-BPS-L1-STACK-ATBD-ARE-010166
version: 1.4.0
date: 02/04/2026
family: atbd
license: Apache-2.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-05'
source_sha256: 9d87e31a8a0f7cf0a6838d25db2683b1ff55d37dfcc02eaee129c1ba07d6dfe4
-->

# BPS L1c Stack ATBD - Algorithms Theoretical Baseline Document

[← Science Guide](../index.md)

:::{admonition} Draft, automatic web conversion
:class: warning

This page is an **automatic web conversion** of the official document, pending review. The official PDF remains the authoritative reference.
[Download the PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_L1_STACK_ATBD_v1_4_0.pdf)
:::

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-L1-STACK-ATBD-ARE-010166` issue **1.4.0** dated **02 April 2026**.
:::

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
| 2022-03-18 | 1.1 | CDR1 issue of the document PDR1-3-57, PDR1-3-80, PDR1-3-99: Added optional SKP phase removal. PDR1-3-58: Corrected the time span of TOM stack. PDR1-3-59: Selection of reference image updated. PDR1-3-60: Removed master and slave terminology. PDR1-3-61, PDR1-3-62: Back-geocoding sections moved to the SAR ATBD. PDR1-3-63: Added missing description of: coregistration shifts computation, geometric phase computation, vertical wavenumbers, and image interpolation. PDR1-3-65: Updated formula in “Model-based inversion of residual shifts”. PDR1-3-66: Corrected the reference to the equation. PDR1-3-67: Added the XYZ product as input. PDR1-3-68, PDR1-3-72: Removed the RFI quality measure for TOM stack. PDR1-3-70: Removed references to DEM processor. PDR1-3-72: Moved the “Data quality measure according to RFI pattern” into the Stacking section. PDR1-3-73: Updated the “Azimuth spectral filtering” section. PDR1-3-74: Added in the I/O sections if one or multiple polarization will used. PDR1-3-75: Clarified in “Slow changing ionosphere estimation” which pre-processing will be applied. PDR1-3-76, PDR1-3-77, PDR1-3-83: Removed the “Multi-baseline/coherence optimization” PDR1-3-77, PDR1-3-84: Added geometric phase removal to in the “Range spectral filtering” and in the “Slow changing ionosphere estimation”. PDR1-3-78: Updated the “Range spectral filtering” section. PDR1-3-79: Updated the Stack processor workflow figure. PDR1-3-81: Updated the inputs of the “Reference image selection”. PDR1-3-82: corrected the correlation formula. PDR1-3-85: Added missing steps description in “Slow changing ionosphere estimation”. PDR1-3-86: Added Weighted Least Square inversion in the Multi-Baseline using the weights computed in the single baseline steps. PDR1-3-87: Removed the autofocus inputs PDR1-3-87, PDR1-3-89: added the formulation of the shifts correction in the “Residual coregistration from data” and in the “Baseline error estimation” PDR1-3-87: The sub-looks computation applied before the coregistration in the “Slow changing ionosphere estimation”. PDR1-3-87: “Residual coregistration from data” added to the calibration block diagram. PDR1-3-90: Added the update of the coregistration shifts annotated in the product in the “Residual coregistration from data” and in the “Baseline error estimation”. PDR1-3-91: Rephrased the sentence. PDR1-3-92: Removed the duplicated computation of the MS InSAR phase PDR1-3-92: Provide reference to the computation of the bandwidth of the azimuth sub-looks in the “Residual ionospheric scintillation estimation”. PDR1-3-94: Updated the I/O section and flow-chart of the “Phase calibration by SKP” PDR1-3-96: Removed the “Residual ionospheric scintillation estimation” algorithm using SVD. PDR1-3-97: Added the scintillation DC component removal in the “Residual ionospheric scintillation estimation” PDR1-3-98: Updated “Stacking” section according to the implementation. | Simone Mancon |
| 2022-05-23 | 1.1.1 | CDR1-224, CDR1-223: New algorithm to compute the RFI impact to the InSAR coherence. CDR1-152, CDR1-153, CDR1-221: Updated {fig}`3` and {sec}`2.3` with proper naming conventions and last algorithmic updates. CDR1-251: Split-band parameters (central frequency and bandwidth of sub-looks) stated as configurable. CDR1-252: The processing chain of the slow ionosphere removal has been restored to PDR1 version (Split-spectrum applied to the coregistered data). CDR1-231: the “Residual range shift estimation” has been removed. CDR1-232: Synthetic phase removal moved before the range spectral filtering. CDR1-233: Corrected the formulation of the central frequency in the “Range spectral filtering”. CDR1-234: Removed from the ”Azimuth spectral filtering” the dependence from the desired resolution. CDR1-235: Specified the type of low pass-filtering used in the “Slow ionosphere estimation”. CDR1-219: Added placemarks for two missing steps: Cross-pol merging and LUT interpolation. CDR1-226: Added list of geometric parameters generated by the back-geocoding. CDR1-236, CDR1-237: Clarified the dimensions of the matrixes in the single-baseline and multi-baseline of the slow ionosphere. CDR1-238: Added the formulation of the quality indicator of the multi-baseline results. CDR1-239: Removed the redundant synthetic phase correction in the “Baseline error estimation”. CDR1-240: Added the coregistration shift among the output of the “Baseline error correction”. CDR1-244: Azimuth filtering removed from the configurable parameters of “Fast ionosphere estimation”. CDR1-246: Clarified the single-baseline “Fast ionosphere estimation” is performed block-wise in range and the block in the same for ionospheric height and ionospheric phase estimation. CDR1-247: Clarified that “Fast ionosphere estimation” need look angle in input if the ionospheric height is passed as external parameter. CDR1-248: Stack calibrated by SKP phases defined as optional output. CDR1-229: Clarified in the formulation that speckle tracking is performed on the intensity of the images. CDR1-230: Modified the terminology: from scintillation to fast ionosphere. | Simone Mancon |
| 2023-02-08 | 1.1.2 draft | Pre-FAT1 issue of the document Updated the methodology to select the reference image for the coregistration by geometry. BPS-42: PDR1-2 - Updated the algorithm of the single-baseline estimation of the residual fast-varying ionosphere. Clarified that the SKP ground phase is composed by both the synthetic phase from geometry plus the residual phase computed by SKP. BPS-58: PDR1-21 – Added the SKP ground phase quality and the related threshold. | Simone Mancon |
| 2023-03-24 | 1.1.2 | Explained the method to perform the cross-pol merging. Updated the methodology to select the reference image for the coregistration using the temporal baseline. Explained the method to perform the oversampling of geometrical quantity. Explained the method to perform the shift and interpolation of the LUTs Add input/output tables to the calibration chapter and other minor refactoring to that chapter. Add the IOS description of the approach used for the ionosphere height estimation. Added the revert of the ionospheric phase screen estimated by L1 processor in the slow varying ionosphere estimation. Added the correction of bias in the slow varying ionosphere estimation due to the combined coregistration mode. Added the formulation of the estimated accuracy for baseline error estimation and the quality of the step has been modified accordingly. Removed the direct geocoding step and added the exploitation of the geometric LUT for coregistration and calibration. The Stack is now split in three sub-processor (according to the code structure of the code): Pre-processing, Stacking and Calibration. | Simone Mancon Federico Boniardi Giovanni Amoroso |
| 2023-01-09 | 1.1.3 | Delta-FAT2 issue of the document Major refactoring of the sections related to calibration: Restructured th AZF sections with major corrections. Rewritten IOB section. Refactored BAE section. Updated IOS section. Changed = with ≈ in equation of SKP phase screen. Added tables in the stacking section. Added description of Automatic coregistration modality. | Federico Boniardi |
| 2024-05-17 | 1.1.4 | Post-Delta-FAT2 issue of the document Fixed typos in calibration tables (IOB, IOS, BAE). Added note on possibility to disable weights in MB estimation of IOS. Updated fitting quality metric for coregistration. Removed “scintillation” in favor of “fast-varying ionosphere”. Added sections that describe how contingency cases are handled: Stacking: {sec}`4.4` IOB: {sec}`5.2.5` BAE: {sec}`5.3.5` IOS: Section 5.4.6 SKP: Section 5.5.3 Refactored {sec}`4.1`, including references to DEM. Finalized incomplete cross-pol merging (formula) and input table Added input/output table in {sec}`3.4.2`. Update quality for BAE’s Single-baseline estimation. Revised references and formulas (new numbering) Revised table of acronyms, including removal of unused entries Major refactoring of the LUT shifting section. Include note on the output resolution of LUTs. | Federico Boniardi |
| 2024-09-06 | 1.1.5 | FAT3 issue of the document Fixed section numbering for AZF. Updated I/O tables in Sections 5.2.1, {sec}`5.3.1`, and {sec}`5.4.1`. Addressed BPSCDR1-25 (multi-look size in azimuth and range direction) Addressed CDR1-245 (how to establish that there’s no peak in the ionosphere profiles) Substitute “Sequestrating” with “suppressing” | Federico Boniardi |
| 2024-12-13 | 1.1.6 | Post-FAT3 issue of the document | Federico Boniardi |
| 2025-03-07 | 1.2.0 | Post-FAT3 issue of the document Fixed contingency handling of Multi-baseline method (IOB, BAE, and IOS). Now a check on factor-graph connectivity is used instead of the previous laxer (and wrong) approach. IOB includes now recommendations from {rd}`25` (BIO-TOMEX). This includes updated I/O tables consistent with those. Updated I/O table for BAE with new parameter associated to MB Added reference to median filtering in SKP and updated I/O table Fixed notations for coregistrationShiftsQuality Fixed numbering of equations and associated references Rewritten {sec}`3.3.3` (same content, better wording) and added missing contingency case in {sec}`3.3.4` Fixed typos and sentences here and there | Federico Boniardi |
| 2025-04-30 | 1.2.1 | Post-FAT3 issue of the document Added more details on selection of interferometric pairs in {sec}`5.2.3` and {sec}`5.3.3`. | Federico Boniardi |
| 2025-07-23 | 1.3.0 | Post-launch issue of the document (associated to BPS V4.0.0) Add notes to explain a few details about the calculation of the vertical wavenumbers Deprecate BAE (former {sec}`5.3`) Deprecate IOS (former {sec}`5.4`) Substitute former {sec}`5.3` with newly added BIC, consistently with {rd}`25` (from BIO-TOMEX) | Federico Boniardi |
| 2025-07-30 | 1.3.1 | Fixes to the post-launch issue of the document Fix stale skpPhaseCorrectionFlag Fix stale “PhasesScreen” Add excludeMPMBPolarizationCrossCovarianceFlag | Federico Boniardi |
| 2025-11-28 | 1.3.2 | Fixes associated to BPS V4.2.0 Deprecated BIC Added Phase Plane Removal (PPR) module Updated coregistration outputs (coreg from orbit LUTs) Added possibility to disable model-based fitting of coreg shifts Added warning to user to avoid using median-filter in SKP Updated figures and diagrams | Federico Boniardi |
| 2026-04-02 | 1.4.0 | Post-launch issue of the document (associated to BPS V4.4.1) Deprecate the range spectral filtering in coregistration Deprecate the PPR Added MSC Added boxcar and Goldstein filter to SKP Removed warning on filtering the SKP phase with median filter | Federico Boniardi |
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
