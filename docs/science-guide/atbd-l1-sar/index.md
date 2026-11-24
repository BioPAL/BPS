<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BPS L1 SAR ATBD - Algorithms Theoretical Baseline Document
doc_code: BIO-BPS-L1-SAR-ATBD-ARE-010165
version: 1.2.5
date: 06/11/2026
family: atbd
license: CC-BY-4.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-23'
source_sha256: dba862f0fc5d16c9a3ecbc4d8c619e0c3a88754dda5c36a3e3dbb742dcb41f90
-->

# BPS L1 SAR ATBD - Algorithms Theoretical Baseline Document

[← Science Guide](../index.md)

::::{only} html

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-L1-SAR-ATBD-ARE-010165` issue **1.2.5** dated **06 November 2026**.
:::

::::

<p class="atbd-licence">Documentation © European Space Agency (ESA) and ARESYS, licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC-BY-4.0</a>.</p>

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BPS L1 SAR ATBD - Algorithms Theoretical Baseline Document |
| External reference | `BIO-BPS-L1-SAR-ATBD-ARE-010165` |
| Issue | 1.2.5 |
| Date | 06 November 2026 |
| Pages | 162 (authoritative PDF) |

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
| Pau Prats |  |
| Jun Su Kim |  |
| Marc Jäger |  |
| Felipe Betancourt |  |

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
| 2022-03-23 | 1.1 | CDR1 issue of the document<br>• Description of framing strategy updated (Sec. 6.1)<br>• Alignment with latest L1 GPP ATBD updates:<br>&nbsp;&nbsp;&nbsp;&nbsp;• Reference included regarding the computation of the CASP based on the CDASP (Sec. 4.6.1)<br>&nbsp;&nbsp;&nbsp;&nbsp;• Reformulation of internal instrument channel imbalance correction based on ICAL data (Sec. 3.6, 4.2.3)<br>• PDR1-2-51, PDR1-3-28: Details regarding the various options for ionosphere height determination added (Sec. 4.10.3)<br>• PDR1-2-52: Chirp function various sources clarified (Sec. 4.1.2)<br>• PDR1-3-26: SWST replaced with SWP (Sec. 4.2.1)<br>• PDR1-3-30: Details about quick-looks generation added (Sec. 5.1.1)<br>• PDR1-3-31, PDR1-3-34: Open points collected in an appendix section (Sec. 8)<br>• PDR1-3-32: Raw Data Correction inputs and outputs clarified (Sec. 3.4)<br>• PDR1-3-35: Instrument Replica Derivation inputs and outputs clarified (Sec. 3.6.3)<br>• PDR1-3-38, PDR1-3-51: Orbit and attitude data management and EO CFI usage clarified (Sec. 3.7, 6.4)<br>• PDR1-3-39: Back-geocoding section fully reviewed, harmonized with other BPS Processors and integrated where needed (Sec. 3.8)<br>• PDR1-3-41: Details about RFI mitigation baseline configuration added (Sec. 4.1)<br>• PDR1-3-42: Wrong equation removed (Sec. 4.3.1)<br>• PDR1-3-43: Antenna pattern added in input (Sec. 4.3.2)<br>• PDR1-3-44: Details about DC estimation block size baseline configuration added (Sec. 4.3.2)<br>• PDR1-3-45: Azimuth resampling remodulation step added (Sec. 4.5)<br>• PDR1-3-46, PDR1-3-49: Details about azimuth resampling step added (Sec. 4.5, 6.3)<br>• PDR1-3-50: Coordinate Frames and Coordinate Frame Transformations section removed to avoid duplications and inconsistencies w.r.t. applicable documentation [AD2] (Sec. 6.6)<br>• PDR1-3-52: Performances analyses sections removed and demanded to reference documentation (ALL)<br>• PDR1-3-53: Details about space-variant threshold added (Sec. 4.1.1)<br>• PDR1-3-54: Link to previous studies on RFI mitigation algorithms underlined (Sec. 4.1.3)<br>• PDR1-3-56: Details about quality parameters computation and optimization algorithm added (Sec. 4.3.2) | Riccardo Piantanida |
| 2022-05-23 | 1.1.1 | Post-CDR1 issue of the document<br>• CDR1-058, CDR1-085: Details on orbit and attitude extraction step added (Sec. 3.7, 6.5)<br>• CDR1-059: Rule for selection of direction geocoding solutions corrected (Sec. 3.8.1)<br>• CDR1-060: Nadir direction formula corrected (Sec. 3.8.3.2)<br>• CDR1-061: Terrain slope description aligned with formulas (Sec. 3.8.3.3)<br>• CDR1-062, CDR1-112, CDR1-114: RFI mitigation step description improved and aligned to latest evolutions; main configuration parameters added to inputs list (Sec. 4.1)<br>• CDR1-063: Computation of unit vector normal to the current pointing plane updated (Sec. 4.3.1)<br>• CDR1-065: Correct sequence of Azimuth Resampling and Doppler Centroid Estimation steps clarified (Sec. 2.3, 4.5)<br>• CDR1-066: Quicklook generation procedure updated to have more homogeneous results (Sec. 5.1.1)<br>• CDR1-067: Open points on ionospheric calibration step added (Sec. 8)<br>• CDR1-068: Open point on BAQ decoding closed, reflecting corresponding action taken at BIOMASS L1 GPP project level (Sec. 3.2, 8)<br>• CDR1-069: Details on interpolation filters constraints added (Sec. 6.3); details about specific solution adopted for azimuth resampling step added (Sec. 4.5)<br>• CDR1-070: Details about PARC processing added, related open point added (Sec. 4.2.4, 8)<br>• CDR1-079: Orbit spline interpolator increased to 5{sup}`th` order (Sec. 6.4)<br>• CDR1-082: Note on first transmitted pulse added (Sec. 3.3)<br>• CDR1-083: Open point on instrument replica derivation added (Sec. 8)<br>• CDR1-084: Open point on instrument internal channel imbalance (Sec. 8)<br>• CDR1-086: Figure updated to add antenna pattern (Sec. 4.3.2)<br>• CDR1-087: Threshold used to select ionosphere height estimation method made configurable (Sec. 4.10.3)<br>• CDR1-113: Morphological operators usage in RFI mitigation step clarified (Sec. 4.1.1)<br>• CDR1-144: Section with considerations on back-geocoding sampling grid added (Sec. 3.8.3)<br>• CDR1-188: Sequence of operations implemented to compute number of looks and filter shape clarified. (Sec. 4.10.2.1)<br>• CDR1-225: List of parameters retrieved by the back-geocoding step added; links to L1 products annotations added (Sec. 3.8.4) | Riccardo Piantanida |
| 2023-03-20 | 1.1.2 | Post-FAT1 issue of the document<br>• Alignment with latest L1 GPP ATBD (I/R 2/2) updates:<br>&nbsp;&nbsp;&nbsp;&nbsp;• Clarification concerning azimuth mis-pointing between polarizations added (Sec. 4.6.1)<br>&nbsp;&nbsp;&nbsp;&nbsp;• New section to describe the updates in the ionospheric filtering step (Sec. 4.10.5)<br>&nbsp;&nbsp;&nbsp;&nbsp;• Clarification on the integration of the IRI model in the GPP in order to retrieve the ionospheric height (Sec. 4.10.6)<br>• BPSCDR1-10, BPSCDR1-11, BPSPDR2-83: Document fully reviewed to be aligned to latest issues of BPS L1\_PFD and AUX\_FMT documents, adding input and output details tables for each processing step (ALL)<br>• BPSCDR1-2: Added clarification on Doppler Centroid differences between polarizations (Sec. 4.3.1)<br>• BPSCDR1-3: Removed quick-look percentile clipping (Sec. 5.1.1)<br>• BPSCDR1-38: Clarified details about peak interpolation function (Sec. 3.4.2)<br>• BPSCDR1-39: Added reference to RDB (Sec. 1.6, 3.2.2) | Riccardo Piantanida |
| 2024-01-19 | 1.1.3 | Delta-FAT2 issue of the document<br>• DLR members added to authors list<br>• Alignment with latest L1 GPP ATBD (I/R 3/0) updates:<br>&nbsp;&nbsp;&nbsp;&nbsp;• Antenna pattern compensation description updated in order to match current implementation (Sec. 4.6)<br>&nbsp;&nbsp;&nbsp;&nbsp;• Improved description of mathematical details for ionosphere calibration step (Sec. 4.10)<br>&nbsp;&nbsp;&nbsp;&nbsp;• Improved description of autofocus approach (Sec. 4.11)<br>• BPS-221: Completed alignment with AUX\_FMT and L1\_PFD document (ALL)<br>• BPSFAT1-80: Improved description of IRI model usage (Sec. 4.10.6) | Riccardo Piantanida |
| 2024-05-17 | 1.1.4 | Post-Delta-FAT2 issue of the document<br>• BPS-505: Clarified strategy for back-geocoding step (Sec. 3.6.3); updated description of LUT in L1 Post-Processing step (Sec. 5.1)<br>• BPSCDR1-16: Added note on Gaussian filter size limitations (Sec. 4.10.5)<br>• BPSFAT1-80: Added clarification on IRI model usage (Sec. 4.10.6) | Riccardo Piantanida |
| 2024-09-06 | 1.1.5 | FAT3 issue of the document<br>• Removed references to Phase Gradient Autofocus (PGA) (Sec. 4.11)<br>• For each section, added in input/output tables references to intermediate products names, in order to have a link with the complete list reported in BPS\_SDD (ALL) | Riccardo Piantanida |
| 2024-12-13 | 1.1.6 | Post-FAT3 issue of the document<br>• Added clarification on RAW data corrections (Sec. 3.3)<br>• Updated reference frame for orbit and attitude from PEF to EF and added details on management (Sec. 3.5)<br>• Added section on 2D antenna patterns extraction (Sec. 3.6)<br>• Added formulas for polarimetric corrections (Sec. 4.9)<br>• Added details on ionosphere height estimation strategies (Sec. 4.10.2.3.3)<br>• Added details on external TEC map management (Sec. 4.10.3.2)<br>• Added details on blocketization strategies (Sec. 4.10.1, 6.1)<br>• Overall update to input/output tables to clarify intermediate products references (ALL) | Riccardo Piantanida |
| 2025-03-07 | 1.2.0 | Post-Delta-FAT3 issue of the document<br>• backgeocodingPosting removed from Back-geocoding step inputs (Sec. 3.7)<br>• Added clarification of potential future enhancements of ionospheric calibration blocketization strategy (Sec. 4.10.1) | Riccardo Piantanida |
| 2025-06-10 | 1.2.1 | Post-launch issue of the document<br>• Added link to [AD6] applicable document (Sec. 3.4)<br>• Updated conventions used for L1a and L1b quick-looks generation (Sec. 5.1.1) | Riccardo Piantanida |
| 2025-07-23 | 1.2.2 | Post-launch issue of the document (associated to BPS V4.0.0)<br>• Removed unused configuration parameters (Sec. 4.6)<br>• Added possibility to make blocks size and overlap configurable (Sec. 4.10.1) | Riccardo Piantanida |
| 2025-11-28 | 1.2.3 | Post-launch issue of the document (associated to BPS V4.2.0)<br>• Added details on new processing options for RFI mitigation step and on annotated RFI statistics (Sec. 4.1)<br>• Added details on range delays corrected during range compression step (Sec. 4.2.2)<br>• Added clarifications on Doppler Centroid combined method fallback and polarization management strategies (Sec. 4.3.2)<br>• Added section on management of geomagnetic equator crossing during ionospheric calibration step (Sec. 4.10.3.3)<br>• List of open points updated (removing closed ones) (Sec. 8) | Riccardo Piantanida |
| 2026-03-27 | 1.2.4 | Post-launch issue of the document (associated to BPS V4.4.0)<br>• Added detailed formulas for annotated RFI statistics (Sec. 4.1)<br>• Added in/out-of-band power ratio computation step (Sec. 3.2.5) | Riccardo Piantanida |
| 2025-11-06 | 1.2.5 | Post-launch issue of the document (associated to BPS V4.5.0)<br>• Added minor clarifications to FR filtering implementation (Sec. 4.10.4) | Riccardo Piantanida |
:::

---

```{toctree}
:maxdepth: 2
:hidden:

1. Introduction <01-introduction>
2. BPS and L1 Processor Overview <02-bps-and-l1-processor-overview>
3. L1 Pre-Processing <03-l1-pre-processing>
4. L1 Core Processing <04-l1-core-processing>
5. L1 Post-Processing <05-l1-post-processing>
6. Ancillary Functions and Parameters <06-ancillary-functions-and-parameters>
7. Configurable Input Parameters <07-configurable-input-parameters>
8. Open Points <08-open-points>
9. Appendix: L1 Framing Tool <09-appendix-l1-framing-tool>
References <references>
```
