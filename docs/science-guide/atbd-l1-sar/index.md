<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


<!--
Pipeline metadata (not rendered):
title: BPS L1 SAR ATBD - Algorithms Theoretical Baseline Document
doc_code: BIO-BPS-L1-SAR-ATBD-ARE-010165
version: 1.2.4
date: 27/03/2026
family: atbd
license: Apache-2.0
backend: docx
status: draft
pipeline_version: 0.1.0
conversion_date: '2026-11-05'
source_sha256: b84379a9befde0623f6bd537aced849b4e2118871134a55c90065fe755287c93
-->

# BPS L1 SAR ATBD - Algorithms Theoretical Baseline Document

[← Science Guide](../index.md)

:::{admonition} Draft, automatic web conversion
:class: warning

This page is an **automatic web conversion** of the official document, pending review. The official PDF remains the authoritative reference.
[Download the PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_L1_SAR_ATBD_v1_2_4.pdf)
:::

:::{admonition} Authoritative document
:class: tip

This page is the web-rendered version of `BIO-BPS-L1-SAR-ATBD-ARE-010165` issue **1.2.4** dated **27 March 2026**.
:::

:::{dropdown} Document information
:icon: info

| Field | Value |
|---|---|
| Title | BPS L1 SAR ATBD - Algorithms Theoretical Baseline Document |
| External reference | `BIO-BPS-L1-SAR-ATBD-ARE-010165` |
| Issue | 1.2.4 |
| Date | 27 March 2026 |
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
| 2022-03-23 | 1.1 | CDR1 issue of the document Description of framing strategy updated ({sec}`6.1`) Alignment with latest L1 GPP ATBD updates: Reference included regarding the computation of the CASP based on the CDASP ({sec}`4.6.1`) Reformulation of internal instrument channel imbalance correction based on ICAL data ({sec}`3.6`, 4.2.3) PDR1-2-51, PDR1-3-28: Details regarding the various options for ionosphere height determination added ({sec}`4.10.3`) PDR1-2-52: Chirp function various sources clarified ({sec}`4.1.2`) PDR1-3-26: SWST replaced with SWP ({sec}`4.2.1`) PDR1-3-30: Details about quick-looks generation added ({sec}`5.1.1`) PDR1-3-31, PDR1-3-34: Open points collected in an appendix section ({sec}`8`) PDR1-3-32: Raw Data Correction inputs and outputs clarified ({sec}`3.4`) PDR1-3-35: Instrument Replica Derivation inputs and outputs clarified ({sec}`3.6.3`) PDR1-3-38, PDR1-3-51: Orbit and attitude data management and EO CFI usage clarified (Sec. 3.7, 6.4) PDR1-3-39: Back-geocoding section fully reviewed, harmonized with other BPS Processors and integrated where needed (Sec. 3.8) PDR1-3-41: Details about RFI mitigation baseline configuration added ({sec}`4.1`) PDR1-3-42: Wrong equation removed ({sec}`4.3.1`) PDR1-3-43: Antenna pattern added in input ({sec}`4.3.2`) PDR1-3-44: Details about DC estimation block size baseline configuration added ({sec}`4.3.2`) PDR1-3-45: Azimuth resampling remodulation step added ({sec}`4.5`) PDR1-3-46, PDR1-3-49: Details about azimuth resampling step added ({sec}`4.5`, 6.3) PDR1-3-50: Coordinate Frames and Coordinate Frame Transformations section removed to avoid duplications and inconsistencies w.r.t. applicable documentation {ad}`2` (Sec. 6.6) PDR1-3-52: Performances analyses sections removed and demanded to reference documentation (ALL) PDR1-3-53: Details about space-variant threshold added ({sec}`4.1.1`) PDR1-3-54: Link to previous studies on RFI mitigation algorithms underlined ({sec}`4.1.3`) PDR1-3-56: Details about quality parameters computation and optimization algorithm added ({sec}`4.3.2`) | Riccardo Piantanida |
| 2022-05-23 | 1.1.1 | Post-CDR1 issue of the document CDR1-058, CDR1-085: Details on orbit and attitude extraction step added (Sec. 3.7, 6.5) CDR1-059: Rule for selection of direction geocoding solutions corrected (Sec. 3.8.1) CDR1-060: Nadir direction formula corrected (Sec. 3.8.3.2) CDR1-061: Terrain slope description aligned with formulas (Sec. 3.8.3.3) CDR1-062, CDR1-112, CDR1-114: RFI mitigation step description improved and aligned to latest evolutions; main configuration parameters added to inputs list ({sec}`4.1`) CDR1-063: Computation of unit vector normal to the current pointing plane updated ({sec}`4.3.1`) CDR1-065: Correct sequence of Azimuth Resampling and Doppler Centroid Estimation steps clarified ({sec}`2.3`, 4.5) CDR1-066: Quicklook generation procedure updated to have more homogeneous results ({sec}`5.1.1`) CDR1-067: Open points on ionospheric calibration step added ({sec}`8`) CDR1-068: Open point on BAQ decoding closed, reflecting corresponding action taken at BIOMASS L1 GPP project level ({sec}`3.2`, 8) CDR1-069: Details on interpolation filters constraints added ({sec}`6.3`); details about specific solution adopted for azimuth resampling step added ({sec}`4.5`) CDR1-070: Details about PARC processing added, related open point added ({sec}`4.2.4`, 8) CDR1-079: Orbit spline interpolator increased to 5th order ({sec}`6.4`) CDR1-082: Note on first transmitted pulse added ({sec}`3.3`) CDR1-083: Open point on instrument replica derivation added ({sec}`8`) CDR1-084: Open point on instrument internal channel imbalance ({sec}`8`) CDR1-086: Figure updated to add antenna pattern ({sec}`4.3.2`) CDR1-087: Threshold used to select ionosphere height estimation method made configurable ({sec}`4.10.3`) CDR1-113: Morphological operators usage in RFI mitigation step clarified ({sec}`4.1.1`) CDR1-144: Section with considerations on back-geocoding sampling grid added (Sec. 3.8.3) CDR1-188: Sequence of operations implemented to compute number of looks and filter shape clarified. ({sec}`4.10.2.1`) CDR1-225: List of parameters retrieved by the back-geocoding step added; links to L1 products annotations added (Sec. 3.8.4) | Riccardo Piantanida |
| 2023-03-20 | 1.1.2 | Post-FAT1 issue of the document Alignment with latest L1 GPP ATBD (I/R 2/2) updates: Clarification concerning azimuth mis-pointing between polarizations added ({sec}`4.6.1`) New section to describe the updates in the ionospheric filtering step ({sec}`4.10.5`) Clarification on the integration of the IRI model in the GPP in order to retrieve the ionospheric height ({sec}`4.10.6`) BPSCDR1-10, BPSCDR1-11, BPSPDR2-83: Document fully reviewed to be aligned to latest issues of BPS L1\_PFD and AUX\_FMT documents, adding input and output details tables for each processing step (ALL) BPSCDR1-2: Added clarification on Doppler Centroid differences between polarizations ({sec}`4.3.1`) BPSCDR1-3: Removed quick-look percentile clipping ({sec}`5.1.1`) BPSCDR1-38: Clarified details about peak interpolation function (Sec. 3.4.2) BPSCDR1-39: Added reference to RDB ({sec}`1.6`, 3.2.2) | Riccardo Piantanida |
| 2024-01-19 | 1.1.3 | Delta-FAT2 issue of the document DLR members added to authors list Alignment with latest L1 GPP ATBD (I/R 3/0) updates: Antenna pattern compensation description updated in order to match current implementation ({sec}`4.6`) Improved description of mathematical details for ionosphere calibration step ({sec}`4.10`) Improved description of autofocus approach ({sec}`4.11`) BPS-221: Completed alignment with AUX\_FMT and L1\_PFD document (ALL) BPSFAT1-80: Improved description of IRI model usage ({sec}`4.10.6`) | Riccardo Piantanida |
| 2024-05-17 | 1.1.4 | Post-Delta-FAT2 issue of the document BPS-505: Clarified strategy for back-geocoding step ({sec}`3.6.3`); updated description of LUT in L1 Post-Processing step ({sec}`5.1`) BPSCDR1-16: Added note on Gaussian filter size limitations ({sec}`4.10.5`) BPSFAT1-80: Added clarification on IRI model usage ({sec}`4.10.6`) | Riccardo Piantanida |
| 2024-09-06 | 1.1.5 | FAT3 issue of the document Removed references to Phase Gradient Autofocus (PGA) ({sec}`4.11`) For each section, added in input/output tables references to intermediate products names, in order to have a link with the complete list reported in BPS\_SDD (ALL) | Riccardo Piantanida |
| 2024-12-13 | 1.1.6 | Post-FAT3 issue of the document Added clarification on RAW data corrections ({sec}`3.3`) Updated reference frame for orbit and attitude from PEF to EF and added details on management ({sec}`3.5`) Added section on 2D antenna patterns extraction ({sec}`3.6`) Added formulas for polarimetric corrections ({sec}`4.9`) Added details on ionosphere height estimation strategies ({sec}`4.10.2.3.3`) Added details on external TEC map management ({sec}`4.10.3.2`) Added details on blocketization strategies ({sec}`4.10.1`, 6.1) Overall update to input/output tables to clarify intermediate products references (ALL) | Riccardo Piantanida |
| 2025-03-07 | 1.2.0 | Post-Delta-FAT3 issue of the document backgeocodingPosting removed from Back-geocoding step inputs (Sec. 3.7) Added clarification of potential future enhancements of ionospheric calibration blocketization strategy ({sec}`4.10.1`) | Riccardo Piantanida |
| 2025-06-10 | 1.2.1 | Post-launch issue of the document Added link to {ad}`6` applicable document ({sec}`3.4`) Updated conventions used for L1a and L1b quick-looks generation ({sec}`5.1.1`) | Riccardo Piantanida |
| 2025-07-23 | 1.2.2 | Post-launch issue of the document (associated to BPS V4.0.0) Removed unused configuration parameters ({sec}`4.6`) Added possibility to make blocks size and overlap configurable ({sec}`4.10.1`) | Riccardo Piantanida |
| 2025-11-28 | 1.2.3 | Post-launch issue of the document (associated to BPS V4.2.0) Added details on new processing options for RFI mitigation step and on annotated RFI statistics ({sec}`4.1`) Added details on range delays corrected during range compression step ({sec}`4.2.2`) Added clarifications on Doppler Centroid combined method fallback and polarization management strategies ({sec}`4.3.2`) Added section on management of geomagnetic equator crossing during ionospheric calibration step ({sec}`4.10.3.3`) List of open points updated (removing closed ones) ({sec}`8`) | Riccardo Piantanida |
| 2026-03-27 | 1.2.4 | Post-launch issue of the document (associated to BPS V4.4.0) Added detailed formulas for annotated RFI statistics ({sec}`4.1`) Added in/out-of-band power ratio computation step ({sec}`3.2.5`) | Riccardo Piantanida |
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
