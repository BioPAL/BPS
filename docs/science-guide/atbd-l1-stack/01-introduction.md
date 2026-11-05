<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-l1-stack-1)=
# 1. Introduction

(sec:atbd-l1-stack-1-1)=
### 1.1 Context

BIOMASS is ESA’s 7th Earth Explorer mission. The purpose of the BIOMASS mission is to reduce the uncertainty in the worldwide spatial distribution and dynamics of forest biomass to improve current assessments and future projections of the global carbon cycle. This objective will be achieved by implementing a P-band SAR mission, providing global maps of forest biomass stocks, forest disturbance, and growth.

The BIOMASS mission will also provide 3D views of forests through the “tomographic” phase and will also support additional science needs arising from the opportunity to explore the Earth for the first time with a P-band SAR system from space offering the possibility to “see” below the vegetated areas and in beneath bare soil or icy regions.

To achieve its challenging objectives, the BIOMASS requires a specific data processing strategy that combines different technologies and techniques at different processing level to generate the final Above Ground Biomass (AGB) global map.

(sec:atbd-l1-stack-1-2)=
### 1.2 Scope of the Document

The purpose of this document is to provide the “Stack Processor Algorithms Theoretical Baseline Document (ATBD)” for the BIOMASS Processing Suite (BPS) project.

This document is organized as follows:

- {sec}`Section|2`: Brief introduction about BPS and Stack Processor;
- {sec}`Section|3`: Description of the Pre-processing algorithms;
- {sec}`Section|4`: Description of the Stacking algorithms;
- {sec}`Section|5`: Description of the Calibration algorithms;
- {sec}`Section|6`: Description of configurable input parameters.

(sec:atbd-l1-stack-1-3)=
### 1.3 Acronyms

:::{dropdown} Acronyms
| Acronym | Description |
|---|---|
| AD | Applicable Document |
| AGB | Above Ground Biomass |
| ARESYS | Advanced Remote Sensing and Systems |
| AT | Along-track (e.g. along-track direction) |
| ATBD | Algorithm Theoretical Baseline Document |
| AUX | Auxiliary (e.g `AUX_PPS`) |
| AWGN | Additive White Gaussian Noise |
| AZF | Azimuth Spectral Filtering |
| BIODEMPP | Biomass DEM Prototype Processor |
| BPS | BIOMASS Processing Suite |
| CFI | Customer Furnished Items |
| CoSCS | Coregistered SCS |
| DEM | Digital Elevation Model |
| DSI | DEM Synthetic Interferogram |
| DTM | Digital Terrain Model |
| ESA | European Space Agency |
| FNF | Forest/Non-forest (e.g. mask) |
| FR | Faraday Rotation |
| InSAR | Interferometric Synthetic Aperture Radar |
| INT | Interferometric Phase |
| IOB | Background Ionosphere Removal (also Slow Varying Ionosphere Removal) |
| IODD | Input&Output Data Definition |
| L1a/b/c | Level-1 a/b/c product |
| L1M | Level-1 Monitoring product |
| L1S | Level-1 Standard product |
| LOS | Line-of-Sight |
| LUT | Look-Up Table |
| MB | Multi-baseline |
| MPMB | Multi-polarimetric Multi-baseline |
| MS | Multi-Squint (e.g. coherence) |
| MSC | Multi-Squint Calibration |
| PFD | Product Format Specification Document |
| RD | Reference Document |
| RFI | Radio-Frequency Interferences |
| SAR | Synthetic Aperture Radar |
| SB | Single-baseline |
| SCS | Single-look Complex Slant-range |
| SKP | Sum of Kronecker Products |
| SNR | Signal-to-Noise Ratio |
| SVD | Single Value Decomposition |
| TEC | Total Electron Content |
| TOM | Tomographic Phase |
:::

(sec:atbd-l1-stack-1-4)=
### 1.4 Applicable documents

| Code | Document |
|---|---|
| <span id="ad1"></span>`[AD1]` | / |

(sec:atbd-l1-stack-1-5)=
### 1.5 Reference documents

| Code | Document |
|---|---|
| <span id="rd1"></span>`[RD1]` | BIOMASS DEM Product Processor Prototype, CoSCS Processor Algorithm Theoretical Baseline Document, BIODEMPP-CoSCS-ATBD-01, Issue 4.0, June 07th 2021. |
| <span id="rd2"></span>`[RD2]` | BIOMASS Report for Mission Selection; SP-1341/1, May 2012 |
| <span id="rd3"></span>`[RD3]` | BIOMASS L2 Implementation Study, Integrated R&D Report. Report to ESA, 2018. |
| <span id="rd4"></span>`[RD4]` | BIOMASS DEM Product Processor Prototype, DEM Processor Algorithm Theoretical Baseline Document, BIODEMPP-CoSCS-DEM-01, Issue 4.0, June 07th 2021. |
| <span id="rd5"></span>`[RD5]` | BIOMASS DEM Product Prototype Processor, Critical review of CoSCS and DEM algorithms, BIODEMPP-TN-01, Issue of 07th of February 2020. |
| <span id="rd6"></span>`[RD6]` | M. Liao, H. Jiang, Y. Wang, T. Wang, L. Zhang, “Improved topographic mapping through high-resolution SAR interferometry with atmospheric effect removal”, ISPRS Journal of Photogrammetry and Remote Sensing, Volume 80, 2013, Pages 72-79 |
| <span id="rd7"></span>`[RD7]` | Z. Hu, J. J. Mallorquí, G. Centolanza and J. Duro, "InSAR atmospheric delays compensation: Case study in Tenerife island," 2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS), Fort Worth, TX, 2017, pp. 3167-3170. |
| <span id="rd8"></span>`[RD8]` | M. Pinheiro, A. Reigber, R. Scheiber, P. Prats-Iraola and A. Moreira, "Generation of Highly Accurate DEMs Over Flat Areas by Means of Dual-Frequency and Dual-Baseline Airborne SAR Interferometry," in IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 8, pp. 4361-4390, Aug. 2018. |
| <span id="rd9"></span>`[RD9]` | G. Gomba, A. Parizzi, F. D. Zan, M. Eineder, and R. Bamler, “Toward Operational Compensation of Ionospheric Effects in SAR Interferograms: The Split-Spectrum Method,” IEEE Transactions on Geoscience and Remote Sensing, vol. 54, no. 3, pp. 1446–1461, Mar. 2016. |
| <span id="rd10"></span>`[RD10]` | S. Mancon, A. M. Guarnieri, D. Giudici, and S. Tebaldini, “On the phase calibration by multisquint analysis in TOPSAR and stripmap interferometry,” IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 1, pp. 134–147, 2016. |
| <span id="rd11"></span>`[RD11]` | S. Tebaldini, M. M. d’Alessandro, J. S. Kim, and K. Papathanassiou, “Ionosphere vertical profiling from biomass multi-squint InSAR,” in 2017 IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2017, pp. 5866–5869. |
| <span id="rd12"></span>`[RD12]` | R. Scheiber and A. Moreira, “Coregistration of interferometric SAR images using spectral diversity,” Geoscience and Remote Sensing, IEEE Transactions on, vol. 38, no. 5, pp. 2179–2191, 2000. |
| <span id="rd13"></span>`[RD13]` | Tebaldini, Stefano, et al. "Phase calibration of airborne tomographic SAR data via phase center double localization." IEEE Transactions on Geoscience and Remote Sensing 54.3 (2016): 1775-1792. |
| <span id="rd14"></span>`[RD14]` | Bamler, Richard und Hartl, Philip (1998) Synthetic Aperture Radar Interferometry. Inverse Problems, 14 (4). |
| <span id="rd15"></span>`[RD15]` | F. Gatelli, A. Monti Guamieri, F. Parizzi, P. Pasquali, C. Prati and F. Rocca, "The wavenumber shift in SAR interferometry," in IEEE Transactions on Geoscience and Remote Sensing, vol. 32, no. 4, pp. 855-865, July 1994. |
| <span id="rd16"></span>`[RD16]` | A. Reigber, "Range dependent spectral filtering to minimize the baseline decorrelation in airborne SAR interferometry," IEEE 1999 International Geoscience and Remote Sensing Symposium. IGARSS'99 (Cat. No.99CH36293), Hamburg, Germany, 1999, pp. 1721-1723 vol.3. |
| <span id="rd17"></span>`[RD17]` | Ferretti, A., Monti-Guarnieri A., Prati, C., Rocca, F., Massonnet, D. (2007). InSAR Principles: Guidelines for SAR Interferometry Processing and Interpretation. ESA Publications, TM-19. ISBN 92-9092-233-8. |
| <span id="rd18"></span>`[RD18]` | [`BPS_AUX_FMT`] BPS Auxiliary Product Format, I/R 1/1 |
| <span id="rd19"></span>`[RD19]` | [`BPS_L1_PFD`] BIOMASS L1a/b/c Products Format Specification, I/R 1/1 |
| <span id="rd20"></span>`[RD20]` | BPS L1 SAR ATBD - Algorithms Theoretical Baseline Document |
| <span id="rd21"></span>`[RD21]` | Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor, and Munther A. Gdeisat, “Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path”, Journal Applied Optics, Vol. 41, No. 35, pp. 7437, 2002 |
| <span id="rd22"></span>`[RD22]` | Laakso, Timo I., et al. “Splitting the unit delay [FIR/all pass filters design].” IEEE Signal Processing Magazine 13.1 (1996): 30-60. |
| <span id="rd23"></span>`[RD23]` | S. Tebaldini, A. M. Guarnieri, and F. Rocca, “Recovering time and space varying phase screens through SAR multi-squint differential interferometry,” in EUSAR 2012; 9th European Conference on Synthetic Aperture Radar, Apr. 2012, pp. 16–19. |
| <span id="rd24"></span>`[RD24]` | [`BPS_IODD`] BIOMASS Processors Input & Output Data Definition, Issue 3.0.5, February 2nd 2024. |
| <span id="rd25"></span>`[RD25]` | [`BIOTOMEX_TN`] BIOTOMEX TN, Technical Note on Performance Analysis, Issue 1.1, November 26th, 2024 |
| <span id="rd26"></span>`[RD26]` | R. Lanari et al. "Generation of digital elevation models by using SIR-C/X-SAR multifrequency two-pass interferometry: The Etna case study." IEEE Transactions on Geoscience and Remote Sensing (1996) |
| <span id="rd27"></span>`[RD27]` | S. Tebaldini, F. Salvaterra, F. Banda, and M. Pinheiro, “Multi-layer ionosphere correction in low-frequency interferometric SAR: modeling, inversion, and application to BIOMASS”. Authorea Preprints (2025) |
