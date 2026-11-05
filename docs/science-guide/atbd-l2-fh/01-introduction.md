<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-fh-1)=
# 1. Introduction

(sec:atbd-fh-1-1)=
### 1.1 Context

BIOMASS is ESA’s 7th Earth Explorer mission. The purpose of the BIOMASS is to reduce the uncertainty in the worldwide spatial distribution and dynamics of forest biomass in order to improve current assessments and future projections of the global carbon cycle. This objective will be achieved by the implementation of a P-band SAR mission, providing global maps of forest biomass stocks, forest disturbance and growth.

The BIOMASS mission will also provide 3D views of forests through the “tomographic” phase and will also support additional science needs arising from the opportunity to explore the Earth for the first time with a P-band SAR system from space offering the possibility to “see” below the vegetated areas and in beneath bare soil or icy regions.

To achieve its challenging objectives, the BIOMASS mission requires a specific data processing strategy that combines different technologies at different processing level in order to generate the final Above Ground Biomass global map.

(sec:atbd-fh-1-2)=
### 1.2 Scope of the document

The purpose of this document is to provide the “Forest Height Algorithms Theoretical Baseline Document (ATBD)” for the BIOMASS Processing Suite (BPS) project.

This document is organized as follows:

- {sec}`Section|1`: Overview of the document;
- {sec}`Section|2`: Brief introduction about BPS and Forest Height processing;
- {sec}`Section|3`: Description of L2a Forest Height algorithms;
- {sec}`Section|4`: Description of L2b/L3 Forest Height algorithms.

(sec:atbd-fh-1-3)=
### 1.3 Acronyms

:::{dropdown} Acronyms
| Acronym | Description |
|---|---|
| AD | Applicable Document |
| ADS | Annotation datasets |
| AGB | Above Ground Biomass |
| ARESYS | Advanced Remote Sensing and Systems |
| ATBD | Algorithm Theoretical Baseline Document |
| AUX | Auxiliary |
| BIODEMPP | Biomass DEM Prototype Processor |
| BPS | BIOMASS Processing Suite |
| CM | Covariance Matrix |
| CFI | Customer Furnished Items |
| CoSCS | Coregistered SCS |
| CP | co-polarization |
| DEM | Digital Elevation Model |
| DGG | Discrete Global Grid |
| DTM | Digital Terrain Model |
| ECSS | European Cooperation for Space Standardization |
| EO-CFI | Earth Observation Custom Furnished Items |
| ESA | European Space Agency |
| GPP | Ground Processor Prototype |
| GPS | Global Positioning System |
| FD | Forest Disturbance |
| FH | Forest Height |
| FNF | Forest/Non-Forest Mask |
| IERS | International Earth Rotation Service |
| InSAR | Interferometric Synthetic Aperture Radar |
| INT | Interferometric Phase |
| L0 | Level-0 product |
| L1c | Level-1c product |
| L2a | Level-2a product |
| L2b | Level-2b product |
| L3 | Level-3 product |
| LOS | Line Of Sight |
| LUT | Look-Up Table |
| PF | Processing Facility |
| PFD | Product Format Specification Document |
| RD | Reference Document |
| SAR | Synthetic Aperture Radar |
| SCS | Single-look Complex Slant-range |
| SKP | Sum of Kronecker Products |
| SNR | Signal-to-Noise Ratio |
| TBD | To Be Defined |
| TBC | To Be Confirmed |
| TOM | Tomographic Phase |
| XML | `eXtended` Markup Language |
| XP | cross-polarization |
:::

(sec:atbd-fh-1-4)=
### 1.4 TBD / TBC Table

| Section | TBD / TBC | Page |
|---|---|---|
| 1.6 | `BTK_WEB`: link must be defined | 9 |

(sec:atbd-fh-1-5)=
### 1.5 Applicable documents

| Code | Document |
|---|---|
| <span id="ad1"></span>`[AD1]` | BIOMASS Production Model, `BIO-ESA-EOPG-EEGS-TN-0046`, issue 3.5 |
| <span id="ad2"></span>`[AD2]` | BIO-BPS-L1-PFD, BIOMASS L1a,b,c products format specification, I/R 1/2/2 |
| <span id="ad3"></span>`[AD3]` | BIO-BPS-TN-PRODMOD, BIOMASS Production Model Technical Note, I/R 1/2/1 |

(sec:atbd-fh-1-6)=
### 1.6 Reference documents

| Code | Document |
|---|---|
| <span id="rd1"></span>`[RD1]` | BIO-BPS-L1-STACK-ATBD, BIOMASS BPS L1c Stack ATBD, I/R 1/1/2 |
| <span id="rd2"></span>`[RD2]` | BIO-BPS-L1-SAR-ATBD, BIOMASS BPS L1 SAR ATBD, I/R 1/1/2 |
| <span id="rd3"></span>`[RD3]` | BIOMASS DEM Product Processor Prototype, DEM Processor Algorithm Theoretical Baseline Document, BIODEMPP-CoSCS-DEM-01, I/R 4/0 |
| <span id="rd4"></span>`[RD4]` | BIO-BPS-FD-ATBD, BIOMASS BPS FD ATBD, I/R 2/1/1 |
| <span id="rd5"></span>`[RD5]` | BIOMASS L2 Implementation Study ATBD, I/R 1/5 |
| <span id="rd6"></span>`[RD6]` | F. Kugler, S. Lee, I. Hajnsek and K. P. Papathanassiou, "Forest Height Estimation by Means of Pol-InSAR Data Inversion: The Role of the Vertical Wavenumber," in IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 10, pp. 5294-5311, Oct. 2015 |
| <span id="rd7"></span>`[RD7]` | BIO-BPS- AUX-FMT, BPS Auxiliary Product Format, I/R 2/1/1 |
| <span id="rd8"></span>`[RD8]` | BIO-BPS-PPD, BIOMASS BPS Product Performance Description, I/R 2/0/2 |
| <span id="rd9"></span>`[RD9]` | F. Kugler, D. Schulze, I. Hajnsek, H. Pretzsch and K. P. Papathanassiou, "TanDEM-X Pol-InSAR Performance for Forest Height Estimation," in IEEE Transactions on Geoscience and Remote Sensing, vol. 52, no. 10, pp. 6404-6422, Oct. 2014 |
| <span id="rd10"></span>`[RD10]` | R. Guliaev, V. Cazcarra-Bes, M. Pardini and K. Papathanassiou, "Forest Height Estimation by Means of TanDEM-X InSAR and Waveform Lidar Data," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 3084-3094, 2021 |
| <span id="rd11"></span>`[RD11]` | Y. Cui, Y. Yamaguchi, H. Yamada and S. Park, "PolInSAR Coherence Region Modeling and Inversion: The Best Normal Matrix Approximation Solution," in IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 2, pp. 1048-1060, Feb. 2015 |
| <span id="rd12"></span>`[RD12]` | BIOMASS L2 Implementation Study ARD, I/R 2/7 |
| <span id="rd13"></span>`[RD13]` | BIO-BPS-IODD, BIOMASS BPS IODD, I/R 1/1/3 |
| <span id="rd14"></span>`[RD14]` | Pinheiro, M.; Amao-Oliva, J.; Scheiber, R.; Jaeger, M.; Horn, R.; Keller, M.; Fischer, J.; Reigber, A. Dual-Frequency Airborne SAR for Large Scale Mapping of Tidal Flats. in Remote Sens., vol. 12, pag. 1827, 2020 |
| <span id="rd15"></span>`[RD15]` | `BTK_WEB`, official link: TBD |
