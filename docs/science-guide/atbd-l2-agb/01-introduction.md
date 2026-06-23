<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

```{include} ../_includes/atbd-logo-banner.md
```


# 1. Introduction

### 1.1 Context

BIOMASS is ESA's 7th Earth Explorer mission. The purpose of BIOMASS is to reduce
the uncertainty in the worldwide spatial distribution and dynamics of forest biomass in
order to improve current assessments and future projections of the global carbon cycle.
This objective will be achieved by the implementation of a P-band SAR mission,
providing global maps of forest biomass stocks, forest disturbance and growth.

The BIOMASS mission will also provide 3D views of forests through the tomographic
phase and will support additional science needs arising from the opportunity to explore
the Earth for the first time with a P-band SAR system from space, offering the
possibility to see below vegetated areas and beneath bare soil or icy regions.

To achieve its challenging objectives, the BIOMASS mission requires a specific data
processing strategy that combines different technologies at different processing levels
in order to generate the final Above Ground Biomass global map.

### 1.2 Scope of the document

The purpose of this document is to provide the Above Ground Biomass Algorithms
Theoretical Baseline Document (ATBD) for the BIOMASS Processing Suite (BPS)
project. This document is organised as follows:

| Section | Content |
|---|---|
| 1 | Overview of the document |
| 2 | Brief introduction about BPS and Above Ground Biomass processing |
| 3 | Description of L2a Above Ground Biomass algorithms |
| 4 | Description of L2b/L3 Above Ground Biomass algorithms |

### 1.3 Acronyms
:::{dropdown} 
| Acronym | Meaning |
|---|---|
| AD | Applicable Document |
| ADS | Annotation datasets |
| ACM | Average Covariance Matrix |
| AGB | Above-ground biomass |
| ARESYS | Advanced Remote Sensing and Systems |
| ATBD | Algorithm Theoretical Baseline Document |
| AUX | Auxiliary |
| BIODEMPP | Biomass DEM Prototype Processor |
| BPS | BIOMASS Processing Suite |
| CFI | Customer Furnished Items |
| CFM | Computed Forest Mask |
| CM | Covariance Matrix |
| CoSCS | Coregistered SCS |
| CP | co-polarization |
| DEM | Digital Elevation Model |
| DGG | Discrete Global Grid |
| DTM | Digital Terrain Model |
| ECSS | European Cooperation for Space Standardization |
| EO-CFI | Earth Observation Custom Furnished Items |
| ESA | European Space Agency |
| FD | Forest Disturbance |
| FH | Forest Height |
| FNF | Forest/Non-Forest Mask |
| GN | ground cancellation (Ground Notching) |
| GPP | Ground Processor Prototype |
| GPS | Global Positioning System |
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
| std | Standard deviation |
| TBD | To Be Defined |
| TBC | To Be Confirmed |
| TOM | Tomographic Phase |
| XML | eXtended Markup Language |
| XP | cross-polarization |
:::

### 1.4 Applicable documents

| Code | Title | Issue |
|---|---|---|
| <span id="ad1"></span>`[AD1]` | BIOMASS Production Model, BIO-ESA-EOPG-EEGS-TN-0046 | 3.5 |
| <span id="ad2"></span>`[AD2]` | BIO-BPS-TN-PRODMOD, BIOMASS Production Model Technical Note | I/R 1/2/1 |
| <span id="ad3"></span>`[AD3]` | BIO-BPS-L1-PFD, BIOMASS L1a,b,c products format specification | I/R 1/2/2 |

### 1.5 Reference documents

| Code | Source |
|---|---|
| <span id="rd1"></span>`[RD1]` | BIO-BPS-L1-STACK-ATBD, BIOMASS BPS L1c Stack ATBD, I/R 1/1/2 |
| <span id="rd2"></span>`[RD2]` | BIO-BPS-L1-SAR-ATBD, BIOMASS BPS L1 SAR ATBD, I/R 1/1/2 |
| <span id="rd3"></span>`[RD3]` | BIOMASS DEM Product Processor Prototype, DEM Processor Algorithm Theoretical Baseline Document, BIODEMPP-CoSCS-DEM-01, Issue 4.0, June 07th 2021 |
| <span id="rd4"></span>`[RD4]` | M. Mariotti d'Alessandro, S. Tebaldini, S. Quegan, M. J. Soja, L. M. H. Ulander and K. Scipal, "Interferometric Ground Cancellation for Above Ground Biomass Estimation," in IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 9, pp. 6410-6419, Sept. 2020 |
| <span id="rd5"></span>`[RD5]` | Snowdon, Peter. "A ratio estimator for bias correction in logarithmic regressions." Canadian Journal of Forest Research 21.5 (1991): 720-724 |
| <span id="rd6"></span>`[RD6]` | Díaz-Francés, Eloísa, and Francisco J. Rubio. "On the existence of a normal approximation to the distribution of the ratio of two independent normal random variables." Statistical Papers 54 (2013): 309-323 |
| <span id="rd7"></span>`[RD7]` | BIO-BPS- AUX-FMT, BPS Auxiliary Product Format, I/R 3/2 |
| <span id="rd8"></span>`[RD8]` | BIO-BPS-PPD, BIOMASS BPS Product Performance Description, I/R 3/0 |
| <span id="rd9"></span>`[RD9]` | L2-ATBD-ARE-011567, BIOMASS L2 Algorithm Theoretical Baseline Document, I/R 1/5 |
| <span id="rd10"></span>`[RD10]` | BIO-BPS-IODD, BIOMASS BPS IODD, I/R 3/0/2 |
| <span id="rd11"></span>`[RD11]` | Soja, Maciej J., et al. "Mapping above-ground biomass in tropical forests with ground-cancelled P-band SAR and limited reference data." Remote Sensing of Environment 253 (2021): 112153 |
