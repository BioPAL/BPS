<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:pfd-l1-1)=
# 1. Introduction

(sec:pfd-l1-1-1)=
### 1.1 Context

BIOMASS is ESA’s 7th Earth Explorer mission. The purpose of the BIOMASS is to reduce the uncertainty in the worldwide spatial distribution and dynamics of forest biomass in order to improve current assessments and future projections of the global carbon cycle. This objective will be achieved by the implementation of a P-band SAR mission, providing global maps of forest biomass stocks, forest disturbance and growth.

The BIOMASS mission will also provide 3D views of forests through the “tomographic” phase and will also support additional science needs arising from the opportunity to explore the Earth for the first time with a P-band SAR system from space offering the possibility to “see” below the vegetated areas and in beneath bare soil or icy regions.

To achieve its challenging objectives, the BIOMASS mission requires a specific data processing strategy that combines different technologies at different processing level in order to generate the final Above Ground Biomass global map.

(sec:pfd-l1-1-2)=
### 1.2 Scope of the document

The purpose of this document is to provide the “L1a/b/c Products Format Specification” for the BIOMASS Processing Suite (BPS) project.

This document is organized as follows:

- {sec}`Section|2`: Brief introduction about BPS and L1 products;
- {sec}`Section|3`: Overview of L1a/b products format;
- {sec}`Section|4`: Detailed description of L1a/b products format specifications;
- {sec}`Section|5`: Overview of L1c products format;
- {sec}`Section|6`: Detailed description of L1c products format specifications;
- {sec}`Section|7` (Appendix): Detailed description of common products format specifications.

(sec:pfd-l1-1-3)=
### 1.3 Acronyms

:::{dropdown} Acronyms
| Acronym | Description |
|---|---|
| ADS | Annotation Data Set |
| ARESYS | Advanced Remote Sensing and Systems |
| BPS | BIOMASS Processing Suite |
| COG | Cloud Optimized GeoTIFF |
| DEM | Digital Elevation Model |
| DGM | Detected Ground-range Multi-looked |
| DSR | Data Set Record |
| L1a | Level-1a product |
| L1b | Level-1b product |
| L1c | Level-1c product |
| L1M | Level-1 Monitoring product |
| L1S | Level-1 Standard product |
| LUT | Look-Up Tables |
| MDS | Measurement Data Set |
| MPH | Main Product Header |
| NetCDF | Network Common Data Form |
| OGC | Open Geospatial Consortium |
| PARC | Polarimetric Active Radar Calibrator |
| RDS | Representation Data Set |
| SAR | Synthetic Aperture Radar |
| SCS | Single-look Complex Slant-range |
:::

(sec:pfd-l1-1-4)=
### 1.4 Symbols and Variables

:::{dropdown} Acronyms
| Acronym | Description |
|---|---|
| $\beta^0$ | Beta-nought quantity |
| $\sigma^0$ | Sigma-nought quantity |
| $\gamma^0$ | Gamma-nought quantity |
:::

(sec:pfd-l1-1-5)=
### 1.5 Applicable documents

| Code | Document |
|---|---|
| <span id="ad1"></span>`[AD1]` | [`L1_PFD`] `BIO-ESA-EOPG-EEGS-TN-0044`, BIOMASS L1 Product Format Definition, I/R 2/0 |
| <span id="ad2"></span>`[AD2]` | [`PROD_NAMING`] `BIO-ESA-EOPG-EEGS-TN-0050`, BIOMASS Products Naming Convention, I/R 2/4 |
| <span id="ad3"></span>`[AD3]` | [MPH] `BIO-ESA-EOPG-EEGS-TN-0051`, BIOMASS Products - Main Product Header Definition, I/R 2/3 |
| <span id="ad4"></span>`[AD4]` | [`AUX_FMT`] `BIO-ESA-EOPG-EEGS-TN-0054`, BIOMASS Auxiliary Product Format Definition, I/R 2/1/1 |
| <span id="ad5"></span>`[AD5]` | `PE-ID-ESA-GS-584`, Earth Observation Mission Software File Format Specification, I/R 1/4 |

(sec:pfd-l1-1-6)=
### 1.6 Reference documents

| Code | Document |
|---|---|
| <span id="rd1"></span>`[RD1]` | Cloud Optimized GeoTIFF format, https://www.cogeo.org/ |
| <span id="rd2"></span>`[RD2]` | GDAL library, https://gdal.org/ |
| <span id="rd3"></span>`[RD3]` | GDAL library Wiki - Cloud Optimized GeoTIFF section, https://trac.osgeo.org/gdal/wiki/CloudOptimizedGeoTIFF#HowtoreaditwithGDAL |
| <span id="rd4"></span>`[RD4]` | LibTIFF - TIFF Library and Utilities, https://libtiff.gitlab.io/libtiff/ |
| <span id="rd5"></span>`[RD5]` | TIFF format tags, https://www.loc.gov/preservation/digital/formats/content/tiff\_tags.shtml |
| <span id="rd6"></span>`[RD6]` | Network Common Data Form (NetCDF) format, https://www.unidata.ucar.edu/software/netcdf/ |
| <span id="rd7"></span>`[RD7]` | Well-Known Text representation of coordinate reference systems, https://www.iso.org/standard/63094.html |
