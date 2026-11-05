<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:pfd-agb-1)=
# 1. Introduction

(sec:pfd-agb-1-1)=
### 1.1 Context

BIOMASS is ESA’s 7th Earth Explorer mission. The purpose of the BIOMASS is to reduce the uncertainty in the worldwide spatial distribution and dynamics of forest biomass in order to improve current assessments and future projections of the global carbon cycle. This objective will be achieved by the implementation of a P-band SAR mission, providing global maps of forest biomass stocks, forest disturbance and growth.

The BIOMASS mission will also provide 3D views of forests through the “tomographic” phase and will also support additional science needs arising from the opportunity to explore the Earth for the first time with a P-band SAR system from space offering the possibility to “see” below the vegetated areas and in beneath bare soil or icy regions.

To achieve its challenging objectives, the BIOMASS mission requires a specific data processing strategy that combines different technologies at different processing level in order to generate the final Above Ground Biomass global map.

(sec:pfd-agb-1-2)=
### 1.2 Scope of the document

The purpose of this document is to provide the “Above Ground Biomass Products Format Specification” for the BIOMASS Processing Suite (BPS) project.

This document is organized as follows:

- {sec}`Section|2`: Brief introduction about BPS and Above Ground Biomass products;
- {sec}`Section|3`: Overview of L2a products format;
- {sec}`Section|4`: Detailed description of L2a products format specifications;
- {sec}`Section|5`: Overview of L2b products format;
- {sec}`Section|6`: Detailed description of L2b products format specifications.
- {sec}`Section|7`: Appendix containing xml primitive product types description.

(sec:pfd-agb-1-3)=
### 1.3 Acronyms

:::{dropdown} Acronyms
| Acronym | Description |
|---|---|
| ADS | Annotation Data Set |
| AGB | Above Ground Biomass |
| ARESYS | Advanced Remote Sensing and Systems |
| BPS | BIOMASS Processing Suite |
| CFM | Computed Forest Mask (by Forest Disturbance L2 processor) |
| COG | Cloud Optimized GeoTIFF |
| COM | Commissioning Phase |
| DEM | Digital Elevation Model |
| DGG | Discrete Global Grid |
| DSR | Data Set Record |
| FD | Forest Disturbance |
| FNF | Forest/Non-Forest Mask |
| GC | Global Cycle |
| GN | Ground Cancelled |
| INT | Interferometric Phase |
| L1c | Level-1c product |
| L2a | Level-2a product |
| L2b | Level-2b product |
| LCM | Land Cover Map (ESA CCI/C3S) |
| LERC | Limited Error Raster Compression |
| LUT | Look-Up Tables |
| MDS | Measurement Data Set |
| MPH | Main Product Header |
| NetCDF | Network Common Data Form |
| OGC | Open Geospatial Consortium |
| RDS | Representation Data Set |
| SAR | Synthetic Aperture Radar |
| STAC | Spatio Temporal Asset Catalogs |
| TBD | To Be Defined |
| TOM | Tomographic Phase |
:::

(sec:pfd-agb-1-4)=
### 1.4 Applicable documents

| Code | Document |
|---|---|
| <span id="ad1"></span>`[AD1]` | [`L1_PFD`] `BIO-ESA-EOPG-EEGS-TN-0044`, BIOMASS L1 Product Format Definition, I/R 2/0 |
| <span id="ad2"></span>`[AD2]` | [`PROD_NAMING`] `BIO-ESA-EOPG-EEGS-TN-0050`, BIOMASS Products Naming Convention, I/R 2/3 |
| <span id="ad3"></span>`[AD3]` | [MPH] `BIO-ESA-EOPG-EEGS-TN-0051`, BIOMASS Products - Main Product Header Definition, I/R 2/3 |
| <span id="ad4"></span>`[AD4]` | [`AUX_FMT`] `BIO-ESA-EOPG-EEGS-TN-0054`, BIOMASS Auxiliary Product Format Definition, I/R 2/1/1 |
| <span id="ad5"></span>`[AD5]` | `PE-ID-ESA-GS-584`, Earth Observation Mission Software File Format Specification, I/R 1/4 |
| <span id="ad6"></span>`[AD6]` | BIOMASS Production Model, `BIO-ESA-EOPG-EEGS-TN-0046`, issue 3.5 |
| <span id="ad7"></span>`[AD7]` | BIO-BPS-TN-PRODMOD, BIOMASS Production Model Technical Note, I/R 1/2/2 |
| <span id="ad8"></span>`[AD8]` | BIOMASS L2 FD ATBD, I/R, 2/1/2 |
| <span id="ad9"></span>`[AD9]` | BIOMASS L2 FH ATBD, I/R, 2/1/2 |
| <span id="ad10"></span>`[AD10]` | BIOMASS L2 AGB ATBD, I/R, 3/1 |

(sec:pfd-agb-1-5)=
### 1.5 Reference documents

| Code | Document |
|---|---|
| <span id="rd1"></span>`[RD1]` | Cloud Optimized GeoTIFF format, https://www.cogeo.org/ |
| <span id="rd2"></span>`[RD2]` | GDAL library, https://gdal.org/ |
| <span id="rd3"></span>`[RD3]` | GDAL library Wiki - Cloud Optimized GeoTIFF section, https://trac.osgeo.org/gdal/wiki/CloudOptimizedGeoTIFF#HowtoreaditwithGDAL |
| <span id="rd4"></span>`[RD4]` | LibTIFF - TIFF Library and Utilities, https://libtiff.gitlab.io/libtiff/ |
| <span id="rd5"></span>`[RD5]` | TIFF format tags, https://www.loc.gov/preservation/digital/formats/content/tiff\_tags.shtml |
| <span id="rd6"></span>`[RD6]` | Network Common Data Form (NetCDF) format, https://www.unidata.ucar.edu/software/netcdf/ |
| <span id="rd7"></span>`[RD7]` | Network Common Data Form (NetCDF), Python |

https://unidata.github.io/netcdf4-python/

| Code | Document |
|---|---|
| <span id="rd8"></span>`[RD8]` | Well-Known Text representation of coordinate reference systems, https://www.iso.org/standard/63094.html |
| <span id="rd9"></span>`[RD9]` | [PDD] `BIO-BPS-PPD-ARE-010083`, BIOMASS Product Performance Description, I/R 3/0 |
| <span id="rd10"></span>`[RD10]` | [L1PDF] `BIO-BPS-L1PFD-ARE-010076`, BIOMASS L1 Product Format Specification, I/R 1/2/6 |
| <span id="rd11"></span>`[RD11]` | https://docs.ogc.org/is/19-008r4/19-008r4.html#\_geotiff\_configuration\_geokeys |
