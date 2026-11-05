<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:pfd-agb-3)=
# 3. L2a Products Format Overview

The BIOMASS Above Ground Biomass L2a product is a folder containing a collection of information, grouped together into files referred to as data sets, and data sets are collected and grouped into several sub-folders in order to achieve a physical tree forming a complete product.

These products contain 3 kinds of datasets:

- the Measurement Data Sets (MDS), containing the images derived from ground cancellation processing applied to L1c data;
- the Annotation Data Sets (ADS), containing metadata that describes the properties and characteristics of the measurement data or how the measurement data was generated;
- the Representation Data Sets (RDS), containing information about the format or syntax of the measurement and annotation data sets and can be used to validate and exploit these data.

The declination and customisation of this format structure for BIOMASS Above Ground Biomass L2a products are summarised in the following figure and table. A typical product contains:

- a main product header, an XML file including high-level information about the product origin, structure and main metadata;
- STAC metadata: the SpatioTemporal Asset Catalog specification (https://stacspec.org/en) provides a common structure for describing and cataloging spatiotemporal assets (it is a file that represents information about the earth captured in a certain space and time);
- Ground cancellation image MDS, including the Ground cancellation image binary raster for all polarizations, organised in a single Cloud Optimized GeoTIFF (COG) file with a layer for each polarization;
- main annotation ADS, including the properties and characteristics of the measurement data or how the measurement data was generated, stored in a single XML file;
- Look-Up Table (LUT) ADS, including those metadata being too voluminous to be coded efficiently in an XML file (i.e., Look-Up Tables) and requiring a binary file;
- quick-look ADS, including an averaged and decimated previews of measurement MDS;
- Overlay ADS, for viewing the Quick-look ADS in any application that supports KML format (like, e.g., Google Earth or GIS applications);
- XML schemas RDS, including the schemas needed for all the XML files included in the product.

**No Data Value**

L2a Data Sets pixels will have a no data value of -9999.0 (float32) or 255 (ubyte) depending on data type, where the input L1c pixels are invalid and for each pixel discarded by the algorithm. The value is reported in the MDS COG metadata (`GDAL_NODATA` tag, see {ref}`Tab.2 <tab:pfd-agb-2>`), in the Raster Image DSR (`noDataValue` element, see {ref}`Tab.10 <tab:pfd-agb-10>`) and in LUT ADS (`noDataValue` variable, see 4.3.2.1).

(fig:pfd-agb-1)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

![Above Ground Biomass ground cancellation L2a products structure](../../_static/figures/pfd-agb/figure_1_section_3.png)

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 1</span><span class="caption-text"> — Above Ground Biomass ground cancellation L2a products structure</span></p>

:::

(tab:pfd-agb-1)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 1</span><span class="caption-text"> — L2a products structure (Ground cancellation)</span></p>

| File / Folder | Cardinality | Format | Description / Content |
|---|---|---|---|
| BIOMASS ground cancellation L2a Product |   | Folder |   |
| ∟ main product header | 1 | XML | Main product header |
| ∟ STAC | 1 | JSON | STAC (https://stacspec.org) metadata. |
| ∟ measurement |   | Folder |   |
| ∟ Ground cancelled backscatter | 3 | COG | Ground cancelled backscatter images MDS (HH/VH/VV, three layers in a single COG file) |
| ∟ annotation |   | Folder |   |
| ∟ main | 1 | XML | Main ADS, including: Product DSR Raster image DSR Input information DSR Processing parameters DSR Annotation LUT DSR |
| ∟ lut | 1 | NetCDF | Look-Up Tables ADS, including: FNF, generated outside BPS Local incidence angle |
| ∟ preview |   | Folder |   |
| ∟ quicklook | 1 | PNG | Quick-look ADS, one RGB from measurement |
| ∟ overlay | 1 | KML | Overlay ADS for the quicklook |
| ∟ schema |   | Folder |   |
| ∟ main `annotation.xsd` | 1 | XSD | RDS |
| ∟ L2a, L2b common `annotation.xsd` | 1 | XSD | RDS |
| ∟ common `types.xsd` | 1 | XSD | RDS |
| ∟ `agb-proc-annotation.xsd` | 1 | XSD | RDS |

:::

(sec:pfd-agb-3-1)=
### 3.1 Product format definition guidelines

The BIOMASS Ground cancellation L2a products format here described has been defined following a set of guidelines and principles, being the main ones:

- follow a structure and an organization into datasets (MDS, ADS and RDS) already adopted with success in other SAR mission (Sentinel-1 above all) and already known to the users;
- rely on existing ESA conventions wherever possible (main product header OGC-compliant, …);
- rely on existing and known-to-the-users standards and formats (COG and NetCDF format for binary files, XML format and corresponding XSD schemas for annotation files, PNG for quick-looks, KML for overlays).

Starting from this consolidated “background”, in order to improve the products usability, a LUT has been introduced to reduce annotation file dimensions and easing its access, management and applicability.

In the end, the L2a products format has been defined together with L2b one, following the same guidelines and trying to identify a structure that can be applied to both with the minimum changes, easing even more their access to the users.

(sec:pfd-agb-3-2)=
### 3.2 Naming convention

The BIOMASS L2a product and all data sets files naming are addressed and described in detail in {ad}`2`.

The naming of the products directories is here proposed following the main points in {ad}`2`:

`<MMM>_<TTTTTTTTTT>_<yyyymmddThhmmss>_<YYYYMMDDTHHMMSS>_<P>_G<CC>_M<NN>_C<___>_T<TTT>_F<FFF>_<BB>_<DDDDDD>`

where:

- `<MMM>`: Satellite ID;
- `<TTTTTTTTTT>`: Product Type/File Type, i.e.:
    - `FP_GN__L2A`: Ground cancellation L2a product;
- `<yyyymmddThhmmss>`: UTC start date and time (min time validity of the stack);
- `<YYYYMMDDTHHMMSS>`: UTC stop date and time (max time validity of the stack);
- `<P>`: Mission phase identifier;
- `<CC>`: Global Coverage identifier;
- `<NN>`: Major Cycle identifier;
- `<___> `Repeat Cycle identifier, set to special value (not used in L2 products);
- `<TTT>`: Track number of the product;
- `<FFF>`: Frame/slice number of the product;
- `<BB>:` Baseline Identifier;
- `<DDDDDD>`: Compact creation date.

The internal files in the product inherit the same naming convention, with the following changes:

- capital letters conversion to lowercase;
- “Baseline Identifier” and “Compact Creation Date” naming parts removal, except for Main Product Header file;
- addition of a dataset type depending suffix and extension, i.e.:
    - `*.xml` for main product header;
    - `*_i_gn.tiff` for image MDS;
    - `*_annot.xml` for main ADS;
    - `*_lut.nc` for Look-Up Tables ADS;
    - `*_gn_ql.png` for quick-look ADS;
    - `*_map.kml` for overlay ADS;
    - `*.xsd` for support schemas RDS;
    - `*.json` for STAC.
