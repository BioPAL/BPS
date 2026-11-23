<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

:::{atbd-logo-banner}
:::


(sec:pfd-fh-4)=
# 4. L2a Products Format Specification

(sec:pfd-fh-4-1)=
### 4.1 Main Product Header

The Main Product Header (MPH) is an XML file present in all the BIOMASS L2 products.

BIOMASS L2 products are folder based and the MPH is top level leaf of the product tree.

The MPH main scope is to provide all the information related to the:

- Genesis of the product;
- Identification of the product for example for cataloguing purposes;
- Product consistency (filesize).

It must be noted that, while the footprint is inherited from lower level products (thus vertices represent the extent of the acquired data), DGG tiles (`tileId` list) are related to the full extent of L2a product (that is geocoded).

For the detailed description of the MPH content and format refer to {ad}`3`. Here just a

schematic representation is reported, , where this notation is applicable:

<span style="color:#0070C0">This is a container of other elements</span>

This element or block of elements can occur more than one time (i.e. it is a list)

<span style="color:#C00000"><b>This is a container of actual metadata</b></span>

<span style="color:#C00000">This is a mandatory element for the standard which is left empty by this specification</span>

This element or block of elements is optional

(fig:pfd-fh-2)=
:::{div} atbd-mermaid-figure atbd-text-figure

<div class="atbd-figure-content">

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree"><span style="color:#0070C0">bio:EarthObservation </span><i>Root element of metadata representation tailored for BIOMASS SAR products</i>
    <span style="color:#0070C0">om:phenomenonTime</span> <i>Time coverage of the product</i>
        <span style="color:#0070C0">gml:TimePeriod</span> <i>Time coverage of the product as a period</i>
            <span style="color:#C00000"><b>gml:beginPosition</b></span><span style="color:#C00000"> </span><b>Start date and time of the product</b>
            <span style="color:#C00000"><b>gml:endPosition</b></span> <b>Stop date and time of the product</b>
    <span style="color:#0070C0">om:resultTime</span> <i>Time when the sensing data has become available</i>
        <span style="color:#0070C0">gml:TimeInstant</span> <i>Identifiable position in time</i>
            <span style="color:#C00000"><b>gml:timePosition</b></span> <b>Date and time of data availability</b>
    <span style="color:#0070C0">om:validTime</span> <i>Time validity of the product</i>
        <span style="color:#0070C0">gml:TimePeriod</span> <i>Time validity of the product as a period</i>
            <span style="color:#C00000"><b>gml:beginPosition </b></span><b>Validity start date and time of the product</b>
            <span style="color:#C00000"><b>gml:endPosition</b></span> <b>Validity stop date and time of the product</b>
    <span style="color:#0070C0">om:procedure</span> <i>Procedure used to sense the data</i>
        <span style="color:#0070C0">eop:EarthObservationEquipment</span> <i>Equipment used to sense the data</i>
            <span style="color:#0070C0">eop:platform</span> <i>Platform description</i>
                <span style="color:#0070C0">eop:Platform</span> <i>Nested element for platform description</i>
                    <span style="color:#C00000"><b>eop:shortName</b></span> <b>Satellite name</b>
            <span style="color:#0070C0">eop:instrument</span> <i>Instrument description</i>
                <span style="color:#0070C0">eop:Instrument</span> <i>Nested element for instrument description</i>
                    <span style="color:#C00000"><b>eop:shortName</b></span> <b>Sensor name</b>
            <span style="color:#0070C0">eop:sensor</span> <i>Sensor description</i>
                <span style="color:#0070C0">eop:Sensor</span> <i>Nested element for sensor description</i>
                    <span style="color:#C00000"><b>eop:sensorType</b></span> <b>Sensor type</b>
                    <span style="color:#C00000"><b>eop:operationalMode</b></span> <b>Sensor operational mode</b>
                    <span style="color:#C00000"><b>eop:swathIdentifier</b></span> <b>Swath identifier</b>
            <span style="color:#0070C0">eop:acquisitionParameters</span> <i>Acquisition parameters</i>
                <span style="color:#0070C0">bio:Acquisition</span> <i>Acquisition description</i>
                    <span style="color:#C00000"><b>eop:orbitDirection</b></span> <b>Orbit direction</b>
                    <span style="color:#C00000"><b>eop:wrsLongitudeGrid</b></span> <b>Track number</b>
                    <span style="color:#C00000"><b>eop:wrsLatitudeGrid</b></span> <b>Frame number</b>
                    <span style="color:#C00000"><b>sar:antennaLookDirection</b></span> <b>SAR antenna look direction</b>
                    <span style="color:#C00000"><b>bio:missionPhase</b></span> <b>BIOMASS mission phase</b>
                    <span style="color:#C00000"><b>bio:globalCoverageID</b></span> <b>BIOMASS global coverage identifier</b> 
                    <span style="color:#C00000"><b>bio:majorCycleID</b></span> <b>BIOMASS major cycle identifier</b>
                    <span style="color:#C00000"><b>bio:</b></span><span style="color:#C00000"><b>t</b></span><span style="color:#C00000"><b>ile</b></span><span style="color:#C00000"><b>ID</b></span> <b>List of DGG geographic cell (tile) identifiers</b>
                    <span style="color:#C00000"><b>bio:basinID</b></span> <b>Retrieval basin identifier</b>
    <span style="color:#C00000">om:observedProperty </span><i>Observed property (empty)</i>
    <span style="color:#0070C0">om:featureOfInterest</span> <i>Observed area</i>
        <span style="color:#0070C0">eop:Footprint</span> <i>Footprint description</i>
            <span style="color:#0070C0">eop:multiExtentOf </span><i>Footprint representation structure, coordinates in posList</i>
                <span style="color:#0070C0">gml:MultiSurface</span>
                    <span style="color:#0070C0">gml:surfaceMember</span>
                        <span style="color:#0070C0">gml:Polygon</span>
                            <span style="color:#0070C0">gml:exterior</span>
                                <span style="color:#0070C0">gml:LinearRing</span>
                                    <span style="color:#C00000"><b>gml:posList</b></span><span style="color:#0070C0"> </span><b>Footprint points in NE,SE,SW,NW</b><b>,</b><b>NE</b><b> order</b>
            <span style="color:#0070C0">eop:centerOf </span><i>Acquisition centre representation structure</i>
                <span style="color:#0070C0">gml:Point</span>
                    <span style="color:#C00000"><b>gml:pos</b></span> <b>Coordinates of the centre of the acquisition</b>
    <span style="color:#0070C0">om:result </span><i>Observation result</i>
        <span style="color:#0070C0">eop:EarthObservationResult</span>
            <span style="color:#0070C0">eop:product</span>
                <span style="color:#0070C0">bio:ProductInformation</span>
                    <span style="color:#0070C0">eop:fileName</span>
                        <span style="color:#C00000"><b>ows:ServiceReference</b></span> <b>Product filename</b>
                            <span style="color:#C00000">ows:RequestMessage</span> <b>download request (empty)</b>
            <span style="color:#C00000"><b>eop:version</b></span> <b>Product baseline</b>
            <span style="color:#0070C0">eop:product </span>(list)
                <span style="color:#0070C0">bio:ProductInformation</span>
                    <span style="color:#0070C0">eop:fileName</span>
                        <span style="color:#C00000"><b>ows:ServiceReference</b></span> <b>Product filename (internal files)</b>
                            <span style="color:#C00000">ows:RequestMessage</span> <b>download request (empty)</b>
                    <span style="color:#C00000"><b>eop:size</b></span> <b>Product size</b>
                    <span style="color:#C00000"><b>bio</b></span><span style="color:#C00000"><b>:rds</b></span><span style="color:#C00000"> </span><b>product schema</b>
    <span style="color:#0070C0">eop:metaDataProperty </span><i>Observation metadata</i>
        <span style="color:#0070C0">bio:EarthObservationMetaData</span>
            <span style="color:#C00000"><b>eop:identifier</b></span> <b>Product identifier</b>
            <span style="color:#C00000"><b>eop:doi</b></span> <b>Digital Object Identifier</b>
            <span style="color:#C00000"><b>eop:acquisitionType</b></span> <b>Type of acquisition</b>
            <span style="color:#C00000"><b>eop:productType</b></span> <b>Product type</b>
            <span style="color:#C00000"><b>eop:status</b></span> <b>Status of the product</b>
                <span style="color:#0070C0">eop:processing </span><i>Data processing information</i>
                    <span style="color:#0070C0">bio:ProcessingInformation</span>
                        <span style="color:#C00000"><b>eop:processingCenter</b></span> <b>Processing centre code</b>
                        <span style="color:#C00000"><b>eop:processingDate</b></span> <b>Processing date and time</b>
                        <span style="color:#C00000"><b>eop:processorName</b></span> <b>Processor name</b>
                        <span style="color:#C00000"><b>eop:processorVersion</b></span> <b>Processor version</b>
                        <span style="color:#C00000"><b>eop:processingLevel</b></span> <b>Processing level</b>
                        <span style="color:#C00000"><b>eop:auxiliaryDataSetFileName</b></span> <b>Auxiliary DS filename</b>
                        <span style="color:#C00000"><b>eop:processingMode</b></span> <b>File Class</b>
                        <span style="color:#C00000"><b>bio:sourceProduct</b></span> <b>BIOMASS source product identifier</b>
            <span style="color:#C00000"><b>bio:refDoc</b></span> <b>Reference document</b></pre></div></div>
```

::::{only} latex

```text
bio:EarthObservation Root element of metadata representation tailored for BIOMASS SAR products
    om:phenomenonTime Time coverage of the product
        gml:TimePeriod Time coverage of the product as a period
            gml:beginPosition Start date and time of the product
            gml:endPosition Stop date and time of the product
    om:resultTime Time when the sensing data has become available
        gml:TimeInstant Identifiable position in time
            gml:timePosition Date and time of data availability
    om:validTime Time validity of the product
        gml:TimePeriod Time validity of the product as a period
            gml:beginPosition Validity start date and time of the product
            gml:endPosition Validity stop date and time of the product
    om:procedure Procedure used to sense the data
        eop:EarthObservationEquipment Equipment used to sense the data
            eop:platform Platform description
                eop:Platform Nested element for platform description
                    eop:shortName Satellite name
            eop:instrument Instrument description
                eop:Instrument Nested element for instrument description
                    eop:shortName Sensor name
            eop:sensor Sensor description
                eop:Sensor Nested element for sensor description
                    eop:sensorType Sensor type
                    eop:operationalMode Sensor operational mode
                    eop:swathIdentifier Swath identifier
            eop:acquisitionParameters Acquisition parameters
                bio:Acquisition Acquisition description
                    eop:orbitDirection Orbit direction
                    eop:wrsLongitudeGrid Track number
                    eop:wrsLatitudeGrid Frame number
                    sar:antennaLookDirection SAR antenna look direction
                    bio:missionPhase BIOMASS mission phase
                    bio:globalCoverageID BIOMASS global coverage identifier
                    bio:majorCycleID BIOMASS major cycle identifier
                    bio:tileID List of DGG geographic cell (tile) identifiers
                    bio:basinID Retrieval basin identifier
    om:observedProperty Observed property (empty)
    om:featureOfInterest Observed area
        eop:Footprint Footprint description
            eop:multiExtentOf Footprint representation structure, coordinates in posList
                gml:MultiSurface
                    gml:surfaceMember
                        gml:Polygon
                            gml:exterior
                                gml:LinearRing
                                    gml:posList Footprint points in NE,SE,SW,NW,NE order
            eop:centerOf Acquisition centre representation structure
                gml:Point
                    gml:pos Coordinates of the centre of the acquisition
    om:result Observation result
        eop:EarthObservationResult
            eop:product
                bio:ProductInformation
                    eop:fileName
                        ows:ServiceReference Product filename
                            ows:RequestMessage download request (empty)
            eop:version Product baseline
            eop:product (list)
                bio:ProductInformation
                    eop:fileName
                        ows:ServiceReference Product filename (internal files)
                            ows:RequestMessage download request (empty)
                    eop:size Product size
                    bio:rds product schema
    eop:metaDataProperty Observation metadata
        bio:EarthObservationMetaData
            eop:identifier Product identifier
            eop:doi Digital Object Identifier
            eop:acquisitionType Type of acquisition
            eop:productType Product type
            eop:status Status of the product
                eop:processing Data processing information
                    bio:ProcessingInformation
                        eop:processingCenter Processing centre code
                        eop:processingDate Processing date and time
                        eop:processorName Processor name
                        eop:processorVersion Processor version
                        eop:processingLevel Processing level
                        eop:auxiliaryDataSetFileName Auxiliary DS filename
                        eop:processingMode File Class
                        bio:sourceProduct BIOMASS source product identifier
            bio:refDoc Reference document
```

::::

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 2</span><span class="caption-text"> — Main product header L2a (metadata schematic representation)</span></p>

:::

(sec:pfd-fh-4-2)=
### 4.2 Measurement Data Sets (MDS)

The MDS of the L2a product is constituted by two images:

- the biophysical measurement consisting of the Forest Height binary raster, representing Top Canopy Height (TCH);
- a layer of quality, consisting of a percentage bias value for each of the Forest Height image pixel, indicating the inversion performance.

The two quantities are organised in separated Cloud Optimized GeoTIFF (COG) files {rd}`1` and coded using a 4-byte float per pixel (float32).

A Cloud Optimized GeoTIFF (COG) is a regular GeoTIFF file, aimed at being hosted on a

HTTP file server, with an internal organization that enables more efficient workflows on the

cloud. It does this by leveraging the ability of clients issuing HTTP GET range requests to ask for just the parts of a file they need.

The COG format is natively supported by GDAL library {rd}`2` {rd}`3` and by libraries and tools based on GDAL library (Dask and Rasterio packages for Python, …).

The MDS images ground coverage is corresponding to the geocoded spatial stack coverage of inputs L1c used to compute L2a product: the L2a MDS images are geocoded to the minimum required latitude-longitude coverage to accommodate for the product on ground, following the same sampling requirements specified in {rd}`9` ensuring a ground sampling always better than the resolution.

**Compression strategy**

The MDS images are composed by real values: it is possible to introduce with benefit a strategy for data compression, making use of native LibTIFF libraries capabilities {rd}`4` adopting a GeoTIFF LERC+ZSTD compression, where:

- LERC compression algorithm can be configured to decide how lossy the compression will be, through the “`MAX_Z_ERROR`” option (loss-less only if set to zero);
- ZSTD is a loss-less compression algorithm with a configurable compression level from 0 to 9, where zero means no compression.

Both set by configuration. More details in {sec}`4.3.1.4`.

**Metadata**

The TIFF and GeoTIFF tags used to annotate the MDS files are described in the following table. For the complete description of each tag refer to official TIFF format documentation {rd}`5` and official GeoTIFF documentation {rd}`11`.

(tab:pfd-fh-2)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 2</span><span class="caption-text"> — MDS – TIFF and GeoTIFF tags</span></p>

| **Tag Name** | **Code** | **Value** | **Description** |
|---|---|---|---|
| ImageWidth | 256 | Variable | Number of columns in the image, i.e. the number of pixels per row. |
| ImageLength | 257 | Variable | Number of rows of pixels in the image. |
| BitsPerSample | 258 | 32 | Number of bits per component. |
| Compression | 259 | 34887, <br> 34926 | Compression scheme used on the image data (LERC and ZSTD) |
| PhotometricInterpretation | 262 | 1 | Colour space of the image. <br> Always set to 1 (minimum value is black). |
| ImageDescription | 270 | Variable | A string that describes the image. Valid Strings are composed by prefix: <br> “BIOMASS L2a FP\_FH\_\_L2A: ” or “BIOMASS L2b FP\_FH\_\_L2B: ” and: <br> “Forest Height [m]” <br> “Forest height quality” <br> “Forest Mask” (L2b only) <br> “Heat Map” (L2b only) <br> "acquisition id image" (L2b only) |
| SamplesPerPixel | 277 | 1 | The number of components per pixel. <br> Always set to 1, since each product is stored in a different file. |
| PlanarConfiguration | 284 | 1 | How the components of each pixel are stored. <br> Always set to 1 (contiguous). |
| Software | 305 | Variable | Software name and version that created the image. |
| DateTime | 306 | Variable | Date and time of image creation in the format: <br> YYYY:MM:DD HH:MM:SS |
| TileWidth | 322 | Variable | The tile width in pixels. <br> This is the number of columns in each tile. |
| TileLength | 323 | Variable | The tile length (height) in pixels. <br> This is the number of rows in each tile. |
| TileOffsets | 324 | Variable | For each tile, the byte offset of that tile, as compressed and stored on disk. |
| TileByteCounts | 325 | Variable | For each tile, the number of (compressed) bytes in that tile. |
| SampleFormat | 339 | 3 or 1 | Specifies how to interpret each data sample in a pixel. <br> Set to 3 (IEEE floating point data) |
| ModelPixelScaleTag | 33550 | Variable | List of doubles, (ScaleX, ScaleY, ScaleZ) where ScaleX and ScaleY give the horizontal and vertical spacing of raster pixels. The ScaleZ is zero. |
| ModelTiePointTag | 33922 | N \* (pixel, line, 0, lat, long, 0) | Maps image pixels to geographic coordinates, where N is the number of tie points in the scene. |
| GeoKeyDirectoryTag | 34735 | Variable | Contains the value of all GeoTIFF keys of type SHORT and the location of the values for GeoTIFF keys of other types. |
| GeoDoubleParamsTag | 34736 | Variable | Contains the value of all GeoTIFF keys of type DOUBLE. |
| GeoAsciiParamsTag | 34737 | Variable | Contains the value of all GeoTIFF keys of type ASCII. |
| `GDAL_METADATA` | 42112 | Variable | Contains GDAL metadata (see table below) |
| `GDAL_NODATA` | 42113 | “-9999” or “255” | Pixel value used in case of invalid data. ASCII string encoded value. (See 3) |
| LercParameters | 50674 | Variable | LERC compression parameters: version, additional compression. <br> e.g. 4,2 for LERC v4 with ZSTD |

:::

The `GDAL_METADATA` tag content is listed in the following table

(tab:pfd-fh-3)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 3</span><span class="caption-text"> — MDS – GDAL_METADATA tiff tag content</span></p>

| **Tag Name** | **Value** | **Description** |
|---|---|---|
| basinID | list of strings | List of DGG basin identifiers (for querying purposes). More than one ID in case of tile covering more than one geographic location. Private Tag code. |
| `MAX_Z_ERROR` | float | Absolute maximum error present in the image data, due to LERC compression. Private Tag code. |
| Swath | string | Image swath. Private Tag code (L2A only) |
| tileID | list of strings | List of DGG geographic cell (tile) identifiers (for querying purposes). Private Tag code. |

:::

**Overviews**

The exploitation of GDAL library allows to easily add to COG file one or more overviews of the L2 image.

Overviews are sub-sampled versions of an L2 image. Adding one or more overviews at different resolution levels allows to use pyramidal viewers directly on L2 image file, without the need of transcoding operations or of further processing.

The following (minimum) number of overviews is foreseen:

- multi-look factors (latitude x longitude) of 2x2.
- multi-look factors (latitude x longitude) of 4x4.

(sec:pfd-fh-4-3)=
### 4.3 Annotation data sets (ADS)

(sec:pfd-fh-4-3-1)=
#### 4.3.1 Main ADS

This section describes in details the content of main ADS for L2a products.

An overview is provided by figure and table below.

(fig:pfd-fh-3)=
::::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<style>.xsd-diagram pre.mermaid > svg { width: auto; }</style>
<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 440px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class mainAnnotationType {
    <<complexType>>
    @schemaVersion : token [1]
    product : productL2aType [1] ⊞
    rasterImage : rasterImageType [1] ⊞
    inputInformation : InputInformationL2aType [1] ⊞
    processingParameters : processingParametersL2aType [1] ⊞
    annotationLUT : layerListType [1] ⊞
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2a-fh-main-annotation.xsd (<code>mainAnnotationType</code>).</small></p>

:::{div} xsd-diagram-legend
Diagram notation: `@name` attribute; `[min..max]` occurrences; `⊞` named type not expanded here; `«inline»` type declared in place, drawn as an «anonymous» class; `choice n ·` alternative n of a choice; `all ·` elements in any order.
:::

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 3</span><span class="caption-text"> — Main L2a ADS</span></p>

::::

(tab:pfd-fh-4)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 4</span><span class="caption-text"> — Main L2a ADS</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *`schemaVersion`* | *PFD version* | *`xsd:token`* | *required* |
| product | Product L2a DSR. This DSR contains the L2a product information. | <u>`productL2aType`</u> | 1 |
| `rasterImage` | Raster image DSR. This DSR contains all the necessary information to exploit the L2a product raster images. | <u>`rasterImageType`</u> | 1 |
| InputInformation | Input Information DSR. This DSR contains the necessary information to identify the input data set to the L2a processing. | <u>InputInformationL2aType</u> | 1 |
| `processingParameters` | Processing parameters DSR. This DSR contains the description of L2a processing parameters. | <u>`processingParametersL2aType`</u> | 1 |
| annotationLUT | Annotation LUT DSR. This DSR contains the list of Look-Up Tables (LUTs) complementing product main annotations. | <u>`layerListType`</u> | 1 |

:::

The content of each DSR is described making use of tables as the one here above. All the tables share the same structure and are filled following the same conventions, i.e.:

- Name: defines the name of the element or attribute;
- Description: describes the purpose of the element or attribute including the units (if applicable). The absence of a unit designation for string data types implies a string literal and for numerical data types (integers, floats, etc.…) the absence of a unit designation implies an absolute number;
- Data Type: defines the data type of the element or attribute;
- Cardinality: for elements, this column defines the number of occurrences of the element in the form `minOccurs..maxOccurs`; for attributes, “optional” means the attribute may or may not be present and “required” means the presence of the attribute is mandatory.

In general, elements are displayed in normal font, attributes are displayed in *italics*.

Data types requiring a dedicated definition table are <u>underlined</u>, for all the others the common reference is {sec}`Sec. 7.1.1 <7.1.1>`.

(sec:pfd-fh-4-3-1-1)=
##### 4.3.1.1 Product L2a DSR

This DSR contains the parameters to describe the L2a products.

An overview is provided by figure and table below.

(fig:pfd-fh-4)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 549px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class productL2aType {
    <<complexType>>
    mission : missionType [1]
    tileID : stringListType [1] ⊞
    basinID : stringListType [1] ⊞
    productType : productType [1]
    startTime : timeType [1]
    stopTime : timeType [1]
    radarCarrierFrequency : doubleWithUnit [1] @units
    missionPhaseID : missionPhaseIDType [1]
    sensorMode : sensorModeType [1]
    globalCoverageID : unsignedShort [1]
    swath : swathType [1]
    majorCycleID : unsignedShort [1]
    absoluteOrbitNumber : integerListType [1] ⊞
    relativeOrbitNumber : unsignedShort [1]
    orbitPass : orbitPassType [1]
    dataTakeID : integerListType [1] ⊞
    frame : unsignedShort [1]
    platformHeading : doubleWithUnit [1] @units
    forestCoveragePercentage : double [1]
    selectedReferenceImage : selectedReferenceImageType [0..1] @acquisitionID
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-common-annotations.xsd (<code>productL2aType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 4</span><span class="caption-text"> — Product L2a DSR (productL2aType)</span></p>

:::

(tab:pfd-fh-5)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 5</span><span class="caption-text"> — Product L2a DSR (productL2aType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| mission | Mission (BIOMASS). | `missionType` | 1 |
| tileID | List of DGG geographic cell (tile) identifiers. | <u>`stringListType`</u> | 1 |
| basinID | List of DGG Basin identifiers. | <u>`stringListType`</u> | 1 |
| `productType` | Product type (“`FH_L2A`”). | `productType` | 1 |
| `startTime` | Start time of the product image [UTC]. | `timeType` | 1 |
| `stopTime` | Stop time of the product image [UTC]. | `timeType` | 1 |
| `radarCarrierFrequency` | Radar carrier frequency [Hz]. | `doubleWithUnit` | 1 |
| `missionPhaseID` | Mission phase identifier. | `missionPhaseIDType` | 1 |
| `sensorMode` | Sensor mode (always Measurement). | `sensorModeType` | 1 |
| `globalCoverageID` | Global coverage identifier. | `xsd:unsignedShort` | 1 |
| swath | Swath (S1, S2, S3). | `swathType` | 1 |
| `majorCycleID` | Major cycle identifier. | `xsd:unsignedShort` | 1 |
| `absoluteOrbitNumber` | List of absolute orbit numbers at start time (one for each input L1c). | <u>`integerListType`</u> | 1 |
| `relativeOrbitNumber` | Relative orbit number (track) at start time | `xsd:unsignedShort` | 1 |
| `orbitPass` | Orbit pass (Ascending, Descending). | `orbitPassType` | 1 |
| `dataTakeID` | List of data take identifiers (one for each input L1c). | <u>`integerListType`</u> | 1 |
| frame | Frame identifier. | `xsd:unsignedShort` | 1 |
| `platformHeading` | Platform heading relative to North [deg]. | `doubleWithUnit` | 1 |
| `forestCoveragePercentage` | Forest coverage of the input STA products, compute as a percentage of all the pixels in the STA footprint. | double | 1 |

:::

(tab:pfd-fh-6)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 6</span><span class="caption-text"> — stringListType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| ID | Tile identifier string | string | 1 .. unbounded |

:::

(tab:pfd-fh-7)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 7</span><span class="caption-text"> — integerListType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| val | Integer value | uint32 | 1 .. unbounded |

:::

(sec:pfd-fh-4-3-1-2)=
##### 4.3.1.2 Raster Image DSR

This DSR contains all the necessary information to exploit the raster images of the product. An overview is provided by figure and table below.

(fig:pfd-fh-5)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 1070px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class rasterImageType {
    <<complexType>>
    footprint : floatArrayWithUnits [1] @count @units
    firstLatitudeValue : floatWithUnit [1] @units
    firstLongitudeValue : floatWithUnit [1] @units
    latitudeSpacing : floatWithUnit [1] @units
    longitudeSpacing : floatWithUnit [1] @units
    numberOfSamples : unsignedInt [1]
    numberOfLines : unsignedInt [1]
    projection : projectionType [1]
    datum : datumType [1]
    pixelRepresentation : pixelRepresentationChoiceType [1]
    pixelType : pixelTypeChoiceType [1]
    noDataValue : noDataValueChoiceType [1]
  }
  class datumType {
    coordinateReferenceSystem : coordinateReferenceSystemType [1]
    geodeticReferenceFrame : geodeticReferenceFrameType [1]
  }
  class pixelRepresentationChoiceType {
    choice 1 · FD : pixelRepresentationType [1]
    choice 1 · CFM : pixelRepresentationType [1]
    choice 1 · probabilityOfChange : pixelRepresentationType [1]
    choice 1 · FD_heatMap : pixelRepresentationType [0..1]
    choice 2 · FH : pixelRepresentationType [1]
    choice 2 · Quality : pixelRepresentationType [1]
    choice 2 · FH_heatMap : pixelRepresentationType [0..1]
    choice 3 · GN : pixelRepresentationType [1]
    choice 4 · AGB : pixelRepresentationType [1]
    choice 4 · AGB_STANDARD_DEVIATION : pixelRepresentationType [1]
    choice 4 · AGB_heatMap : pixelRepresentationType [1]
    BPS_FNF : pixelRepresentationType [0..1]
    Acquisition_id_image : pixelRepresentationType [0..1]
  }
  class pixelTypeChoiceType {
    floatPixelType : pixelTypeType [0..1]
    intPixelType : pixelTypeType [0..1]
  }
  class noDataValueChoiceType {
    floatNoDataValue : float [0..1]
    intNoDataValue : unsignedByte [0..1]
  }
  rasterImageType --> datumType : datum
  rasterImageType --> pixelRepresentationChoiceType : pixelRepresentation
  rasterImageType --> pixelTypeChoiceType : pixelType
  rasterImageType --> noDataValueChoiceType : noDataValue
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-common-annotations.xsd (<code>rasterImageType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 5</span><span class="caption-text"> — Raster image DSR (rasterImageType)</span></p>

:::

(tab:pfd-fh-8)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 8</span><span class="caption-text"> — Raster image DSR (rasterImageType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| footprint | Image footprint, expressed as a list of latitude, longitude tuples in NE, SE, SW, NW order [deg]. | `floatArrayWithUnits` | 1 |
| `firstLatitudeValue` | First latitude value of the image [deg]. | `floatWithUnit` | 1 |
| `firstLongitudeValue` | First longitude value of the image [deg]. | `floatWithUnit` | 1 |
| `latitudeSpacing` | Latitude spacing between samples [deg]. | `floatWithUnit` | 1 |
| `longitudeSpacing` | Longitude spacing between samples [deg]. | `floatWithUnit` | 1 |
| `numberOfSamples` | Total number of samples in the image. | `xsd:unsignedInt` | 1 |
| `numberOfLines` | Total number of lines in the image. | `xsd:unsignedInt` | 1 |
| projection | Projection of the image: “Latitude longitude based on DGG”. | `projectionType` | 1 |
| datum | Datum used during processing. | `datumType` | 1 |
| `pixelRepresentation` | Representation of the image pixels within the image MDS. | <u>`pixelRepresentationChoiceType`</u> | 1 |
| `pixelType` | Data type of output pixels within the image MDS. | <u>`pixelTypeChoiceType`</u> | 1 |
| `noDataValue` | Pixel value in case of invalid data. The value is always set to -9999.0 float32 or 255 ubyte, depending on data type, see {sec}`section 3 <3>`, 5. | <u>`noDataValueChoiceType`</u> | 1 |

:::

(tab:pfd-fh-9)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 9</span><span class="caption-text"> — pixelRepresentationChoiceType(L2a/L2b FH described here)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| FH | Representation of the image pixels within the image MDS: Forest Height [m] | `pixelRepresentationType` | 1 |
| Quality | Representation of the image pixels within the image MDS: Forest height quality | `pixelRepresentationType` | 1 |
| `FH_heatMap` | Representation of the image pixels within the image MDS: Heat Map (Only L2b) | `pixelRepresentationType` | 0,1 |
| `BPS_FNF` | Representation of the image pixels within the image MDS: Forest Mask (Only L2b) | `pixelRepresentationType` | 0,1 |
| Acquisition\_Id\_Image | Representation of the image pixels within the image MDS: acquisition\_id\_image (Only L2b) | `pixelRepresentationType` | 0,1 |

:::

(tab:pfd-fh-10)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 10</span><span class="caption-text"> — pixelTypeChoiceType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `floatPixelType` | Data type of output pixels within the image MDS: 32 bit Float | `pixelTypeType` | 1 |
| `intPixelType` | Integer data type of output pixels within the image MDS: 8 bit Unsigned Integer (L2b only) | `pixelTypeType` | 0,1 |

:::

(tab:pfd-fh-11)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 11</span><span class="caption-text"> — noDataValueChoiceType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `floatNoDataValue` | Pixel value in case of invalid data. The value is always set to <br> -9999.0 | `xsd:float` | 1 |
| `intNoDataValue` | Pixel value in case of invalid data. The value is always set to +255 (L2b only) | `xsd:unsignedByte` | 0,1 |

:::

(sec:pfd-fh-4-3-1-3)=
##### 4.3.1.3 Input Information DSR

This DSR contains information about the L1c acquisitions in input to the L2a processor.

An overview is provided by figures and tables below.

(fig:pfd-fh-6)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 395px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class InputInformationL2aType {
    <<complexType>>
    productType : productType [1]
    overallProductsQualityIndex : unsignedByte [1]
    nominalStack : bool [1]
    polarisationList : polarisationListType [1] ⊞
    projection : projectionType [1]
    footprint : floatArrayWithUnits [1] @count @units
    verticalWavenumbers : minMaxTypeWithUnit [1] ⊞
    heightOfAmbiguity : minMaxTypeWithUnit [1] ⊞
    acquisitionList : acquisitionListType [1] ⊞
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-common-annotations.xsd (<code>InputInformationL2aType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 6</span><span class="caption-text"> — Input Information DSR (InputInformationL2aType)</span></p>

:::

(fig:pfd-fh-7)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 325px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class acquisitionListType {
    <<complexType>>
    @count : unsignedInt [1]
    acquisition : acquisitionType [2..8] ⊞
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-common-annotations.xsd (<code>acquisitionListType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 7</span><span class="caption-text"> — Input Information DSR, acquisition list detail (acquisitionListType)</span></p>

:::

(fig:pfd-fh-8)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 293px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class acquisitionType {
    <<complexType>>
    @referenceImage : bool [1]
    @averageWavenumber : float [0..1]
    FolderName : string [1]
    staQuality : staQualityType [1] ⊞
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-common-annotations.xsd (<code>acquisitionType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 8</span><span class="caption-text"> — Input Information DSR, acquisition detail (acquisitionType)</span></p>

:::

(fig:pfd-fh-9)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 472px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class staQualityType {
    <<complexType>>
    overallProductQualityIndex : unsignedInt [1]
    staQualityParametersList : staQualityParametersListType [1] ⊞
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-common-annotations.xsd (<code>staQualityType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 9</span><span class="caption-text"> — Input information DSR, staQualityType</span></p>

:::

(fig:pfd-fh-10)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 440px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class staQualityParametersListType {
    <<complexType>>
    @count : unsignedInt [1]
    staQualityParameters : staQualityParametersType [1..4] ⊞
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-common-annotations.xsd (<code>staQualityParametersListType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 10</span><span class="caption-text"> — Input information DSR, staQualityParametersListType</span></p>

:::

(fig:pfd-fh-11)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 427px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class staQualityParametersType {
    <<complexType>>
    @polarisation : polarisationType [1]
    invalidL1aDataSamples : float [1]
    rfiDecorrelation : float [1]
    rfiDecorrelationThreshold : float [1]
    faradayDecorrelation : float [1]
    faradayDecorrelationThreshold : float [1]
    invalidResidualShiftsFraction : float [1]
    residualShiftsQualityThreshold : float [1]
    invalidGroundPhasesScreenEstimatesFraction : float [1]
    groundPhasesScreenQualityThreshold : float [1]
    skpDecompositionIndex : unsignedInt [1]
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-common-annotations.xsd (<code>staQualityParametersType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 11</span><span class="caption-text"> — Input information DSR, staQualityParametersType</span></p>

:::

(tab:pfd-fh-12)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 12</span><span class="caption-text"> — Input Information DSR (InputInformationL2aType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `productType` | Input product type: always “STA” (L1c STA) | `productType` | 1 |
| `overallProductsQualityIndex` | Quality index based on the "`overallProductQualityIndex`" of the single STA input data, giving a qualitative overall idea of the input stack. Equal to "0" if all STA feature "`overallProductQualityIndex`"=False and to "1" if at least one is degraded. | `xsd:unsignedByte` | 1 |
| `nominalStack` | True if it is a nominal stack (3 STA products for INT phase or 7 STA products for TOM phase), False otherwise | bool | 1 |
| `polarisationList` | List of polarisations composing input L1c data: for each polarisation it is also specified, with an attribute, the decorrelation factors due to RFI strength. | <u>`polarisationListType`</u> | 1 |
| projection | Projection of the input acquisitions, always “Slant Range”. | `projectionType` | 1 |
| footprint | Image footprint of all the acquisitions composing the stack, expressed as a list of latitude, longitude tuples in NE, SE, SW, NW order [deg]. | `floatArrayWithUnits` | 1 |
| `verticalWavenumbers` | Minimum and maximum values of the vertical wavenumbers [rad/m] in the read and used Input STA products. | `minMaxTypeWithUnit` | 1 |
| `heightOfAmbiguity` | Minimum and maximum values of the height of ambiguity (HoA) [m], computed from the vertical wavenumbers in the read and used Input STA products. | `minMaxTypeWithUnit` | 1 |
| `acquisitionsList` | Info of each acquisition composing the stack, in input to L2a. | <u>`acquisitionListType`</u> | 1 |

:::

The `acquisitionListType` is composed by 2 to 8 entries, where each entry is an L1c acquisition folder name string, as defined in {rd}`10`

(tab:pfd-fh-13)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 13</span><span class="caption-text"> — Input information DSR (acquisitionListType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *count* | *Number of acquisitions.* | *`unsignedInt`* | *required* |
| acquisition | Info about acquisition of the stack. | `acquisitionType` | 2..8 |

:::

(tab:pfd-fh-14)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 14</span><span class="caption-text"> — Input information DSR (acquisitionType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *`referenceImage`* | *Flag to indicate if this is the reference image or not.* | *bool* | *1* |
| *`averageWavenumber`* | *Optional Average Wavenumber, it is not used in FH (only in L2A FD)* | *float* | *0* |
| FolderName | Folder name which univocally identifies an acquisition of the stack. | `xsd:string` | 1 |
| `staQuality` | <u>Folder name which univocally identifies an acquisition of the stack, as defined in [RD10].</u> | <u>`staQualityType`</u> | 1 |

:::

(tab:pfd-fh-15)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 15</span><span class="caption-text"> — Input information DSR (staQualityType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `overallProductQualityIndex` | Overall product quality index. This annotation is calculated based on specific quality parameters and gives an overall quality value to the product. Equal to 0 for valid products and to 1 for invalid ones. | `xsd:unsignedInt` | 1 |
| `staQualityParametersList` | Quality parameters list. The list contains an entry for each output product’s polarisation. | <u>`staQualityParametersListType`</u> | 1 |

:::

(tab:pfd-fh-16)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 16</span><span class="caption-text"> — STA quality DSR (staQualityParametersListType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *count* |   | *`xsd:unsignedInt`* | *required* |
| `staQualityParameters` | Quality parameters for the current polarisation. | <u>`staQualityParametersType`</u> | 1 .. 4 |

:::

(tab:pfd-fh-17)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 17</span><span class="caption-text"> — STA quality DSR (staQualityParametersType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *polarisation* |   | *`polarisationType`* | *required* |
| `invalidL1aDataSamples` | Number of invalid L1a data samples (according to their mean and standard deviation), expressed as a normalized fraction of the total number of samples. The value is between 0 and 1. | `xsd:float` | 1 |
| `rfiDecorrelation` | Decorrelation factor due to RFI strength. The value is between 0 and 1. | `xsd:float` | 1 |
| `rfiDecorrelationThreshold` | Maximum decorrelation factor due to RFI strength admissible to use the image as the primary image. The value is between 0 and 1 and copied from the AUX-PPS. | `xsd:float` | 1 |
| `faradayDecorrelation` | Decorrelation factor due to Faraday residual. The value is between 0 and 1. | `xsd:float` | 1 |
| `faradayDecorrelationThreshold` | Maximum decorrelation factor due to Faraday residual admissible to use the image as the primary image. The value is between 0 and 1 and copied from AUX-PPS. | `xsd:float` | 1 |
| `invalidResidualShiftsFraction` | Number of invalid residual coregistration shifts, expressed as a normalized fraction of the total number of residual coregistration shifts. The value is between 0 and 1. | `xsd:float` | 1 |
| `residualShiftsQualityThreshold` | Threshold on residual coregistration shifts quality. The value is between 0 and 1 and copied from AUX-PPS. Residual shifts with a quality lower than this threshold are not used for processing. | `xsd:float` | 1 |
| `invalidGroundPhasesScreenEstimatesFraction` | Number of invalid ground phase screen estimates, expressed as a normalized fraction of the total number of ground phase screen estimates. The value is between 0 and 1. | `xsd:float` | 1 |
| `groundPhasesScreenQualityThreshold` | Threshold on ground phase screen estimation quality. The value is between 0 and 1 and copied from the AUX-PPS. Estimates with a quality lower than this threshold are not corrected. Used only in case `skpPhaseCorrectionFlag` is set to True. | `xsd:float` | 1 |
| `skpDecompositionIndex` | Error code for the SKP decomposition. Positive if the SKP decomposition hit a nonblocking contingency case (e.g. SVD decomposition failure etc.) and the exported SKP-related LUTs contain values obtained from a contingency handling procedure. 0 otherwise. | `xsd:unsignedInt` | 1 |

:::

(tab:pfd-fh-18)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 18</span><span class="caption-text"> — Input information DSR (polarisationListType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *count* | *Number of polarization images.* | *`xsd:unsignedInt`* | *required* |
| polarisation | Polarisation (HH, VH, VV). | `polarisationType` | 3 |

:::

(sec:pfd-fh-4-3-1-4)=
##### 4.3.1.4 Processing parameters L2a DSR

This DSR contains the exhaustive list of L2a Forest Height inversion processing parameters applied. An overview is provided by figure and table below.

(fig:pfd-fh-12)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 555px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class processingParametersL2aType {
    <<complexType>>
    processorVersion : string [1]
    productGenerationTime : timeType [1]
    generalConfigurationParameters : generalConfigurationParametersType [1] ⊞
    verticalReflectivityOption : verticalProfileOptionType [1]
    verticalReflectivityDefaultProfile : verticalReflectivityProfileType [1] ⊞
    modelInversion : modelInversionType [1]
    spectralDecorrelationCompensationFlag : bool [1]
    SNRDecorrelationCompensation : bool [1]
    correctTerrainSlopesFlag : bool [1]
    normalisedHeightEstimationRange : minMaxType [1] ⊞
    normalisedWavenumberEstimationRange : minMaxNumType [1] ⊞
    groundToVolumeRatioRange : minMaxNumType [1] ⊞
    temporalDecorrelationEstimationRange : minMaxNumType [1] ⊞
    temporalDecorrelationGroundToVolumeRatio : float [1]
    residualDecorrelation : float [1]
    productResolution : floatWithUnit [1] @units
    uncertaintyValidvaluesLimits : minMaxType [1] ⊞
    verticalWavenumberValidvaluesLimits : minMaxType [1] ⊞
    lowerHeightLimit : floatWithUnit [1] @units
    upsamplingFactor : unsignedByte [1]
    compressionOptions : compressionOptionsL2a [1] ⊞
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-fh-proc-annotations.xsd (<code>processingParametersL2aType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 12</span><span class="caption-text"> — Processing parameters L2a DSR (processingparametersL2aType)</span></p>

:::

(tab:pfd-fh-19)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 19</span><span class="caption-text"> — Processing parameters L2a DSR (processingparametersL2aType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `processorVersion` | Version of the processor used to generate the product. | `xsd:string` | 1 |
| `productGenerationTime` | Product generation time [UTC]. | `timeType` | 1 |
| `generalConfigurationParameters` | General processing flags (not specific to the Forest Height processing). | <u>`generalConfigurationParametersType`</u> | 1 |
| `verticalReflectivityOption` | Specify which vertical reflectivity profile has been used: default modelled profile or tomographic profile (out of BPS scope). <br> See also `verticalReflectivityDefaultProfile`. | <u>`verticalProfileOptionType`</u> | 1 |
| `verticalReflectivityDefaultProfile` | Used if `verticalReflectivityOption` is set to “default profile”, in this case the profile stored to xml is used: <br> a modeled tomographic volume-only reflectivity profile normalized to unit height; deterministic and fixed for all the globe {ad}`9`. | <u>`verticalReflectivityProfileType`</u> | 1 |
| `modelInversion` | Model inversion algorithm used between single or dual (baseline). | <u>`modelInversionType`</u> | 1 |
| `spectralDecorrelationCompensationFlag` | True if range spectral decorrelation compensation has been applied, False otherwise. | bool | 1 |
| SNRDecorrelationCompensation | True if SNR decorrelation compensation has been applied, False otherwise. | bool | <u>1</u> |
| `correctTerrainSlopesFlag` | True if terrain slope correction has been applied, False otherwise. | bool | <u>1</u> |
| `normalisedHeightEstimationRange` | Validity range of heights used for the canopy height estimation process, normalized. | `minMaxType` | 1 |
| `normalisedWavenumberEstimationRange` | Validity range of wavenumbers used for the canopy height estimation process, normalized. | `minMaxNumType` | 1 |
| `groundToVolumeRatioRange` | Validity ground to volume ratio range used for the canopy height estimation process. | `minMaxNumType` | 1 |
| `temporalDecorrelationEstimationRange` | Validity temporal decorrelation range used for the canopy height estimation process. | `minMaxNumType` | 1 |
| `temporalDecorrelationGroundToVolumeRatio` | Ratio of temporal decorrelation between ground and volume (0.0 means no temporal decorrelation for ground, while 1.0 means ground and volume are equally impacted by temporal decorrelation) | `xsd:float` | 1 |
| `residualDecorrelation` | Residual non-volumetric decorrelation used in error model computation. | `xsd:float` | 1 |
| `productResolution` | Value in [m] used as the resolution on ground range map and also to perform the coherence averaging in radar coordinates. | `floatWithUnit` | 1 |
| `uncertaintyValidvaluesLimits` | Uncertainty valid values limits applied, as a percentage: estimates out of this limits have been discarded and set to no data value. | `minMaxType` | 1 |
| `verticalWavenumberValidvaluesLimits` | Vertical wavenumber valid values limits in [rad/m] applied: estimates out of this limits have been discarded and set to no data value. | `minMaxType` | 1 |
| `lowerHeightLimit` | FH estimates lower this limit [m] were discarded and set to no data value | `floatWithUnit` | 1 |
| `upsamplingFactor` | Upsampling factor used for coherence. | `xsd:unsignedByte` | 1 |
| `compressionOptions` | Configurable compression options for all the L2 MDS COG and ADS NetCDF LUT variables (see 4.3.2 for LUT “ZLIB” compression details). | <u>`compressionOptionsL2Type`</u> | 1 |

:::

(tab:pfd-fh-20)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 20</span><span class="caption-text"> — Processing parameters L2a DSR (verticalReflectivityProfileType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *count* | *Number of values composing the function.* | *`xsd:unsignedInt`* | *required* |
| val | N floating point values which compose the default modelled tomographic volume-only reflectivity profile function. | `xsd:float` | 1..N |

:::

(fig:pfd-fh-13)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 523px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class generalConfigurationParametersType {
    <<complexType>>
    applyCalibrationScreen : calibrationScreenType [1]
    forestCoverageThreshold : float [1]
    forestMaskInterpolationThreshold : float [1]
    subsettingRule : subsettingRuleType [1]
    polarisationCombinationMethod : polarisationCombinationMethodType [1]
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-common-annotations.xsd (<code>generalConfigurationParametersType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 13</span><span class="caption-text"> — Processing parameters L2a DSR (generalConfigurationParametersType)</span></p>

:::

(tab:pfd-fh-21)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 21</span><span class="caption-text"> — Processing parameters L2a DSR (generalConfigurationParametersType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `applyCalibrationScreen` | Phase calibration performed: <br> “none”: no phase screen has been applied <br> “geometry”: only flattening phase screen has been applied (i.e., as computed from acquisition geometry) <br> “skp”: complete phase screen has been applied (default) | `calibrationScreenType` | 1 |
| `forestCoverageThreshold` | Minimum percentage forest coverage in L2a product footprint, used to trigger L2a processing. | `xsd:float` | 1 |
| `forestMaskInterpolationThreshold` | This parameter is a threshold which has been used to fix rounding of pixels with decimal values originated from binary FNF interpolation onto L2a grid. <br> This created a safety buffer around forest border. | `xsd:float` | 1 |
| `subsettingRule` | In case of more than three acquisitions the user selected subsetting rule is here annotated (always “geometry” for FH L2a processor). | <u>`subsettingRuleType`</u> | 1 |
| `polarisationCombinationMethod` | Polarisations combination method: “HV” or “VH”, if just one of the two is selected (in addition to HH and VV ones); “Average”, if the average of HV and VH is computed and used; “None” if no combination is performed (all the four polarizations are used). | `polarisationCombinationMethodType` | 1 |

:::

(fig:pfd-fh-14)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 1032px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class compressionOptionsL2a {
    <<complexType>>
    MDS : «inline» [1]
    ADS : «inline» [1]
    MDS_blockSize : unsignedByte [1]
    ADS_blockSize : unsignedByte [1]
  }
  class compressionOptionsL2a_MDS["compressionOptionsL2a / MDS"] {
    <<anonymous>>
    FH : «inline» [1]
    Quality : «inline» [1]
  }
  class compressionOptionsL2a_ADS["compressionOptionsL2a / ADS"] {
    <<anonymous>>
    FNF : «inline» [1]
    geometricLUTs : «inline» [1]
  }
  class compressionOptionsL2a_MDS_FH["compressionOptionsL2a / MDS / FH"] {
    <<anonymous>>
    compressionFactor : unsignedByte [1]
    MAX_Z_ERROR : float [1]
  }
  class compressionOptionsL2a_MDS_Quality["compressionOptionsL2a / MDS / Quality"] {
    <<anonymous>>
    compressionFactor : unsignedByte [1]
    MAX_Z_ERROR : float [1]
  }
  class compressionOptionsL2a_ADS_FNF["compressionOptionsL2a / ADS / FNF"] {
    <<anonymous>>
    compressionFactor : unsignedByte [1]
  }
  class compressionOptionsL2a_ADS_geometricLUTs["compressionOptionsL2a / ADS / geometricLUTs"] {
    <<anonymous>>
    compressionFactor : unsignedByte [1]
    least_significant_digit : unsignedByte [1]
  }
  compressionOptionsL2a --> compressionOptionsL2a_MDS : MDS
  compressionOptionsL2a --> compressionOptionsL2a_ADS : ADS
  compressionOptionsL2a_MDS --> compressionOptionsL2a_MDS_FH : FH
  compressionOptionsL2a_MDS --> compressionOptionsL2a_MDS_Quality : Quality
  compressionOptionsL2a_ADS --> compressionOptionsL2a_ADS_FNF : FNF
  compressionOptionsL2a_ADS --> compressionOptionsL2a_ADS_geometricLUTs : geometricLUTs
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-fh-proc-annotations.xsd (<code>compressionOptionsL2a</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 14</span><span class="caption-text"> — Processing parameters L2a DSR (compressionOptionsL2Type)</span></p>

:::

(tab:pfd-fh-22)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 22</span><span class="caption-text"> — Processing parameters L2a DSR (compressionOptionsL2aType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| MDS →FH → `compressionFactor` <br> MDS →Quality→ `compressionFactor` | ZSTD algorithm compression factor for both FH and quality images MDS. From 0 to 9, where zero means no compression. | `xsd:unsignedByte` | 1 |
| MDS → FH → `MAX_Z_ERROR` <br> MDS → Quality→ `MAX_Z_ERROR` | For both FH and quality images MDS, define exactly how lossy the LERC compression algorithm is allowed to be, specifying the absolute maximum error admitted. <br> Zero means loss-less compression. | `xsd:float` | 1 |
| ADS →FNF → `compressionFactor` | ZLIB algorithm compression factor for the FNF ADS. From 0 to 9, where zero means no compression. | `xsd:unsignedByte` | 1 |
| ADS → GeometricLuts → `compressionFactor` | ZLIB algorithm compression factor valid for all the Geometric LUTs ADS layers. From 0 to 9, where zero means no compression. | `xsd:unsignedByte` | 1 |
| ADS → GeometricLuts → least\_significant\_digit | For all Geometric LUTs ADS, define exactly how lossy the ZLIB compression algorithm is allowed to be, specifying the power of ten of the smallest decimal place in the data that is a reliable value. | `xsd:unsignedByte` | 1 |
| `MDS_blockSize` | MDS COG blocking algorithm size. Same value is used for both data array dimensions. | `xsd:unsignedByte` | 1 |
| `ADS_blockSize` | ADS NetCDF chunking algorithm size. Same value is used for both data array dimensions. | `xsd:unsignedByte` | 1 |

:::

(sec:pfd-fh-4-3-1-5)=
##### 4.3.1.5 Annotation LUT DSR

This DSR contains the list of Look-Up Tables (LUTs) complementing product main annotations. An overview is provided by figure and table below.

(fig:pfd-fh-15)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 242px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class layerListType {
    <<complexType>>
    @count : unsignedInt [1]
    layer : layerType [1..30]
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-common-types.xsd (<code>layerListType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 15</span><span class="caption-text"> — Annotation LUT DSR (layerListType)</span></p>

:::

(tab:pfd-fh-23)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 23</span><span class="caption-text"> — Annotation LUT DSR (layerListType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *count* | *Number of layers.* | *`xsd:unsignedInt`* | *required* |
| layer | LUT layer content. | `layerType` | 1 .. unbounded |

:::

(sec:pfd-fh-4-3-2)=
#### 4.3.2 Look-Up Tables (LUT) ADS

This section describes in details the content of Look-Up Tables (LUT) ADS.

This ADS complements the Main ADS one, including those annotations that can be represented as 2D maps of values. This approach has been chosen in order to avoid worsening the Main ADS readability and handling, and also to provide an easier access to such information.

The format adopted for LUT ADS is NetCDF one {rd}`6` and in particular the used implementation is the Python NetCDF4 {rd}`7`. Thanks to the flexibility provided by this format, all the components of this ADS are stored in the same file.

The LUT ADS provides the Forest/Non-Forest mask (generated outside of BPS {ad}`6` {ad}`7` {ad}`8` {ad}`9`).

{fig}`16` provides a schematic representation of the LUT ADS structure. Note that an XSD diagram has been used for this purpose, even if there is no link between the two formats. The ADS is organised as follows:

- a series of root level *Attributes*, containing all those global parameters referring to the product as a whole;
- a *Dimensions* group, where *names* and *lengths* of the dimensions used when creating data arrays are here defined;
- a series of *Coordinate* *variables*, special variables to store the axis data values, for each dimension in the *Dimensions* group;
- a series of components (groups of variables), one for each LUT stored in the ADS. Each component is composed by a group of variables which can be divided in:
  - the data arrays, storing the annotation Look-Up Tables themselves. A variable represents an array of values of the same data type and it is defined over two or more dimensions from the *Dimensions* group;
  - ancillary scalar variables containing other useful metadata to complete the LUT definition.

(fig:pfd-fh-16)=
::::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

```{mermaid}
%%{init: {"flowchart": {"useMaxWidth": false}, "class": {"useMaxWidth": false}, "theme": "default"}}%%
classDiagram
  accTitle: Look-Up Tables (LUT) ADS
  accDescr: XML schema diagram. The element lutAnnotation has the type lutAnnotationType, an extension. Its sequence holds Attributes (18), Dimensions (2), Coordinate_variables (2) and FNF. FNF has the type fnf_variables, a sequence of 10 Variable elements.
  direction LR
  %% A number alone under a stacked (double-frame) XMLSpy element is its occurrence count
  %% (minOccurs = maxOccurs), written [18], [2], [10]; it is not a child count.
  %% Types of Attributes, Dimensions, Coordinate_variables and Variable are not shown: none written.
  class lutAnnotation
  class lutAnnotationType["lutAnnotationType (extension)"] {
    <<complexType>>
    Attributes [18]
    Dimensions [2]
    Coordinate_variables [2]
    FNF : fnf_variables [1]
  }
  class fnf_variables {
    <<complexType>>
    Variable [10]
  }
  lutAnnotation --> lutAnnotationType
  lutAnnotationType --> fnf_variables : FNF
```

<!-- source image: figure_16_section_4.3.2 sha256 88c9cb3fdff6c5a58df4d938ec655c2e8075feb3fe4ced732cb3e127d0c8ef4a -->

:::{div} xsd-diagram-legend
Diagram notation: `@name` attribute; `[min..max]` occurrences; `⊞` named type not expanded here; `«inline»` type declared in place, drawn as an «anonymous» class; `choice n ·` alternative n of a choice; `all ·` elements in any order.
:::

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 16</span><span class="caption-text"> — Look-Up Tables (LUT) ADS</span></p>

::::

**Compression strategy**

The NetCDF data arrays variable is composed by binary values, it is possible to introduce with benefit a strategy for data compression making use of native Python NetCDF4 capabilities {rd}`7`, using the ZLIB (loss-less) compression algorithm, with a compression level from 0 to 9, where zero means no compression, set by configuration, more details in {sec}`4.3.1.4`.

(sec:pfd-fh-4-3-2-1)=
##### 4.3.2.1 Look-Up tables detailed content

The content of L2a LUT ADS is reported hereafter, where:

- the series of *Attributes* contain the same list of parameters described in {sec}`Sec. 4.3.1.1 <4.3.1.1>` for the Product L2a DSR (for all the details about these parameters refer to this section);
- the *Dimensions* group is composed by Latitude and Longitude, as the data array variable is defined in the Latitude Longitude based on DGG projection;
- there is only one data array variable, which is the Forest/Non-Forest mask;
- for the geometric LUTs (local incidence angle, height, latitude, longitude, incidence/elevation angles, range/azimuth terrain slopes), the repeated variables are described only once.
- this notation is applicable:
  - **this is the attribute or variable name**
  - this is the actual content of constant values over the mission
  - *this is a description of values content which depends on the processing*

<u>LUT ADS Attributes</u>:

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        <b>mission</b>: “BIOMASS”
        <b>t</b><b>ileID</b><i>: </i><i>[string] json encoded list of DGG geographical tile identifiers (NxxEyyy)</i>
        <b>b</b><b>asinID</b>: <i>[string] json encoded list of </i><i>DGG Basin identifier</i><i>s</i><i> </i>
        <b>pro</b><b>d</b><b>uctType</b>: FP_FH__L2A
        <b>startTime</b>:<i> [str: YYYY-MM-DD hh:mm:ss.ppp] Start time of the product image, UTC</i>
        <b>stopTime</b>: <i>[str: YYYY-MM-DD hh:mm:ss.ppp] Stop time of the product image, UTC </i>
        <b>radarCarrierFrequency</b>: <i>[</i><i>float32</i><i>] Radar carrier frequency [Hz]</i>
        <b>missionPhaseID</b>: <i>[str: COM, TOM or INT] Mission phase identifier</i>
        <b>sensorMode</b>: “Measurement”
        <b>globalCoverageID</b>: <i>[uint16] Global coverage identifier</i>
        <b>swath</b>: <i>[str: S1, S2 or S3] BIOMASS swath identifier</i>
        <b>majorCycleID</b>: <i>[uint16] Major cycle identifier</i>
        <b>absoluteOrbitNumber</b>:<i> [list of uint32] List of absolute orbit numbers at start time (one for each input L1c)</i>
        <b>relativeOrbitNumber</b><i>: [uint16] Relative orbit number (track) at start time</i>
        <b>orbitPass</b>: <i>[str: Ascending or Descending] Orbit pass</i>
        <b>dataTakeID</b>: <i>[string] json encoded list of </i><i>data take identifiers (one for each input L1c)</i>
        <b>frame</b>: <i>[uint16] Frame identifier</i>
        <b>platform</b><b>_h</b><b>eading</b>: <i>[</i><i>float32</i><i>] Platform heading relative to North, deg</i>
        <b>forest</b><b>_c</b><b>overage</b><b>_p</b><b>ercentage</b>: <i>[</i><i>float32</i><i>] </i><i>Forest coverage of the input STA products, compute as a percentage of all the pixels in the </i>                <i>STA footprint</i>
         <b>formatVersion </b><i>[</i><i>str</i><i>]</i><i> </i><i>Version of the product</i></pre></div></div>
```

::::{only} latex

```text
        mission: “BIOMASS”
        tileID: [string] json encoded list of DGG geographical tile identifiers (NxxEyyy)
        basinID: [string] json encoded list of DGG Basin identifiers
        productType: FP_FH__L2A
        startTime: [str: YYYY-MM-DD hh:mm:ss.ppp] Start time of the product image, UTC
        stopTime: [str: YYYY-MM-DD hh:mm:ss.ppp] Stop time of the product image, UTC
        radarCarrierFrequency: [float32] Radar carrier frequency [Hz]
        missionPhaseID: [str: COM, TOM or INT] Mission phase identifier
        sensorMode: “Measurement”
        globalCoverageID: [uint16] Global coverage identifier
        swath: [str: S1, S2 or S3] BIOMASS swath identifier
        majorCycleID: [uint16] Major cycle identifier
        absoluteOrbitNumber: [list of uint32] List of absolute orbit numbers at start time (one for each input L1c)
        relativeOrbitNumber: [uint16] Relative orbit number (track) at start time
        orbitPass: [str: Ascending or Descending] Orbit pass
        dataTakeID: [string] json encoded list of data take identifiers (one for each input L1c)
        frame: [uint16] Frame identifier
        platform_heading: [float32] Platform heading relative to North, deg
        forest_coverage_percentage: [float32] Forest coverage of the input STA products, compute as a percentage of all the pixels in the                 STA footprint
         formatVersion [str] Version of the product
```

::::

<u>LUT ADS Dimensions / Coordinate variables</u>:

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        Name: <b>Latitude</b>
                size: <i>[int] size</i>
                type: dtype(float32)
                units: deg
                description: FNF LUT latitude axis
        Name: <b>Longitude</b>
                size<i>: [int] size</i>
                type: dtype(float32)
                units: deg
                description: FNF LUT longitude axis</pre></div></div>
```

::::{only} latex

```text
        Name: Latitude
                size: [int] size
                type: dtype(float32)
                units: deg
                description: FNF LUT latitude axis
        Name: Longitude
                size: [int] size
                type: dtype(float32)
                units: deg
                description: FNF LUT longitude axis
```

::::

<u>LUT ADS group of variables</u>:

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        Group Name: <b>FNF</b>
        FNF group contents:
                Name: <b>FNF</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(ubyte)
                        description: Forest/Non-Forest mask : this is an external image, defined outside of BPS from aggregated C3S LCM, representing the initial two-dimensional mask, here defined in the same latitude-longitude based on DGG projection used for L2a images and cropped to have a coverage containing the L2a product image boundaries. It defines, for each geographical pixel, the initial presence or absence of forest.
                Name: <b>firstSample</b>
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS sample the first LUT sample corresponds to
                Name: <b>firstLine</b>
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS line the first LUT line corresponds to
                Name: <b>samplesInterval</b>
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS samples one LUT sample corresponds to
                Name: <b>linesInterval</b>
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS lines one LUT line corresponds to
                Name: <b>pixelType</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: Data type of output LUT pixels
                Name: <b>noDataValue</b>
                        shape: (1, 1)
                        type: dtype(ubyte)
                        description: Pixel value in case of invalid data. The value is always set to 255. (See 3)
                Name: <b>projection</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: LUT projection, DGG, as described in [RD8]
                Name: <b>coordinateReferenceSystem</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: datum, <b>geodeticReferenceFrame</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: Geodetic reference frame
        Group Name: <b> </b><b>height</b>
        height group contents:
                Name: <b> </b><b>height</b>
                        dimensions: (Longitude, Latitude)
                        shape:<i> ([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the height coordinates of each (sub-sampled) pixel, derived from the Digital Elevation Model (DEM) used during processing. Values are expressed in [m]
    units: m
                Name: <b>firstSample</b>
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS sample the first LUT sample corresponds to
                Name: <b>firstLine</b>
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS line the first LUT line corresponds to
                Name: <b>samplesInterval</b>
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS samples one LUT sample corresponds to
                Name: <b>linesInterval</b>
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS lines one LUT line corresponds to
                Name: <b>pixelType</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: Data type of output LUT pixels
                Name: <b>noDataValue</b>
                        shape: (1, 1)
                        type: dtype(float32)
                        description: Pixel value in case of invalid data. The value is always set to -9999.0. (See 3)
                Name: <b>projection</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: LUT projection, DGG, as described in [RD8]
                Name: <b>coordinateReferenceSystem</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: <b>geodeticReferenceFrame</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: Geodetic reference frame
                Name: <b>least_significant_digit</b>
                        shape: (1, 1)
                        type: unsignedByte
                        description: Power of ten of the smallest decimal place in the data that is a reliable value, adopted to the data during ZLIB compression.
        Group Name: <b> </b><b>latitude</b>
        height group contents:
                Name: <b> </b><b>latitude</b>
                        dimensions: (Longitude, Latitude)
                        shape:<i> ([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the latitude coordinates of each (sub-sampled) pixel. Values are expressed in [deg].
    units: deg</pre></div></div>
```

::::{only} latex

```text
        Group Name: FNF
        FNF group contents:
                Name: FNF
                        dimensions: (Longitude, Latitude)
                        shape: ([int] x shape, [int] y shape)
                        type: dtype(ubyte)
                        description: Forest/Non-Forest mask : this is an external image, defined outside of BPS from aggregated C3S LCM, representing the initial two-dimensional mask, here defined in the same latitude-longitude based on DGG projection used for L2a images and cropped to have a coverage containing the L2a product image boundaries. It defines, for each geographical pixel, the initial presence or absence of forest.
                Name: firstSample
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS sample the first LUT sample corresponds to
                Name: firstLine
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS line the first LUT line corresponds to
                Name: samplesInterval
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS samples one LUT sample corresponds to
                Name: linesInterval
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS lines one LUT line corresponds to
                Name: pixelType
                        shape: (1, 1)
                        type: <class str>
                        description: Data type of output LUT pixels
                Name: noDataValue
                        shape: (1, 1)
                        type: dtype(ubyte)
                        description: Pixel value in case of invalid data. The value is always set to 255. (See 3)
                Name: projection
                        shape: (1, 1)
                        type: <class str>
                        description: LUT projection, DGG, as described in [RD8]
                Name: coordinateReferenceSystem
                        shape: (1, 1)
                        type: <class str>
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: datum, geodeticReferenceFrame
                        shape: (1, 1)
                        type: <class str>
                        description: Geodetic reference frame
        Group Name:  height
        height group contents:
                Name:  height
                        dimensions: (Longitude, Latitude)
                        shape: ([int] x shape, [int] y shape)
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the height coordinates of each (sub-sampled) pixel, derived from the Digital Elevation Model (DEM) used during processing. Values are expressed in [m]
    units: m
                Name: firstSample
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS sample the first LUT sample corresponds to
                Name: firstLine
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS line the first LUT line corresponds to
                Name: samplesInterval
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS samples one LUT sample corresponds to
                Name: linesInterval
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS lines one LUT line corresponds to
                Name: pixelType
                        shape: (1, 1)
                        type: <class str>
                        description: Data type of output LUT pixels
                Name: noDataValue
                        shape: (1, 1)
                        type: dtype(float32)
                        description: Pixel value in case of invalid data. The value is always set to -9999.0. (See 3)
                Name: projection
                        shape: (1, 1)
                        type: <class str>
                        description: LUT projection, DGG, as described in [RD8]
                Name: coordinateReferenceSystem
                        shape: (1, 1)
                        type: <class str>
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: geodeticReferenceFrame
                        shape: (1, 1)
                        type: <class str>
                        description: Geodetic reference frame
                Name: least_significant_digit
                        shape: (1, 1)
                        type: unsignedByte
                        description: Power of ten of the smallest decimal place in the data that is a reliable value, adopted to the data during ZLIB compression.
        Group Name:  latitude
        height group contents:
                Name:  latitude
                        dimensions: (Longitude, Latitude)
                        shape: ([int] x shape, [int] y shape)
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the latitude coordinates of each (sub-sampled) pixel. Values are expressed in [deg].
    units: deg
```

::::

The remaining variables, which are identical across all geometric LUTs, are omitted from the detailed description (`firstSample`, `firstLine`, `samplesInterval`, `pixelType`, `noDataValue`, projection, `coordinateReferenceSystem`, `geodeticReferenceFrame`, least\_significant\_digit).

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        Group Name: <b> </b><b>longitude</b>
        height group contents:
                Name: <b> </b><b>longitude</b>
                        dimensions: (Longitude, Latitude)
                        shape:<i> ([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the longitude coordinates of each (sub-sampled) pixel. Values are expressed in [deg]
    units: deg</pre></div></div>
```

::::{only} latex

```text
        Group Name:  longitude
        height group contents:
                Name:  longitude
                        dimensions: (Longitude, Latitude)
                        shape: ([int] x shape, [int] y shape)
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the longitude coordinates of each (sub-sampled) pixel. Values are expressed in [deg]
    units: deg
```

::::

The remaining variables, which are identical across all geometric LUTs, are omitted from the detailed description (`firstSample`, `firstLine`, `samplesInterval`, `pixelType`, `noDataValue`, projection, `coordinateReferenceSystem`, `geodeticReferenceFrame`, least\_significant\_digit).

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        Group Name: <b> </b><b>incidenceAngle</b>
        height group contents:
                Name: <b> </b><b>incidenceAngle</b>
                        dimensions: (Longitude, Latitude)
                        shape:<i> ([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the incidence angle of each (sub-sampled) pixel. Values are computed during L1 processing considering an average height. Values are expressed in [deg].
    units: deg</pre></div></div>
```

::::{only} latex

```text
        Group Name:  incidenceAngle
        height group contents:
                Name:  incidenceAngle
                        dimensions: (Longitude, Latitude)
                        shape: ([int] x shape, [int] y shape)
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the incidence angle of each (sub-sampled) pixel. Values are computed during L1 processing considering an average height. Values are expressed in [deg].
    units: deg
```

::::

The remaining variables, which are identical across all geometric LUTs, are omitted from the detailed description (`firstSample`, `firstLine`, `samplesInterval`, `pixelType`, `noDataValue`, projection, `coordinateReferenceSystem`, `geodeticReferenceFrame`, least\_significant\_digit).

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        Group Name: <b> </b><b>elevationAngle</b>
        height group contents:
                Name: <b> </b><b>elevationAngle</b>
                        dimensions: (Longitude, Latitude)
                        shape:<i> ([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the elevation angle of each (sub-sampled) pixel. Values are computed during L1 processing considering an average height. Values are expressed in [deg].
    units: deg</pre></div></div>
```

::::{only} latex

```text
        Group Name:  elevationAngle
        height group contents:
                Name:  elevationAngle
                        dimensions: (Longitude, Latitude)
                        shape: ([int] x shape, [int] y shape)
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the elevation angle of each (sub-sampled) pixel. Values are computed during L1 processing considering an average height. Values are expressed in [deg].
    units: deg
```

::::

The remaining variables, which are identical across all geometric LUTs, are omitted from the detailed description (`firstSample`, `firstLine`, `samplesInterval`, `pixelType`, `noDataValue`, projection, `coordinateReferenceSystem`, `geodeticReferenceFrame`, least\_significant\_digit).

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        Group Name: <b> </b><b>rangeTerrainSlope</b>
        height group contents:
                Name: <b> </b><b>rangeTerrainSlope</b>
                        dimensions: (Longitude, Latitude)
                        shape:<i> ([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the local DEM slopes along range direction computed using the Digital Elevation Model (DEM). Values are expressed in [deg].
    units: deg</pre></div></div>
```

::::{only} latex

```text
        Group Name:  rangeTerrainSlope
        height group contents:
                Name:  rangeTerrainSlope
                        dimensions: (Longitude, Latitude)
                        shape: ([int] x shape, [int] y shape)
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the local DEM slopes along range direction computed using the Digital Elevation Model (DEM). Values are expressed in [deg].
    units: deg
```

::::

The remaining variables, which are identical across all geometric LUTs, are omitted from the detailed description (`firstSample`, `firstLine`, `samplesInterval`, `pixelType`, `noDataValue`, projection, `coordinateReferenceSystem`, `geodeticReferenceFrame`, least\_significant\_digit).

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        Group Name: <b> </b><b>azimuthTerrainSlope</b>
        height group contents:
                Name: <b> </b><b>azimuthTerrainSlope</b>
                        dimensions: (Longitude, Latitude)
                        shape:<i> ([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the local DEM slopes along azimuth direction computed using the Digital Elevation Model (DEM). Values are expressed in [deg].
    units: deg</pre></div></div>
```

::::{only} latex

```text
        Group Name:  azimuthTerrainSlope
        height group contents:
                Name:  azimuthTerrainSlope
                        dimensions: (Longitude, Latitude)
                        shape: ([int] x shape, [int] y shape)
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the local DEM slopes along azimuth direction computed using the Digital Elevation Model (DEM). Values are expressed in [deg].
    units: deg
```

::::

The remaining variables, which are identical across all geometric LUTs, are omitted from the detailed description (`firstSample`, `firstLine`, `samplesInterval`, `pixelType`, `noDataValue`, projection, `coordinateReferenceSystem`, `geodeticReferenceFrame`, least\_significant\_digit).

(sec:pfd-fh-4-3-3)=
#### 4.3.3 Quick-look ADS

The Quick-look ADS is composed by lower resolution versions of the MDS Forest Height and quality layer images; they are stored in PNG file format and they are always included with any BIOMASS L2 product.

The quantities are composed by real values, so the images are generated without any other processing than averaging and decimation.

(sec:pfd-fh-4-3-4)=
#### 4.3.4 Overlay ADS

The Overlay ADS is a Keyhole Markup Language (KML) file that describes the product coverage area and is suitable for viewing the Quick-look ADS in any application that supports KML format (like, e.g., Google Earth or GIS applications).

Its structure is described in the following figure and tables.

(fig:pfd-fh-17)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 1450px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction TB
  class kmlType {
    <<complexType>>
    Document : DocumentType [1]
  }
  class DocumentType {
    name : string [1]
    description : string [1]
    GroundOverlay : GroundOverlayType [1]
    Placemark : PlacemarkType [1]
  }
  class GroundOverlayType {
    name : string [1]
    visibility : unsignedInt [1]
    TimeStamp : TimeStampType [1]
    Icon : IconType [1]
    gx:LatLonQuad : «inline» [1]
  }
  class PlacemarkType {
    name : string [1]
    visibility : unsignedInt [1]
    TimeStamp : TimeStampType [1]
    Style : StyleType [1]
    ExtendedData : ExtendedDataType [1]
    Polygon : PolygonType [1]
  }
  class TimeStampType {
    when : string [1]
  }
  class IconType {
    href : string [1]
  }
  class LatLonQuad {
    <<anonymous>>
    coordinates : string [1]
  }
  class StyleType {
    LineStyle : LineStyleType [1]
    PolyStyle : PolyStyleType [1]
  }
  class ExtendedDataType {
    Data : DataType [1..∞]
  }
  class PolygonType {
    tessellate : unsignedInt [1]
    altitudeMode : string [1]
    outerBoundaryIs : outerBoundaryIsType [1]
  }
  class LineStyleType {
    color : string [1]
    width : unsignedInt [1]
  }
  class PolyStyleType {
    color : string [1]
  }
  class DataType {
    @name : string [1]
    value : string [1]
  }
  class outerBoundaryIsType {
    LinearRing : LinearRingType [1]
  }
  class LinearRingType {
    coordinates : string [1]
  }
  kmlType --> DocumentType : Document
  DocumentType --> GroundOverlayType : GroundOverlay
  DocumentType --> PlacemarkType : Placemark
  GroundOverlayType --> TimeStampType : TimeStamp
  GroundOverlayType --> IconType : Icon
  GroundOverlayType --> LatLonQuad : LatLonQuad
  PlacemarkType --> TimeStampType : TimeStamp
  PlacemarkType --> StyleType : Style
  PlacemarkType --> ExtendedDataType : ExtendedData
  PlacemarkType --> PolygonType : Polygon
  StyleType --> LineStyleType : LineStyle
  StyleType --> PolyStyleType : PolyStyle
  ExtendedDataType --> DataType : Data
  PolygonType --> outerBoundaryIsType : outerBoundaryIs
  outerBoundaryIsType --> LinearRingType : LinearRing
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l1-overlay.xsd (<code>kmlType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 17</span><span class="caption-text"> — Overlay ADS</span></p>

:::

(tab:pfd-fh-24)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 24</span><span class="caption-text"> — Overlay ADS</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| Document | Document container for KML components. | DocumentType | 1 |

:::

(tab:pfd-fh-25)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 25</span><span class="caption-text"> — Overlay ADS (DocumentType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| name | Name of the document. | `xsd:string` | 1 |
| description | Description of the document. | `xsd:string` | 1 |
| GroundOverlay | Contains the parameters required to specify the footprint of the image and overlay the quicklook image on a map. | GroundOverlayType | 1 |
| Placemark | Contains the metadata associated to the overlay. | PlacemarkType | 1 |

:::

(tab:pfd-fh-26)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 26</span><span class="caption-text"> — Overlay ADS (GroundOverlayType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| name | Name of the ground overlay. | `xsd:string` | 1 |
| visibility | Specifies whether the feature is drawn in the 3D viewer when it is initially loaded (0: not visible, 1: visible). | `xsd:unsignedInt` | 1 |
| TimeStamp | Structure containing the time stamp of the overlay that can be used for dynamic visualizations over time. | TimeStampType | 1 |
| Icon | Structure describing the image file used on the map overlay. | IconType | 1 |
| LatLonQuad | Structure containing the latitude and longitude coordinates used to position the image overlay on the map. | LatLonQuadType | 1 |

:::

(tab:pfd-fh-27)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 27</span><span class="caption-text"> — Overlay ADS (TimeStampType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| when | Overlay time stamp in ISO 8601 format (YYYY-MM-DDThh:mm:ss). | `xsd:string` | 1 |

:::

(tab:pfd-fh-28)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 28</span><span class="caption-text"> — Overlay ADS (IconType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| href | A local file specification or URL used to load the desired image and overlay it on the map. | `xsd:string` | 1 |

:::

(tab:pfd-fh-29)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 29</span><span class="caption-text"> — Overlay ADS (LatLonQuadType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| coordinates | A string of 5 lon,lat coordinate pairs which describe the corners of the image. The string is of the form: <br> *lon,lat lon,lat lon,lat lon,lat lon,lat* <br> The coordinates must appear in the following order: <br> *last line first pixel, last line last pixel, first line last pixel, first line first pixel, last line first pixel* | `xsd:string` | 1 |

:::

(tab:pfd-fh-30)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 30</span><span class="caption-text"> — Overlay ADS (PlacemarkType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| Name | Name of the placemark. | `xsd:string` | 1 |
| visibility | Specifies whether the feature is drawn in the 3D viewer when it is initially loaded (0: not visible, 1: visible). | `xsd:unsignedInt` | 1 |
| TimeStamp | Structure containing the time stamp of the placemark that can be used for dynamic visualizations over time | TimeStampType | 1 |
| Style | Structure describing the stype used to display the placemark. | StyleType | 1 |
| ExtendedData | Structure containing the metadata associated to the overlay. | ExtendedDataType | 1 |
| Polygon | Structure containing the latitude and longitude coordinates used to position the image overlay on the map. | PolygonType | 1 |

:::

(tab:pfd-fh-31)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 31</span><span class="caption-text"> — Overlay ADS (StyleType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| LyneStyle | Line style. | LyneStyleType | 1 |
| PolyStyle | Polygon style. | PolyStyleType | 1 |

:::

(tab:pfd-fh-32)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 32</span><span class="caption-text"> — Overlay ADS (LyneStyleType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| color | Line color in 32 bit hex format (AABBGGRR order). | `xsd:string` | 1 |
| width | Line width. | `xsd:unsignedInt` | 1 |

:::

(tab:pfd-fh-33)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 33</span><span class="caption-text"> — Overlay ADS (PolyStyleType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| color | Polygon color in 32 bit hex format (AABBGGRR order). | `xsd:string` | 1 |

:::

(tab:pfd-fh-34)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 34</span><span class="caption-text"> — Overlay ADS (ExtendedDataType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| Data | Metadata. | DataType | 1..inf |

:::

(tab:pfd-fh-35)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 35</span><span class="caption-text"> — Overlay ADS (DataType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| value | Metadata value. | `xsd:string` | 1 |

:::

(tab:pfd-fh-36)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 36</span><span class="caption-text"> — Overlay ADS ()</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| tessellate | Tessellate option. | `xsd:unsignedInt` | 1 |
| `altitudeMode` | Altitude mode option. | `xsd:string` | 1 |
| `outerBoundaryIs` | Polygon outer boundary description. | `outerBoundaryIsType` | 1 |

:::

(tab:pfd-fh-37)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 37</span><span class="caption-text"> — Overlay ADS (outerBoundaryIsType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| LinearRing | Linear ring description. | LinearRingType | 1 |

:::

(tab:pfd-fh-38)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 38</span><span class="caption-text"> — Overlay ADS (LinearRingType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| coordinates | A string of 5 lon,lat,height coordinate tuples which describe the corners of the image. The string is of the form: <br> *lon,lat,height lon,lat,height lon,lat,height lon,lat,height lon,lat,height* <br> The coordinates must appear in the following order: <br> *last line first pixel, last line last pixel, first line last pixel, first line first pixel, last line first pixel* | `xsd:string` | 1 |

:::
