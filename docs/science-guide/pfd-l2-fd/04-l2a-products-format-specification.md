<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

:::{atbd-logo-banner}
:::


(sec:pfd-fd-4)=
# 4. L2a Products Format Specification

(sec:pfd-fd-4-1)=
### 4.1 Main Product Header

The Main Product Header (MPH) is an XML file present in all the BIOMASS L2 products.

BIOMASS L2 products are folder based and the MPH is top level leaf of the product tree.

The MPH main scope is to provide all the information related to the:

- Genesis of the product;
- Identification of the product for example for cataloguing purposes;
- Product consistency (filesize).

It must be noted that, while the footprint is inherited from lower level products (thus vertices represent the extent of the acquired data), DGG tiles (`tileId` list) are related to the full extent of L2a product (that is geocoded).

For the detailed description of the MPH content and format refer to {ad}`3`. Here just a

schematic representation is reported, where this notation is applicable:

<span style="color:#0070C0">This is a container of other elements</span>

This element or block of elements can occur more than one time (i.e. it is a list)

<span style="color:#C00000"><b>This is a container of actual metadata</b></span>

<span style="color:#C00000">This is a mandatory element for the standard which is l</span><span style="color:#C00000">eft empty by this specification</span>

This element or block of elements is optional

(fig:pfd-fd-2)=
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
                    <span style="color:#C00000"><b>bio:</b></span><span style="color:#C00000"><b>t</b></span><span style="color:#C00000"><b>ileID</b></span> <b>List of DGG geographic cell (tile) identifiers</b>
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
                    <span style="color:#C00000"><b>bio</b></span><span style="color:#C00000"><b>:rds </b></span><b>Product schema</b>
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

```{raw} latex
\begin{Verbatim}[commandchars=\\\{\},fontsize=\small]
\textcolor[HTML]{0070C0}{bio:EarthObservation }\textit{Root element of metadata representation tailored for BIOMASS SAR products}
    \textcolor[HTML]{0070C0}{om:phenomenonTime} \textit{Time coverage of the product}
        \textcolor[HTML]{0070C0}{gml:TimePeriod} \textit{Time coverage of the product as a period}
            \textcolor[HTML]{C00000}{\textbf{gml:beginPosition}}\textcolor[HTML]{C00000}{ }\textbf{Start date and time of the product}
            \textcolor[HTML]{C00000}{\textbf{gml:endPosition}} \textbf{Stop date and time of the product}
    \textcolor[HTML]{0070C0}{om:resultTime} \textit{Time when the sensing data has become available}
        \textcolor[HTML]{0070C0}{gml:TimeInstant} \textit{Identifiable position in time}
            \textcolor[HTML]{C00000}{\textbf{gml:timePosition}} \textbf{Date and time of data availability}
    \textcolor[HTML]{0070C0}{om:validTime} \textit{Time validity of the product}
        \textcolor[HTML]{0070C0}{gml:TimePeriod} \textit{Time validity of the product as a period}
            \textcolor[HTML]{C00000}{\textbf{gml:beginPosition }}\textbf{Validity start date and time of the product}
            \textcolor[HTML]{C00000}{\textbf{gml:endPosition}} \textbf{Validity stop date and time of the product}
    \textcolor[HTML]{0070C0}{om:procedure} \textit{Procedure used to sense the data}
        \textcolor[HTML]{0070C0}{eop:EarthObservationEquipment} \textit{Equipment used to sense the data}
            \textcolor[HTML]{0070C0}{eop:platform} \textit{Platform description}
                \textcolor[HTML]{0070C0}{eop:Platform} \textit{Nested element for platform description}
                    \textcolor[HTML]{C00000}{\textbf{eop:shortName}} \textbf{Satellite name}
            \textcolor[HTML]{0070C0}{eop:instrument} \textit{Instrument description}
                \textcolor[HTML]{0070C0}{eop:Instrument} \textit{Nested element for instrument description}
                    \textcolor[HTML]{C00000}{\textbf{eop:shortName}} \textbf{Sensor name}
            \textcolor[HTML]{0070C0}{eop:sensor} \textit{Sensor description}
                \textcolor[HTML]{0070C0}{eop:Sensor} \textit{Nested element for sensor description}
                    \textcolor[HTML]{C00000}{\textbf{eop:sensorType}} \textbf{Sensor type}
                    \textcolor[HTML]{C00000}{\textbf{eop:operationalMode}} \textbf{Sensor operational mode}
                    \textcolor[HTML]{C00000}{\textbf{eop:swathIdentifier}} \textbf{Swath identifier}
            \textcolor[HTML]{0070C0}{eop:acquisitionParameters} \textit{Acquisition parameters}
                \textcolor[HTML]{0070C0}{bio:Acquisition} \textit{Acquisition description}
                    \textcolor[HTML]{C00000}{\textbf{eop:orbitDirection}} \textbf{Orbit direction}
                    \textcolor[HTML]{C00000}{\textbf{eop:wrsLongitudeGrid}} \textbf{Track number}
                    \textcolor[HTML]{C00000}{\textbf{eop:wrsLatitudeGrid}} \textbf{Frame number}
                    \textcolor[HTML]{C00000}{\textbf{sar:antennaLookDirection}} \textbf{SAR antenna look direction}
                    \textcolor[HTML]{C00000}{\textbf{bio:missionPhase}} \textbf{BIOMASS mission phase}
                    \textcolor[HTML]{C00000}{\textbf{bio:globalCoverageID}} \textbf{BIOMASS global coverage identifier} 
                    \textcolor[HTML]{C00000}{\textbf{bio:majorCycleID}} \textbf{BIOMASS major cycle identifier}
                    \textcolor[HTML]{C00000}{\textbf{bio:}}\textcolor[HTML]{C00000}{\textbf{t}}\textcolor[HTML]{C00000}{\textbf{ileID}} \textbf{List of DGG geographic cell (tile) identifiers}
                    \textcolor[HTML]{C00000}{\textbf{bio:basinID}} \textbf{Retrieval basin identifier}
    \textcolor[HTML]{C00000}{om:observedProperty }\textit{Observed property (empty)}
    \textcolor[HTML]{0070C0}{om:featureOfInterest} \textit{Observed area}
        \textcolor[HTML]{0070C0}{eop:Footprint} \textit{Footprint description}
            \textcolor[HTML]{0070C0}{eop:multiExtentOf }\textit{Footprint representation structure, coordinates in posList}
                \textcolor[HTML]{0070C0}{gml:MultiSurface}
                    \textcolor[HTML]{0070C0}{gml:surfaceMember}
                        \textcolor[HTML]{0070C0}{gml:Polygon}
                            \textcolor[HTML]{0070C0}{gml:exterior}
                                \textcolor[HTML]{0070C0}{gml:LinearRing}
                                    \textcolor[HTML]{C00000}{\textbf{gml:posList}}\textcolor[HTML]{0070C0}{ }\textbf{Footprint points in NE,SE,SW,NW}\textbf{,}\textbf{NE}\textbf{ order}
            \textcolor[HTML]{0070C0}{eop:centerOf }\textit{Acquisition centre representation structure}
                \textcolor[HTML]{0070C0}{gml:Point}
                    \textcolor[HTML]{C00000}{\textbf{gml:pos}} \textbf{Coordinates of the centre of the acquisition}
    \textcolor[HTML]{0070C0}{om:result }\textit{Observation result}
        \textcolor[HTML]{0070C0}{eop:EarthObservationResult}
            \textcolor[HTML]{0070C0}{eop:product}
                \textcolor[HTML]{0070C0}{bio:ProductInformation}
                    \textcolor[HTML]{0070C0}{eop:fileName}
                        \textcolor[HTML]{C00000}{\textbf{ows:ServiceReference}} \textbf{Product filename}
                            \textcolor[HTML]{C00000}{ows:RequestMessage} \textbf{download request (empty)}
            \textcolor[HTML]{C00000}{\textbf{eop:version}} \textbf{Product baseline}
            \textcolor[HTML]{0070C0}{eop:product }(list)
                \textcolor[HTML]{0070C0}{bio:ProductInformation}
                    \textcolor[HTML]{0070C0}{eop:fileName}
                        \textcolor[HTML]{C00000}{\textbf{ows:ServiceReference}} \textbf{Product filename (internal files)}
                            \textcolor[HTML]{C00000}{ows:RequestMessage} \textbf{download request (empty)}
                    \textcolor[HTML]{C00000}{\textbf{eop:size}} \textbf{Product size}
                    \textcolor[HTML]{C00000}{\textbf{bio}}\textcolor[HTML]{C00000}{\textbf{:rds }}\textbf{Product schema}
    \textcolor[HTML]{0070C0}{eop:metaDataProperty }\textit{Observation metadata}
        \textcolor[HTML]{0070C0}{bio:EarthObservationMetaData}
            \textcolor[HTML]{C00000}{\textbf{eop:identifier}} \textbf{Product identifier}
            \textcolor[HTML]{C00000}{\textbf{eop:doi}} \textbf{Digital Object Identifier}
            \textcolor[HTML]{C00000}{\textbf{eop:acquisitionType}} \textbf{Type of acquisition}
            \textcolor[HTML]{C00000}{\textbf{eop:productType}} \textbf{Product type}
            \textcolor[HTML]{C00000}{\textbf{eop:status}} \textbf{Status of the product}
                \textcolor[HTML]{0070C0}{eop:processing }\textit{Data processing information}
                    \textcolor[HTML]{0070C0}{bio:ProcessingInformation}
                        \textcolor[HTML]{C00000}{\textbf{eop:processingCenter}} \textbf{Processing centre code}
                        \textcolor[HTML]{C00000}{\textbf{eop:processingDate}} \textbf{Processing date and time}
                        \textcolor[HTML]{C00000}{\textbf{eop:processorName}} \textbf{Processor name}
                        \textcolor[HTML]{C00000}{\textbf{eop:processorVersion}} \textbf{Processor version}
                        \textcolor[HTML]{C00000}{\textbf{eop:processingLevel}} \textbf{Processing level}
                        \textcolor[HTML]{C00000}{\textbf{eop:auxiliaryDataSetFileName}} \textbf{Auxiliary DS filename}
                        \textcolor[HTML]{C00000}{\textbf{eop:processingMode}} \textbf{File Class}
                        \textcolor[HTML]{C00000}{\textbf{bio:sourceProduct}} \textbf{BIOMASS source product identifier}
            \textcolor[HTML]{C00000}{\textbf{bio:refDoc}} \textbf{Reference document}
\end{Verbatim}
```

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 2</span><span class="caption-text"> — Main product header L2a (metadata schematic representation)</span></p>

:::

(sec:pfd-fd-4-2)=
### 4.2 Measurement Data Sets (MDS)

The MDS of the L2a product is obtained by processing L1c images at the given Global cycle, along with the processing results from the previous Global cycle and it is composed by three images, each organised in separated Cloud Optimized GeoTIFF (COG) files {rd}`1`:

- the probability of change, being the quality layer, which identifies changes in areas at a given Global cycle; each pixel of the probability raster is a real value from 0 to 1, coded using a 4-byte float per pixel (float32);
- the forest disturbance is a binary classification of intact/deforested areas, coded using a 1-byte per pixel (ubyte);
- the CFM image is a mask computed from the CFM inside L2a product of previous Global Cycle and given as input, or from the FNF in input if it is first Global Cycle; basing on the forest disturbance MDS and coded using a 1-byte per pixel (ubyte).

A Cloud Optimized GeoTIFF (COG) is a regular GeoTIFF file, aimed at being hosted on a

HTTP file server, with an internal organization that enables more efficient workflows on the

cloud. It does this by leveraging the ability of clients issuing HTTP GET range requests to ask for just the parts of a file they need.

The COG format is natively supported by GDAL library {rd}`2` {rd}`3` and by libraries and tools based on GDAL library (Dask and Rasterio packages for Python, …).

The MDS images ground coverage is corresponding to the geocoded spatial stack coverage of input L1c used to compute L2a product: the L2a MDS images are geocoded to the minimum required latitude-longitude coverage to accommodate for the product on ground, following the same sampling requirements specified in {rd}`9`, ensuring a ground sampling always better than the resolution.

**Compression strategy**

MDS is composed by real and binary images: it is possible to introduce with benefit a strategy for data compression making use of native LibTIFF libraries capabilities {rd}`4`; depending on the data type, the strategy will be a GeoTIFF LERC+ZSTD compression for the real data and a ZSTD only (loss-less) for the binary one, in details:

- LERC compression algorithm, for the real data only, can be configured to decide how lossy the compression will be, through the “`MAX_Z_ERROR`” option (loss-less only if set to zero);
- ZSTD compression algorithm, for both data types, with a configurable compression level from 0 to 9 (loss-less), where zero means no compression.

Both set by configuration: more details in sec. {ref}`Tab.19 <tab:pfd-fd-19>`.

**Metadata**

The TIFF and GeoTIFF tags used to annotate the MDS files are described in the following table. For the complete description of each tag refer to official TIFF format documentation {rd}`5` and official GeoTIFF documentation {rd}`11`

(tab:pfd-fd-2)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 2</span><span class="caption-text"> — MDS – TIFF and GeoTIFF tags</span></p>

| **Tag Name** | **Code** | **Value** | **Description** |
|---|---|---|---|
| ImageWidth | 256 | Variable | Number of columns in the image, i.e. the number of pixels per row. |
| ImageLength | 257 | Variable | Number of rows of pixels in the image. |
| BitsPerSample | 258 | 32 or 8 | Number of bits per component. |
| Compression | 259 | 34887, <br> 34926 | Compression scheme used on the image data (LERC and ZSTD) |
| PhotometricInterpretation | 262 | 1 | Colour space of the image. <br> Always set to 1 (minimum value is black). |
| ImageDescription | 270 | Variable | A string that describes the image. Valid Strings are composed by prefix: <br> “BIOMASS L2a FP\_FD\_\_L2A: ” or “BIOMASS L2b FP\_FD\_\_L2B: ” and: <br> “Probability of change” <br> “Computed forest mask” <br> “Forest Disturbance” <br> “Heat Map” (L2b only) <br> "acquisition id image" (L2b only) |
| SamplesPerPixel | 277 | 1 | The number of components per pixel. <br> Always set to 1, since each product is stored in a different file. |
| PlanarConfiguration | 284 | 1 | How the components of each pixel are stored. <br> Always set to 1 (contiguous). |
| Software | 305 | Variable | Software name and version that created the image. |
| DateTime | 306 | Variable | Date and time of image creation in the format: <br> YYYY:MM:DD HH:MM:SS |
| TileWidth | 322 | Variable | The tile width in pixels. <br> This is the number of columns in each tile. |
| TileLength | 323 | Variable | The tile length (height) in pixels. <br> This is the number of rows in each tile. |
| TileOffsets | 324 | Variable | For each tile, the byte offset of that tile, as compressed and stored on disk. |
| TileByteCounts | 325 | Variable | For each tile, the number of (compressed) bytes in that tile. |
| SampleFormat | 339 | 3 or 1 | Specifies how to interpret each data sample in a pixel. <br> Set to 3 (IEEE floating point data) or 1 (unsigned integer data). |
| ModelPixelScaleTag | 33550 | Variable | List of doubles, (ScaleX, ScaleY, ScaleZ) where ScaleX and ScaleY give the horizontal and vertical spacing of raster pixels. The ScaleZ is zero. |
| ModelTiePointTag | 33922 | N \* (pixel, line, 0, lat, long, 0) | Maps image pixels to geographic coordinates, where N is the number of tie points in the scene. |
| GeoKeyDirectoryTag | 34735 | Variable | Contains the value of all GeoTIFF keys of type SHORT and the location of the values for GeoTIFF keys of other types. |
| GeoDoubleParamsTag | 34736 | Variable | Contains the value of all GeoTIFF keys of type DOUBLE. |
| GeoAsciiParamsTag | 34737 | Variable | Contains the value of all GeoTIFF keys of type ASCII. |
| `GDAL_METADATA` | 42112 | Variable | Contains GDAL metadata (see table below) |
| `GDAL_NODATA` | 42113 | “-9999.0” or “255” | Pixel value used in case of invalid data. ASCII string encoded value. (See 3) |
| LercParameters | 50674 | Variable | LERC compression parameters: version, additional compression. <br> e.g. 4,2 for LERC v4 with ZSTD <br> Not present in binary tiffs (FD and CFM) |
| Predictor | 317 | Variable | Integer of value “1” (`PREDICTOR_NONE`), meaning that no prediction scheme is used before coding. Present only in binary tiffs (FD and CFM) |

:::

The `GDAL_METADATA` tag content is listed in the following table

(tab:pfd-fd-3)=
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

(sec:pfd-fd-4-3)=
### 4.3 Annotation data sets (ADS)

(sec:pfd-fd-4-3-1)=
#### 4.3.1 Main ADS

This section describes in details the content of main ADS for L2a products.

An overview is provided by figure and table below.

(fig:pfd-fd-3)=
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

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2a-fd-main-annotation.xsd (<code>mainAnnotationType</code>).</small></p>

:::{div} xsd-diagram-legend
Diagram notation: `@name` attribute; `[min..max]` occurrences; `⊞` named type not expanded here; `«inline»` type declared in place, drawn as an «anonymous» class; `choice n ·` alternative n of a choice; `all ·` elements in any order.
:::

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 3</span><span class="caption-text"> — Main L2a ADS</span></p>

::::

(tab:pfd-fd-4)=
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

(sec:pfd-fd-4-3-1-1)=
##### 4.3.1.1 Product L2a DSR

This DSR contains the parameters to describe the L2a product.

An overview is provided by figure and table below.

(fig:pfd-fd-4)=
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

(tab:pfd-fd-5)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 5</span><span class="caption-text"> — Product L2a DSR (productL2aType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| mission | Mission (BIOMASS). | `missionType` | 1 |
| tileID | List of DGG geographic cell (tile) identifiers. | <u>`stringListType`</u> | 1 |
| basinID | List of DGG Basin identifiers. | <u>`stringListType`</u> | 1 |
| `productType` | Product type (“`FD_L2A`”). | `productType` | 1 |
| `startTime` | Start time of the product image [UTC]. | `timeType` | 1 |
| `stopTime` | Stop time of the product image [UTC]. | `timeType` | 1 |
| `radarCarrierFrequency` | Radar carrier frequency [Hz]. | `doubleWithUnit` | 1 |
| `missionPhaseID` | Mission phase identifier. | `missionPhaseIDType` | 1 |
| `sensorMode` | Sensor mode (always Measurement). | `sensorModeType` | 1 |
| `globalCoverageID` | Global coverage identifier. | `xsd:unsignedShort` | 1 |
| swath | Swath (S1, S2, S3). | `swathType` | 1 |
| `majorCycleID` | Major cycle identifier. | `xsd:unsignedShort` | 1 |
| `absoluteOrbitNumber` | List of absolute orbit numbers at start time (one for each input L1c). | <u>`integerListType`</u> | 1 |
| `relativeOrbitNumber` | Relative orbit number (track) at start time. | `xsd:unsignedShort` | 1 |
| `orbitPass` | Orbit pass (Ascending, Descending). | `orbitPassType` | 1 |
| `dataTakeID` | List of data take identifiers (one for each input L1c). | <u>`integerListType`</u> | 1 |
| frame | Frame identifier. | `xsd:unsignedShort` | 1 |
| `platformHeading` | Platform heading relative to North [deg]. | `doubleWithUnit` | 1 |
| `forestCoveragePercentage` | Forest coverage of the input STA products, compute as a percentage of all the pixels in the STA footprint. | double | 1 |
| `selectedReferenceImage` | Index of the input STA acquisition selected as optimal reference image (guaranteeing the widest scene coverage),used during ground cancellation. <br> Optional element, present only when ground cancellation is performed (FD, GN) and ***`operationalMode`*** configuration has been set to “***single reference***”. | <u>`selectedReferenceImageType`</u> | 0,1 |

:::

(tab:pfd-fd-6)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 6</span><span class="caption-text"> — stringListType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| ID | Tile identifier string | string | 1 .. unbounded |

:::

(tab:pfd-fd-7)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 7</span><span class="caption-text"> — integerListType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| val | Integer value | uint32 | 1 .. unbounded |

:::

(tab:pfd-fd-8)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 8</span><span class="caption-text"> — selectedReferenceImageType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *acquisitionID* | *Acquisition folder name, to better identify the input image referred by the integer value.* | *`xsd:string`* | *1* |
| `selectedReferenceImage` | Zero based Integer value of the selected reference image, based on the image reading order. | `xsd:unsignedByte` | 1 |

:::

(sec:pfd-fd-4-3-1-2)=
##### 4.3.1.2 Raster Image DSR

This DSR contains all the necessary information to exploit the raster images of the product. An overview is provided by figure and table below.

(fig:pfd-fd-5)=
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

(tab:pfd-fd-9)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 9</span><span class="caption-text"> — Raster image DSR (rasterImageType)</span></p>

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
| `noDataValue` | Pixel value in case of invalid data. The value is always set to -9999.0 float32 or 255 ubyte, depending on data type, see {sec}`section 3 <3>`. | <u>`noDataValueChoiceType`</u> | 1 |

:::

(tab:pfd-fd-10)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 10</span><span class="caption-text"> — pixelRepresentationChoiceType (only L2a/L2b FD choice described here)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| FD | Representation of the image pixels within the image MDS: Forest Disturbance | `pixelRepresentationType` | 1 |
| CFM | Representation of the image pixels within the image MDS: Computed forest mask | `pixelRepresentationType` | 1 |
| `probabilityOfChange` | Representation of the image pixels within the image MDS: Probability of change | `pixelRepresentationType` | 1 |
| `FD_heatMap` | Representation of the image pixels within the image MDS: Heat Map (Only L2b) | `pixelRepresentationType` | 0,1 |
| Acquisition\_Id\_Image | Representation of the image pixels within the image MDS: acquisition\_id\_image (Only L2b) | `pixelRepresentationType` | 0,1 |

:::

(tab:pfd-fd-11)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 11</span><span class="caption-text"> — pixelTypeChoiceType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `floatPixelType` | Data type of output pixels within the image MDS: 32 bit Float | `pixelTypeType` | 1 |
| `intPixelType` | Integer data type of output pixels within the image MDS: 8 bit Unsigned Integer | `pixelTypeType` | 1 |

:::

(tab:pfd-fd-12)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 12</span><span class="caption-text"> — noDataValueChoiceType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `floatNoDataValue` | Pixel value in case of invalid data. The value is always set to <br> -9999.0 | `xsd:float` | 1 |
| `intNoDataValue` | Pixel value in case of invalid data. The value is always set to 255 | `xsd:unsignedByte` | 1 |

:::

(sec:pfd-fd-4-3-1-3)=
##### 4.3.1.3 Input Information DSR

This DSR contains information about the L1c acquisitions in input to the L2a processor.

An overview is provided by figures and tables below.

(fig:pfd-fd-6)=
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

(fig:pfd-fd-7)=
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

(fig:pfd-fd-8)=
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

(fig:pfd-fd-9)=
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

(fig:pfd-fd-10)=
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

(fig:pfd-fd-11)=
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

(tab:pfd-fd-13)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 13</span><span class="caption-text"> — Input Information DSR (InputInformationL2aType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `productType` | Input product type: always “STA” (L1c STA). | `productType` | 1 |
| `overallProductsQualityIndex` | Quality index based on the "`overallProductQualityIndex`" of the single STA input data, giving a qualitative overall idea of the input stack. Equal to "0" if all STA feature "`overallProductQualityIndex`"=False and to "1" if at least one is degraded. | `xsd:unsignedByte` | 1 |
| `nominalStack` | True if it is a nominal stack (3 STA products for INT phase or 7 STA products for TOM phase), False otherwise | bool | 1 |
| `polarisationList` | List of polarisations composing input L1c data: for each polarisation it is also specified, with an attribute, the decorrelation factors due to RFI strength. | <u>`polarisationListType`</u> | 1 |
| projection | Projection of the input acquisitions, always “Slant Range”. | `projectionType` | 1 |
| footprint | Image footprint of all the acquisitions composing the stack, expressed as a list of latitude, longitude tuples in NE, SE, SW, NW order [deg]. | `floatArrayWithUnits` | 1 |
| `verticalWavenumbers` | Minimum and maximum values of the vertical wavenumbers [rad/m] in the read and used Input STA products. | `minMaxTypeWithUnit` | 1 |
| `heightOfAmbiguity` | Minimum and maximum values of the height of ambiguity (HoA) [m], computed from the vertical wavenumbers in the read and used Input STA products. | `minMaxTypeWithUnit` | 1 |
| `acquisitionList` | Info about acquisition of the stack. | `acquisitionType` | 1 |

:::

The `acquisitionListType` is composed by 2 to 8 entries, where each entry is an L1c acquisition folder name string, as defined in {rd}`10` (or a couple of images for the images pair selection for ground cancellation, 4.3.1.4)

(tab:pfd-fd-14)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 14</span><span class="caption-text"> — Input information DSR (acquisitionListType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *count* | *Number of acquisitions.* | *`unsignedInt`* | *required* |
| acquisition | Info about acquisition of the stack. | `acquisitionType` | 2..8 |

:::

(tab:pfd-fd-15)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 15</span><span class="caption-text"> — Input information DSR (acquisitionType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *`referenceImage`* | *Flag to indicate if this is the reference image or not.* | *bool* | *1* |
| *`averageWavenumber`* | *Average Wavenumber* | *float* | *1* |
| FolderName | Folder name which univocally identifies an acquisition of the stack. | `xsd:string` | 1 |
| `staQuality` | <u>Folder name which univocally identifies an acquisition of the stack, as defined in [RD10]</u> | <u>`staQualityType`</u> | 1 |

:::

(tab:pfd-fd-16)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 16</span><span class="caption-text"> — Input information DSR (staQualityType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `overallProductQualityIndex` | Overall product quality index. This annotation is calculated based on specific quality parameters and gives an overall quality value to the product. Equal to 0 for valid products and to 1 for invalid ones. | `xsd:unsignedInt` | 1 |
| `staQualityParametersList` | Quality parameters list. The list contains an entry for each output product’s polarisation. | <u>`staQualityParametersListType`</u> | 1 |

:::

(tab:pfd-fd-17)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 17</span><span class="caption-text"> — STA quality DSR (staQualityParametersListType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *count* |   | *`xsd:unsignedInt`* | *required* |
| `staQualityParameters` | Quality parameters for the current polarisation. | <u>`staQualityParametersType`</u> | 1 .. 4 |

:::

(tab:pfd-fd-18)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 18</span><span class="caption-text"> — STA quality DSR (staQualityParametersType)</span></p>

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

(tab:pfd-fd-19)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 19</span><span class="caption-text"> — Input information DSR (polarisationListType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *count* |   | *`xsd:unsignedInt`* | *required* |
| polarisation | Polarisation (HH, VH, VV). | `polarisationType` | 3 |

:::

(sec:pfd-fd-4-3-1-4)=
##### 4.3.1.4 Processing parameters L2a DSR

This DSR contains the exhaustive list of L2a Forest Disturbance processing parameters applied. An overview is provided by figure and table below.

(fig:pfd-fd-12)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 549px">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "state": {"useMaxWidth": false}}}%%
classDiagram
  direction LR
  class processingParametersL2aType {
    <<complexType>>
    processorVersion : string [1]
    productGenerationTime : timeType [1]
    generalConfigurationParameters : generalConfigurationParametersType [1] ⊞
    emphasizedForestHeight : floatWithUnit [1] @units
    operationalMode : operationalModeType [1]
    imagesPairSelection : acquisitionListType [0..1] ⊞
    disableGroundCancellationFlag : bool [0..1]
    significanceLevel : float [1]
    productResolution : floatWithUnit [1] @units
    numericalDeterminantLimit : float [1]
    upsamplingFactor : unsignedByte [1]
    compressionOptions : compressionOptionsL2a [1] ⊞
  }
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-fd-proc-annotations.xsd (<code>processingParametersL2aType</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 12</span><span class="caption-text"> — Processing parameters L2a DSR (processingparametersL2aType)</span></p>

:::

(tab:pfd-fd-20)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 20</span><span class="caption-text"> — Processing parameters L2a DSR (processingparametersL2aType</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `processorVersion` | Version of the processor used to generate the product. | `xsd:string` | 1 |
| `productGenerationTime` | Product generation time [UTC]. | `timeType` | 1 |
| `generalConfigurationParameters` | General processing flags (not specific to the Forest Disturbance processing). | <u>`generalConfigurationParametersType`</u> | 1 |
| `emphasizedForestHeight` | Value of the height [m] which has been emphasized during ground cancellation {ad}`8`. | `floatWithUnit` | 1 |
| `operationalMode` | Ground Cancellation method used between the followings. <br> “single reference”: direct computation with a preliminary automatic reference image selection. <br> *or* <br> “insar pair”: debug operational mode, perform computation with only the two images specified in the optional element `imagesPairSelection`. <br> Note: <br> in case of only two images available, the `operationalMode` is automatically set to “insar pair” (without the need of `imagesPairSelection` element). | <u>`operationalModeType`</u> | 1 |
| `imagesPairSelection` | Optional: <br> if *`operationalMode`* is “insar pair” and if this element is present, this means that ground cancellation has being performed using only the two images specified here. | `acquisitionListType` | 0..1 |
| `disableGroundCancellationFlag` | False, if the ground cancellation has been performed. <br> True, if ground cancellation has been disabled. <br> Optional, default is False. | bool | 0..1 |
| `significanceLevel` | Significance level, expressed as a percentage, applied in the change detection algorithm {ad}`8`. | `xsd:float` | 1 |
| `productResolution` | Value in [m] used as the resolution on ground grange map and also to perform the covariance averaging in radar coordinates. | `floatWithUnit` | 1 |
| `numericalDeterminantLimit` | Lower threshold to avoid numerical errors when computing pol covariance matrices determinants. Positive float number, | `xsd:float` | 1 |
| `compressionOptions` | Configurable compression options for all the L2a MDS COG and ADS NetCDF LUT variables (see 4.3.2 for LUT “ZLIB” compression details). | <u>`compressionOptionsL2aType`</u> | 1 |
| `upsamplingFactor` | Upsampling factor used during covariance computation. | `xsd:unsignedByte` | 1 |

:::

(fig:pfd-fd-13)=
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

(tab:pfd-fd-21)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 21</span><span class="caption-text"> — Processing parameters L2a DSR (generalConfigurationParametersType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| `applyCalibrationScreen` | Phase calibration performed: <br> “none”: no phase screen has been applied <br> “geometry”: only flattening phase screen has been applied (i.e., as computed from acquisition geometry) <br> “skp”: complete phase screen has been applied (default) | `calibrationScreenType` | 1 |
| `forestCoverageThreshold` | Minimum percentage forest coverage in L2a product footprint, used to trigger L2a processing. | `xsd:float` | 1 |
| `forestMaskInterpolationThreshold` | This parameter is a threshold which has been used to fix rounding of pixels with decimal values originated from binary FNF interpolation onto L2a grid. <br> This created a safety buffer around forest border. | `xsd:float` | 1 |
| `subsettingRule` | In case of more than three acquisitions the user selected subsetting rule is here annotated (always “geometry” for FD L2a processor). | <u>`subsettingRuleType`</u> | 1 |
| `polarisationCombinationMethod` | Polarisations combination method: “HV” or “VH”, if just one of the two is selected (in addition to HH and VV ones); “Average”, if the average of HV and VH is computed and used; “None” if no combination is performed (all the four polarizations are used). | `polarisationCombinationMethodType` | 1 |

:::

(fig:pfd-fd-14)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

<div class="xsd-diagram" style="overflow-x: auto">
<div style="min-width: 1157px">

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
    FD : «inline» [1]
    probabilityOFChange : «inline» [1]
    CFM : «inline» [1]
  }
  class compressionOptionsL2a_ADS["compressionOptionsL2a / ADS"] {
    <<anonymous>>
    FNF : «inline» [1]
    ACM : «inline» [1]
    numberOfAverages : «inline» [1]
    geometricLUTs : «inline» [1]
  }
  class compressionOptionsL2a_MDS_FD["compressionOptionsL2a / MDS / FD"] {
    <<anonymous>>
    compressionFactor : unsignedByte [1]
  }
  class compressionOptionsL2a_MDS_probabilityOFChange["compressionOptionsL2a / MDS / probabilityOFChange"] {
    <<anonymous>>
    compressionFactor : unsignedByte [1]
    MAX_Z_ERROR : float [1]
  }
  class compressionOptionsL2a_MDS_CFM["compressionOptionsL2a / MDS / CFM"] {
    <<anonymous>>
    compressionFactor : unsignedByte [1]
  }
  class compressionOptionsL2a_ADS_FNF["compressionOptionsL2a / ADS / FNF"] {
    <<anonymous>>
    compressionFactor : unsignedByte [1]
  }
  class compressionOptionsL2a_ADS_ACM["compressionOptionsL2a / ADS / ACM"] {
    <<anonymous>>
    compressionFactor : unsignedByte [1]
    least_significant_digit : unsignedByte [1]
  }
  class compressionOptionsL2a_ADS_numberOfAverages["compressionOptionsL2a / ADS / numberOfAverages"] {
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
  compressionOptionsL2a_MDS --> compressionOptionsL2a_MDS_FD : FD
  compressionOptionsL2a_MDS --> compressionOptionsL2a_MDS_probabilityOFChange : probabilityOFChange
  compressionOptionsL2a_MDS --> compressionOptionsL2a_MDS_CFM : CFM
  compressionOptionsL2a_ADS --> compressionOptionsL2a_ADS_FNF : FNF
  compressionOptionsL2a_ADS --> compressionOptionsL2a_ADS_ACM : ACM
  compressionOptionsL2a_ADS --> compressionOptionsL2a_ADS_numberOfAverages : numberOfAverages
  compressionOptionsL2a_ADS --> compressionOptionsL2a_ADS_geometricLUTs : geometricLUTs
```

</div>
</div>

<p class="xsd-diagram-source"><small>Generated from biomass-xsd v4.5.0, bio-l2l3-fd-proc-annotations.xsd (<code>compressionOptionsL2a</code>).</small></p>

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 14</span><span class="caption-text"> — Processing parameters L2a DSR (compressionOptionsL2aType)</span></p>

:::

(tab:pfd-fd-22)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 22</span><span class="caption-text"> — Processing parameters L2a DSR (compressionOptionsL2aType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| MDS → `probabilityOfChange` → `compressionFactor` | `LERC_ZSTD` algorithm compression factor for the probability of change image MDS. From 0 to 9, where zero means no compression. | `xsd:unsignedByte` | 1 |
| MDS → `probabilityOfChange` → `MAX_Z_ERROR` | For the probability of change image MDS, define exactly how lossy the LERC compression algorithm is allowed to be, specifying the absolute maximum error admitted. <br> Zero means loss-less compression. | `xsd:float` | 1 |
| MDS → FD → `compressionFactor` <br> and <br> MDS → CFM → `compressionFactor` | ZSTD algorithm compression factor for the FD and CFM binary images MDS. From 0 to 9, where zero means no compression. | `xsd:unsignedByte` | 1 |
| ADS →FNF → `compressionFactor` <br> and <br> ADS →`numberOfAverages` → `compressionFactor` | ZLIB algorithm compression factor for the FNF and `numberOfAverages` ADS. From 0 to 9, where zero means no compression. | `xsd:unsignedByte` | 1 |
| ADS → ACM → `compressionFactor` | ZLIB algorithm compression factor valid for all the ACM ADS LUT layers. From 0 to 9, where zero means no compression. | `xsd:unsignedByte` | 1 |
| ADS → ACM → least\_significant\_digit | For all the layers of ACM LUT ADS, define exactly how lossy the ZLIB compression algorithm is allowed to be, specifying the power of ten of the smallest decimal place in the data that is a reliable value. <br> See 4.3.2. | `xsd:unsignedByte` | 1 |
| ADS → GeometricLuts → `compressionFactor` | ZLIB algorithm compression factor valid for all the Geometric LUTs ADS layers. From 0 to 9, where zero means no compression. | `xsd:unsignedByte` | 1 |
| ADS → GeometricLuts → least\_significant\_digit | For all Geometric LUTs ADS, define exactly how lossy the ZLIB compression algorithm is allowed to be, specifying the power of ten of the smallest decimal place in the data that is a reliable value. | `xsd:unsignedByte` | 1 |
| `MDS_blockSize` | MDS COG blocking algorithm size. Same value is used for both data array dimensions. | `xsd:unsignedByte` | 1 |
| `ADS_blockSize` | ADS NetCDF chunking algorithm size. Same value is used for both data array dimensions. | `xsd:unsignedByte` | 1 |

:::

(sec:pfd-fd-4-3-1-5)=
##### 4.3.1.5 Annotation LUT DSR

This DSR contains the list of Look-Up Tables (LUTs) complementing product main annotations. An overview is provided by figure and table below.

(fig:pfd-fd-15)=
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

(tab:pfd-fd-23)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 23</span><span class="caption-text"> — Annotation LUT DSR (layerListType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| *count* | *Number of layers.* | *`xsd:unsignedInt`* | *required* |
| layer | LUT layer content. | `layerType` | 1 .. unbounded |

:::

(sec:pfd-fd-4-3-2)=
#### 4.3.2 Look-Up Tables (LUT) ADS

This section describes in details the content of Look-Up Tables (LUT) ADS.

This ADS complements the Main ADS one, including those annotations that can be represented as 2D maps of values. This approach has been chosen in order to avoid worsening the Main ADS readability and handling, and also to provide an easier access to such information.

The format adopted for LUT ADS is NetCDF one {rd}`6` and in particular the used implementation is the Python NetCDF4 {rd}`7`. Thanks to the flexibility provided by this format, all the components of this ADS are stored in the same file.

**Complex images disk storage details**

Differently from other SAR missions (and from the standard way of storing SAR data), for the BIOMASS mission, complex data are stored separating amplitude and phase values of each pixel instead of its real and imaginary parts. In particular for FD, the ACM ADS is a matrix of 6 images of which three real and three complex, so in total there will be 9 layers, because the three complex images will be separated each in two different NetCDF layers:

- one layer with the amplitude values of each pixel;
- one layer with the phase values (radians) of each pixel.

The main rationale behind this choice stands in the products usability. Amplitude and phase representation has in fact several advantages from the users’ point of view:

- ease image visualization on a map, either using a data browser on a cloud computing platform or through a GIS software on a local PC, and the generation of overviews;
- run applications that exploit the amplitude only;
- run fast workflows on a cloud computing platform on a custom subset of data;
- speed-up applications working on all the polarisations together;

and more.

The LUT ADS provides:

- external Forest/Non-Forest mask (generated outside of BPS {ad}`6` {ad}`7` {ad}`8` {ad}`9`).
- ACM, averaging of the computed covariance from input L1c data stack at the given Global cycle with the previous cycle ACM from L2a given in input, with a total of 9 layers needed to store the three real images and the three complex images which are separated in amplitude and phase layers.
- Number of averages, the number of ACM covariances averaged so far at given Global cycle, pixel wise.

{fig}`16` provides a schematic representation of the LUT ADS structure. Note that an XSD diagram has been used for this purpose, even if there is no link between the two formats. The ADS is organised as follows:

- a series of root level *Attributes*, containing all those global parameters referring to the product as a whole;
- a *Dimensions* group, where *names* and *lengths* of the dimensions used when creating data arrays are here defined;
- a series of *Coordinate variables*, special variables to store the axis data values, for each dimension in the *Dimensions* group;
- a series of components (groups of variables), one for each LUT stored in the ADS. Each component is composed by a group of variables which can be divided in:
  - the data arrays, storing the annotation Look-Up Tables themselves. A variable represents an array of values of the same data type and it is defined over two or more dimensions from the *Dimensions* group;
  - ancillary scalar variables containing other useful metadata to complete the LUT definition.

(fig:pfd-fd-16)=
::::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

```{mermaid}
%%{init: {"flowchart": {"useMaxWidth": false}, "class": {"useMaxWidth": false}, "theme": "default"}}%%
classDiagram
  accTitle: Look-Up Tables (LUT) ADS
  accDescr: XML schema diagram. The element lutAnnotation has the type lutAnnotationType, an extension. Its sequence holds Attributes (18), Dimensions (2), Coordinate_variables (2), FNF, ACM and numberOfAverages. FNF has the type fnf_variables with 11 Variable elements, ACM has the type ACM_variables with 19, and numberOfAverages has the type numberOfAverages_variables with 11.
  direction TB
  %% A number alone under a stacked (double-frame) XMLSpy element is its occurrence count
  %% (minOccurs = maxOccurs), written [18], [2], [11], [19]; it is not a child count.
  %% Types of Attributes, Dimensions, Coordinate_variables and Variable are not shown: none written.
  class lutAnnotation
  class lutAnnotationType["lutAnnotationType (extension)"] {
    <<complexType>>
    Attributes [18]
    Dimensions [2]
    Coordinate_variables [2]
    FNF : fnf_variables [1]
    ACM : ACM_variables [1]
    numberOfAverages : numberOfAverages_variables [1]
  }
  class fnf_variables {
    <<complexType>>
    Variable [11]
  }
  class ACM_variables {
    <<complexType>>
    Variable [19]
  }
  class numberOfAverages_variables {
    <<complexType>>
    Variable [11]
  }
  lutAnnotation --> lutAnnotationType
  lutAnnotationType --> fnf_variables : FNF
  lutAnnotationType --> ACM_variables : ACM
  lutAnnotationType --> numberOfAverages_variables : numberOfAverages
```

<!-- source image: figure_16_section_4.3.2 sha256 3e5d3b1e5ebb9c25bc33c3362bd26a1f612b8d26e36ea9e71ed1e1e8517276de -->

:::{div} xsd-diagram-legend
Diagram notation: `@name` attribute; `[min..max]` occurrences; `⊞` named type not expanded here; `«inline»` type declared in place, drawn as an «anonymous» class; `choice n ·` alternative n of a choice; `all ·` elements in any order.
:::

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 16</span><span class="caption-text"> — Look-Up Tables (LUT) ADS</span></p>

::::

**Compression strategy**

ACM LUT ADS data variable is composed by real float values, FNF is binary integer and `numberOfAverages` is integer. Depending on data type it is possible to introduce with benefit the following strategy for data compression making use of native Python NetCDF4 capabilities {rd}`7`, using the ZLIB compression algorithm and featuring:

- for binary data (FNF) and integer data (`numberOfAverages`) a compression level from 0 to 9 (loss-less), where zero means no compression;
- for real float data variable (ACM) a compression level from 0 to 9 and an additional option “*least\_significant\_digit*” to decide how lossy the compression will be.

All set by configuration, more details in sec. {ref}`Tab.19 <tab:pfd-fd-19>`.

(sec:pfd-fd-4-3-2-1)=
##### 4.3.2.1 Look-Up tables detailed content

The content of L2a LUT ADS is reported hereafter, where:

- the series of *Attributes* contain the same list of parameters described in {sec}`Sec. 4.3.1.1 <4.3.1.1>` for the Product L2a DSR (for all the details about these parameters refer to this section);
- the *Dimensions* group is composed by Latitude and Longitude, as the data array variables are defined in the Latitude Longitude based on DGG projection;
- there are three components, FNF, ACM and `numberOfAverages` matrices
- for the geometric LUTs (local incidence angle, height, latitude, longitude, incidence/elevation angles, range/azimuth terrain slopes), the repeated variables are described only once.
- this notation is applicable:
  - **this is the attribute or variable name**
  - this is the actual content of constant values over the mission
  - *this is a description of values content which depends on the processing*

<u>LUT ADS Attributes</u>:

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        <b>mission</b>: “BIOMASS”
        <b>t</b><b>ileID</b>: <i>[</i><i>string] json encoded list of </i><i>DGG geographic tile identifiers</i><i> (</i><i>NxxEyyy)</i><i>)</i>
        <b>b</b><b>asinID</b>: <i>[</i><i>string] json encoded list of </i><i>DGG Basin identifier</i><i>s</i><i> </i>
        <b>pro</b><b>d</b><b>uctType</b>: “FP_FD__L2A”
        <b>startTime</b>: <i>[str: YYYY-MM-DD hh:mm:ss.ppp] Start time of the product image, UTC</i>
        <b>stopTime</b><i>: [str: YYYY-MM-DD hh:mm:ss.ppp] Stop time of the product image, UTC</i> 
        <b>radarCarrierFrequency</b>: <i>[float</i><i>32</i><i>] Radar carrier frequency [Hz]</i>
        <b>missionPhaseID</b>: <i>[str: COM, TOM or INT] Mission phase identifier</i>
        <b>sensorMode</b>: “Measurement”
        <b>globalCoverageID</b>: <i>[uint16] Global coverage identifier</i>
        <b>swath</b>: <i>[str: S1, S2 or S3] BIOMASS swath identifier</i>
        <b>majorCycleID</b>: <i>[uint16] Major cycle identifier</i>
        <b>absoluteOrbitNumber</b>: <i>[list of uint32] List of absolute orbit numbers at start time (one for each input L1c)</i>
        <b>relativeOrbitNumber</b>: <i>[uint16] Relative orbit number (track) at start time</i>
        <b>orbitPass</b><i>: [str: Ascending or Descending] Orbit pass</i>
        <b>dataTakeID</b><b>:</b><i> </i><i>[string] json encoded list of </i><i>data take identifiers (one for each input L1c)</i><i> </i>
        <b>frame</b>: <i>[uint16] Frame identifier</i>
        <b>platform</b><b>_h</b><b>eading</b>: <i>[</i><i>float32</i><i>] Platform heading relative to North, deg</i>
        <b>forest</b><b>_c</b><b>overage</b><b>_p</b><b>ercentage</b>: <i>[</i><i>float32</i><i>] </i><i>Forest coverage of the input STA products, compute as a percentage of all the pixels in the </i>        <i>STA footprint</i>
        <b>selected</b><b>_r</b><b>eference</b><b>_i</b><b>mage</b><b> </b><i>Optional</i><i> </i><i>[uint16]</i><i> </i><i>Index of the input STA acquisition selected as optimal reference image (guaranteeing the widest scene coverage), used during ground cancellation. Optional element, present only when ground cancellation is performed (FD, GN) and  operationalMode configuration has been set to “single reference”</i>
        <b>acquisition_id_reference_image</b><b> </b><i>Optional</i><i> </i><i>[</i><i>str</i><i>]</i><i> Acquisition folder name, to better identify the input image referred by the </i><i>selectedReferenceImag</i><i>e</i></pre></div></div>
```

```{raw} latex
\begin{Verbatim}[commandchars=\\\{\},fontsize=\small]
        \textbf{mission}: “BIOMASS”
        \textbf{t}\textbf{ileID}: \textit{[}\textit{string] json encoded list of }\textit{DGG geographic tile identifiers}\textit{ (}\textit{NxxEyyy)}\textit{)}
        \textbf{b}\textbf{asinID}: \textit{[}\textit{string] json encoded list of }\textit{DGG Basin identifier}\textit{s}\textit{ }
        \textbf{pro}\textbf{d}\textbf{uctType}: “FP_FD__L2A”
        \textbf{startTime}: \textit{[str: YYYY-MM-DD hh:mm:ss.ppp] Start time of the product image, UTC}
        \textbf{stopTime}\textit{: [str: YYYY-MM-DD hh:mm:ss.ppp] Stop time of the product image, UTC} 
        \textbf{radarCarrierFrequency}: \textit{[float}\textit{32}\textit{] Radar carrier frequency [Hz]}
        \textbf{missionPhaseID}: \textit{[str: COM, TOM or INT] Mission phase identifier}
        \textbf{sensorMode}: “Measurement”
        \textbf{globalCoverageID}: \textit{[uint16] Global coverage identifier}
        \textbf{swath}: \textit{[str: S1, S2 or S3] BIOMASS swath identifier}
        \textbf{majorCycleID}: \textit{[uint16] Major cycle identifier}
        \textbf{absoluteOrbitNumber}: \textit{[list of uint32] List of absolute orbit numbers at start time (one for each input L1c)}
        \textbf{relativeOrbitNumber}: \textit{[uint16] Relative orbit number (track) at start time}
        \textbf{orbitPass}\textit{: [str: Ascending or Descending] Orbit pass}
        \textbf{dataTakeID}\textbf{:}\textit{ }\textit{[string] json encoded list of }\textit{data take identifiers (one for each input L1c)}\textit{ }
        \textbf{frame}: \textit{[uint16] Frame identifier}
        \textbf{platform}\textbf{_h}\textbf{eading}: \textit{[}\textit{float32}\textit{] Platform heading relative to North, deg}
        \textbf{forest}\textbf{_c}\textbf{overage}\textbf{_p}\textbf{ercentage}: \textit{[}\textit{float32}\textit{] }\textit{Forest coverage of the input STA products, compute as a percentage of all the pixels in the }        \textit{STA footprint}
        \textbf{selected}\textbf{_r}\textbf{eference}\textbf{_i}\textbf{mage}\textbf{ }\textit{Optional}\textit{ }\textit{[uint16]}\textit{ }\textit{Index of the input STA acquisition selected as optimal reference image (guaranteeing the widest scene coverage), used during ground cancellation. Optional element, present only when ground cancellation is performed (FD, GN) and  operationalMode configuration has been set to “single reference”}
        \textbf{acquisition_id_reference_image}\textbf{ }\textit{Optional}\textit{ }\textit{[}\textit{str}\textit{]}\textit{ Acquisition folder name, to better identify the input image referred by the }\textit{selectedReferenceImag}\textit{e}
\end{Verbatim}
```

`formatVersion` *[str]* *Version of the product*

<u>LUT ADS Dimensions / Coordinate variables</u>:

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        Name: <b>Latitude</b>
                size: <i>[int] size</i>
                type: dtype(float32)
                units: deg
                description: LUTs latitude axis
        Name: <b>Longitude</b>
                size<i>: [int] size</i>
                type: dtype(float32)
                units: deg
                description: LUTs longitude axis</pre></div></div>
```

```{raw} latex
\begin{Verbatim}[commandchars=\\\{\},fontsize=\small]
        Name: \textbf{Latitude}
                size: \textit{[int] size}
                type: dtype(float32)
                units: deg
                description: LUTs latitude axis
        Name: \textbf{Longitude}
                size\textit{: [int] size}
                type: dtype(float32)
                units: deg
                description: LUTs longitude axis
\end{Verbatim}
```

<u>LUT ADS group of variables</u>:

```{raw} html
<div class="highlight-text notranslate"><div class="highlight"><pre class="atbd-text-tree">        Group Name: <b>FNF</b>
        FNF group contents:
                Name: <b>FNF</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(ubyte)
                        description: Forest/Non-Forest mask: this is an external image, defined outside of BPS from aggregated C3S LCM, representing the initial two-dimensional mask, here defined in the same latitude-longitude based on DGG projection used for L2a images and cropped to have a coverage containing the L2a product image boundaries. It defines, for each geographical pixel, the initial presence or absence of forest.
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
                        description: LUT projection, is the latitude-longitude based on DGG
                Name: <b>coordinateReferenceSystem</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: <b>geodeticReferenceFrame</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: Geodetic reference frame
        Group Name: <b>ACM</b>
                group description: ACM represents the covariance history of all previous cycles (see also <i>numberOfAverages</i> variable): it is obtained by averaging the covariance computed with FD algorithm at the given Global cycle with the previous ACM in input to the processor [AD8]. At first Global cycle it is the current covariance. It consists of six images of which three real and three complex, stored in nine NetCDF layers as three layers for the three real images and three complex images, each split in the amplitude value layer and the phase value (radians) layer.
        ACM group contents:
                Name: <b>layer1</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: layer 1: ACM image 1 of 6, real, HH-HH
                Name: <b>layer2</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: layer 2: ACM image 2 of 6, amplitude value of complex image, HH-XP
                Name: <b>layer3</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: layer 3: ACM image 2 of 6, phase value of complex image [rad], HH-XP
     units: rad
                Name: <b>layer4</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: layer 4: ACM image 3 of 6, amplitude value of complex image, HH-VV 
                Name: <b>layer5</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: layer 5: ACM image 3 of 6, phase value of complex image [rad], HH-VV
      units: rad
                Name: <b>layer6</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: layer 6: ACM image 4 of 6, real, XP-XP
                Name: <b>layer7</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: layer 7: ACM image 5 of 6, amplitude value of complex image, XP-VV
                Name: <b>layer8</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: layer 8: ACM image 5 of 6, phase value of complex image [rad], XP-VV
     units: rad
                Name: <b>layer9</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(float32)
                        description: layer 9: ACM image 6 of 6, real, VV-VV
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
                        description: LUT projection, is the latitude-longitude based on DGG
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
        Group Name: <b>numberOfAverages</b>
        numberOfAverages group contents:
                Name: <b>numberOfAverages</b>
                        dimensions: (Longitude, Latitude)
                        shape: <i>([int] x shape, [int] y shape)</i>
                        type: dtype(ubyte)
                        description: It represents, for the ACM, the number of covariances averaged so far at given Global cycle, pixel wise (see also <i>ACM</i> variable and [AD8]). At first Global cycle it is zero for each pixel.
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
                        description: Pixel value in case of invalid data. The value is always set to float 255. (See 3)
                Name: <b>projection</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: LUT projection, is the latitude-longitude based on DGG
                Name: <b>coordinateReferenceSystem</b>
                        shape: (1, 1)
                        type: &lt;class str&gt;
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: <b>geodeticReferenceFrame</b>
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

```{raw} latex
\begin{Verbatim}[commandchars=\\\{\},fontsize=\small]
        Group Name: \textbf{FNF}
        FNF group contents:
                Name: \textbf{FNF}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(ubyte)
                        description: Forest/Non-Forest mask: this is an external image, defined outside of BPS from aggregated C3S LCM, representing the initial two-dimensional mask, here defined in the same latitude-longitude based on DGG projection used for L2a images and cropped to have a coverage containing the L2a product image boundaries. It defines, for each geographical pixel, the initial presence or absence of forest.
                Name: \textbf{firstSample}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS sample the first LUT sample corresponds to
                Name: \textbf{firstLine}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS line the first LUT line corresponds to
                Name: \textbf{samplesInterval}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS samples one LUT sample corresponds to
                Name: \textbf{linesInterval}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS lines one LUT line corresponds to
                Name: \textbf{pixelType}
                        shape: (1, 1)
                        type: <class str>
                        description: Data type of output LUT pixels
                Name: \textbf{noDataValue}
                        shape: (1, 1)
                        type: dtype(ubyte)
                        description: Pixel value in case of invalid data. The value is always set to 255. (See 3)
                Name: \textbf{projection}
                        shape: (1, 1)
                        type: <class str>
                        description: LUT projection, is the latitude-longitude based on DGG
                Name: \textbf{coordinateReferenceSystem}
                        shape: (1, 1)
                        type: <class str>
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: \textbf{geodeticReferenceFrame}
                        shape: (1, 1)
                        type: <class str>
                        description: Geodetic reference frame
        Group Name: \textbf{ACM}
                group description: ACM represents the covariance history of all previous cycles (see also \textit{numberOfAverages} variable): it is obtained by averaging the covariance computed with FD algorithm at the given Global cycle with the previous ACM in input to the processor [AD8]. At first Global cycle it is the current covariance. It consists of six images of which three real and three complex, stored in nine NetCDF layers as three layers for the three real images and three complex images, each split in the amplitude value layer and the phase value (radians) layer.
        ACM group contents:
                Name: \textbf{layer1}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: layer 1: ACM image 1 of 6, real, HH-HH
                Name: \textbf{layer2}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: layer 2: ACM image 2 of 6, amplitude value of complex image, HH-XP
                Name: \textbf{layer3}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: layer 3: ACM image 2 of 6, phase value of complex image [rad], HH-XP
     units: rad
                Name: \textbf{layer4}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: layer 4: ACM image 3 of 6, amplitude value of complex image, HH-VV 
                Name: \textbf{layer5}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: layer 5: ACM image 3 of 6, phase value of complex image [rad], HH-VV
      units: rad
                Name: \textbf{layer6}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: layer 6: ACM image 4 of 6, real, XP-XP
                Name: \textbf{layer7}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: layer 7: ACM image 5 of 6, amplitude value of complex image, XP-VV
                Name: \textbf{layer8}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: layer 8: ACM image 5 of 6, phase value of complex image [rad], XP-VV
     units: rad
                Name: \textbf{layer9}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: layer 9: ACM image 6 of 6, real, VV-VV
                Name: \textbf{firstSample}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS sample the first LUT sample corresponds to
                Name: \textbf{firstLine}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS line the first LUT line corresponds to
                Name: \textbf{samplesInterval}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS samples one LUT sample corresponds to
                Name: \textbf{linesInterval}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS lines one LUT line corresponds to
                Name: \textbf{pixelType}
                        shape: (1, 1)
                        type: <class str>
                        description: Data type of output LUT pixels
                Name: \textbf{noDataValue}
                        shape: (1, 1)
                        type: dtype(float32)
                        description: Pixel value in case of invalid data. The value is always set to -9999.0. (See 3)
                Name: \textbf{projection}
                        shape: (1, 1)
                        type: <class str>
                        description: LUT projection, is the latitude-longitude based on DGG
                Name: \textbf{coordinateReferenceSystem}
                        shape: (1, 1)
                        type: <class str>
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: \textbf{geodeticReferenceFrame}
                        shape: (1, 1)
                        type: <class str>
                        description: Geodetic reference frame
                Name: \textbf{least_significant_digit}
                        shape: (1, 1)
                        type: unsignedByte
                        description: Power of ten of the smallest decimal place in the data that is a reliable value, adopted to the data during ZLIB compression.
        Group Name: \textbf{numberOfAverages}
        numberOfAverages group contents:
                Name: \textbf{numberOfAverages}
                        dimensions: (Longitude, Latitude)
                        shape: \textit{([int] x shape, [int] y shape)}
                        type: dtype(ubyte)
                        description: It represents, for the ACM, the number of covariances averaged so far at given Global cycle, pixel wise (see also \textit{ACM} variable and [AD8]). At first Global cycle it is zero for each pixel.
                Name: \textbf{firstSample}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS sample the first LUT sample corresponds to
                Name: \textbf{firstLine}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS line the first LUT line corresponds to
                Name: \textbf{samplesInterval}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS samples one LUT sample corresponds to
                Name: \textbf{linesInterval}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS lines one LUT line corresponds to
                Name: \textbf{pixelType}
                        shape: (1, 1)
                        type: <class str>
                        description: Data type of output LUT pixels
                Name: \textbf{noDataValue}
                        shape: (1, 1)
                        type: dtype(ubyte)
                        description: Pixel value in case of invalid data. The value is always set to float 255. (See 3)
                Name: \textbf{projection}
                        shape: (1, 1)
                        type: <class str>
                        description: LUT projection, is the latitude-longitude based on DGG
                Name: \textbf{coordinateReferenceSystem}
                        shape: (1, 1)
                        type: <class str>
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: \textbf{geodeticReferenceFrame}
                        shape: (1, 1)
                        type: <class str>
                        description: Geodetic reference frame
        Group Name: \textbf{ }\textbf{height}
        height group contents:
                Name: \textbf{ }\textbf{height}
                        dimensions: (Longitude, Latitude)
                        shape:\textit{ ([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the height coordinates of each (sub-sampled) pixel, derived from the Digital Elevation Model (DEM) used during processing. Values are expressed in [m]
    units: m
                Name: \textbf{firstSample}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS sample the first LUT sample corresponds to
                Name: \textbf{firstLine}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Product image MDS line the first LUT line corresponds to
                Name: \textbf{samplesInterval}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS samples one LUT sample corresponds to
                Name: \textbf{linesInterval}
                        shape: (1, 1)
                        type: dtype(uint32)
                        description: Number of product image MDS lines one LUT line corresponds to
                Name: \textbf{pixelType}
                        shape: (1, 1)
                        type: <class str>
                        description: Data type of output LUT pixels
                Name: \textbf{noDataValue}
                        shape: (1, 1)
                        type: dtype(float32)
                        description: Pixel value in case of invalid data. The value is always set to -9999.0. (See 3)
                Name: \textbf{projection}
                        shape: (1, 1)
                        type: <class str>
                        description: LUT projection, DGG, as described in [RD8]
                Name: \textbf{coordinateReferenceSystem}
                        shape: (1, 1)
                        type: <class str>
                        description: Coordinate reference systems, as per WKT representation [RD7]
                Name: \textbf{geodeticReferenceFrame}
                        shape: (1, 1)
                        type: <class str>
                        description: Geodetic reference frame
                Name: \textbf{least_significant_digit}
                        shape: (1, 1)
                        type: unsignedByte
                        description: Power of ten of the smallest decimal place in the data that is a reliable value, adopted to the data during ZLIB compression.
        Group Name: \textbf{ }\textbf{latitude}
        height group contents:
                Name: \textbf{ }\textbf{latitude}
                        dimensions: (Longitude, Latitude)
                        shape:\textit{ ([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the latitude coordinates of each (sub-sampled) pixel. Values are expressed in [deg].
    units: deg
\end{Verbatim}
```

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

```{raw} latex
\begin{Verbatim}[commandchars=\\\{\},fontsize=\small]
        Group Name: \textbf{ }\textbf{longitude}
        height group contents:
                Name: \textbf{ }\textbf{longitude}
                        dimensions: (Longitude, Latitude)
                        shape:\textit{ ([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the longitude coordinates of each (sub-sampled) pixel. Values are expressed in [deg]
    units: deg
\end{Verbatim}
```

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

```{raw} latex
\begin{Verbatim}[commandchars=\\\{\},fontsize=\small]
        Group Name: \textbf{ }\textbf{incidenceAngle}
        height group contents:
                Name: \textbf{ }\textbf{incidenceAngle}
                        dimensions: (Longitude, Latitude)
                        shape:\textit{ ([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the incidence angle of each (sub-sampled) pixel. Values are computed during L1 processing considering an average height. Values are expressed in [deg].
    units: deg
\end{Verbatim}
```

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

```{raw} latex
\begin{Verbatim}[commandchars=\\\{\},fontsize=\small]
        Group Name: \textbf{ }\textbf{elevationAngle}
        height group contents:
                Name: \textbf{ }\textbf{elevationAngle}
                        dimensions: (Longitude, Latitude)
                        shape:\textit{ ([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the elevation angle of each (sub-sampled) pixel. Values are computed during L1 processing considering an average height. Values are expressed in [deg].
    units: deg
\end{Verbatim}
```

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

```{raw} latex
\begin{Verbatim}[commandchars=\\\{\},fontsize=\small]
        Group Name: \textbf{ }\textbf{rangeTerrainSlope}
        height group contents:
                Name: \textbf{ }\textbf{rangeTerrainSlope}
                        dimensions: (Longitude, Latitude)
                        shape:\textit{ ([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the local DEM slopes along range direction computed using the Digital Elevation Model (DEM). Values are expressed in [deg].
    units: deg
\end{Verbatim}
```

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

```{raw} latex
\begin{Verbatim}[commandchars=\\\{\},fontsize=\small]
        Group Name: \textbf{ }\textbf{azimuthTerrainSlope}
        height group contents:
                Name: \textbf{ }\textbf{azimuthTerrainSlope}
                        dimensions: (Longitude, Latitude)
                        shape:\textit{ ([int] x shape, [int] y shape)}
                        type: dtype(float32)
                        description: Two-dimensional map covering the geocoded image and describing the local DEM slopes along azimuth direction computed using the Digital Elevation Model (DEM). Values are expressed in [deg].
    units: deg
\end{Verbatim}
```

The remaining variables, which are identical across all geometric LUTs, are omitted from the detailed description (`firstSample`, `firstLine`, `samplesInterval`, `pixelType`, `noDataValue`, projection, `coordinateReferenceSystem`, `geodeticReferenceFrame`, least\_significant\_digit).

(sec:pfd-fd-4-3-3)=
#### 4.3.3 Quick-look ADS

The Quick-look ADS is composed by lower resolution versions of all the MDS images; they are stored in PNG file format, and they are always included with any BIOMASS L2 product.

The quantities are composed by real values, so the images are generated without any other processing than averaging and decimation.

(sec:pfd-fd-4-3-4)=
#### 4.3.4 Overlay ADS

The Overlay ADS is a Keyhole Markup Language (KML) file that describes the product coverage area and is suitable for viewing the Quick-look ADS in any application that supports KML format (like, e.g., Google Earth or GIS applications).

Its structure is described in the following figure and tables.

(fig:pfd-fd-17)=
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

(tab:pfd-fd-24)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 24</span><span class="caption-text"> — Overlay ADS</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| Document | Document container for KML components. | DocumentType | 1 |

:::

(tab:pfd-fd-25)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 25</span><span class="caption-text"> — Overlay ADS (DocumentType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| name | Name of the document. | `xsd:string` | 1 |
| description | Description of the document. | `xsd:string` | 1 |
| GroundOverlay | Contains the parameters required to specify the footprint of the image and overlay the quicklook image on a map. | GroundOverlayType | 1 |
| Placemark | Contains the metadata associated to the overlay. | PlacemarkType | 1 |

:::

(tab:pfd-fd-26)=
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

(tab:pfd-fd-27)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 27</span><span class="caption-text"> — Overlay ADS (TimeStampType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| when | Overlay time stamp in ISO 8601 format (YYYY-MM-DDThh:mm:ss). | `xsd:string` | 1 |

:::

(tab:pfd-fd-28)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 28</span><span class="caption-text"> — Overlay ADS (IconType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| href | A local file specification or URL used to load the desired image and overlay it on the map. | `xsd:string` | 1 |

:::

(tab:pfd-fd-29)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 29</span><span class="caption-text"> — Overlay ADS (LatLonQuadType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| coordinates | A string of 5 lon,lat coordinate pairs which describe the corners of the image. The string is of the form: <br> *lon,lat lon,lat lon,lat lon,lat lon,lat* <br> The coordinates must appear in the following order: <br> *last line first pixel, last line last pixel, first line last pixel, first line first pixel, last line first pixel* | `xsd:string` | 1 |

:::

(tab:pfd-fd-30)=
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

(tab:pfd-fd-31)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 31</span><span class="caption-text"> — Overlay ADS (StyleType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| LyneStyle | Line style. | LyneStyleType | 1 |
| PolyStyle | Polygon style. | PolyStyleType | 1 |

:::

(tab:pfd-fd-32)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 32</span><span class="caption-text"> — Overlay ADS (LyneStyleType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| color | Line color in 32 bit hex format (AABBGGRR order). | `xsd:string` | 1 |
| width | Line width. | `xsd:unsignedInt` | 1 |

:::

(tab:pfd-fd-33)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 33</span><span class="caption-text"> — Overlay ADS (PolyStyleType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| color | Polygon color in 32 bit hex format (AABBGGRR order). | `xsd:string` | 1 |

:::

(tab:pfd-fd-34)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 34</span><span class="caption-text"> — Overlay ADS (ExtendedDataType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| Data | Metadata. | DataType | 1..inf |

:::

(tab:pfd-fd-35)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 35</span><span class="caption-text"> — Overlay ADS (DataType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| value | Metadata value. | `xsd:string` | 1 |

:::

(tab:pfd-fd-36)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 36</span><span class="caption-text"> — Overlay ADS ()</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| tessellate | Tessellate option. | `xsd:unsignedInt` | 1 |
| `altitudeMode` | Altitude mode option. | `xsd:string` | 1 |
| `outerBoundaryIs` | Polygon outer boundary description. | `outerBoundaryIsType` | 1 |

:::

(tab:pfd-fd-37)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 37</span><span class="caption-text"> — Overlay ADS (outerBoundaryIsType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| LinearRing | Linear ring description. | LinearRingType | 1 |

:::

(tab:pfd-fd-38)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 38</span><span class="caption-text"> — Overlay ADS (LinearRingType)</span></p>

| **Name** | **Description** | **Data Type** | **Cardinality** |
|---|---|---|---|
| coordinates | A string of 5 lon,lat,height coordinate tuples which describe the corners of the image. The string is of the form: <br> *lon,lat,height lon,lat,height lon,lat,height lon,lat,height lon,lat,height* <br> The coordinates must appear in the following order: <br> *last line first pixel, last line last pixel, first line last pixel, first line first pixel, last line first pixel* | `xsd:string` | 1 |

:::
