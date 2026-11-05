<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:pfd-l1-7)=
# 7. Appendix: Common Products Format Specification

(sec:pfd-l1-7-1)=
### 7.1 Annotation data sets (ADS)

(sec:pfd-l1-7-1-1)=
#### 7.1.1 Primitive and common types

The following table summarizes and describes primitive product types used throughout the document.

(tab:pfd-l1-90)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 90</span><span class="caption-text"> — Primitive product types</span></p>

| Name | Description | Base Type | Values |
|---|---|---|---|
| `missionType` | Enumeration of valid BIOMASS mission names. | `xsd:string` | BIOMASS |
| `swathType` | Enumeration of all valid swath identifiers. | `xsd:string` | S1 S2 S3 |
| `acquisitionModeIDType` | Enumeration of all valid acquisition modes. | `xsd:string` | S1 INT S2 INT S3 INT S1 TOM S2 TOM S3 TOM Rx Only |
| `productType` | Enumeration of valid product types. | `xsd:string` | SCS DGM STA `FH_L2A` `FH_L2B` `FD_L2A` `FD_L2B` `GN_L2A` `AGB_L2B` `TFH_L2A` |
| `polarisationType` | Enumeration of valid polarisations. | `xsd:string` | H V HH HV VH VV XX |
| `groupType` | Enumeration of valid LUT groups of variables. | `xsd:string` | DEM based LUT RFI based LUT Image based LUT |
| `channelType` | Enumeration of valid RGB channels. | `xsd:string` | RED GREEN BLUE |
| `timeType` | UTC time expressed according to ISO 8601 standard (1 microsecond resolution). | `xsd:dateTime` |   |
| `missionPhaseIDType` | Enumeration of all valid BIOMASS mission phases. | `xsd:string` | COM TOM INT |
| `sensorModeType` | Enumeration of valid sensor mode abbreviations. | `xsd:string` | Measurement RX Only External Calibration |
| l1acQlColourCodingMethodType | Enumeration of L1a and L1c products quick-looks colour coding methods. | `xsd:string` | Pauli HSV Pauli and HSV |
| l1bQlColourCodingMethodType | Enumeration of L1b products quick-looks colour coding methods. | `xsd:string` | Lexicographic |
| `orbitPassType` | Enumeration of the orbit pass direction values. | `xsd:string` | Ascending Descending |
| `projectionType` | Enumeration of the image projection. | `xsd:string` | Slant Range Ground Range |
| `pixelRepresentationType` | Enumeration of image pixel representation types. | `xsd:string` | I Q Abs Phase Abs |
| `pixelTypeType` | Enumeration of image pixel data types. | `xsd:string` | 32 bit Float 16 bit Signed Integer 16 bit Unsigned Integer |
| `pixelQuantityType` | Enumeration of pixel quantity types. | `xsd:string` | Beta-Nought Sigma-Nought Gamma-Nought |
| `compressionMethodType` | Enumeration of TIFF compression methods. | `xsd:string` | NONE DEFLATE ZSTD LERC `LERC_DEFLATE` `LERC_ZSTD` |
| `productCompositionType` | Enumeration of product composition indicators. The valid values are: “Nominal”, to indicate a framed product of nominal length; “Merged”, if the product is resulting from the merging of a “short” frame with the contiguous one; “Partial”, if the product length is not nominal, but it is not merged to a contiguous one since it is “long enough”; “Incomplete”, if the product length is not nominal due to contingency (e.g., data loss); “Not Framed”, if the product is not framed, i.e., as long as input L0 one. | `xsd:string` | Nominal Merged Partial Incomplete Not Framed |
| `dataFormatModeType` | Enumeration of compression method names. | `xsd:string` | BAQ 4 Bit BAQ 5 Bit BAQ 6 Bit Bypass |
| `processingModeType` | Enumeration of processing modes. | `xsd:string` | Nominal PARC |
| `orbitAttitudeSourceType` | Enumeration of value sources of orbit and attitude data. "Downlink" or "Auxiliary". | `xsd:string` | Downlink Auxiliary |
| `rfiMitigationMethodType` | Enumeration of RFI mitigation methods. | `xsd:string` | Time Frequency Time and Frequency Frequency and Time |
| `rfiMaskType` | Enumeration of RFI masks. | `xsd:string` | Single Multiple |
| `rfiMaskGenerationMethodType` | Enumeration of RFI mask generation methods. | `xsd:string` | AND OR |
| `rangeReferenceFunctionType` | Enumeration of the available range reference functions. | `xsd:string` | Nominal Replica Internal |
| `rangeCompressionMethodType` | Enumeration of the available range compression methods. | `xsd:string` | Matched Filter Inverse Filter |
| rfiFMMitigationMethodType | Enumeration of the available RFI Frequency Domain mitigation methods. | `xsd:string` | `NOTCH_FILTER` `NEAREST_NEIGHBOUR_INTERPOLATION` |
| `dcMethodType` | Enumeration of Doppler Centroid calculation/estimation methods. | `xsd:string` | Geometry Combined Fixed |
| `weightingWindowType` | Enumeration of weighting window names. | `xsd:string` | Kaiser Hamming None |
| `bistaticDelayCorrectionMethodType` | Enumeration of bistatic delay correction methods. | `xsd:string` | Bulk Full |
| `ionosphereHeightEstimationMethodType` | Enumeration of ionosphere height estimation methods. | `xsd:string` | Automatic Feature Tracking Squint Sensitivity Model Fixed NA |
| `ionosphereHeightSourceType` | Enumeration of stack ionosphere height sources. | `xsd:string` | None Reference Image Manual |
| `autofocusMethodType` | Enumeration of autofocus methods. | `xsd:string` | Map Drift |
| `internalCalibrationSourceType` | Enumeration of the available internal calibration sources. | `xsd:string` | Extracted Model |
| `heightModelType` Attribute: version | Enumeration of Digital Elevation Models (DEM). Attribute: DEM version. Default values: COP-`DEM_GLO`-90-DGED-2021\_1 `EGM2008-2`.5 for Copernicus DEM (convention: \<format\>-\<delivery name\> \<geoid\>-\<resolution\>) `SRTM_v1` EGM96 for SRTM (convention: SRTM\_\<version\> \<geoid\>) Empty value for Ellipsoid Version value has a one-to-one mapping with the physical folder where DEM to be used has been installed. | `xsd:string` Attribute: `xsd:string` | Ellipsoid SRTM Copernicus DEM BIOMASS DTM |
| `coordinateReferenceSystemType` | Coordinate reference systems, as per WKT representation {rd}`7`. | `xsd:string` |   |
| `geodeticReferenceFrameType` | Enumeration of geodetic reference frames. | `xsd:string` | WGS84 ITRF2000 |
| `polarisationCombinationMethodType` | Enumeration of polarisations combination methods. | `xsd:string` | HV VH Average None |
| `primaryImageSelectionMethodType` | Enumeration of coregistration primary image selection methods. | `xsd:string` | Geometry Geometry and Quality Temporal Baseline |
| `coregistrationMethodType` | Enumeration of coregistration methods. | `xsd:string` | Geometry Geometry and Data Automatic |
| `baselineMethodType` | Enumeration of baseline method types (e.g. SingleBaseline or MultiBaseline). | `xsd:string` | Single-Baseline Multi-Baseline |
| `ionosphereTypeType` | Enumeration of the ionosphere types estimated by the BIC. | `xsd:string` | Linear Fast-varying None |
| `coregistrationExecutionPolicyType` | Enumeration of coregistration execution policies | `xsd:string` | Nominal Shift Estimation Only Warping Only |
| `skpPhaseCorrectionType` | Enumeration of supported corrections executed by SKP. | `xsd:string` | None Flattening Phase Screen Ground Phase Screen |
| `skpPhaseCalibrationPostprocessingType` | Postprocessing methods for the SKP calibration phase screen. | `xsd:string` | None Goldstein Boxcar Median |
| `primaryImageSelectionInformationType` | Enumeration of information used to select coregistration primary image. | `xsd:string` | Geometry Geometry and RFI Correction Geometry and FR Correction Geometry and RFI+FR Correction Temporal Baseline |
| `layerType` | Enumeration of LUT layer contents. | `xsd:string` | Faraday Rotation plane [rad] Faraday Rotation [rad] Faraday Rotation std [rad] Phase screen [rad] TEC [TECU] Range shifts [m] Azimuth shifts [m] Autofocus phase screen [rad] Autofocus phase screen std [rad] RFI time domain mask HH RFI time domain mask HV RFI time domain mask VH RFI time domain mask VV RFI frequency domain mask HH RFI frequency domain mask HV RFI frequency domain mask VH RFI frequency domain mask VV Sigma-nought LUT Gamma-nought LUT Denoising map HH Denoising map HV Denoising map VH Denoising map VV Denoising map XX Latitude [deg] Longitude [deg] Height [m] Incidence angle [deg] Elevation angle [deg] Range terrain slope [deg] Azimuth terrain slope [deg] FNF ACM `numberOfAverages` Azimuth coregistration shifts [m] Azimuth orbit coregistration shifts [m] Range coregistration shifts [m] Range orbit coregistration shifts [m] Coregistration shifts quality Wavenumbers [rad/m] Flattening phase screen [rad] Ionosphere Phase Screens [rad] SKP calibration phase screen [rad] SKP calibration phase screen quality Coherence Interferogram [rad] |
| `uomType` | Enumeration of unit of measures. | `xsd:string` | s UTC m samples lines deg rad Hz Hz/s MHz dB Mbps C K deg/s rad/s mm/Km deg/m rad/m 1/m t/ha (rad/nT)^2 |
| bool | Boolean (true or false). | `xsd:boolean` | (false)\|(true) |

:::

The following table summarizes and describes those primitive product types for which attributes have been added with respect to base types.

(tab:pfd-l1-91)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 91</span><span class="caption-text"> — Primitive product types with custom attributes</span></p>

| Name | Description | Base Type | Attributes |
|---|---|---|---|
| `timeTypeWithPolarisation` | Extension of `timeType` with the indication of the corresponding polarisation. | `timeType` | polarisation |
| `unsignedIntWithGroup` | Extension of `unsignedInt` with the indication of the corresponding LUT group of variables. Attribute type is `groupType` | `xsd:unsignedInt` | group |
| `unitInteger` | Extension of the integer data type to include an optional "units" attribute. | `xsd:integer` | units |
| `unitNonNegativeInteger` | Extension of the `nonNegativeInteger` data type to include an optional "units" attribute. | `xsd:nonNegativeInteger` | units |
| float | 32 bit single precision floating point number. | `xsd:float` | units |
| `floatWithPolarisation` | Extension of float with the indication of the corresponding polarisation. | `xsd:float` | polarisation |
| `floatWithChannel` | Extension of float with the indication of the corresponding RGB channel. | `xsd:float` | channel |
| double | 64 bit double precision floating point number. | `xsd:double` | units |
| `intArray` | String containing an array of int values separated by spaces. The mandatory count attribute defines the number of elements in the array. | `xsd:string` | count |
| `floatArray` | String containing an array of float values separated by spaces. The mandatory count attribute defines the number of elements in the array. | `xsd:string` | count |
| `doubleArray` | String containing an array of double precision floating point values separated by spaces. The mandatory count attribute defines the number of elements in the array. | `xsd:string` | count |
| `floatArray` | String containing an array of float coefficient values separated by spaces. The mandatory count attribute defines the number of elements in the array. | `xsd:string` | count |
| `floatArrayWithUnits` | String containing an array of float values separated by spaces. The mandatory count attribute defines the number of elements in the array. | `xsd:string` | count units |
| `doubleArrayWithUnits` | String containing an array of double precision floating point coefficient values separated by spaces. The mandatory count attribute defines the number of elements in the array. | `xsd:string` | count units |
| `complexArray` | String containing an array of complex value pairs separated by spaces in the form of I Q I Q I Q ... The mandatory count attribute defines the number of complex elements in the array. | `xsd:string` | count |

:::

The following tables describe common product types used throughout the document.

(tab:pfd-l1-92)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 92</span><span class="caption-text"> — Common product types (complex)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| re | 32 bit single precision floating point real number. | `xsd:float` | 1 |
| im | 32 bit single precision floating point imaginary number. | `xsd:float` | 1 |

:::

(tab:pfd-l1-93)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 93</span><span class="caption-text"> — Common product types (stateType)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| `azimuthTime` | Zero Doppler azimuth time of current value [UTC]. | `timeType` | 1 |
| value | Current value. | `floatWithUnit` | 1 |

:::

(tab:pfd-l1-94)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 94</span><span class="caption-text"> — Common product types (slantRangePolynomialType)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| `azimuthTime` | Zero Doppler azimuth time corresponding to the start of the current polynomial validity area [UTC]. | `timeType` | 1 |
| t0 | Two-way slant range time origin for polynomial [s]. | `doubleWithUnit` | 1 |
| polynomial | Local estimate expressed as the following polynomial: `X = c0+c1(tSR-t0)+c2(tSR-t0)^2+...` where `tSR` is the two-way slant range time. | `doubleArray` | 1 |

:::

(tab:pfd-l1-95)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 95</span><span class="caption-text"> — Common product types (azimuthPolynomialType)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| `slantRangeTime` | Two-way slant range time corresponding to the start of the current polynomial validity area [s]. | `doubleWithUnit` | 1 |
| t0 | Zero Doppler azimuth time origin for polynomial [UTC]. | `timeType` | 1 |
| polynomial | Local estimate expressed as the following polynomial: `X = c0+c1(tSR-t0)+c2(tSR-t0)^2+...` where tAZ is the Zero Doppler azimuth time. | `doubleArray` | 1 |

:::

(tab:pfd-l1-96)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 96</span><span class="caption-text"> — Common product types (datumType)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| `coordinateReferenceSystem` | Coordinate reference system. | `coordinateReferenceSystemType` | 1 |
| `geodeticReferenceFrame` | Geodetic reference frame. | `geodeticReferenceFrameType` | 1 |

:::

(tab:pfd-l1-97)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 97</span><span class="caption-text"> — Common product types (minMaxTypeWithUnit)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| min | Minimum value. | `floatWithUnit` | 1 |
| max | Maximum value. | `floatWithUnit` | 1 |

:::

(tab:pfd-l1-98)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 98</span><span class="caption-text"> — Common product types (floatArrayWithUnits)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| count | counter | `xsd:unsignedInt` | 1 |
| units | units | `uomType` | 1 |
| `floatArrayWithUnits` | String containing an array of float values separated by spaces. The mandatory count attribute defines the number of elements in the array. | `xsd:string` | 1 |

:::

(tab:pfd-l1-99)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 99</span><span class="caption-text"> — Common product types (floatWithUnit)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| units | units | `uomType` | 1 |
| `floatWithUnit` | 32 bit single precision floating point number. | `xsd:float` | 1 |

:::

(tab:pfd-l1-100)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 100</span><span class="caption-text"> — Common product types (doubleWithUnit)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| units | units | `uomType` | 1 |
| `doubleWithUnit` | 64 bit double precision floating point number. | xsd: double | 1 |

:::

(tab:pfd-l1-101)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 101</span><span class="caption-text"> — Common product types (doubleArrayWithUnits)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| count | counter | `xsd:unsignedInt` | 1 |
| units | units | `uomType` | 1 |
| `doubleArrayWithUnits` | String containing an array of double precision floating point values separated by spaces. The mandatory count attribute defines the number of elements in the array. | `xsd:string` | 1 |

:::

(tab:pfd-l1-102)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 102</span><span class="caption-text"> — Common product types (interferometricPairType)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| primary | Index of the primary inteferometric image with respect to a stack of images. | `xsd:unisignedInt` | 1 |
| secondary | Index of the secondary interferometric image with respect to a stack of images. | `xsd:unsignedInt` | 1 |

:::
