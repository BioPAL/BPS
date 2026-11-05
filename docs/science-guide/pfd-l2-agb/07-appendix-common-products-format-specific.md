<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:pfd-agb-7)=
# 7. Appendix: Common Products Format Specification

(sec:pfd-agb-7-1)=
### 7.1 Annotation data sets (ADS)

(sec:pfd-agb-7-1-1)=
#### 7.1.1 Primitive and common types

The following table summarizes and describes primitive product types used throughout the document.

(tab:pfd-agb-62)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 62</span><span class="caption-text"> — Primitive product types</span></p>

| Name | Description | Base Type | Values |
|---|---|---|---|
| `subsettingRuleType` | Enumeration of available rules for INT subsetting: select three acquisitions from all the ones in input (when input is TOM phase). | `xsd:string` | geometry maintain all |
| `calibrationScreenType` | Type of calibration screen applied (none, if not applied) | `xsd:string` | none geometry skp |
| `polarisationCombinationMethodType` | Enumeration of polarisations combination methods. | `xsd:string` | HV VH Average None |
| bps\_fnfType | Enumeration of Forest Mask: CFM from FD processor or FNF. | `xsd:string` | FNF CFM |
| `layerType` | Enumeration of LUT layer contents. | `xsd:string` | FNF ACM Incidence angle [deg] `numberOfAverages` |
| `operationalModeType` | Enumeration of Ground Cancellation algorithm to be used. | `xsd:string` | single reference multi reference insar pair |
| `missionType` | Enumeration of valid BIOMASS mission names. | `xsd:string` | BIOMASS |
| `swathType` | Enumeration of all valid swath identifiers. | `xsd:string` | S1 S2 S3 |
| `productType` | Enumeration of valid product types. | `xsd:string` | SCS DGM STA `FH_L2A` `FH_L2B` `FD_L2A` `FD_L2B` `GN_L2A` `AGB_L2B` |
| `polarisationType` | Enumeration of valid polarisations. | `xsd:string` | HH HV VH VV XX |
| `timeType` | UTC time expressed according to ISO 8601 standard (1 microsecond resolution). | `xsd:dateTime` |   |
| `missionPhaseIDType` | Enumeration of all valid BIOMASS mission phases. | `xsd:string` | COM INT TOM |
| `sensorModeType` | Enumeration of valid sensor mode abbreviations. | `xsd:string` | Measurement Rx Only External Calibration |
| `orbitPassType` | Enumeration of the orbit pass direction values. | `xsd:string` | Ascending Descending |
| `projectionType` | Enumeration of the image projection. | `xsd:string` | Slant Range Ground Range Latitude longitude based on DGG DGG |
| `pixelRepresentationType` | Representation of the image pixels within the image MDS. | `xsd:string` | I Q Abs Phase Abs Probability of change Computed forest mask Forest Disturbance Heat Map Forest Height [m] Forest height quality Forest mask Ground Cancelled Backscatter Above Ground Biomass [t/ha] Above Ground Biomass quality [t/ha] Acquisition ID image |
| `referenceSelectionType` | Enumeration of AGB selection of reference data (second iteration only). String | `xsd:string` | `refOnly` `firstIterationOnly` `weightedMean` |
| `agbIndexingType` | Enumeration of AGB algorithm indexings. | `xsd:string` | none p j k pj pk jk pjk |
| `pixelTypeType` | Enumeration of image pixel data types. | `xsd:string` | 32 bit Float 16 bit Signed Integer 16 bit Unsigned Integer 8 bit Unsigned Integer |
| `coordinateReferenceSystemType` | Coordinate reference systems, as per WKT representation {rd}`8`. | `xsd:string` |   |
| `geodeticReferenceFrameType` | Enumeration of geodetic reference frames. | `xsd:string` | WGS84 ITRF2000 |
| `uomType` | Enumeration of unit of measures. | `xsd:string` | s UTC m samples lines deg rad Hz Hz/s dB Mbps C K rad/s mm/Km rad/m 1/m t/ha (rad/nT)^2 |
| float | 32 bit single precision floating point number. | `xsd:float` | units |
| double | 64 bit double precision floating point number. | `xsd:double` | units |
| bool | Boolean (true or false). | `xsd:boolean` | (false)\|(true) |

:::

The following table summarizes and describes those primitive product types for which attributes have been added with respect to base types.

(tab:pfd-agb-63)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 63</span><span class="caption-text"> — Primitive product types with custom attributes</span></p>

| Name | Description | Base Type | Attributes |
|---|---|---|---|
| `acquisitionType` | String containing an acquisition folder name; The mandatory `referenceImage` attribute is a bool to indicate if the acquisition is the reference one. The optional `averageWavenumber` attribute is a float with the average value of the vertical wavenumber of the acquisition. Not used in image pair acquisitions (4.3.1.4) | `xsd:string` | `referenceImage` (mandatory, bool) `averageWavenumber` (optional, float) |

:::

(tab:pfd-agb-64)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 64</span><span class="caption-text"> — Common product types (datumType)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| `coordinateReferenceSystem` | Coordinate reference system. | `coordinateReferenceSystemType` | 1 |
| `geodeticReferenceFrame` | Geodetic reference frame. | `geodeticReferenceFrameType` | 1 |

:::

(tab:pfd-agb-65)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 65</span><span class="caption-text"> — Common product types (minMaxTypeWithUnit)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| min | Minimum value. | `floatWithUnit` | 1 |
| max | Maximum value. | `floatWithUnit` | 1 |

:::

(tab:pfd-agb-66)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 66</span><span class="caption-text"> — Common product types (floatArrayWithUnits)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| count | counter | `xsd:unsignedInt` | 1 |
| units | units | `uomType` | 1 |
| `floatArrayWithUnits` | String containing an array of float values separated by spaces. The mandatory count attribute defines the number of elements in the array. | `xsd:string` | 1 |

:::

(tab:pfd-agb-67)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 67</span><span class="caption-text"> — Common product types (floatWithUnit)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| units | units | `uomType` | 1 |
| `floatWithUnit` | 32 bit single precision floating point number. | `xsd:float` | 1 |

:::

(tab:pfd-agb-68)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 68</span><span class="caption-text"> — Common product types (doubleWithUnit)</span></p>

| Name | Description | Data Type | Cardinality |
|---|---|---|---|
| units | units | `uomType` | 1 |
| `doubleWithUnit` | 64 bit double precision floating point number. | xsd: double | 1 |

:::
