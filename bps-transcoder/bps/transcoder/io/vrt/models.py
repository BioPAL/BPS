# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD VRT models
--------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(kw_only=True)
class AlgorithmOptionsType:
    any_element: list[object] = field(
        default_factory=list, metadata={"type": "Wildcard", "namespace": "##any", "process_contents": "skip"}
    )


@dataclass(kw_only=True)
class AttributeType:
    data_type: str = field(
        metadata={
            "name": "DataType",
            "type": "Element",
        },
    )
    value: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Value",
            "type": "Element",
        },
    )
    name: str = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class CategoryNamesType:
    category: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Category",
            "type": "Element",
        },
    )


class ColorInterpType(Enum):
    GRAY = "Gray"
    PALETTE = "Palette"
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    ALPHA = "Alpha"
    HUE = "Hue"
    SATURATION = "Saturation"
    LIGHTNESS = "Lightness"
    CYAN = "Cyan"
    MAGENTA = "Magenta"
    YELLOW = "Yellow"
    BLACK = "Black"
    YCB_CR_Y = "YCbCr_Y"
    YCB_CR_CB = "YCbCr_Cb"
    YCB_CR_CR = "YCbCr_Cr"
    UNDEFINED = "Undefined"


@dataclass(kw_only=True)
class ColorTableEntryType:
    c1: int = field(
        metadata={
            "type": "Attribute",
        },
    )
    c2: int = field(
        metadata={
            "type": "Attribute",
        },
    )
    c3: int = field(
        metadata={
            "type": "Attribute",
        },
    )
    c4: None | int = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class ConstantValueType:
    value: str = field(default="")
    offset: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    count: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


class DataTypeType(Enum):
    BYTE = "Byte"
    UINT16 = "UInt16"
    INT16 = "Int16"
    UINT32 = "UInt32"
    INT32 = "Int32"
    UINT64 = "UInt64"
    INT64 = "Int64"
    FLOAT32 = "Float32"
    FLOAT64 = "Float64"
    CINT16 = "CInt16"
    CINT32 = "CInt32"
    CFLOAT32 = "CFloat32"
    CFLOAT64 = "CFloat64"


@dataclass(kw_only=True)
class DestSlabType:
    offset: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class DimensionRefType:
    ref: str = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class DimensionType:
    name: str = field(
        metadata={
            "type": "Attribute",
        },
    )
    type_value: None | str = field(
        default=None,
        metadata={
            "name": "type",
            "type": "Attribute",
        },
    )
    direction: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    size: int = field(
        metadata={
            "type": "Attribute",
        },
    )
    indexing_variable: None | str = field(
        default=None,
        metadata={
            "name": "indexingVariable",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class FieldDefnType:
    name: str = field(
        metadata={
            "name": "Name",
            "type": "Element",
        },
    )
    type_value: int = field(
        metadata={
            "name": "Type",
            "type": "Element",
        },
    )
    usage: int = field(
        metadata={
            "name": "Usage",
            "type": "Element",
        },
    )
    index: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class Gcptype:
    class Meta:
        name = "GCPType"

    id: None | str = field(
        default=None,
        metadata={
            "name": "Id",
            "type": "Attribute",
        },
    )
    info: None | str = field(
        default=None,
        metadata={
            "name": "Info",
            "type": "Attribute",
        },
    )
    pixel: float = field(
        metadata={
            "name": "Pixel",
            "type": "Attribute",
        },
    )
    line: float = field(
        metadata={
            "name": "Line",
            "type": "Attribute",
        },
    )
    x: float = field(
        metadata={
            "name": "X",
            "type": "Attribute",
        },
    )
    y: float = field(
        metadata={
            "name": "Y",
            "type": "Attribute",
        },
    )
    z: None | float = field(
        default=None,
        metadata={
            "name": "Z",
            "type": "Attribute",
        },
    )
    gcpz: None | float = field(
        default=None,
        metadata={
            "name": "GCPZ",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class GdalwarpOptionsType:
    class Meta:
        name = "GDALWarpOptionsType"

    any_element: list[object] = field(
        default_factory=list, metadata={"type": "Wildcard", "namespace": "##any", "process_contents": "skip"}
    )


@dataclass(kw_only=True)
class InlineValuesType:
    value: str = field(default="")
    offset: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    count: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class InlineValuesWithValueElementType:
    value: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Value",
            "type": "Element",
            "min_occurs": 1,
        },
    )
    offset: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    count: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class Mditype:
    class Meta:
        name = "MDIType"

    value: str = field(default="")
    key: str = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class MetadataType:
    any_element: list[object] = field(
        default_factory=list, metadata={"type": "Wildcard", "namespace": "##any", "process_contents": "skip"}
    )
    domain: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    format: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


class Nantype(Enum):
    NAN = "nan"
    NAN_1 = "NAN"


class OgrbooleanType(Enum):
    VALUE_1 = "1"
    VALUE_0 = "0"
    ON = "ON"
    OFF = "OFF"
    ON_1 = "on"
    OFF_1 = "off"
    YES = "YES"
    NO = "NO"
    YES_1 = "yes"
    NO_1 = "no"
    TRUE = "TRUE"
    FALSE = "FALSE"
    TRUE_1 = "true"
    FALSE_1 = "false"
    TRUE_2 = "True"
    FALSE_2 = "False"


@dataclass(kw_only=True)
class Ooitype:
    class Meta:
        name = "OOIType"

    value: str = field(default="")
    key: str = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class OverviewListType:
    value: list[int] = field(default_factory=list, metadata={"tokens": True})
    resampling: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class RectType:
    x_off: None | float = field(
        default=None,
        metadata={
            "name": "xOff",
            "type": "Attribute",
        },
    )
    y_off: None | float = field(
        default=None,
        metadata={
            "name": "yOff",
            "type": "Attribute",
        },
    )
    x_size: None | float = field(default=None, metadata={"name": "xSize", "type": "Attribute", "min_exclusive": 0.0})
    y_size: None | float = field(default=None, metadata={"name": "ySize", "type": "Attribute", "min_exclusive": 0.0})


@dataclass(kw_only=True)
class RegularlySpacedValuesType:
    start: float = field(
        metadata={
            "type": "Attribute",
        },
    )
    increment: float = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class RowType:
    f: list[object] = field(
        default_factory=list,
        metadata={
            "name": "F",
            "type": "Element",
        },
    )
    index: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class Srstype:
    class Meta:
        name = "SRSType"

    value: str = field(default="")
    data_axis_to_srsaxis_mapping: None | str = field(
        default=None,
        metadata={
            "name": "dataAxisToSRSAxisMapping",
            "type": "Attribute",
        },
    )
    coordinate_epoch: None | float = field(
        default=None,
        metadata={
            "name": "coordinateEpoch",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class SourceSlabType:
    offset: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    count: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    step: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


class VrtrasterBandSubTypeType(Enum):
    VRTWARPED_RASTER_BAND = "VRTWarpedRasterBand"
    VRTDERIVED_RASTER_BAND = "VRTDerivedRasterBand"
    VRTRAW_RASTER_BAND = "VRTRawRasterBand"
    VRTPANSHARPENED_RASTER_BAND = "VRTPansharpenedRasterBand"


class ZeroOrOne(Enum):
    VALUE_0 = 0
    VALUE_1 = 1


@dataclass(kw_only=True)
class ColorTableType:
    entry: list[ColorTableEntryType] = field(
        default_factory=list,
        metadata={
            "name": "Entry",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class GcplistType:
    class Meta:
        name = "GCPListType"

    gcp: list[Gcptype] = field(
        default_factory=list,
        metadata={
            "name": "GCP",
            "type": "Element",
        },
    )
    projection: None | str = field(
        default=None,
        metadata={
            "name": "Projection",
            "type": "Attribute",
        },
    )
    data_axis_to_srsaxis_mapping: None | str = field(
        default=None,
        metadata={
            "name": "dataAxisToSRSAxisMapping",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class GdalrasterAttributeTableType:
    class Meta:
        name = "GDALRasterAttributeTableType"

    field_defn: list[FieldDefnType] = field(
        default_factory=list,
        metadata={
            "name": "FieldDefn",
            "type": "Element",
        },
    )
    row: list[RowType] = field(
        default_factory=list,
        metadata={
            "name": "Row",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class HistItemType:
    hist_min: list[float] = field(
        default_factory=list,
        metadata={
            "name": "HistMin",
            "type": "Element",
        },
    )
    hist_max: list[float] = field(
        default_factory=list,
        metadata={
            "name": "HistMax",
            "type": "Element",
        },
    )
    bucket_count: list[int] = field(
        default_factory=list,
        metadata={
            "name": "BucketCount",
            "type": "Element",
        },
    )
    include_out_of_range: list[ZeroOrOne] = field(
        default_factory=list,
        metadata={
            "name": "IncludeOutOfRange",
            "type": "Element",
        },
    )
    approximate: list[ZeroOrOne] = field(
        default_factory=list,
        metadata={
            "name": "Approximate",
            "type": "Element",
        },
    )
    hist_counts: list[str] = field(
        default_factory=list,
        metadata={
            "name": "HistCounts",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class KernelType:
    size: int = field(
        metadata={
            "name": "Size",
            "type": "Element",
        },
    )
    coefs: str = field(
        metadata={
            "name": "Coefs",
            "type": "Element",
        },
    )
    normalized: None | ZeroOrOne = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class OpenOptionsType:
    ooi: list[Ooitype] = field(
        default_factory=list,
        metadata={
            "name": "OOI",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class SourceFilenameType:
    value: str = field(default="")
    relative_to_vrt: None | ZeroOrOne = field(
        default=None,
        metadata={
            "name": "relativeToVRT",
            "type": "Attribute",
        },
    )
    relativeto_vrt_attribute: None | ZeroOrOne = field(
        default=None,
        metadata={
            "name": "relativetoVRT",
            "type": "Attribute",
        },
    )
    shared: None | OgrbooleanType = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class SourcePropertiesType:
    raster_xsize: None | int = field(
        default=None, metadata={"name": "RasterXSize", "type": "Attribute", "max_inclusive": 2147483647}
    )
    raster_ysize: None | int = field(
        default=None, metadata={"name": "RasterYSize", "type": "Attribute", "max_inclusive": 2147483647}
    )
    data_type: None | DataTypeType = field(
        default=None,
        metadata={
            "name": "DataType",
            "type": "Attribute",
        },
    )
    block_xsize: None | int = field(
        default=None, metadata={"name": "BlockXSize", "type": "Attribute", "max_inclusive": 2147483647}
    )
    block_ysize: None | int = field(
        default=None, metadata={"name": "BlockYSize", "type": "Attribute", "max_inclusive": 2147483647}
    )


@dataclass(kw_only=True)
class ComplexSourceType:
    source_filename: list[SourceFilenameType] = field(
        default_factory=list,
        metadata={
            "name": "SourceFilename",
            "type": "Element",
        },
    )
    open_options: list[OpenOptionsType] = field(
        default_factory=list,
        metadata={
            "name": "OpenOptions",
            "type": "Element",
        },
    )
    source_band: list[str] = field(
        default_factory=list,
        metadata={
            "name": "SourceBand",
            "type": "Element",
        },
    )
    source_properties: list[SourcePropertiesType] = field(
        default_factory=list,
        metadata={
            "name": "SourceProperties",
            "type": "Element",
        },
    )
    src_rect: list[RectType] = field(
        default_factory=list,
        metadata={
            "name": "SrcRect",
            "type": "Element",
        },
    )
    dst_rect: list[RectType] = field(
        default_factory=list,
        metadata={
            "name": "DstRect",
            "type": "Element",
        },
    )
    scale_offset: list[float] = field(
        default_factory=list,
        metadata={
            "name": "ScaleOffset",
            "type": "Element",
        },
    )
    scale_ratio: list[float] = field(
        default_factory=list,
        metadata={
            "name": "ScaleRatio",
            "type": "Element",
        },
    )
    color_table_component: list[int] = field(
        default_factory=list,
        metadata={
            "name": "ColorTableComponent",
            "type": "Element",
        },
    )
    exponent: list[float] = field(
        default_factory=list,
        metadata={
            "name": "Exponent",
            "type": "Element",
        },
    )
    src_min: list[float] = field(
        default_factory=list,
        metadata={
            "name": "SrcMin",
            "type": "Element",
        },
    )
    src_max: list[float] = field(
        default_factory=list,
        metadata={
            "name": "SrcMax",
            "type": "Element",
        },
    )
    dst_min: list[float] = field(
        default_factory=list,
        metadata={
            "name": "DstMin",
            "type": "Element",
        },
    )
    dst_max: list[float] = field(
        default_factory=list,
        metadata={
            "name": "DstMax",
            "type": "Element",
        },
    )
    nodata: list[float | Nantype] = field(
        default_factory=list,
        metadata={
            "name": "NODATA",
            "type": "Element",
        },
    )
    use_mask_band: list[bool] = field(
        default_factory=list,
        metadata={
            "name": "UseMaskBand",
            "type": "Element",
        },
    )
    lut: list[str] = field(
        default_factory=list,
        metadata={
            "name": "LUT",
            "type": "Element",
        },
    )
    resampling: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class HistogramsType:
    hist_item: list[HistItemType] = field(
        default_factory=list,
        metadata={
            "name": "HistItem",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class KernelFilteredSourceType:
    source_filename: list[SourceFilenameType] = field(
        default_factory=list,
        metadata={
            "name": "SourceFilename",
            "type": "Element",
        },
    )
    open_options: list[OpenOptionsType] = field(
        default_factory=list,
        metadata={
            "name": "OpenOptions",
            "type": "Element",
        },
    )
    source_band: list[str] = field(
        default_factory=list,
        metadata={
            "name": "SourceBand",
            "type": "Element",
        },
    )
    source_properties: list[SourcePropertiesType] = field(
        default_factory=list,
        metadata={
            "name": "SourceProperties",
            "type": "Element",
        },
    )
    src_rect: list[RectType] = field(
        default_factory=list,
        metadata={
            "name": "SrcRect",
            "type": "Element",
        },
    )
    dst_rect: list[RectType] = field(
        default_factory=list,
        metadata={
            "name": "DstRect",
            "type": "Element",
        },
    )
    scale_offset: list[float] = field(
        default_factory=list,
        metadata={
            "name": "ScaleOffset",
            "type": "Element",
        },
    )
    scale_ratio: list[float] = field(
        default_factory=list,
        metadata={
            "name": "ScaleRatio",
            "type": "Element",
        },
    )
    color_table_component: list[int] = field(
        default_factory=list,
        metadata={
            "name": "ColorTableComponent",
            "type": "Element",
        },
    )
    exponent: list[float] = field(
        default_factory=list,
        metadata={
            "name": "Exponent",
            "type": "Element",
        },
    )
    src_min: list[float] = field(
        default_factory=list,
        metadata={
            "name": "SrcMin",
            "type": "Element",
        },
    )
    src_max: list[float] = field(
        default_factory=list,
        metadata={
            "name": "SrcMax",
            "type": "Element",
        },
    )
    dst_min: list[float] = field(
        default_factory=list,
        metadata={
            "name": "DstMin",
            "type": "Element",
        },
    )
    dst_max: list[float] = field(
        default_factory=list,
        metadata={
            "name": "DstMax",
            "type": "Element",
        },
    )
    nodata: list[float | Nantype] = field(
        default_factory=list,
        metadata={
            "name": "NODATA",
            "type": "Element",
        },
    )
    use_mask_band: list[bool] = field(
        default_factory=list,
        metadata={
            "name": "UseMaskBand",
            "type": "Element",
        },
    )
    lut: list[str] = field(
        default_factory=list,
        metadata={
            "name": "LUT",
            "type": "Element",
        },
    )
    kernel: list[KernelType] = field(
        default_factory=list,
        metadata={
            "name": "Kernel",
            "type": "Element",
        },
    )
    resampling: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class OverviewType:
    source_filename: list[SourceFilenameType] = field(
        default_factory=list,
        metadata={
            "name": "SourceFilename",
            "type": "Element",
        },
    )
    source_band: list[str] = field(
        default_factory=list,
        metadata={
            "name": "SourceBand",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class PanchroBandType:
    source_filename: SourceFilenameType = field(
        metadata={
            "name": "SourceFilename",
            "type": "Element",
        },
    )
    source_band: str = field(
        metadata={
            "name": "SourceBand",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class SimpleSourceType:
    source_filename: list[SourceFilenameType] = field(
        default_factory=list,
        metadata={
            "name": "SourceFilename",
            "type": "Element",
        },
    )
    open_options: list[OpenOptionsType] = field(
        default_factory=list,
        metadata={
            "name": "OpenOptions",
            "type": "Element",
        },
    )
    source_band: list[str] = field(
        default_factory=list,
        metadata={
            "name": "SourceBand",
            "type": "Element",
        },
    )
    source_properties: list[SourcePropertiesType] = field(
        default_factory=list,
        metadata={
            "name": "SourceProperties",
            "type": "Element",
        },
    )
    src_rect: list[RectType] = field(
        default_factory=list,
        metadata={
            "name": "SrcRect",
            "type": "Element",
        },
    )
    dst_rect: list[RectType] = field(
        default_factory=list,
        metadata={
            "name": "DstRect",
            "type": "Element",
        },
    )
    resampling: None | str = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class SourceType:
    source_filename: SourceFilenameType = field(
        metadata={
            "name": "SourceFilename",
            "type": "Element",
        },
    )
    source_array: None | str = field(
        default=None,
        metadata={
            "name": "SourceArray",
            "type": "Element",
        },
    )
    source_band: None | str = field(
        default=None,
        metadata={
            "name": "SourceBand",
            "type": "Element",
        },
    )
    source_transpose: None | str = field(
        default=None,
        metadata={
            "name": "SourceTranspose",
            "type": "Element",
        },
    )
    source_view: None | str = field(
        default=None,
        metadata={
            "name": "SourceView",
            "type": "Element",
        },
    )
    source_slab: None | SourceSlabType = field(
        default=None,
        metadata={
            "name": "SourceSlab",
            "type": "Element",
        },
    )
    dest_slab: None | DestSlabType = field(
        default=None,
        metadata={
            "name": "DestSlab",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class SpectralBandType:
    source_filename: SourceFilenameType = field(
        metadata={
            "name": "SourceFilename",
            "type": "Element",
        },
    )
    source_band: str = field(
        metadata={
            "name": "SourceBand",
            "type": "Element",
        },
    )
    dst_band: None | int = field(
        default=None,
        metadata={
            "name": "dstBand",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class ArrayType:
    data_type: str = field(
        metadata={
            "name": "DataType",
            "type": "Element",
        },
    )
    dimension: list[DimensionType] = field(
        default_factory=list,
        metadata={
            "name": "Dimension",
            "type": "Element",
        },
    )
    dimension_ref: list[DimensionRefType] = field(
        default_factory=list,
        metadata={
            "name": "DimensionRef",
            "type": "Element",
        },
    )
    srs: None | Srstype = field(
        default=None,
        metadata={
            "name": "SRS",
            "type": "Element",
        },
    )
    unit: None | str = field(
        default=None,
        metadata={
            "name": "Unit",
            "type": "Element",
        },
    )
    no_data_value: None | float | Nantype = field(
        default=None,
        metadata={
            "name": "NoDataValue",
            "type": "Element",
        },
    )
    offset: None | float = field(
        default=None,
        metadata={
            "name": "Offset",
            "type": "Element",
        },
    )
    scale: None | float = field(
        default=None,
        metadata={
            "name": "Scale",
            "type": "Element",
        },
    )
    regularly_spaced_values: None | RegularlySpacedValuesType = field(
        default=None,
        metadata={
            "name": "RegularlySpacedValues",
            "type": "Element",
        },
    )
    constant_value: list[ConstantValueType] = field(
        default_factory=list,
        metadata={
            "name": "ConstantValue",
            "type": "Element",
        },
    )
    inline_values: list[InlineValuesType] = field(
        default_factory=list,
        metadata={
            "name": "InlineValues",
            "type": "Element",
        },
    )
    inline_values_with_value_element: list[InlineValuesWithValueElementType] = field(
        default_factory=list,
        metadata={
            "name": "InlineValuesWithValueElement",
            "type": "Element",
        },
    )
    source: list[SourceType] = field(
        default_factory=list,
        metadata={
            "name": "Source",
            "type": "Element",
        },
    )
    attribute: list[AttributeType] = field(
        default_factory=list,
        metadata={
            "name": "Attribute",
            "type": "Element",
        },
    )
    name: str = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class PansharpeningOptionsType:
    algorithm: None | str = field(
        default=None,
        metadata={
            "name": "Algorithm",
            "type": "Element",
        },
    )
    algorithm_options: None | AlgorithmOptionsType = field(
        default=None,
        metadata={
            "name": "AlgorithmOptions",
            "type": "Element",
        },
    )
    resampling: None | str = field(
        default=None,
        metadata={
            "name": "Resampling",
            "type": "Element",
        },
    )
    num_threads: None | str = field(
        default=None,
        metadata={
            "name": "NumThreads",
            "type": "Element",
        },
    )
    bit_depth: None | str = field(
        default=None,
        metadata={
            "name": "BitDepth",
            "type": "Element",
        },
    )
    no_data: None | float | str = field(
        default=None,
        metadata={
            "name": "NoData",
            "type": "Element",
        },
    )
    spatial_extent_adjustment: None | str = field(
        default=None,
        metadata={
            "name": "SpatialExtentAdjustment",
            "type": "Element",
        },
    )
    panchro_band: PanchroBandType = field(
        metadata={
            "name": "PanchroBand",
            "type": "Element",
        },
    )
    spectral_band: list[SpectralBandType] = field(
        default_factory=list,
        metadata={
            "name": "SpectralBand",
            "type": "Element",
            "min_occurs": 1,
        },
    )


@dataclass(kw_only=True)
class VrtrasterBandType:
    class Meta:
        name = "VRTRasterBandType"

    description: list[str] = field(
        default_factory=list,
        metadata={
            "name": "Description",
            "type": "Element",
        },
    )
    unit_type: list[str] = field(
        default_factory=list,
        metadata={
            "name": "UnitType",
            "type": "Element",
        },
    )
    offset: list[float] = field(
        default_factory=list,
        metadata={
            "name": "Offset",
            "type": "Element",
        },
    )
    scale: list[float] = field(
        default_factory=list,
        metadata={
            "name": "Scale",
            "type": "Element",
        },
    )
    category_names: list[CategoryNamesType] = field(
        default_factory=list,
        metadata={
            "name": "CategoryNames",
            "type": "Element",
        },
    )
    color_table: list[ColorTableType] = field(
        default_factory=list,
        metadata={
            "name": "ColorTable",
            "type": "Element",
        },
    )
    gdalraster_attribute_table: list[GdalrasterAttributeTableType] = field(
        default_factory=list,
        metadata={
            "name": "GDALRasterAttributeTable",
            "type": "Element",
        },
    )
    no_data_value_element: list[float | Nantype] = field(
        default_factory=list,
        metadata={
            "name": "NoDataValue",
            "type": "Element",
        },
    )
    nodata_value: list[float] = field(
        default_factory=list,
        metadata={
            "name": "NodataValue",
            "type": "Element",
        },
    )
    hide_no_data_value: list[ZeroOrOne] = field(
        default_factory=list,
        metadata={
            "name": "HideNoDataValue",
            "type": "Element",
        },
    )
    metadata: list[MetadataType] = field(
        default_factory=list,
        metadata={
            "name": "Metadata",
            "type": "Element",
        },
    )
    color_interp: list[ColorInterpType] = field(
        default_factory=list,
        metadata={
            "name": "ColorInterp",
            "type": "Element",
        },
    )
    overview: list[OverviewType] = field(
        default_factory=list,
        metadata={
            "name": "Overview",
            "type": "Element",
        },
    )
    mask_band: list[MaskBandType] = field(
        default_factory=list,
        metadata={
            "name": "MaskBand",
            "type": "Element",
        },
    )
    histograms: list[HistogramsType] = field(
        default_factory=list,
        metadata={
            "name": "Histograms",
            "type": "Element",
        },
    )
    simple_source: list[SimpleSourceType] = field(
        default_factory=list,
        metadata={
            "name": "SimpleSource",
            "type": "Element",
        },
    )
    complex_source: list[ComplexSourceType] = field(
        default_factory=list,
        metadata={
            "name": "ComplexSource",
            "type": "Element",
        },
    )
    averaged_source: list[SimpleSourceType] = field(
        default_factory=list,
        metadata={
            "name": "AveragedSource",
            "type": "Element",
        },
    )
    kernel_filtered_source: list[KernelFilteredSourceType] = field(
        default_factory=list,
        metadata={
            "name": "KernelFilteredSource",
            "type": "Element",
        },
    )
    pixel_function_type: list[str] = field(
        default_factory=list,
        metadata={
            "name": "PixelFunctionType",
            "type": "Element",
        },
    )
    source_transfer_type: list[DataTypeType] = field(
        default_factory=list,
        metadata={
            "name": "SourceTransferType",
            "type": "Element",
        },
    )
    pixel_function_language: list[str] = field(
        default_factory=list,
        metadata={
            "name": "PixelFunctionLanguage",
            "type": "Element",
        },
    )
    pixel_function_code: list[str] = field(
        default_factory=list,
        metadata={
            "name": "PixelFunctionCode",
            "type": "Element",
        },
    )
    pixel_function_arguments: list[VrtrasterBandType.PixelFunctionArguments] = field(
        default_factory=list,
        metadata={
            "name": "PixelFunctionArguments",
            "type": "Element",
        },
    )
    buffer_radius: list[int] = field(
        default_factory=list,
        metadata={
            "name": "BufferRadius",
            "type": "Element",
        },
    )
    source_filename: list[SourceFilenameType] = field(
        default_factory=list,
        metadata={
            "name": "SourceFilename",
            "type": "Element",
        },
    )
    image_offset: list[int] = field(
        default_factory=list,
        metadata={
            "name": "ImageOffset",
            "type": "Element",
        },
    )
    pixel_offset: list[int] = field(
        default_factory=list,
        metadata={
            "name": "PixelOffset",
            "type": "Element",
        },
    )
    line_offset: list[int] = field(
        default_factory=list,
        metadata={
            "name": "LineOffset",
            "type": "Element",
        },
    )
    byte_order: list[str] = field(
        default_factory=list,
        metadata={
            "name": "ByteOrder",
            "type": "Element",
        },
    )
    data_type: None | DataTypeType = field(
        default=None,
        metadata={
            "name": "dataType",
            "type": "Attribute",
        },
    )
    band: None | int = field(
        default=None,
        metadata={
            "type": "Attribute",
        },
    )
    block_xsize: None | int = field(
        default=None, metadata={"name": "blockXSize", "type": "Attribute", "max_inclusive": 2147483647}
    )
    block_ysize: None | int = field(
        default=None, metadata={"name": "blockYSize", "type": "Attribute", "max_inclusive": 2147483647}
    )
    sub_class: None | VrtrasterBandSubTypeType = field(
        default=None,
        metadata={
            "name": "subClass",
            "type": "Attribute",
        },
    )

    @dataclass(kw_only=True)
    class PixelFunctionArguments:
        any_attributes: dict[str, str] = field(
            default_factory=dict, metadata={"type": "Attributes", "namespace": "##any"}
        )


@dataclass(kw_only=True)
class GroupType:
    dimension: list[DimensionType] = field(
        default_factory=list,
        metadata={
            "name": "Dimension",
            "type": "Element",
        },
    )
    attribute: list[AttributeType] = field(
        default_factory=list,
        metadata={
            "name": "Attribute",
            "type": "Element",
        },
    )
    array: list[ArrayType] = field(
        default_factory=list,
        metadata={
            "name": "Array",
            "type": "Element",
        },
    )
    group: list[GroupType] = field(
        default_factory=list,
        metadata={
            "name": "Group",
            "type": "Element",
        },
    )
    name: str = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class MaskBandType:
    vrtraster_band: VrtrasterBandType = field(
        metadata={
            "name": "VRTRasterBand",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class Vrtdataset:
    class Meta:
        name = "VRTDataset"

    srs: list[Srstype] = field(
        default_factory=list,
        metadata={
            "name": "SRS",
            "type": "Element",
        },
    )
    geo_transform: list[str] = field(
        default_factory=list,
        metadata={
            "name": "GeoTransform",
            "type": "Element",
        },
    )
    gcplist: list[GcplistType] = field(
        default_factory=list,
        metadata={
            "name": "GCPList",
            "type": "Element",
        },
    )
    block_xsize: list[int] = field(
        default_factory=list, metadata={"name": "BlockXSize", "type": "Element", "max_inclusive": 2147483647}
    )
    block_ysize: list[int] = field(
        default_factory=list, metadata={"name": "BlockYSize", "type": "Element", "max_inclusive": 2147483647}
    )
    metadata: list[MetadataType] = field(
        default_factory=list,
        metadata={
            "name": "Metadata",
            "type": "Element",
        },
    )
    vrtraster_band: list[VrtrasterBandType] = field(
        default_factory=list,
        metadata={
            "name": "VRTRasterBand",
            "type": "Element",
        },
    )
    mask_band: list[MaskBandType] = field(
        default_factory=list,
        metadata={
            "name": "MaskBand",
            "type": "Element",
        },
    )
    gdalwarp_options: list[GdalwarpOptionsType] = field(
        default_factory=list,
        metadata={
            "name": "GDALWarpOptions",
            "type": "Element",
        },
    )
    pansharpening_options: list[PansharpeningOptionsType] = field(
        default_factory=list,
        metadata={
            "name": "PansharpeningOptions",
            "type": "Element",
        },
    )
    group: list[GroupType] = field(
        default_factory=list,
        metadata={
            "name": "Group",
            "type": "Element",
        },
    )
    overview_list: list[OverviewListType] = field(
        default_factory=list,
        metadata={
            "name": "OverviewList",
            "type": "Element",
        },
    )
    sub_class: None | str = field(
        default=None,
        metadata={
            "name": "subClass",
            "type": "Attribute",
        },
    )
    raster_xsize: None | int = field(
        default=None, metadata={"name": "rasterXSize", "type": "Attribute", "max_inclusive": 2147483647}
    )
    raster_ysize: None | int = field(
        default=None, metadata={"name": "rasterYSize", "type": "Attribute", "max_inclusive": 2147483647}
    )
