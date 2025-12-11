# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD common types
----------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChannelType(Enum):
    """
    Enumeration of valid RGB channels.
    """

    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"


@dataclass
class Complex:
    """
    64 bit complex number consisting of a 32 bit single precision floating point
    real part and a 32 bit single precision floating point imaginary part.

    Parameters
    ----------
    re
        32 bit single precision floating point real number.
    im
        32 bit single precision floating point imaginary number.
    """

    class Meta:
        name = "complex"

    re: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
    im: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class ComplexArray:
    """String containing an array of complex value pairs separated by spaces in the
    form of I Q I Q I Q ...

    The mandatory count attribute defines the number of complex elements in the array.
    """

    class Meta:
        name = "complexArray"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class DoubleArray:
    """String containing an array of double precision floating point values
    separated by spaces.

    The mandatory count attribute defines the number of elements in the array.
    """

    class Meta:
        name = "doubleArray"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class FloatArray:
    """String containing an array of float values separated by spaces.

    The mandatory count attribute defines the number of elements in the array.
    """

    class Meta:
        name = "floatArray"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


class GeodeticReferenceFrameType(Enum):
    """
    Enumeration of geodetic reference frames.
    """

    WGS84 = "WGS84"
    ITRF2000 = "ITRF2000"


class GroupType(Enum):
    """
    Enumeration of valid LUT groups of variables.
    """

    DEM_BASED_LUT = "DEM based LUT"
    RFI_BASED_LUT = "RFI based LUT"
    IMAGE_BASED_LUT = "Image based LUT"


class HeightModelBaseType(Enum):
    ELLIPSOID = "Ellipsoid"
    SRTM = "SRTM"
    COPERNICUS_DEM = "Copernicus DEM"
    BIOMASS_DTM = "BIOMASS DTM"


@dataclass
class IntArray:
    """String containing an array of int values separated by spaces.

    The mandatory count attribute defines the number of elements in the array.
    """

    class Meta:
        name = "intArray"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class InterferometricPairType:
    """
    Parameters
    ----------
    primary
        Index of the primary inteferometric image with respect to a stack of images.
    secondary
        Index of the secondary interferometric image with respect to a stack of images.
    """

    class Meta:
        name = "interferometricPairType"

    primary: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
    secondary: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )


class LayerType(Enum):
    """
    Enumeration of LUT layer contents.
    """

    FARADAY_ROTATION_PLANE_RAD = "Faraday Rotation plane [rad]"
    FARADAY_ROTATION_RAD = "Faraday Rotation [rad]"
    FARADAY_ROTATION_STD_RAD = "Faraday Rotation std [rad]"
    PHASE_SCREEN_RAD = "Phase screen [rad]"
    TEC_TECU = "TEC [TECU]"
    RANGE_SHIFTS_M = "Range shifts [m]"
    AZIMUTH_SHIFTS_M = "Azimuth shifts [m]"
    AUTOFOCUS_PHASE_SCREEN_RAD = "Autofocus phase screen [rad]"
    AUTOFOCUS_PHASE_SCREEN_STD_RAD = "Autofocus phase screen std [rad]"
    RFI_TIME_DOMAIN_MASK_HH = "RFI time domain mask HH"
    RFI_TIME_DOMAIN_MASK_HV = "RFI time domain mask HV"
    RFI_TIME_DOMAIN_MASK_VH = "RFI time domain mask VH"
    RFI_TIME_DOMAIN_MASK_VV = "RFI time domain mask VV"
    RFI_FREQUENCY_DOMAIN_MASK_HH = "RFI frequency domain mask HH"
    RFI_FREQUENCY_DOMAIN_MASK_HV = "RFI frequency domain mask HV"
    RFI_FREQUENCY_DOMAIN_MASK_VH = "RFI frequency domain mask VH"
    RFI_FREQUENCY_DOMAIN_MASK_VV = "RFI frequency domain mask VV"
    SIGMA_NOUGHT_LUT = "Sigma-nought LUT"
    GAMMA_NOUGHT_LUT = "Gamma-nought LUT"
    DENOISING_MAP_HH = "Denoising map HH"
    DENOISING_MAP_HV = "Denoising map HV"
    DENOISING_MAP_XX = "Denoising map XX"
    DENOISING_MAP_VH = "Denoising map VH"
    DENOISING_MAP_VV = "Denoising map VV"
    LATITUDE_DEG = "Latitude [deg]"
    LONGITUDE_DEG = "Longitude [deg]"
    HEIGHT_M = "Height [m]"
    INCIDENCE_ANGLE_DEG = "Incidence angle [deg]"
    ELEVATION_ANGLE_DEG = "Elevation angle [deg]"
    TERRAIN_SLOPE_DEG = "Terrain slope [deg]"
    FNF = "FNF"
    ACM = "ACM"
    NUMBER_OF_AVERAGES = "numberOfAverages"
    AZIMUTH_COREGISTRATION_SHIFTS_M = "Azimuth coregistration shifts [m]"
    AZIMUTH_ORBIT_COREGISTRATION_SHIFTS_M = "Azimuth orbit coregistration shifts [m]"
    RANGE_COREGISTRATION_SHIFTS_M = "Range coregistration shifts [m]"
    RANGE_ORBIT_COREGISTRATION_SHIFTS_M = "Range orbit coregistration shifts [m]"
    COREGISTRATION_SHIFTS_QUALITY = "Coregistration shifts quality"
    WAVENUMBERS_RAD_M = "Wavenumbers [rad/m]"
    FLATTENING_PHASE_SCREEN_RAD = "Flattening phase screen [rad]"
    SKP_CALIBRATION_PHASE_SCREEN_RAD = "SKP calibration phase screen [rad]"
    SKP_CALIBRATION_PHASE_SCREEN_QUALITY = "SKP calibration phase screen quality"


@dataclass
class MinMaxType:
    class Meta:
        name = "minMaxType"

    min: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
    max: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )


class PolarisationType(Enum):
    """
    Enumeration of valid polarisations.
    """

    H = "H"
    V = "V"
    HH = "HH"
    HV = "HV"
    VH = "VH"
    VV = "VV"
    XX = "XX"


class UomType(Enum):
    """
    Enumeration of unit of measures.
    """

    S = "s"
    M = "m"
    SAMPLES = "samples"
    LINES = "lines"
    DEG = "deg"
    RAD = "rad"
    HZ = "Hz"
    HZ_S = "Hz/s"
    MHZ = "MHz"
    D_B = "dB"
    MBPS = "Mbps"
    C = "C"
    K = "K"
    DEG_S = "deg/s"
    RAD_S = "rad/s"
    MM_KM = "mm/Km"
    DEG_M = "deg/m"
    RAD_M = "rad/m"
    VALUE_1_M = "1/m"
    T_HA = "t/ha"
    RAD_N_T_2 = "(rad/nT)^2"


@dataclass
class ChannelImbalanceList:
    """
    Parameters
    ----------
    channel_imbal_hvrx
        H to V channel imbalance on receive across swath.
    channel_imbal_hvtx
        H to V channel imbalance on transmit across swath.
    """

    class Meta:
        name = "channelImbalanceList"

    channel_imbal_hvrx: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "channelImbalHVRx",
            "type": "Element",
            "required": True,
        },
    )
    channel_imbal_hvtx: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "channelImbalHVTx",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class CrossTalkList:
    """
    Parameters
    ----------
    cross_talk_hvrx
        H to V cross-talk on receive across swath.
    cross_talk_vhrx
        V to H cross-talk on receive across swath.
    cross_talk_vhtx
        V to H cross-talk on transmit across swath.
    cross_talk_hvtx
        H to V cross-talk on transmit across swath.
    """

    class Meta:
        name = "crossTalkList"

    cross_talk_hvrx: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "crossTalkHVRx",
            "type": "Element",
            "required": True,
        },
    )
    cross_talk_vhrx: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "crossTalkVHRx",
            "type": "Element",
            "required": True,
        },
    )
    cross_talk_vhtx: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "crossTalkVHTx",
            "type": "Element",
            "required": True,
        },
    )
    cross_talk_hvtx: Optional[Complex] = field(
        default=None,
        metadata={
            "name": "crossTalkHVTx",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class DatumType:
    """
    Parameters
    ----------
    coordinate_reference_system
        Coordinate reference system.
    geodetic_reference_frame
        Geodetic reference frame.
    """

    class Meta:
        name = "datumType"

    coordinate_reference_system: Optional[str] = field(
        default=None,
        metadata={
            "name": "coordinateReferenceSystem",
            "type": "Element",
            "required": True,
        },
    )
    geodetic_reference_frame: Optional[GeodeticReferenceFrameType] = field(
        default=None,
        metadata={
            "name": "geodeticReferenceFrame",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class DoubleArrayWithUnits:
    """String containing an array of double precision floating point values
    separated by spaces.

    The mandatory count attribute defines the number of elements in the array.
    """

    class Meta:
        name = "doubleArrayWithUnits"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    units: Optional[UomType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class DoubleWithUnit:
    """
    64 bit double precision floating point number.
    """

    class Meta:
        name = "doubleWithUnit"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    units: Optional[UomType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class FloatArrayWithUnits:
    """String containing an array of float values separated by spaces.

    The mandatory count attribute defines the number of elements in the array.
    """

    class Meta:
        name = "floatArrayWithUnits"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    units: Optional[UomType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class FloatWithChannel:
    """
    Extension of float with the indication of the corresponding RGB channel.
    """

    class Meta:
        name = "floatWithChannel"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    channel: Optional[ChannelType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class FloatWithPolarisation:
    """
    Extension of float with the indication of the corresponding polarisation.
    """

    class Meta:
        name = "floatWithPolarisation"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class FloatWithUnit:
    """
    32 bit single precision floating point number.
    """

    class Meta:
        name = "floatWithUnit"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    units: Optional[UomType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class HeightModelType:
    class Meta:
        name = "heightModelType"

    value: Optional[HeightModelBaseType] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    version: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class InterferometricPairListType:
    """
    Parameters
    ----------
    interferometric_pairs
        List of interferometric pairs expressed as indices.
    count
        Number of interferometric pairs.
    """

    class Meta:
        name = "interferometricPairListType"

    interferometric_pairs: list[InterferometricPairType] = field(
        default_factory=list,
        metadata={
            "name": "interferometricPairs",
            "type": "Element",
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class LayerListType:
    """
    Parameters
    ----------
    layer
        LUT layer content.
    count
        Number of layers.
    """

    class Meta:
        name = "layerListType"

    layer: list[LayerType] = field(
        default_factory=list, metadata={"type": "Element", "min_occurs": 1, "max_occurs": 29}
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class TimeTypeWithPolarisation:
    """
    Extension of timeType with the indication of the corresponding polarisation.
    """

    class Meta:
        name = "timeTypeWithPolarisation"

    value: str = field(
        default="", metadata={"required": True, "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}"}
    )
    polarisation: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class UnsignedIntWithGroup:
    """
    Extension of unsignedInt with the indication of the corresponding LUT group of
    variables.
    """

    class Meta:
        name = "unsignedIntWithGroup"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    group: Optional[GroupType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class AzimuthPolynomialType:
    """
    Parameters
    ----------
    slant_range_time
        Two-way slant range time corresponding to the start of the current polynomial validity area [s].
    t0
        Zero Doppler azimuth time origin for polynomial [UTC].
    polynomial
        Local estimate expressed as the following polynomial: X=c0+c1(tAZ-t0)+c2(tAZ-t0)^2+..., where tAZ is the
        Zero Doppler azimuth time.
    """

    class Meta:
        name = "azimuthPolynomialType"

    slant_range_time: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "name": "slantRangeTime",
            "type": "Element",
            "required": True,
        },
    )
    t0: Optional[str] = field(
        default=None,
        metadata={"type": "Element", "required": True, "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}"},
    )
    polynomial: Optional[DoubleArray] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class MinMaxTypeWithUnit:
    class Meta:
        name = "minMaxTypeWithUnit"

    min: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
    max: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class SlantRangePolynomialType:
    """
    Parameters
    ----------
    azimuth_time
        Zero Doppler azimuth time corresponding to the start of the current polynomial validity area [UTC].
    t0
        Two-way slant range time origin for polynomial [s].
    polynomial
        Local estimate expressed as the following polynomial: X=c0+c1(tSR-t0)+c2(tSR-t0)^2+..., where tSR is the
        two-way slant range time.
    """

    class Meta:
        name = "slantRangePolynomialType"

    azimuth_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    t0: Optional[DoubleWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
    polynomial: Optional[DoubleArray] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )


class AutofocusMethodType(Enum):
    """
    Enumeration of autofocus methods.
    """

    MAP_DRIFT = "Map Drift"


class BistaticDelayCorrectionMethodType(Enum):
    """
    Enumeration of bistatic delay correction methods.
    """

    BULK = "Bulk"
    FULL = "Full"


class DataFormatModeType(Enum):
    """
    Enumeration of compression method names.
    """

    BAQ_4_BIT = "BAQ 4 Bit"
    BAQ_5_BIT = "BAQ 5 Bit"
    BAQ_6_BIT = "BAQ 6 Bit"
    BYPASS = "Bypass"


class DcMethodType(Enum):
    """
    Enumeration of Doppler Centroid calculation/estimation methods.
    """

    GEOMETRY = "Geometry"
    COMBINED = "Combined"
    FIXED = "Fixed"


class InternalCalibrationSourceType(Enum):
    """
    Enumeration of the available internal calibration sources.
    """

    EXTRACTED = "Extracted"
    MODEL = "Model"


class IonosphereHeightEstimationMethodType(Enum):
    """
    Enumeration of ionosphere height estimation methods.
    """

    AUTOMATIC = "Automatic"
    FEATURE_TRACKING = "Feature Tracking"
    SQUINT_SENSITIVITY = "Squint Sensitivity"
    MODEL = "Model"
    FIXED = "Fixed"
    NA = "NA"


class MissionPhaseIdtype(Enum):
    """
    Enumeration of all valid BIOMASS mission phases.
    """

    COM = "COM"
    TOM = "TOM"
    INT = "INT"


class MissionType(Enum):
    """
    Enumeration of valid BIOMASS mission names.
    """

    BIOMASS = "BIOMASS"


class OrbitAttitudeSourceType(Enum):
    """Enumeration of value sources of orbit and attitude data.

    "Downlink" or "Auxiliary".
    """

    DOWNLINK = "Downlink"
    AUXILIARY = "Auxiliary"


class OrbitPassType(Enum):
    """
    Enumeration of the orbit pass direction values.
    """

    ASCENDING = "Ascending"
    DESCENDING = "Descending"


class PixelQuantityType(Enum):
    """
    Enumeration of pixel quantity types.
    """

    BETA_NOUGHT = "Beta-Nought"
    SIGMA_NOUGHT = "Sigma-Nought"
    GAMMA_NOUGHT = "Gamma-Nought"


class PixelRepresentationType(Enum):
    """
    Enumeration of image pixel representation types.
    """

    I_Q = "I Q"
    ABS_PHASE = "Abs Phase"
    ABS = "Abs"
    PROBABILITY_OF_CHANGE = "Probability of change"
    COMPUTED_FOREST_MASK = "Computed forest mask"
    FOREST_DISTURBANCE = "Forest Disturbance"
    HEAT_MAP = "Heat Map"
    FOREST_HEIGHT_M = "Forest Height [m]"
    FOREST_HEIGHT_QUALITY = "Forest height quality"
    FOREST_MASK = "Forest Mask"
    GROUND_CANCELLED_BACKSCATTER = "Ground Cancelled Backscatter"
    ABOVE_GROUND_BIOMASS_T_HA = "Above Ground Biomass [t/ha]"
    ABOVE_GROUND_BIOMASS_QUALITY_T_HA = "Above Ground Biomass quality [t/ha]"
    ACQUISITION_ID_IMAGE = "Acquisition ID image"


class PixelTypeType(Enum):
    """
    Enumeration of image pixel data types.
    """

    VALUE_32_BIT_FLOAT = "32 bit Float"
    VALUE_16_BIT_SIGNED_INTEGER = "16 bit Signed Integer"
    VALUE_16_BIT_UNSIGNED_INTEGER = "16 bit Unsigned Integer"
    VALUE_8_BIT_UNSIGNED_INTEGER = "8 bit Unsigned Integer"


class ProcessingModeType(Enum):
    """
    Enumeration of processing modes.
    """

    NOMINAL = "Nominal"
    PARC = "PARC"


class ProductCompositionType(Enum):
    """Enumeration of product composition indicators.

    The valid values are: “Nominal”, to indicate a framed product of nominal length; “Merged”, if the product is resulting from the merging of a “short” frame with the contiguous one; “Partial”, if the product length is not nominal, but it is not merged to a contiguous one since it is “long enough”; “Incomplete”, if the product length is not nominal due to contingency (e.g., data loss); “Not Framed”, if the product is not framed, i.e., as long as input L0 one.
    """

    NOMINAL = "Nominal"
    MERGED = "Merged"
    PARTIAL = "Partial"
    INCOMPLETE = "Incomplete"
    NOT_FRAMED = "Not Framed"


class ProductType(Enum):
    """
    Enumeration of valid product types.
    """

    SCS = "SCS"
    DGM = "DGM"
    STA = "STA"
    FH_L2_A = "FH_L2A"
    FH_L2_B = "FH_L2B"
    FD_L2_A = "FD_L2A"
    FD_L2_B = "FD_L2B"
    GN_L2_A = "GN_L2A"
    AGB_L2_B = "AGB_L2B"
    TFH_L2_A = "TFH_L2A"


class ProjectionType(Enum):
    """
    Enumeration of the image projection.
    """

    SLANT_RANGE = "Slant Range"
    GROUND_RANGE = "Ground Range"
    LATITUDE_LONGITUDE_BASED_ON_DGG = "Latitude-Longitude based on DGG"
    DGG = "DGG"


class RangeCompressionMethodType(Enum):
    """
    Enumeration of the available range compression methods.
    """

    MATCHED_FILTER = "Matched Filter"
    INVERSE_FILTER = "Inverse Filter"


class RangeReferenceFunctionType(Enum):
    """
    Enumeration of the available range reference functions.
    """

    NOMINAL = "Nominal"
    REPLICA = "Replica"
    INTERNAL = "Internal"


class RfiFmmitigationMethodType(Enum):
    NOTCH_FILTER = "NOTCH_FILTER"
    NEAREST_NEIGHBOUR_INTERPOLATION = "NEAREST_NEIGHBOUR_INTERPOLATION"


class RfiMaskGenerationMethodType(Enum):
    """
    Enumeration of RFI mask generation methods.
    """

    AND = "AND"
    OR = "OR"


class RfiMaskType(Enum):
    """
    Enumeration of RFI masks.
    """

    SINGLE = "Single"
    MULTIPLE = "Multiple"


class RfiMitigationMethodType(Enum):
    """
    Enumeration of RFI mitigation methods.
    """

    TIME = "Time"
    FREQUENCY = "Frequency"
    TIME_AND_FREQUENCY = "Time and Frequency"
    FREQUENCY_AND_TIME = "Frequency and Time"


class SensorModeType(Enum):
    """
    Enumeration of valid sensor mode abbreviations.
    """

    MEASUREMENT = "Measurement"
    RX_ONLY = "RX Only"
    EXTERNAL_CALIBRATION = "External Calibration"


class SwathType(Enum):
    """
    Enumeration of all valid swath identifiers.
    """

    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


class WeightingWindowType(Enum):
    """
    Enumeration of weighting window names.
    """

    KAISER = "Kaiser"
    HAMMING = "Hamming"
    NONE = "None"


class CompressionMethodType(Enum):
    """
    Enumeration of TIFF compression methods.
    """

    NONE = "NONE"
    DEFLATE = "DEFLATE"
    ZSTD = "ZSTD"
    LERC = "LERC"
    LERC_DEFLATE = "LERC_DEFLATE"
    LERC_ZSTD = "LERC_ZSTD"


class BaselineMethodType(Enum):
    """
    Enumeration of baseline method types (e.g. SingleBaseline or MultiBaseline).
    """

    SINGLE_BASELINE = "Single-Baseline"
    MULTI_BASELINE = "Multi-Baseline"


class CoregistrationExecutionPolicyType(Enum):
    """
    Enumeration of coregistration execution policy.
    """

    NOMINAL = "Nominal"
    SHIFT_ESTIMATION_ONLY = "Shift Estimation Only"
    WARPING_ONLY = "Warping Only"


class CoregistrationMethodType(Enum):
    """
    Enumeration of coregistration methods.
    """

    GEOMETRY = "Geometry"
    GEOMETRY_AND_DATA = "Geometry and Data"
    AUTOMATIC = "Automatic"


class PolarisationCombinationMethodType(Enum):
    """
    Enumeration of polarisations combination methods.
    """

    HV = "HV"
    VH = "VH"
    AVERAGE = "Average"
    NONE = "None"


class PrimaryImageSelectionInformationType(Enum):
    """
    Enumeration of information used to select coregistration primary image.
    """

    GEOMETRY = "Geometry"
    GEOMETRY_AND_RFI_CORRECTION = "Geometry and RFI Correction"
    GEOMETRY_AND_FR_CORRECTION = "Geometry and FR Correction"
    GEOMETRY_AND_RFI_FR_CORRECTIONS = "Geometry and RFI+FR Corrections"
    TEMPORAL_BASELINE = "Temporal Baseline"


class SkpPhaseCorrectionType(Enum):
    """
    Enumeration of supported corrections executed by SKP.
    """

    NONE = "None"
    FLATTENING_PHASE_SCREEN = "Flattening Phase Screen"
    GROUND_PHASE_SCREEN = "Ground Phase Screen"


class PrimaryImageSelectionMethodType(Enum):
    """
    Enumeration of coregistration primary image selection methods.
    """

    GEOMETRY = "Geometry"
    GEOMETRY_AND_QUALITY = "Geometry and Quality"
    TEMPORAL_BASELINE = "Temporal Baseline"


class AcquisitionModeIdtype(Enum):
    """
    Enumeration of all valid acquisition modes.
    """

    S1_INT = "S1 INT"
    S2_INT = "S2 INT"
    S3_INT = "S3 INT"
    S1_TOM = "S1 TOM"
    S2_TOM = "S2 TOM"
    S3_TOM = "S3 TOM"
    RX_ONLY = "RX Only"


class SignalType(Enum):
    """
    Enumeration of valid signal types.
    """

    TX_VAND_RX = "TxVandRx"
    TX_HAND_RX = "TxHandRx"
    RX_ONLY = "RxOnly"
    TX_ONLY_V = "TxOnly-V"
    TX_ONLY_H = "TxOnly-H"
    TX_CAL_V = "TxCal-V"
    TX_CAL_H = "TxCal-H"
    TX_CAL_V1 = "TxCal-V1"
    TX_CAL_V2 = "TxCal-V2"
    TX_CAL_H1 = "TxCal-H1"
    TX_CAL_H2 = "TxCal-H2"
    RX_CAL = "RxCal"
    SH_CAL = "ShCal"
    IDLE = "Idle"
    TX_VAND_RX_S = "TxVandRx_S"
    TX_HAND_RX_S = "TxHandRx_S"
    NOISE = "Noise"


@dataclass
class StateType:
    """
    Parameters
    ----------
    azimuth_time
        Zero Doppler azimuth time of current value [UTC].
    value
        Current value.
    """

    class Meta:
        name = "stateType"

    azimuth_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "azimuthTime",
            "type": "Element",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    value: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
