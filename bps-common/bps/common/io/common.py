# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Common
------
"""

from dataclasses import dataclass
from enum import Enum

from arepytools.timing.precisedatetime import PreciseDateTime


class ProcessingModeType(Enum):
    """Processing mode"""

    NOMINAL = "Nominal"
    PARC = "PARC"


class OrbitAttitudeSourceType(Enum):
    """Navigation source"""

    DOWNLINK = "Downlink"
    AUXILIARY = "Auxiliary"


class RfiMaskType(Enum):
    """RFI mask type"""

    SINGLE = "Single"
    MULTIPLE = "Multiple"


class RfiMitigationMethodType(Enum):
    """RFI mitigation methods"""

    TIME = "Time"
    FREQUENCY = "Frequency"
    TIME_AND_FREQUENCY = "Time and Frequency"
    FREQUENCY_AND_TIME = "Frequency and Time"


class RfiMaskGenerationMethodType(Enum):
    """RFI mask generation methods"""

    AND = "AND"
    OR = "OR"


class RangeCompressionMethodType(Enum):
    """Available range compression methods"""

    MATCHED_FILTER = "Matched Filter"
    INVERSE_FILTER = "Inverse Filter"


class RangeReferenceFunctionType(Enum):
    """Available range reference functions"""

    NOMINAL = "Nominal"
    REPLICA = "Replica"
    INTERNAL = "Internal"


class DcMethodType(Enum):
    """Doppler Centroid calculation/estimation methods"""

    GEOMETRY = "Geometry"
    COMBINED = "Combined"
    FIXED = "Fixed"


class WeightingWindowType(Enum):
    """Weighting window names"""

    KAISER = "Kaiser"
    HAMMING = "Hamming"
    NONE = "None"


class BistaticDelayCorrectionMethodType(Enum):
    """bistatic delay correction methods"""

    BULK = "Bulk"
    FULL = "Full"


class PolarisationType(Enum):
    """valid polarisations"""

    H = "H"
    V = "V"
    HH = "HH"
    HV = "HV"
    VH = "VH"
    VV = "VV"
    XX = "XX"


FloatWithPolarisation = tuple[float, PolarisationType]


class IonosphereHeightEstimationMethodType(Enum):
    """Ionosphere height estimation methods"""

    AUTOMATIC = "Automatic"
    FEATURE_TRACKING = "Feature Tracking"
    SQUINT_SENSITIVITY = "Squint Sensitivity"
    MODEL = "Model"
    FIXED = "Fixed"
    NA = "NA"


class IonosphereType(Enum):
    """Ionosphere types estiamted by the BIC module."""

    LINEAR = "Linear"
    FAST_VARYING = "Fast-varying"


class AutofocusMethodType(Enum):
    """Autofocus methods."""

    MAP_DRIFT = "Map Drift"


class UomType(Enum):
    """Unit of measures."""

    S = "s"
    UTC = "UTC"
    M = "m"
    SAMPLES = "samples"
    LINES = "lines"
    DEG = "deg"
    RAD = "rad"
    HZ = "Hz"
    HZ_S = "Hz/s"
    D_B = "dB"
    MBPS = "Mbps"
    C = "C"
    K = "K"
    DEG_S = "deg/s"
    MM_KM = "mm/Km"
    DEG_M = "deg/m"
    VALUE_1_M = "1/m"
    T_HA = "t/ha"
    RAD_N_T_2 = "(rad/nT)^2"
    VALUE = ""


class ProjectionType(Enum):
    """Image projection"""

    SLANT_RANGE = "Slant Range"
    GROUND_RANGE = "Ground Range"
    LATITUDE_LONGITUDE_BASED_ON_DGG = "Latitude-Longitude based on DGG"
    DGG = "DGG"


class GeodeticReferenceFrameType(Enum):
    """Reference frame"""

    WGS84 = "WGS84"
    ITRF2000 = "ITRF2000"


@dataclass
class DatumType:
    """Datum type"""

    coordinate_reference_system: str
    geodetic_reference_frame: GeodeticReferenceFrameType


class PixelRepresentationType(Enum):
    """Image pixel representation types"""

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
    """Image pixel data types"""

    VALUE_32_BIT_FLOAT = "32 bit Float"
    VALUE_16_BIT_SIGNED_INTEGER = "16 bit Signed Integer"
    VALUE_16_BIT_UNSIGNED_INTEGER = "16 bit Unsigned Integer"
    VALUE_8_BIT_UNSIGNED_INTEGER = "8 bit Unsigned Integer"


class PixelQuantityType(Enum):
    """Pixel quantity types"""

    BETA_NOUGHT = "Beta-Nought"
    SIGMA_NOUGHT = "Sigma-Nought"
    GAMMA_NOUGHT = "Gamma-Nought"


class MissionType(Enum):
    """BIOMASS mission name"""

    BIOMASS = "BIOMASS"


class SwathType(Enum):
    """All valid swath identifiers"""

    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


class ProductType(Enum):
    """Valid product types"""

    SCS = "SCS"
    DGM = "DGM"
    STA = "STA"
    FH_L2_A = "FH_L2A"
    FH_L2_B = "FH_L2B"
    FD_L2_A = "FD_L2A"
    FD_L2_B = "FD_L2B"
    GN_L2_A = "GN_L2A"
    AGB_L2_B = "AGB_L2B"


class MissionPhaseIdtype(Enum):
    """All valid BIOMASS mission phases"""

    COM = "COM"
    TOM = "TOM"
    INT = "INT"


class SensorModeType(Enum):
    """Valid sensor modes"""

    MEASUREMENT = "Measurement"
    RX_ONLY = "RX Only"
    EXTERNAL_CALIBRATION = "External Calibration"


class OrbitPassType(Enum):
    """Orbit pass direction"""

    ASCENDING = "Ascending"
    DESCENDING = "Descending"


class ProductCompositionType(Enum):
    """Product composition indicators"""

    NOMINAL = "Nominal"
    MERGED = "Merged"
    PARTIAL = "Partial"
    INCOMPLETE = "Incomplete"
    NOT_FRAMED = "Not Framed"


class DataFormatModeType(Enum):
    """compression method names"""

    BAQ_4_BIT = "BAQ 4 Bit"
    BAQ_5_BIT = "BAQ 5 Bit"
    BAQ_6_BIT = "BAQ 6 Bit"
    BYPASS = "Bypass"


@dataclass
class SlantRangePolynomialType:
    """Slant range polynomial type"""

    azimuth_time: PreciseDateTime
    t0: float
    polynomial: list[float]


class InternalCalibrationSourceType(Enum):
    """Available internal calibration sources"""

    EXTRACTED = "Extracted"
    MODEL = "Model"


class HeightModelBaseType(Enum):
    """Height models"""

    ELLIPSOID = "Ellipsoid"
    SRTM = "SRTM"
    COPERNICUS_DEM = "Copernicus DEM"
    BIOMASS_DTM = "BIOMASS DTM"


@dataclass
class HeightModelType:
    """Height model with version"""

    value: HeightModelBaseType
    version: str


class LayerType(Enum):
    """LUT layer contents"""

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
    RANGE_COREGISTRATION_SHIFTS_M = "Range coregistration shifts [m]"
    COREGISTRATION_SHIFTS_QUALITY = "Coregistration shifts quality"
    WAVENUMBERS_RAD_M = "Wavenumbers [rad/m]"
    FLATTENING_PHASE_SCREEN_RAD = "Flattening phase screen [rad]"
    BASELINE_ERROR_PHASE_SCREEN_RAD = "Baseline error phase screen [rad]"
    RESIDUAL_IONOSPHERE_PHASE_SCREEN_RAD = "Residual ionosphere phase screen [rad]"
    RESIDUAL_IONOSPHERE_PHASE_SCREEN_QUALITY = "Residual ionosphere phase screen quality"
    SKP_CALIBRATION_PHASE_SCREEN_RAD = "SKP calibration phase screen [rad]"
    SKP_CALIBRATION_PHASE_SCREEN_QUALITY = "SKP calibration phase screen quality"


class PolarisationCombinationMethodType(Enum):
    """Polarisations combination methods"""

    HV = "HV"
    VH = "VH"
    AVERAGE = "Average"


class PrimaryImageSelectionInformationType(Enum):
    """information used to select coregistration primary image"""

    GEOMETRY = "Geometry"
    GEOMETRY_AND_RFI_CORRECTION = "Geometry and RFI Correction"
    GEOMETRY_AND_FR_CORRECTION = "Geometry and FR Correction"
    GEOMETRY_AND_RFI_FR_CORRECTIONS = "Geometry and RFI+FR Corrections"
    TEMPORAL_BASELINE = "Temporal Baseline"


class CoregistrationMethodType(Enum):
    """Coregistration methods"""

    GEOMETRY = "Geometry"
    GEOMETRY_AND_DATA = "Geometry and Data"
    AUTOMATIC = "Automatic"


class CoregistrationExecutionPolicyType(Enum):
    """Coregistration execution policy."""

    NOMINAL = "Nominal"
    SHIFT_ESTIMATION_ONLY = "Shift Estimation Only"
    WARPING_ONLY = "Warping Only"


@dataclass
class CrossTalkList:
    """Cross talk list"""

    hv_rx: complex
    vh_rx: complex
    vh_tx: complex
    hv_tx: complex


@dataclass
class ChannelImbalanceList:
    """Channel imbalance list"""

    hv_rx: complex
    hv_tx: complex
