# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD BPS Configuration file models
---------------------------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class ChannelType:
    number: Optional[int] = field(
        default=None,
        metadata={
            "name": "Number",
            "type": "Attribute",
        },
    )
    total: Optional[int] = field(
        default=None,
        metadata={
            "name": "Total",
            "type": "Attribute",
        },
    )


class AttitudeFittingTypes(Enum):
    LINEAR = "LINEAR"
    AVERAGE = "AVERAGE"
    DISABLED = "DISABLED"


class AntennaShiftCompensationModeType(Enum):
    """
    Antenna shift compensation mode (DISABLED: antenna shift compensation disabled,
    FORCED: antenna shift compensation always performed)
    """

    DISABLED = "DISABLED"
    FORCED = "FORCED"


class BistaticDelayCorrectionTypes(Enum):
    BIAS_ONLY = "BIAS_ONLY"
    NEAR_RANGE = "NEAR_RANGE"
    MIDDLE_RANGE = "MIDDLE_RANGE"
    SCENE_CENTER = "SCENE_CENTER"
    RANGE_DEPENDENT = "RANGE_DEPENDENT"


class BpsantennaPatternCompensationType(Enum):
    DISABLED = "DISABLED"
    APC_PRE_ONLY = "APC_PRE_ONLY"
    APC_PRE_POST_ONLY = "APC_PRE_POST_ONLY"
    APC_PRE_CROSS_ONLY = "APC_PRE_CROSS_ONLY"
    FULL = "FULL"


class Bpsloglevels(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    PROGRESS = "PROGRESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class BaseConfType:
    class Meta:
        target_namespace = "aresysConfTypes"

    beam: Optional[str] = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


class BiomassIntCalConfTypeInternalCalibrationSource(Enum):
    EXTRACTED = "EXTRACTED"
    MODEL = "MODEL"


@dataclass
class BiomassL0ImportConfType:
    class Meta:
        target_namespace = "aresysConfTypes"

    max_ispgap: Optional[int] = field(
        default=None,
        metadata={
            "name": "MaxISPGap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_time_gap: Optional[float] = field(
        default=None,
        metadata={
            "name": "MaxTimeGap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_mean_expected: Optional[float] = field(
        default=None,
        metadata={
            "name": "RawMeanExpected",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_mean_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "RawMeanThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_std_expected: Optional[float] = field(
        default=None,
        metadata={
            "name": "RawStdExpected",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_std_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "RawStdThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    beam: Optional[str] = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


@dataclass
class BiomassRawDataCorrectionsConfType:
    class Meta:
        target_namespace = "aresysConfTypes"

    correct_bias: Optional[bool] = field(
        default=None,
        metadata={
            "name": "CorrectBias",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    correct_gain_imbalance: Optional[bool] = field(
        default=None,
        metadata={
            "name": "CorrectGainImbalance",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    correct_non_orthogonality: Optional[bool] = field(
        default=None,
        metadata={
            "name": "CorrectNonOrthogonality",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    beam: Optional[str] = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


@dataclass
class ComplexAlg:
    class Meta:
        name = "COMPLEX_ALG"
        target_namespace = "aresysConfTypes"

    real_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "RealValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    imaginary_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "ImaginaryValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class ComplexPol:
    class Meta:
        name = "COMPLEX_POL"
        target_namespace = "aresysConfTypes"

    abs_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "AbsValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    phase_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "PhaseValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


class CoregMode(Enum):
    GEOMETRY = "GEOMETRY"
    FULL_ACCURACY = "FULL_ACCURACY"
    AUTOMATIC = "AUTOMATIC"


@dataclass
class ComplexNumberType:
    class Meta:
        target_namespace = "aresysConfTypes"

    amplitude: Optional[float] = field(
        default=None,
        metadata={
            "name": "Amplitude",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    phase: Optional[float] = field(
        default=None,
        metadata={
            "name": "Phase",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class CoregistrationOutputProductsConfType:
    """
    Parameters
    ----------
    remove_ancillary_coregistration_data
    provide_coregistration_shifts
    provide_coregistration_accuracy_stats
    xcorr_azimuth_min_overlap
    provide_products_for_each_polarization
        Coregistration byproducts outputs (shifts, dsi and wavenumbers) are computed on the reference polarization
        only. This flag allows to choose between saving only a file for each byproduct (0), or duplicate it for the
        remaining polarizations (1, default).
    provide_wavenumbers
    shifts_only_estimation
    provide_absolute_primary_distance
    provide_geometry_shifts
    provide_full_accuracy_shifts
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    remove_ancillary_coregistration_data: int = field(
        default=0,
        metadata={
            "name": "RemoveAncillaryCoregistrationData",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_coregistration_shifts: int = field(
        default=0,
        metadata={
            "name": "ProvideCoregistrationShifts",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_coregistration_accuracy_stats: int = field(
        default=0,
        metadata={
            "name": "ProvideCoregistrationAccuracyStats",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    xcorr_azimuth_min_overlap: Optional[int] = field(
        default=None,
        metadata={
            "name": "XCorrAzimuthMinOverlap",
            "type": "Element",
            "namespace": "",
        },
    )
    provide_products_for_each_polarization: Optional[int] = field(
        default=None,
        metadata={
            "name": "ProvideProductsForEachPolarization",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_wavenumbers: Optional[int] = field(
        default=None,
        metadata={
            "name": "ProvideWavenumbers",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    shifts_only_estimation: Optional[int] = field(
        default=None,
        metadata={
            "name": "ShiftsOnlyEstimation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_absolute_primary_distance: Optional[int] = field(
        default=None,
        metadata={
            "name": "ProvideAbsolutePrimaryDistance",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_geometry_shifts: Optional[int] = field(
        default=None,
        metadata={
            "name": "ProvideGeometryShifts",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_full_accuracy_shifts: Optional[int] = field(
        default=None,
        metadata={
            "name": "ProvideFullAccuracyShifts",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )


@dataclass
class DccoreAlgorithmType:
    class Meta:
        name = "DCCoreAlgorithmType"
        target_namespace = "aresysConfTypes"

    mle: Optional["DccoreAlgorithmType.Mle"] = field(
        default=None,
        metadata={
            "name": "MLE",
            "type": "Element",
            "namespace": "",
        },
    )
    jpl: Optional[object] = field(
        default=None,
        metadata={
            "name": "JPL",
            "type": "Element",
            "namespace": "",
        },
    )
    cde: Optional[object] = field(
        default=None,
        metadata={
            "name": "CDE",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class Mle:
        iterations: Optional[int] = field(
            default=None,
            metadata={
                "name": "Iterations",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        delta_f: Optional[float] = field(
            default=None,
            metadata={
                "name": "DeltaF",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        antenna_wrapping_lateral_bands: Optional[int] = field(
            default=None,
            metadata={
                "name": "AntennaWrappingLateralBands",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )


class DcEstimationMethodsTypes(Enum):
    GEOMETRICAL = "GEOMETRICAL"
    DATA = "DATA"
    COMBINED = "COMBINED"


class DemTypes(Enum):
    SRTM = "SRTM"
    GETASSE = "GETASSE"
    COPERNICUS = "COPERNICUS"
    WGS84 = "WGS84"


class EarthGeometry(Enum):
    ELLIPSOID = "ELLIPSOID"
    DEM = "DEM"


class FilterMask(Enum):
    AVERAGE = "AVERAGE"
    GAUSSIAN = "GAUSSIAN"


class FocusingMethodTypes(Enum):
    WK = "WK"
    CZT = "CZT"
    BP = "BP"


class FormatType(Enum):
    BIN_XML = "BIN+XML"
    TIFF_XML = "TIFF+XML"


class FullAccuracyPostProcessingConfTypeResidualShiftFittingModel(Enum):
    POLYNOMIAL = "POLYNOMIAL"
    MODEL_BASED = "MODEL_BASED"


class InterpType(Enum):
    LINEAR = "LINEAR"
    FFT = "FFT"


class IonosphericCalibrationConfTypeIonosphericHeightEstimationMethod(Enum):
    NONE = "None"
    FEATURE_TRACKING = "FeatureTracking"
    SQUINT_SENSITIVITY = "SquintSensitivity"
    MODEL = "Model"
    AUTO = "Auto"


class Loglevels(Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    DEBUG = "DEBUG"


@dataclass
class MemorySizeType:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    uo_m: str = field(
        init=False,
        default="MB",
        metadata={
            "name": "UoM",
            "type": "Attribute",
            "required": True,
        },
    )


class OutputBorderPolicyType(Enum):
    """
    Output border policy (CUT: remove border, PAD: replace data at border with a
    padding value, DATA: keep data at border)
    """

    CUT = "CUT"
    PAD = "PAD"
    DATA = "DATA"


class OutputQuantityType(Enum):
    SIGMA = "SIGMA"
    GAMMA = "GAMMA"


class PolyEstimationConstraintTypes(Enum):
    FULL = "FULL"
    UNCONSTRAINED = "UNCONSTRAINED"


class PolarizationType(Enum):
    H_H = "H/H"
    H_V = "H/V"
    V_H = "V/H"
    V_V = "V/V"
    X_X = "X/X"


class RfifrequencyDomainRemovalConfTypeFilteringMode(Enum):
    NEAREST_NEIGHBOUR = "NEAREST_NEIGHBOUR"
    NOISE = "NOISE"
    NOTCH = "NOTCH"


class RfimaskCompositionMethodsType(Enum):
    AND = "AND"
    OR = "OR"
    NONE = "NONE"


class RfimitigationMethodsType(Enum):
    FREQUENCY = "FREQUENCY"
    TIME = "TIME"
    TIME_AND_FREQUENCY = "TIME_AND_FREQUENCY"
    FREQUENCY_AND_TIME = "FREQUENCY_AND_TIME"


class RfimitigationSettingsChirpSource(Enum):
    ANNOTATION = "ANNOTATION"
    PRODUCT = "PRODUCT"
    AUTO = "AUTO"


class RfimitigationSettingsMode(Enum):
    DETECTION_ONLY = "DETECTION_ONLY"
    DETECTION_AND_MITIGATION = "DETECTION_AND_MITIGATION"


class RfitimeDomainRemovalConfTypeCorrectionMode(Enum):
    NEAREST = "NEAREST"
    ZERO = "ZERO"
    GAUSSNOISE = "GAUSSNOISE"


class RangeFocusingMethodType(Enum):
    """
    Range focusing method (MATCHED_FILTER: matched filter method, INVERSE_FILTER:
    inverse filter method, INVERSE_FFT: focusing method for dechirped data)
    """

    MATCHED_FILTER = "MATCHED_FILTER"
    INVERSE_FILTER = "INVERSE_FILTER"
    INVERSE_FFT = "INVERSE_FFT"


@dataclass
class RealValueDegrees:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    units: str = field(
        init=False,
        default="deg",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RealValueMeters:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    units: str = field(
        init=False,
        default="m",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ReinterpolationConfType:
    """
    Parameters
    ----------
    filter_length
        Length of the Filter used for the Coregistration
    bank_size
    bandwidth
    range_overlap
    demodulation_type
        Generic configuration parameters
    unsigned_flag
    memory
    verbose
    report_level
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    filter_length: int = field(
        default=11,
        metadata={
            "name": "FilterLength",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    bank_size: int = field(
        default=101,
        metadata={
            "name": "BankSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    bandwidth: float = field(
        default=0.80000001,
        metadata={
            "name": "Bandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_overlap: int = field(
        default=1,
        metadata={
            "name": "RangeOverlap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    demodulation_type: int = field(
        default=0,
        metadata={
            "name": "DemodulationType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    unsigned_flag: int = field(
        default=0,
        metadata={
            "name": "UnsignedFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    memory: int = field(
        default=256,
        metadata={
            "name": "Memory",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    verbose: int = field(
        default=0,
        metadata={
            "name": "Verbose",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    report_level: int = field(
        default=0,
        metadata={
            "name": "ReportLevel",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class SarfocProcessingStepType:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: Optional[bool] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "name": "ID",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class SarfocProductType:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "name": "ID",
            "type": "Attribute",
            "required": True,
        },
    )


class UnitTypes(Enum):
    NORMALIZED = "Normalized"
    HZ = "Hz"
    S = "s"


class Windows(Enum):
    HAMMING = "HAMMING"
    KAISER = "KAISER"


@dataclass
class Bpsl1CoreProcessorInterfaceSettingsType:
    class Meta:
        name = "BPSL1CoreProcessorInterfaceSettingsType"
        target_namespace = "aresysConfTypes"

    products_format: Optional[FormatType] = field(
        default=None,
        metadata={
            "name": "ProductsFormat",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    enable_quick_look_generation: Optional[bool] = field(
        default=None,
        metadata={
            "name": "EnableQuickLookGeneration",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    remove_intermediate_products: Optional[bool] = field(
        default=None,
        metadata={
            "name": "RemoveIntermediateProducts",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class BpsloggerConfType:
    """
    BPS Logger configuration parameters.

    Parameters
    ----------
    node_name
        Name of the node on which the processor is running on
    processor_name
        Processor name
    processor_version
        Processor version
    task_name
        Name of the task issuing the message
    std_out_log_level
        Report level: DEBUG, INFO, PROGRESS, WARNING, ERROR
    std_err_log_level
        Report level: DEBUG, INFO, PROGRESS, WARNING, ERROR
    """

    class Meta:
        name = "BPSLoggerConfType"
        target_namespace = "aresysConfTypes"

    node_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "NodeName",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processor_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "ProcessorName",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processor_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "ProcessorVersion",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "TaskName",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    std_out_log_level: Optional[Bpsloglevels] = field(
        default=None,
        metadata={
            "name": "StdOutLogLevel",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    std_err_log_level: Optional[Bpsloglevels] = field(
        default=None,
        metadata={
            "name": "StdErrLogLevel",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class BiomassIntCalConfType:
    class Meta:
        target_namespace = "aresysConfTypes"

    internal_calibration_source: Optional[BiomassIntCalConfTypeInternalCalibrationSource] = field(
        default=None,
        metadata={
            "name": "InternalCalibrationSource",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_drift_amplitude_std_fraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "MaxDriftAmplitudeStdFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_drift_phase_std_fraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "MaxDriftPhaseStdFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_drift_amplitude_error: Optional[float] = field(
        default=None,
        metadata={
            "name": "MaxDriftAmplitudeError",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_drift_phase_error: Optional[float] = field(
        default=None,
        metadata={
            "name": "MaxDriftPhaseError",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_invalid_drift_fraction: Optional[float] = field(
        default=None,
        metadata={
            "name": "MaxInvalidDriftFraction",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    beam: Optional[str] = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


@dataclass
class CalibrationConstantsConfType:
    """
    Polarimetric Distortion Optimization configuration parameters.
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    radiometric: Optional["CalibrationConstantsConfType.Radiometric"] = field(
        default=None,
        metadata={
            "name": "Radiometric",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    geometric: Optional["CalibrationConstantsConfType.Geometric"] = field(
        default=None,
        metadata={
            "name": "Geometric",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class Radiometric:
        """
        Parameters
        ----------
        channel_imbalance_values
            Channel Imbalance complex value normalized against ICAL  []
        cross_talk_correction
            Xtalks correction parameters to override scene based processor estimations
        """

        channel_imbalance_values: Optional["CalibrationConstantsConfType.Radiometric.ChannelImbalanceValues"] = field(
            default=None,
            metadata={
                "name": "ChannelImbalanceValues",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        cross_talk_correction: Optional["CalibrationConstantsConfType.Radiometric.CrossTalkCorrection"] = field(
            default=None,
            metadata={
                "name": "CrossTalkCorrection",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class ChannelImbalanceValues:
            rx: Optional[ComplexNumberType] = field(
                default=None,
                metadata={
                    "name": "Rx",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            tx: Optional[ComplexNumberType] = field(
                default=None,
                metadata={
                    "name": "Tx",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )

        @dataclass
        class CrossTalkCorrection:
            xtalk1: Optional[ComplexNumberType] = field(
                default=None,
                metadata={
                    "name": "Xtalk1",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            xtalk2: Optional[ComplexNumberType] = field(
                default=None,
                metadata={
                    "name": "Xtalk2",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            xtalk3: Optional[ComplexNumberType] = field(
                default=None,
                metadata={
                    "name": "Xtalk3",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            xtalk4: Optional[ComplexNumberType] = field(
                default=None,
                metadata={
                    "name": "Xtalk4",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )
            alpha: Optional[ComplexNumberType] = field(
                default=None,
                metadata={
                    "name": "Alpha",
                    "type": "Element",
                    "namespace": "",
                },
            )

    @dataclass
    class Geometric:
        """
        Parameters
        ----------
        internal_delay_hh
            Time delay associate to polarization HH expressed in [seconds]
        internal_delay_hv
            Time delay associate to polarization HV expressed in [seconds]
        internal_delay_vh
            Time delay associate to polarization VH expressed in [seconds]
        internal_delay_vv
            Time delay associate to polarization VV expressed in [seconds]
        """

        internal_delay_hh: Optional[float] = field(
            default=None,
            metadata={
                "name": "InternalDelayHH",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        internal_delay_hv: Optional[float] = field(
            default=None,
            metadata={
                "name": "InternalDelayHV",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        internal_delay_vh: Optional[float] = field(
            default=None,
            metadata={
                "name": "InternalDelayVH",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        internal_delay_vv: Optional[float] = field(
            default=None,
            metadata={
                "name": "InternalDelayVV",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )


@dataclass
class DigitalElevationModelType:
    class Meta:
        target_namespace = "aresysConfTypes"

    type_value: Optional[DemTypes] = field(
        default=None,
        metadata={
            "name": "Type",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    repository: Optional[str] = field(
        default=None,
        metadata={
            "name": "Repository",
            "type": "Element",
            "namespace": "",
        },
    )
    index_file_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "IndexFileName",
            "type": "Element",
            "namespace": "",
        },
    )
    geoid_file_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "GeoidFileName",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class Fcomplex:
    class Meta:
        name = "FCOMPLEX"
        target_namespace = "aresysConfTypes"

    complex_alg: Optional[ComplexAlg] = field(
        default=None,
        metadata={
            "name": "ComplexAlg",
            "type": "Element",
            "namespace": "",
        },
    )
    complex_pol: Optional[ComplexPol] = field(
        default=None,
        metadata={
            "name": "ComplexPol",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class FullAccuracyPreProcessingConfType:
    """
    Generic configuration parameters.

    Parameters
    ----------
    coreg_reference_polarization
    range_max_shift
        Maxima Shift in Range
    azimuth_max_shift
        Maxima Shift in Azimuth
    range_block_size
        Block Size in Range
    azimuth_block_size
        Block Size in Azimuth
    coarse_input
        Coarse input flag
    range_min_overlap
        Minuma Overlap on Range
    azimuth_min_overlap
        Minuma Overlap on Azimuth
    memory
    verbose
    report_level
    enable_common_band_range_filter
        Enable Common Bandwidth Range Filter
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    coreg_reference_polarization: Optional[PolarizationType] = field(
        default=None,
        metadata={
            "name": "CoregReferencePolarization",
            "type": "Element",
            "namespace": "",
        },
    )
    range_max_shift: int = field(
        default=4,
        metadata={
            "name": "RangeMaxShift",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_max_shift: int = field(
        default=4,
        metadata={
            "name": "AzimuthMaxShift",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_block_size: int = field(
        default=0,
        metadata={
            "name": "RangeBlockSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_block_size: int = field(
        default=0,
        metadata={
            "name": "AzimuthBlockSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    coarse_input: int = field(
        default=0,
        metadata={
            "name": "CoarseInput",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_min_overlap: int = field(
        default=1000,
        metadata={
            "name": "RangeMinOverlap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_min_overlap: int = field(
        default=1000,
        metadata={
            "name": "AzimuthMinOverlap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    memory: int = field(
        default=256,
        metadata={
            "name": "Memory",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    verbose: int = field(
        default=0,
        metadata={
            "name": "Verbose",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    report_level: int = field(
        default=0,
        metadata={
            "name": "ReportLevel",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    enable_common_band_range_filter: Optional[bool] = field(
        default=None,
        metadata={
            "name": "EnableCommonBandRangeFilter",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class IonosphericCalibrationConfType(BaseConfType):
    class Meta:
        target_namespace = "aresysConfTypes"

    perform_defocusing_on_ionospheric_height: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformDefocusingOnIonosphericHeight",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    perform_faraday_rotation_correction: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformFaradayRotationCorrection",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    perform_phase_screen_correction: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformPhaseScreenCorrection",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    perform_group_delay_correction: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformGroupDelayCorrection",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    ionospheric_height_estimation_method: Optional[IonosphericCalibrationConfTypeIonosphericHeightEstimationMethod] = (
        field(
            default=None,
            metadata={
                "name": "IonosphericHeightEstimationMethod",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
    )
    squint_sensitivity: Optional["IonosphericCalibrationConfType.SquintSensitivity"] = field(
        default=None,
        metadata={
            "name": "SquintSensitivity",
            "type": "Element",
            "namespace": "",
        },
    )
    feature_tracking: Optional["IonosphericCalibrationConfType.FeatureTracking"] = field(
        default=None,
        metadata={
            "name": "FeatureTracking",
            "type": "Element",
            "namespace": "",
        },
    )
    zthreshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "ZThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    gaussian_filter_max_size_azimuth: Optional[int] = field(
        default=None,
        metadata={
            "name": "GaussianFilterMaxSizeAzimuth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    gaussian_filter_max_size_range: Optional[int] = field(
        default=None,
        metadata={
            "name": "GaussianFilterMaxSizeRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    gaussian_filter_default_size_azimuth: Optional[int] = field(
        default=None,
        metadata={
            "name": "GaussianFilterDefaultSizeAzimuth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    gaussian_filter_default_size_range: Optional[int] = field(
        default=None,
        metadata={
            "name": "GaussianFilterDefaultSizeRange",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    default_ionospheric_height: Optional[RealValueMeters] = field(
        default=None,
        metadata={
            "name": "DefaultIonosphericHeight",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    max_ionospheric_height: Optional[RealValueMeters] = field(
        default=None,
        metadata={
            "name": "MaxIonosphericHeight",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    min_ionospheric_height: Optional[RealValueMeters] = field(
        default=None,
        metadata={
            "name": "MinIonosphericHeight",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_block_overlap: Optional[int] = field(
        default=None,
        metadata={
            "name": "AzimuthBlockOverlap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_block_size: Optional[int] = field(
        default=None,
        metadata={
            "name": "AzimuthBlockSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class SquintSensitivity:
        number_of_looks: Optional[int] = field(
            default=None,
            metadata={
                "name": "NumberOfLooks",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        height_step: Optional[float] = field(
            default=None,
            metadata={
                "name": "HeightStep",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        faraday_rotation_bias: Optional[RealValueDegrees] = field(
            default=None,
            metadata={
                "name": "FaradayRotationBias",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

    @dataclass
    class FeatureTracking:
        max_offset: Optional[int] = field(
            default=None,
            metadata={
                "name": "MaxOffset",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        profile_step: Optional[int] = field(
            default=None,
            metadata={
                "name": "ProfileStep",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        normalized_min_value_threshold: Optional[float] = field(
            default=None,
            metadata={
                "name": "NormalizedMinValueThreshold",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )


@dataclass
class LoggerConfType:
    """
    Report configuration parameters.

    Parameters
    ----------
    enable_log_file
        Report on file flag (0: report file not written, 1: report file written)
    enable_std_output
        Report on standard output flag (0: report not displayed, 1: report displayed)
    report_level
        Report level: NONE, LOW, MEDIUM, HIGH, DEBUG
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    enable_log_file: Optional[int] = field(
        default=None,
        metadata={
            "name": "EnableLogFile",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    enable_std_output: Optional[int] = field(
        default=None,
        metadata={
            "name": "EnableStdOutput",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    report_level: Optional[Loglevels] = field(
        default=None,
        metadata={
            "name": "ReportLevel",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class NoiseMapGeneratorConfType(BaseConfType):
    """
    Parameters
    ----------
    noise_normalization_constant
        Noise data normalization factor
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    noise_normalization_constant: Optional[float] = field(
        default=None,
        metadata={
            "name": "NoiseNormalizationConstant",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class NonStationaryCoregConfType:
    """
    Parameters
    ----------
    lpfilter_type
        Low pass filter type [Default: AVERAGE]
    order
        Filter order type [DEFAULT: 9]
    parameter
        Adding Parameters according to Filter type. Active only for GAUSSIAN filtering [ Gaussian Sigma: ( DEFAULT:
        0.84089642) ]
    interp_type
        Interpolation Type [DEPRECATED always setted to LINEAR]
    azimuth_adaptive_step_second
        Polynomial adaptive regression step [DEPRECATED]
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    lpfilter_type: Optional[FilterMask] = field(
        default=None,
        metadata={
            "name": "LPFilterType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    order: Optional[int] = field(
        default=None,
        metadata={
            "name": "Order",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    parameter: Optional[float] = field(
        default=None,
        metadata={
            "name": "Parameter",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    interp_type: Optional[InterpType] = field(
        default=None,
        metadata={
            "name": "InterpType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_adaptive_step_second: Optional[float] = field(
        default=None,
        metadata={
            "name": "AzimuthAdaptiveStepSecond",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class PolarimetricProcessorConfType(BaseConfType):
    """
    Parameters
    ----------
    enable_cross_talk_compensation
        Enable Cross-Talk Compensation (0: No Cross-Talk Compensation, 1: Cross-Talk Compensation)
    enable_channel_imbalance_compensation
        Enable Channel Imbalance Compensation (0: No Channel Imbalance Compensation, 1: Channel Imbalance
        Compensation)
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    enable_cross_talk_compensation: Optional[int] = field(
        default=None,
        metadata={
            "name": "EnableCrossTalkCompensation",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    enable_channel_imbalance_compensation: Optional[int] = field(
        default=None,
        metadata={
            "name": "EnableChannelImbalanceCompensation",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )


@dataclass
class RfifrequencyDomainRemovalConfType(BaseConfType):
    """
    Frequency Domain Interference Removal Configuration.

    Parameters
    ----------
    remove_interferences
        Interference removal flag (0: not performed, 1: performed)
    block_size
        Report configuration parameters
    periodgram_size
        Report configuration parameters
    persistent_rfithreshold
        Report configuration parameters
    isolated_rfithreshold
        Report configuration parameters
    power_loss_threshold
        Report configuration parameters
    threshold_std
        Report configuration parameters
    percentile_low
        Report configuration parameters
    percentile_high
        Report configuration parameters
    filtering_mode
    """

    class Meta:
        name = "RFIFrequencyDomainRemovalConfType"
        target_namespace = "aresysConfTypes"

    remove_interferences: Optional[int] = field(
        default=None,
        metadata={
            "name": "RemoveInterferences",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "name": "BlockSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    periodgram_size: Optional[int] = field(
        default=None,
        metadata={
            "name": "PeriodgramSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    persistent_rfithreshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "PersistentRFIThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    isolated_rfithreshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "IsolatedRFIThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    power_loss_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "PowerLossThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    threshold_std: Optional[float] = field(
        default=None,
        metadata={
            "name": "ThresholdStd",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    percentile_low: Optional[float] = field(
        default=None,
        metadata={
            "name": "PercentileLow",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    percentile_high: Optional[float] = field(
        default=None,
        metadata={
            "name": "PercentileHigh",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    filtering_mode: Optional[RfifrequencyDomainRemovalConfTypeFilteringMode] = field(
        default=None,
        metadata={
            "name": "FilteringMode",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class RfitimeDomainRemovalConfType(BaseConfType):
    """
    Time Domain Interference Removal Configuration.

    Parameters
    ----------
    correction_mode
        Correction mode to be applied: DISABLED, NEAREST (substitute with the nearest valid neighbour), ZERO
        (substitute with zero).
    percentile_threshold
        Percentile of the signal distribution model to be tuned. Lower values imply more false positives.
    median_filter_block_lines
        Number of range lines to be used to compute the local median filter.
    lines_in_estimate_block
        Number of range lines in each computation block.
    box_filter_azimuth_dimension
        Dimension along the azimuth direction of the convolution block used to compute the local percentiles of the
        signal distribution.
    box_filter_range_dimension
        Dimension along the range direction of the convolution block used to compute the local percentiles of the
        signal distribution.
    morph_open_line_length
        Length of the line patter used for the morphological-open filter.
    morph_close_line_length
        Length of the line patter used for the morphological-close filter.
    morph_close_before_open
        Flag used to chose the order of the morphological operations.
    morph_open_close_iterations
        Number of iterations of open/close or close/open morphological couple operations.
    """

    class Meta:
        name = "RFITimeDomainRemovalConfType"
        target_namespace = "aresysConfTypes"

    correction_mode: Optional[RfitimeDomainRemovalConfTypeCorrectionMode] = field(
        default=None,
        metadata={
            "name": "CorrectionMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    percentile_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "PercentileThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    median_filter_block_lines: Optional[int] = field(
        default=None,
        metadata={
            "name": "MedianFilterBlockLines",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lines_in_estimate_block: Optional[int] = field(
        default=None,
        metadata={
            "name": "LinesInEstimateBlock",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    box_filter_azimuth_dimension: Optional[int] = field(
        default=None,
        metadata={
            "name": "BoxFilterAzimuthDimension",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    box_filter_range_dimension: Optional[int] = field(
        default=None,
        metadata={
            "name": "BoxFilterRangeDimension",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    morph_open_line_length: Optional[int] = field(
        default=None,
        metadata={
            "name": "MorphOpenLineLength",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    morph_close_line_length: Optional[int] = field(
        default=None,
        metadata={
            "name": "MorphCloseLineLength",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    morph_close_before_open: Optional[bool] = field(
        default=None,
        metadata={
            "name": "MorphCloseBeforeOpen",
            "type": "Element",
            "namespace": "",
        },
    )
    morph_open_close_iterations: Optional[int] = field(
        default=None,
        metadata={
            "name": "MorphOpenCloseIterations",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class SarfocDigitalElevationModelType:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: Optional[DemTypes] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "name": "ID",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class SarfocOutputProductsType:
    class Meta:
        target_namespace = "aresysConfTypes"

    product: list[SarfocProductType] = field(
        default_factory=list,
        metadata={
            "name": "Product",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )


@dataclass
class SarfocProcessingStepsType:
    class Meta:
        target_namespace = "aresysConfTypes"

    processing_step: list[SarfocProcessingStepType] = field(
        default_factory=list,
        metadata={
            "name": "ProcessingStep",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )


@dataclass
class Slant2GroundConfType(BaseConfType):
    """
    Slant to ground configuration parameters.

    Parameters
    ----------
    ground_step
        Ground sampling step
    invalid_value
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    ground_step: Optional[float] = field(
        default=None,
        metadata={
            "name": "GroundStep",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_value: Optional[ComplexAlg] = field(
        default=None,
        metadata={
            "name": "InvalidValue",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class StripmapDcConfType(BaseConfType):
    """
    DC estimator configuration parameters for stripmap.

    Parameters
    ----------
    blocks
        Number of samples of each contrast block
    blockl
        Number of lines of each contrast block
    undersampling_snrdcazimuth_ratio
        SNR and Doppler Centroid grids azimuth undersampling factor (will be multiplied by Blockl to get the DC
        estimation block size)
    undersampling_snrdcrange_ratio
        SNR and Doppler Centroid grids range undersampling factor (will be multiplied by Blocks to get the DC
        estimation block size)
    az_max_frequency_search_bin_number
        Number of azimuth samples for the CZT computation
    rg_max_frequency_search_bin_number
        Number of range samples for the CZT computation
    az_max_frequency_search_norm_band
        Azimuth spectral window for the CZT computation
    rg_max_frequency_search_norm_band
        Range spectral window for the CZT computation
    nummlbf
        Number of spectral looks in each Mlbf block
    nbestblocks
        Number of blocks to be used in Mlbf algorithm
    rg_band
        Range processed bandwidth (normalized)
    an_len
        Azimuth antenna length [m]
    lookbf
        Single spectral look bandwidth (normalized)
    lookbt
        Spectral separation between two looks (normalized)
    lookrp
        Filter pass-band (normalized)
    lookrs
        Filter transition band (normalized)
    decfac
        Decimation factor
    flength
        Filter length [samples]
    dftstep
        Discrete Fourier Transform step (normalized)
    peakwid
        Spectral width of the Mlfb peak
    minamb
        Minimum Doppler Centroid ambiguity number
    maxamb
        Maximum Doppler Centroid ambiguity number
    sthr
        SNR threshold for processing block selection
    varth
        Kramer Rao variance threshold for processing block selection
    pol_weights
        Doppler Centroid and Doppler Rate polynomial coefficients selection flags (0: not used, 1: used)
    dc_estimation_method
        Doppler estimation method: GEOMETRICAL (only attitude is used), DATA (only data are used), COMBINED (both
        attitude and data are used)
    attitude_fitting
        Sensor attitude fitting method: LINEAR, AVERAGE, DISABLED
    dccore_algorithm
    poly_changing_freq
        Frequency of doppler polynomial changing
    poly_estimation_constraint
        Doppler poly estimation constraint: FULL, UNCONSTRAINED
    perform_joint_estimation
        Perform joint estimation of Doppler Centroid polynomials across copolarizations (0: not applied, 1: applied)
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    blocks: Optional[int] = field(
        default=None,
        metadata={
            "name": "Blocks",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    blockl: Optional[int] = field(
        default=None,
        metadata={
            "name": "Blockl",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    undersampling_snrdcazimuth_ratio: Optional[int] = field(
        default=None,
        metadata={
            "name": "UndersamplingSNRDCazimuthRatio",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    undersampling_snrdcrange_ratio: Optional[int] = field(
        default=None,
        metadata={
            "name": "UndersamplingSNRDCrangeRatio",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    az_max_frequency_search_bin_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "azMaxFrequencySearchBinNumber",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rg_max_frequency_search_bin_number: Optional[int] = field(
        default=None,
        metadata={
            "name": "rgMaxFrequencySearchBinNumber",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    az_max_frequency_search_norm_band: Optional[float] = field(
        default=None,
        metadata={
            "name": "azMaxFrequencySearchNormBand",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rg_max_frequency_search_norm_band: Optional[float] = field(
        default=None,
        metadata={
            "name": "rgMaxFrequencySearchNormBand",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    nummlbf: Optional[int] = field(
        default=None,
        metadata={
            "name": "Nummlbf",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    nbestblocks: Optional[int] = field(
        default=None,
        metadata={
            "name": "Nbestblocks",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rg_band: Optional[float] = field(
        default=None,
        metadata={
            "name": "RgBand",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    an_len: Optional[float] = field(
        default=None,
        metadata={
            "name": "AnLen",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lookbf: Optional[float] = field(
        default=None,
        metadata={
            "name": "Lookbf",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lookbt: Optional[float] = field(
        default=None,
        metadata={
            "name": "Lookbt",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lookrp: Optional[float] = field(
        default=None,
        metadata={
            "name": "Lookrp",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lookrs: Optional[float] = field(
        default=None,
        metadata={
            "name": "Lookrs",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    decfac: Optional[int] = field(
        default=None,
        metadata={
            "name": "Decfac",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    flength: Optional[int] = field(
        default=None,
        metadata={
            "name": "Flength",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    dftstep: Optional[float] = field(
        default=None,
        metadata={
            "name": "Dftstep",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    peakwid: Optional[float] = field(
        default=None,
        metadata={
            "name": "Peakwid",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    minamb: Optional[float] = field(
        default=None,
        metadata={
            "name": "Minamb",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    maxamb: Optional[float] = field(
        default=None,
        metadata={
            "name": "Maxamb",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sthr: Optional[float] = field(
        default=None,
        metadata={
            "name": "Sthr",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    varth: Optional[float] = field(
        default=None,
        metadata={
            "name": "Varth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pol_weights: Optional["StripmapDcConfType.PolWeights"] = field(
        default=None,
        metadata={
            "name": "Pol_weights",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    dc_estimation_method: Optional[DcEstimationMethodsTypes] = field(
        default=None,
        metadata={
            "name": "DcEstimationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    attitude_fitting: Optional[AttitudeFittingTypes] = field(
        default=None,
        metadata={
            "name": "AttitudeFitting",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    dccore_algorithm: Optional[DccoreAlgorithmType] = field(
        default=None,
        metadata={
            "name": "DCCoreAlgorithm",
            "type": "Element",
            "namespace": "",
        },
    )
    poly_changing_freq: Optional[float] = field(
        default=None,
        metadata={
            "name": "PolyChangingFreq",
            "type": "Element",
            "namespace": "",
        },
    )
    poly_estimation_constraint: Optional[PolyEstimationConstraintTypes] = field(
        default=None,
        metadata={
            "name": "PolyEstimationConstraint",
            "type": "Element",
            "namespace": "",
        },
    )
    perform_joint_estimation: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformJointEstimation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )

    @dataclass
    class PolWeights:
        w: list[int] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "namespace": "",
                "min_occurs": 7,
                "max_occurs": 7,
            },
        )


@dataclass
class WindowConfType:
    """
    Weighting window parameters.

    Parameters
    ----------
    window_type
        Window type: HAMMING, KAISER
    window_parameter
        Window parameter
    window_look_bandwidth
        Window look bandwidth (assumed Normalized if not present)
    window_transition_bandwidth
        Window transition bandwidth (assumed Normalized if not present)
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    window_type: Optional[Windows] = field(
        default=None,
        metadata={
            "name": "WindowType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    window_parameter: Optional[float] = field(
        default=None,
        metadata={
            "name": "WindowParameter",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    window_look_bandwidth: Optional["WindowConfType.WindowLookBandwidth"] = field(
        default=None,
        metadata={
            "name": "WindowLookBandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    window_transition_bandwidth: Optional["WindowConfType.WindowTransitionBandwidth"] = field(
        default=None,
        metadata={
            "name": "WindowTransitionBandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class WindowLookBandwidth:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[UnitTypes] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class WindowTransitionBandwidth:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[UnitTypes] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass
class AzimuthConfType(BaseConfType):
    """
    Azimuth focusing configuration parameters.

    Parameters
    ----------
    lines_in_block
        Number of lines of each processing block
    samples_in_block
        Number of samples of each processing block
    azimuth_overlap
        Number of overlapping lines in each processing block
    range_overlap
        Number of overlapping samples in each processing block
    perform_interpolation
        WK interpolation parameter (0: Stolt Interpolation (SI) not performed, 1: SI approximated with CZT and
        linear interpolation, n: SI performed with a GLS filter of length 2*n+1)
    stolt_padding
        Range interpolation zero padding as fraction of the whole image
    range_modulation
        Range modulation flag (0: not performed, 1: performed)
    apply_azimuth_spectral_weighting_window
        Azimuth spectral weighting window flag (0: not apllied, 1: applied)
    azimuth_spectral_weighting_window
        Azimuth spectral weighting window parameters
    apply_rg_shift
        Image range recentering flag (0: not performed, 1: performed)
    apply_az_shift
        Image azimuth recentering flag (0: not performed, 1: performed)
    whitening_flag
        Antenna whitening flag (0: not performed, 1: performed)
    antenna_length
        Azimuth antenna equivalent length [m]
    pad_result
        Focused data zero-padding flag (0: not performed, 1: performed, 2: not performed in range direction)
    lines_to_skip_dc_fr
        Doppler Centroid and Doppler Rate grids azimuth step [lines]
    samples_to_skip_dc_fr
        Doppler Centroid and Doppler Rate grids range step [samples]
    focusing_method
    az_proc_bandwidth
        Bandwidth to process during azimuth focalization. If not specified, the bandwidth is considered to be 1
        (Normalized).
    bistatic_delay_correction_mode
        Bistatic delay correction mode to apply during azimuth focalization.
    azimuth_time_bias
        Azimuth time bias to apply during azimuth focalization.
    apply_pol_channels_coregistration
        Coregistration of polarimetric channels on same azimuth start time (0: not performed, 1: performed)
    antenna_shift_compensation_mode
        Antenna shift compensation mode
    nominal_block_memory_size
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    lines_in_block: Optional[int] = field(
        default=None,
        metadata={
            "name": "LinesInBlock",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    samples_in_block: Optional[int] = field(
        default=None,
        metadata={
            "name": "SamplesInBlock",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_overlap: Optional[int] = field(
        default=None,
        metadata={
            "name": "AzimuthOverlap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_overlap: Optional[int] = field(
        default=None,
        metadata={
            "name": "RangeOverlap",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    perform_interpolation: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformInterpolation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    stolt_padding: Optional[float] = field(
        default=None,
        metadata={
            "name": "StoltPadding",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_modulation: Optional[int] = field(
        default=None,
        metadata={
            "name": "RangeModulation",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    apply_azimuth_spectral_weighting_window: Optional[int] = field(
        default=None,
        metadata={
            "name": "ApplyAzimuthSpectralWeightingWindow",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    azimuth_spectral_weighting_window: Optional[WindowConfType] = field(
        default=None,
        metadata={
            "name": "AzimuthSpectralWeightingWindow",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    apply_rg_shift: Optional[int] = field(
        default=None,
        metadata={
            "name": "ApplyRgShift",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    apply_az_shift: Optional[int] = field(
        default=None,
        metadata={
            "name": "ApplyAzShift",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    whitening_flag: Optional[int] = field(
        default=None,
        metadata={
            "name": "WhiteningFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    antenna_length: Optional[float] = field(
        default=None,
        metadata={
            "name": "AntennaLength",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pad_result: Optional[int] = field(
        default=None,
        metadata={
            "name": "PadResult",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lines_to_skip_dc_fr: Optional[int] = field(
        default=None,
        metadata={
            "name": "LinesToSkipDcFr",
            "type": "Element",
            "namespace": "",
        },
    )
    samples_to_skip_dc_fr: Optional[int] = field(
        default=None,
        metadata={
            "name": "SamplesToSkipDcFr",
            "type": "Element",
            "namespace": "",
        },
    )
    focusing_method: Optional[FocusingMethodTypes] = field(
        default=None,
        metadata={
            "name": "FocusingMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    az_proc_bandwidth: Optional["AzimuthConfType.AzProcBandwidth"] = field(
        default=None,
        metadata={
            "name": "AzProcBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    bistatic_delay_correction_mode: Optional[BistaticDelayCorrectionTypes] = field(
        default=None,
        metadata={
            "name": "BistaticDelayCorrectionMode",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_time_bias: Optional[float] = field(
        default=None,
        metadata={
            "name": "AzimuthTimeBias",
            "type": "Element",
            "namespace": "",
        },
    )
    apply_pol_channels_coregistration: Optional[int] = field(
        default=None,
        metadata={
            "name": "ApplyPolChannelsCoregistration",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    antenna_shift_compensation_mode: Optional[AntennaShiftCompensationModeType] = field(
        default=None,
        metadata={
            "name": "AntennaShiftCompensationMode",
            "type": "Element",
            "namespace": "",
        },
    )
    nominal_block_memory_size: Optional["AzimuthConfType.NominalBlockMemorySize"] = field(
        default=None,
        metadata={
            "name": "NominalBlockMemorySize",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class AzProcBandwidth:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[UnitTypes] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass
    class NominalBlockMemorySize:
        gpu: list[MemorySizeType] = field(
            default_factory=list,
            metadata={
                "name": "GPU",
                "type": "Element",
                "namespace": "",
                "max_occurs": 2,
            },
        )
        cpu: Optional[MemorySizeType] = field(
            default=None,
            metadata={
                "name": "CPU",
                "type": "Element",
                "namespace": "",
            },
        )


@dataclass
class BpsconfType:
    """
    BPS global configuration parameters.

    Parameters
    ----------
    bpslogger_conf
        BPS Logger configuration parameters
    """

    class Meta:
        name = "BPSConfType"
        target_namespace = "aresysConfTypes"

    bpslogger_conf: Optional[BpsloggerConfType] = field(
        default=None,
        metadata={
            "name": "BPSLoggerConf",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class BiomassL0ImportPreProcConfType:
    """
    Parameters
    ----------
    biomass_l0_import_conf
    biomass_raw_data_corrections_conf
    biomass_int_cal_conf
    enable_channel_delays_annotation
        Enable the annotation of channel delays in extracted Raw product metadata
    enable_int_cal
        Enable/disable internal calibration step (enabled by default if not specified)
    beam
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    biomass_l0_import_conf: Optional[BiomassL0ImportConfType] = field(
        default=None,
        metadata={
            "name": "BiomassL0ImportConf",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    biomass_raw_data_corrections_conf: Optional[BiomassRawDataCorrectionsConfType] = field(
        default=None,
        metadata={
            "name": "BiomassRawDataCorrectionsConf",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    biomass_int_cal_conf: Optional[BiomassIntCalConfType] = field(
        default=None,
        metadata={
            "name": "BiomassIntCalConf",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    enable_channel_delays_annotation: Optional[bool] = field(
        default=None,
        metadata={
            "name": "EnableChannelDelaysAnnotation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    enable_int_cal: Optional[bool] = field(
        default=None,
        metadata={
            "name": "EnableIntCal",
            "type": "Element",
            "namespace": "",
        },
    )
    beam: Optional[str] = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


@dataclass
class FullAccuracyPostProcessingConfType:
    """
    Parameters
    ----------
    non_stationary_coreg_conf
    quality_threshold_for_automatic_mode
        R squared quality threshold
    residual_shift_fitting_model
    min_valid_blocks
        Minimum Percentage of Valid Blocks
    weight_threshold_refine_rg
    weight_threshold_refine_az
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    non_stationary_coreg_conf: Optional[NonStationaryCoregConfType] = field(
        default=None,
        metadata={
            "name": "NonStationaryCoregConf",
            "type": "Element",
            "namespace": "",
        },
    )
    quality_threshold_for_automatic_mode: Optional[float] = field(
        default=None,
        metadata={
            "name": "QualityThresholdForAutomaticMode",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    residual_shift_fitting_model: Optional[FullAccuracyPostProcessingConfTypeResidualShiftFittingModel] = field(
        default=None,
        metadata={
            "name": "ResidualShiftFittingModel",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    min_valid_blocks: int = field(
        default=50,
        metadata={
            "name": "MinValidBlocks",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    weight_threshold_refine_rg: Optional[float] = field(
        default=None,
        metadata={
            "name": "WeightThresholdRefineRg",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    weight_threshold_refine_az: Optional[float] = field(
        default=None,
        metadata={
            "name": "WeightThresholdRefineAz",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class MultilookDirectionConfType:
    """
    Parameters
    ----------
    pfactor
        P Factor applied
    qfactor
        Q Factor applied
    weighting_window
        Weighting window parameters
    central_frequency
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    pfactor: Optional[int] = field(
        default=None,
        metadata={
            "name": "PFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    qfactor: Optional[int] = field(
        default=None,
        metadata={
            "name": "QFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    weighting_window: Optional[WindowConfType] = field(
        default=None,
        metadata={
            "name": "WeightingWindow",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    central_frequency: list["MultilookDirectionConfType.CentralFrequency"] = field(
        default_factory=list,
        metadata={
            "name": "CentralFrequency",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class CentralFrequency:
        value: Optional[float] = field(
            default=None,
            metadata={
                "required": True,
            },
        )
        unit: Optional[UnitTypes] = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass
class RfimitigationConfType(BaseConfType):
    """
    Generic Interference Removal Configuration.

    Parameters
    ----------
    rfimitigation_method
        Interference Removal method: FREQUENCY (Frequency Domain Interference Removal is used), TIME (Time Domain
        Interference Removal used), TIME_AND_FREQUENCY (Time Domain Interference Removal then Frequency Domain
        Interference Removal used), FREQUENCY_AND_TIME (Frequency Domain Interference Removal then Time Domain
        Interference Removal used)
    rfimask_composition_method
        Optional Mask Composition method across polarizations: AND (intersection of all masks), OR (union of all
        masks). The resulting mask is used for mitigation of all polarizations
    rfimitigation_time_domain_conf
    rfimitigation_frequency_domain_conf
    """

    class Meta:
        name = "RFIMitigationConfType"
        target_namespace = "aresysConfTypes"

    rfimitigation_method: Optional[RfimitigationMethodsType] = field(
        default=None,
        metadata={
            "name": "RFIMitigationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfimask_composition_method: Optional[RfimaskCompositionMethodsType] = field(
        default=None,
        metadata={
            "name": "RFIMaskCompositionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    rfimitigation_time_domain_conf: Optional[RfitimeDomainRemovalConfType] = field(
        default=None,
        metadata={
            "name": "RFIMitigationTimeDomainConf",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfimitigation_frequency_domain_conf: list[RfifrequencyDomainRemovalConfType] = field(
        default_factory=list,
        metadata={
            "name": "RFIMitigationFrequencyDomainConf",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
            "max_occurs": 2,
            "sequence": 1,
        },
    )


@dataclass
class RangeCompensatorConfType(BaseConfType):
    """
    Parameters
    ----------
    rslreference_distance
        Range Spreading Loss reference distance [m]
    perform_rslcompensation
        Perform Range Spreading Loss compensation flag (0: do not perform RSL compensation; 1: perform RSL
        compensation)
    perform_incidence_compensation
        Perform incidence angle compensation flag (0: do not perform incidence angle compensation; 1: perform
        incidence angle compensation)
    perform_pattern_compensation
        Perform range antena pattern compensation flag (0: do not perform range antenna pattern compensation; 1:
        perform range antenna pattern compensation)
    perform_roll_compensation
        (Deprecated) Perform roll angle compensation flag (0: Do not perform roll angle compensation; 1: perform
        roll angle compensation)
    perform_line_correction
        Perform line correction flag (0: do not perform line correction; 1: perform line correction)
    fast_mode
        WGS84 fast mode flag (0: fast mode disabled; 1: fast mode enabled)
    external_calibration_factor
        External calibration factor
    processing_gain
        Processing normalization factor
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    rslreference_distance: Optional[float] = field(
        default=None,
        metadata={
            "name": "RSLReferenceDistance",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    perform_rslcompensation: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformRSLCompensation",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    perform_incidence_compensation: Optional["RangeCompensatorConfType.PerformIncidenceCompensation"] = field(
        default=None,
        metadata={
            "name": "PerformIncidenceCompensation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    perform_pattern_compensation: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformPatternCompensation",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    perform_roll_compensation: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformRollCompensation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    perform_line_correction: Optional[int] = field(
        default=None,
        metadata={
            "name": "PerformLineCorrection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    fast_mode: Optional[int] = field(
        default=None,
        metadata={
            "name": "FastMode",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    external_calibration_factor: Optional["RangeCompensatorConfType.ExternalCalibrationFactor"] = field(
        default=None,
        metadata={
            "name": "ExternalCalibrationFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_gain: Optional[Fcomplex] = field(
        default=None,
        metadata={
            "name": "ProcessingGain",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class PerformIncidenceCompensation:
        value: Optional[int] = field(
            default=None,
            metadata={
                "required": True,
                "min_inclusive": 0,
                "max_inclusive": 1,
            },
        )
        output_quantity: Optional[OutputQuantityType] = field(
            default=None,
            metadata={
                "name": "OutputQuantity",
                "type": "Attribute",
            },
        )

    @dataclass
    class ExternalCalibrationFactor(Fcomplex):
        apply: Optional[int] = field(
            default=None,
            metadata={
                "name": "Apply",
                "type": "Attribute",
                "min_inclusive": 0,
                "max_inclusive": 1,
            },
        )


@dataclass
class RangeConfType(BaseConfType):
    """
    Range focusing configuration parameters.

    Parameters
    ----------
    flag_ortog
        Data ortogonalization flag (0: not performed, 1: performed)
    apply_range_spectral_weighting_window
        Range spectral weighting window flag (0: not apllied, 1: applied)
    range_spectral_weighting_window
        Range spectral weighting window parameters
    swstbias
        SWST bias in seconds
    range_decimation_factor
        Range decimation factor
    apply_rx_gain_correction
        Correct Receiver attenuation (0: not applied, 1: applied)
    focusing_method
        Range focusing method
    output_border_policies
    post_processing
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    flag_ortog: Optional[int] = field(
        default=None,
        metadata={
            "name": "Flag_ortog",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    apply_range_spectral_weighting_window: Optional[int] = field(
        default=None,
        metadata={
            "name": "ApplyRangeSpectralWeightingWindow",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    range_spectral_weighting_window: Optional[WindowConfType] = field(
        default=None,
        metadata={
            "name": "RangeSpectralWeightingWindow",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    swstbias: Optional[float] = field(
        default=None,
        metadata={
            "name": "SWSTBias",
            "type": "Element",
            "namespace": "",
        },
    )
    range_decimation_factor: Optional[int] = field(
        default=None,
        metadata={
            "name": "RangeDecimationFactor",
            "type": "Element",
            "namespace": "",
        },
    )
    apply_rx_gain_correction: Optional[int] = field(
        default=None,
        metadata={
            "name": "ApplyRxGainCorrection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    focusing_method: Optional[RangeFocusingMethodType] = field(
        default=None,
        metadata={
            "name": "FocusingMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    output_border_policies: Optional["RangeConfType.OutputBorderPolicies"] = field(
        default=None,
        metadata={
            "name": "OutputBorderPolicies",
            "type": "Element",
            "namespace": "",
        },
    )
    post_processing: Optional["RangeConfType.PostProcessing"] = field(
        default=None,
        metadata={
            "name": "PostProcessing",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class OutputBorderPolicies:
        """
        Parameters
        ----------
        range
            Output range border policy
        """

        range: Optional[OutputBorderPolicyType] = field(
            default=None,
            metadata={
                "name": "Range",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

    @dataclass
    class PostProcessing:
        azimuth_resampling: Optional["RangeConfType.PostProcessing.AzimuthResampling"] = field(
            default=None,
            metadata={
                "name": "AzimuthResampling",
                "type": "Element",
                "namespace": "",
            },
        )
        enable_prfchange_data_post_processing: Optional[bool] = field(
            default=None,
            metadata={
                "name": "EnablePRFChangeDataPostProcessing",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass
        class AzimuthResampling:
            output_prf: Optional["RangeConfType.PostProcessing.AzimuthResampling.OutputPrf"] = field(
                default=None,
                metadata={
                    "name": "OutputPRF",
                    "type": "Element",
                    "namespace": "",
                    "required": True,
                },
            )

            @dataclass
            class OutputPrf:
                """
                Parameters
                ----------
                value
                    Output pulse repetition frequency value [Hz]
                """

                value: Optional[float] = field(
                    default=None,
                    metadata={
                        "name": "Value",
                        "type": "Element",
                        "namespace": "",
                        "required": True,
                    },
                )


@dataclass
class SarfocExternalResourcesType:
    class Meta:
        target_namespace = "aresysConfTypes"

    digital_elevation_model: list[DigitalElevationModelType] = field(
        default_factory=list,
        metadata={
            "name": "DigitalElevationModel",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    prfresampling_filter_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "PRFResamplingFilterProduct",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class SarfocProcessingSettingsType:
    class Meta:
        target_namespace = "aresysConfTypes"

    digital_elevation_model: list[SarfocDigitalElevationModelType] = field(
        default_factory=list,
        metadata={
            "name": "DigitalElevationModel",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    prfchange_data_post_processing: Optional[bool] = field(
        default=None,
        metadata={
            "name": "PRFChangeDataPostProcessing",
            "type": "Element",
            "namespace": "",
        },
    )
    rfimitigation_settings: Optional["SarfocProcessingSettingsType.RfimitigationSettings"] = field(
        default=None,
        metadata={
            "name": "RFIMitigationSettings",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class RfimitigationSettings:
        chirp_source: Optional[RfimitigationSettingsChirpSource] = field(
            default=None,
            metadata={
                "name": "ChirpSource",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        mode: Optional[RfimitigationSettingsMode] = field(
            default=None,
            metadata={
                "name": "Mode",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )


@dataclass
class Bpsl1CoreProcessingSettingsType:
    class Meta:
        name = "BPSL1CoreProcessingSettingsType"
        target_namespace = "aresysConfTypes"

    core_processing_settings: Optional[SarfocProcessingSettingsType] = field(
        default=None,
        metadata={
            "name": "CoreProcessingSettings",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_compression: Optional["Bpsl1CoreProcessingSettingsType.AzimuthCompression"] = field(
        default=None,
        metadata={
            "name": "AzimuthCompression",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarimetric_compensator: Optional["Bpsl1CoreProcessingSettingsType.PolarimetricCompensator"] = field(
        default=None,
        metadata={
            "name": "PolarimetricCompensator",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )

    @dataclass
    class AzimuthCompression:
        antenna_pattern_compensation: Optional[BpsantennaPatternCompensationType] = field(
            default=None,
            metadata={
                "name": "AntennaPatternCompensation",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        elevation_mispointing_deg: Optional[float] = field(
            default=None,
            metadata={
                "name": "ElevationMispointingDeg",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )

    @dataclass
    class PolarimetricCompensator:
        enable_ionospheric_calibration: Optional[bool] = field(
            default=None,
            metadata={
                "name": "EnableIonosphericCalibration",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )


@dataclass
class MultilookConfType(BaseConfType):
    """
    Multilooking configuration parameters.

    Parameters
    ----------
    multilook_conf_name
        Multilook configuration name
    apply_azimuth_time_weighting_window
    azimuth_time_weighting_window
        Burst based azimuth time weighting window
    normalization_factor
        Normalization factor applied to data during multilooking
    slow_multilook
        Multilook parameters in Slow Direction
    fast_multilook
        Multilook parameters in Fast Direction
    presum
    invalid_value
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    multilook_conf_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "MultilookConfName",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    apply_azimuth_time_weighting_window: Optional[int] = field(
        default=None,
        metadata={
            "name": "ApplyAzimuthTimeWeightingWindow",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    azimuth_time_weighting_window: Optional[WindowConfType] = field(
        default=None,
        metadata={
            "name": "AzimuthTimeWeightingWindow",
            "type": "Element",
            "namespace": "",
        },
    )
    normalization_factor: Optional[float] = field(
        default=None,
        metadata={
            "name": "NormalizationFactor",
            "type": "Element",
            "namespace": "",
        },
    )
    slow_multilook: Optional[MultilookDirectionConfType] = field(
        default=None,
        metadata={
            "name": "SlowMultilook",
            "type": "Element",
            "namespace": "",
        },
    )
    fast_multilook: Optional[MultilookDirectionConfType] = field(
        default=None,
        metadata={
            "name": "FastMultilook",
            "type": "Element",
            "namespace": "",
        },
    )
    presum: Optional["MultilookConfType.Presum"] = field(
        default=None,
        metadata={
            "name": "Presum",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_value: Optional[ComplexAlg] = field(
        default=None,
        metadata={
            "name": "InvalidValue",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass
    class Presum:
        fast_factor: Optional[int] = field(
            default=None,
            metadata={
                "name": "FastFactor",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )
        slow_factor: Optional[int] = field(
            default=None,
            metadata={
                "name": "SlowFactor",
                "type": "Element",
                "namespace": "",
                "required": True,
            },
        )


@dataclass
class StaprocessorConfType:
    class Meta:
        name = "STAProcessorConfType"
        target_namespace = "aresysConfTypes"

    full_accuracy_pre_processing_conf: Optional[FullAccuracyPreProcessingConfType] = field(
        default=None,
        metadata={
            "name": "FullAccuracyPreProcessingConf",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    full_accuracy_post_processing_conf: Optional[FullAccuracyPostProcessingConfType] = field(
        default=None,
        metadata={
            "name": "FullAccuracyPostProcessingConf",
            "type": "Element",
            "namespace": "",
        },
    )
    reinterpolation_conf: Optional[ReinterpolationConfType] = field(
        default=None,
        metadata={
            "name": "ReinterpolationConf",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    coreg_mode: Optional[CoregMode] = field(
        default=None,
        metadata={
            "name": "CoregMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    skip_geometry_shifts_computation: Optional[bool] = field(
        default=None,
        metadata={
            "name": "SkipGeometryShiftsComputation",
            "type": "Element",
            "namespace": "",
        },
    )
    earth_geometry: Optional[EarthGeometry] = field(
        default=None,
        metadata={
            "name": "EarthGeometry",
            "type": "Element",
            "namespace": "",
        },
    )
    coregistration_output_products_conf: Optional[CoregistrationOutputProductsConfType] = field(
        default=None,
        metadata={
            "name": "CoregistrationOutputProductsConf",
            "type": "Element",
            "namespace": "",
        },
    )
    remove_temporary_products: int = field(
        default=0,
        metadata={
            "name": "RemoveTemporaryProducts",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    memory_sargeometry: int = field(
        default=256,
        metadata={
            "name": "MemorySARGeometry",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    digital_elevation_model_repository: Optional[str] = field(
        default=None,
        metadata={
            "name": "DigitalElevationModelRepository",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    xsd_schema_repository: Optional[str] = field(
        default=None,
        metadata={
            "name": "XsdSchemaRepository",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_adaptive_step_second: Optional[float] = field(
        default=None,
        metadata={
            "name": "AzimuthAdaptiveStepSecond",
            "type": "Element",
            "namespace": "",
        },
    )
    do_slave_synthetic_compensation: Optional[bool] = field(
        default=None,
        metadata={
            "name": "DoSlaveSyntheticCompensation",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class Bpsl1CoreProcessorConfType(BaseConfType):
    """
    BPSL1CoreProcessor configuration parameters.
    """

    class Meta:
        name = "BPSL1CoreProcessorConfType"
        target_namespace = "aresysConfTypes"

    processing_steps: Optional[SarfocProcessingStepsType] = field(
        default=None,
        metadata={
            "name": "ProcessingSteps",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_settings: Optional[Bpsl1CoreProcessingSettingsType] = field(
        default=None,
        metadata={
            "name": "ProcessingSettings",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    output_products: Optional[SarfocOutputProductsType] = field(
        default=None,
        metadata={
            "name": "OutputProducts",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    external_resources: Optional[SarfocExternalResourcesType] = field(
        default=None,
        metadata={
            "name": "ExternalResources",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    interface_settings: Optional[Bpsl1CoreProcessorInterfaceSettingsType] = field(
        default=None,
        metadata={
            "name": "InterfaceSettings",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AresysXmlDocType:
    number_of_channels: Optional[int] = field(
        default=None,
        metadata={
            "name": "NumberOfChannels",
            "type": "Element",
            "required": True,
        },
    )
    version_number: Optional[float] = field(
        default=None,
        metadata={
            "name": "VersionNumber",
            "type": "Element",
            "required": True,
        },
    )
    description: Optional[str] = field(
        default=None,
        metadata={
            "name": "Description",
            "type": "Element",
            "required": True,
        },
    )
    channel: list["AresysXmlDocType.Channel"] = field(
        default_factory=list,
        metadata={
            "name": "Channel",
            "type": "Element",
        },
    )

    @dataclass
    class Channel(ChannelType):
        range_conf: list[RangeConfType] = field(
            default_factory=list,
            metadata={
                "name": "RangeConf",
                "type": "Element",
            },
        )
        dcest_conf_stripmap: list[StripmapDcConfType] = field(
            default_factory=list,
            metadata={
                "name": "DCEstConfStripmap",
                "type": "Element",
            },
        )
        azimuth_conf: list[AzimuthConfType] = field(
            default_factory=list,
            metadata={
                "name": "AzimuthConf",
                "type": "Element",
            },
        )
        range_compensator_conf: list[RangeCompensatorConfType] = field(
            default_factory=list,
            metadata={
                "name": "RangeCompensatorConf",
                "type": "Element",
            },
        )
        multi_processor_conf: list[MultilookConfType] = field(
            default_factory=list,
            metadata={
                "name": "MultiProcessorConf",
                "type": "Element",
            },
        )
        bpsconf: list[BpsconfType] = field(
            default_factory=list,
            metadata={
                "name": "BPSConf",
                "type": "Element",
            },
        )
        bpsl1_core_processor_conf: list[Bpsl1CoreProcessorConfType] = field(
            default_factory=list,
            metadata={
                "name": "BPSL1CoreProcessorConf",
                "type": "Element",
            },
        )
        slant2_ground_conf: list[Slant2GroundConfType] = field(
            default_factory=list,
            metadata={
                "name": "Slant2GroundConf",
                "type": "Element",
            },
        )
        staprocessor_conf: list[StaprocessorConfType] = field(
            default_factory=list,
            metadata={
                "name": "STAProcessorConf",
                "type": "Element",
            },
        )
        rfimitigation_conf: list[RfimitigationConfType] = field(
            default_factory=list,
            metadata={
                "name": "RFIMitigationConf",
                "type": "Element",
            },
        )
        polarimetric_processor_conf: list[PolarimetricProcessorConfType] = field(
            default_factory=list,
            metadata={
                "name": "PolarimetricProcessorConf",
                "type": "Element",
            },
        )
        calibration_constants_conf: list[CalibrationConstantsConfType] = field(
            default_factory=list,
            metadata={
                "name": "CalibrationConstantsConf",
                "type": "Element",
            },
        )
        noise_map_generator_conf: list[NoiseMapGeneratorConfType] = field(
            default_factory=list,
            metadata={
                "name": "NoiseMapGeneratorConf",
                "type": "Element",
            },
        )
        ionospheric_calibration_conf: list[IonosphericCalibrationConfType] = field(
            default_factory=list,
            metadata={
                "name": "IonosphericCalibrationConf",
                "type": "Element",
            },
        )
        biomass_l0_import_pre_proc_conf: list[BiomassL0ImportPreProcConfType] = field(
            default_factory=list,
            metadata={
                "name": "BiomassL0ImportPreProcConf",
                "type": "Element",
            },
        )


@dataclass
class AresysXmlDoc(AresysXmlDocType):
    pass
