# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD BPS Configuration file models
---------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(kw_only=True)
class ChannelType:
    number: None | int = field(
        default=None,
        metadata={
            "name": "Number",
            "type": "Attribute",
        },
    )
    total: None | int = field(
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
    Antenna shift compensation mode (DISABLED: antenna shift compensation disabled, FORCED: antenna shift
    compensation always performed).
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


@dataclass(kw_only=True)
class BaseConfType:
    class Meta:
        target_namespace = "aresysConfTypes"

    beam: None | str = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


class BiomassIntCalConfTypeInternalCalibrationSource(Enum):
    EXTRACTED = "EXTRACTED"
    MODEL = "MODEL"


@dataclass(kw_only=True)
class BiomassL0ImportConfType:
    class Meta:
        target_namespace = "aresysConfTypes"

    max_ispgap: int = field(
        metadata={
            "name": "MaxISPGap",
            "type": "Element",
            "namespace": "",
        }
    )
    max_time_gap: float = field(
        metadata={
            "name": "MaxTimeGap",
            "type": "Element",
            "namespace": "",
        }
    )
    raw_mean_expected: float = field(
        metadata={
            "name": "RawMeanExpected",
            "type": "Element",
            "namespace": "",
        }
    )
    raw_mean_threshold: float = field(
        metadata={
            "name": "RawMeanThreshold",
            "type": "Element",
            "namespace": "",
        }
    )
    raw_std_expected: float = field(
        metadata={
            "name": "RawStdExpected",
            "type": "Element",
            "namespace": "",
        }
    )
    raw_std_threshold: float = field(
        metadata={
            "name": "RawStdThreshold",
            "type": "Element",
            "namespace": "",
        }
    )
    beam: None | str = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class BiomassRawDataCorrectionsConfType:
    class Meta:
        target_namespace = "aresysConfTypes"

    correct_bias: bool = field(
        metadata={
            "name": "CorrectBias",
            "type": "Element",
            "namespace": "",
        }
    )
    correct_gain_imbalance: bool = field(
        metadata={
            "name": "CorrectGainImbalance",
            "type": "Element",
            "namespace": "",
        }
    )
    correct_non_orthogonality: bool = field(
        metadata={
            "name": "CorrectNonOrthogonality",
            "type": "Element",
            "namespace": "",
        }
    )
    beam: None | str = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class ComplexAlg:
    class Meta:
        name = "COMPLEX_ALG"
        target_namespace = "aresysConfTypes"

    real_value: float = field(
        metadata={
            "name": "RealValue",
            "type": "Element",
            "namespace": "",
        }
    )
    imaginary_value: float = field(
        metadata={
            "name": "ImaginaryValue",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
class ComplexPol:
    class Meta:
        name = "COMPLEX_POL"
        target_namespace = "aresysConfTypes"

    abs_value: float = field(
        metadata={
            "name": "AbsValue",
            "type": "Element",
            "namespace": "",
        }
    )
    phase_value: float = field(
        metadata={
            "name": "PhaseValue",
            "type": "Element",
            "namespace": "",
        }
    )


class CoregMode(Enum):
    GEOMETRY = "GEOMETRY"
    FULL_ACCURACY = "FULL_ACCURACY"
    AUTOMATIC = "AUTOMATIC"


@dataclass(kw_only=True)
class ComplexNumberType:
    class Meta:
        target_namespace = "aresysConfTypes"

    amplitude: float = field(
        metadata={
            "name": "Amplitude",
            "type": "Element",
            "namespace": "",
        }
    )
    phase: float = field(
        metadata={
            "name": "Phase",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
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
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    xcorr_azimuth_min_overlap: None | int = field(
        default=None,
        metadata={
            "name": "XCorrAzimuthMinOverlap",
            "type": "Element",
            "namespace": "",
        },
    )
    provide_products_for_each_polarization: None | int = field(
        default=None,
        metadata={
            "name": "ProvideProductsForEachPolarization",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_wavenumbers: None | int = field(
        default=None,
        metadata={
            "name": "ProvideWavenumbers",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    shifts_only_estimation: None | int = field(
        default=None,
        metadata={
            "name": "ShiftsOnlyEstimation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_absolute_primary_distance: None | int = field(
        default=None,
        metadata={
            "name": "ProvideAbsolutePrimaryDistance",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_geometry_shifts: None | int = field(
        default=None,
        metadata={
            "name": "ProvideGeometryShifts",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    provide_full_accuracy_shifts: None | int = field(
        default=None,
        metadata={
            "name": "ProvideFullAccuracyShifts",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )


@dataclass(kw_only=True)
class DccoreAlgorithmType:
    class Meta:
        name = "DCCoreAlgorithmType"
        target_namespace = "aresysConfTypes"

    mle: None | DccoreAlgorithmType.Mle = field(
        default=None,
        metadata={
            "name": "MLE",
            "type": "Element",
            "namespace": "",
        },
    )
    jpl: None | object = field(
        default=None,
        metadata={
            "name": "JPL",
            "type": "Element",
            "namespace": "",
        },
    )
    cde: None | object = field(
        default=None,
        metadata={
            "name": "CDE",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class Mle:
        iterations: int = field(
            metadata={
                "name": "Iterations",
                "type": "Element",
                "namespace": "",
            }
        )
        delta_f: float = field(
            metadata={
                "name": "DeltaF",
                "type": "Element",
                "namespace": "",
            }
        )
        antenna_wrapping_lateral_bands: int = field(
            metadata={
                "name": "AntennaWrappingLateralBands",
                "type": "Element",
                "namespace": "",
            }
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


@dataclass(kw_only=True)
class MemorySizeType:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: int = field()
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
    Output border policy (CUT: remove border, PAD: replace data at border with a padding value, DATA: keep data at
    border).
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
    Range focusing method (MATCHED_FILTER: matched filter method, INVERSE_FILTER: inverse filter method,
    INVERSE_FFT: focusing method for dechirped data).
    """

    MATCHED_FILTER = "MATCHED_FILTER"
    INVERSE_FILTER = "INVERSE_FILTER"
    INVERSE_FFT = "INVERSE_FFT"


@dataclass(kw_only=True)
class RealValueDegrees:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: float = field()
    units: str = field(
        init=False,
        default="deg",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass(kw_only=True)
class RealValueMeters:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: float = field()
    units: str = field(
        init=False,
        default="m",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass(kw_only=True)
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
        },
    )
    bank_size: int = field(
        default=101,
        metadata={
            "name": "BankSize",
            "type": "Element",
            "namespace": "",
        },
    )
    bandwidth: float = field(
        default=0.80000001,
        metadata={
            "name": "Bandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    range_overlap: int = field(
        default=1,
        metadata={
            "name": "RangeOverlap",
            "type": "Element",
            "namespace": "",
        },
    )
    demodulation_type: int = field(
        default=0,
        metadata={
            "name": "DemodulationType",
            "type": "Element",
            "namespace": "",
        },
    )
    unsigned_flag: int = field(
        default=0,
        metadata={
            "name": "UnsignedFlag",
            "type": "Element",
            "namespace": "",
        },
    )
    memory: int = field(
        default=256,
        metadata={
            "name": "Memory",
            "type": "Element",
            "namespace": "",
        },
    )
    verbose: int = field(
        default=0,
        metadata={
            "name": "Verbose",
            "type": "Element",
            "namespace": "",
        },
    )
    report_level: int = field(
        default=0,
        metadata={
            "name": "ReportLevel",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class SarfocProcessingStepType:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: bool = field()
    id: str = field(
        metadata={
            "name": "ID",
            "type": "Attribute",
        }
    )


@dataclass(kw_only=True)
class SarfocProductType:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: str = field(default="")
    id: str = field(
        metadata={
            "name": "ID",
            "type": "Attribute",
        }
    )


class UnitTypes(Enum):
    NORMALIZED = "Normalized"
    HZ = "Hz"
    S = "s"


class Windows(Enum):
    HAMMING = "HAMMING"
    KAISER = "KAISER"


@dataclass(kw_only=True)
class Bpsl1CoreProcessorInterfaceSettingsType:
    class Meta:
        name = "BPSL1CoreProcessorInterfaceSettingsType"
        target_namespace = "aresysConfTypes"

    products_format: FormatType = field(
        metadata={
            "name": "ProductsFormat",
            "type": "Element",
            "namespace": "",
        }
    )
    enable_quick_look_generation: bool = field(
        metadata={
            "name": "EnableQuickLookGeneration",
            "type": "Element",
            "namespace": "",
        }
    )
    remove_intermediate_products: bool = field(
        metadata={
            "name": "RemoveIntermediateProducts",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
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

    node_name: str = field(
        metadata={
            "name": "NodeName",
            "type": "Element",
            "namespace": "",
        }
    )
    processor_name: str = field(
        metadata={
            "name": "ProcessorName",
            "type": "Element",
            "namespace": "",
        }
    )
    processor_version: str = field(
        metadata={
            "name": "ProcessorVersion",
            "type": "Element",
            "namespace": "",
        }
    )
    task_name: str = field(
        metadata={
            "name": "TaskName",
            "type": "Element",
            "namespace": "",
        }
    )
    std_out_log_level: Bpsloglevels = field(
        metadata={
            "name": "StdOutLogLevel",
            "type": "Element",
            "namespace": "",
        }
    )
    std_err_log_level: Bpsloglevels = field(
        metadata={
            "name": "StdErrLogLevel",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
class BiomassIntCalConfType:
    class Meta:
        target_namespace = "aresysConfTypes"

    internal_calibration_source: BiomassIntCalConfTypeInternalCalibrationSource = field(
        metadata={
            "name": "InternalCalibrationSource",
            "type": "Element",
            "namespace": "",
        }
    )
    max_drift_amplitude_std_fraction: float = field(
        metadata={
            "name": "MaxDriftAmplitudeStdFraction",
            "type": "Element",
            "namespace": "",
        }
    )
    max_drift_phase_std_fraction: float = field(
        metadata={
            "name": "MaxDriftPhaseStdFraction",
            "type": "Element",
            "namespace": "",
        }
    )
    max_drift_amplitude_error: float = field(
        metadata={
            "name": "MaxDriftAmplitudeError",
            "type": "Element",
            "namespace": "",
        }
    )
    max_drift_phase_error: float = field(
        metadata={
            "name": "MaxDriftPhaseError",
            "type": "Element",
            "namespace": "",
        }
    )
    max_invalid_drift_fraction: float = field(
        metadata={
            "name": "MaxInvalidDriftFraction",
            "type": "Element",
            "namespace": "",
        }
    )
    beam: None | str = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class CalibrationConstantsConfType:
    """
    Polarimetric Distortion Optimization configuration parameters.
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    radiometric: CalibrationConstantsConfType.Radiometric = field(
        metadata={
            "name": "Radiometric",
            "type": "Element",
            "namespace": "",
        }
    )
    geometric: CalibrationConstantsConfType.Geometric = field(
        metadata={
            "name": "Geometric",
            "type": "Element",
            "namespace": "",
        }
    )

    @dataclass(kw_only=True)
    class Radiometric:
        """
        Parameters
        ----------
        channel_imbalance_values
            Channel Imbalance complex value normalized against ICAL  []
        cross_talk_correction
            Xtalks correction parameters to override scene based processor estimations
        """

        channel_imbalance_values: CalibrationConstantsConfType.Radiometric.ChannelImbalanceValues = field(
            metadata={
                "name": "ChannelImbalanceValues",
                "type": "Element",
                "namespace": "",
            }
        )
        cross_talk_correction: None | CalibrationConstantsConfType.Radiometric.CrossTalkCorrection = field(
            default=None,
            metadata={
                "name": "CrossTalkCorrection",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass(kw_only=True)
        class ChannelImbalanceValues:
            rx: ComplexNumberType = field(
                metadata={
                    "name": "Rx",
                    "type": "Element",
                    "namespace": "",
                }
            )
            tx: ComplexNumberType = field(
                metadata={
                    "name": "Tx",
                    "type": "Element",
                    "namespace": "",
                }
            )

        @dataclass(kw_only=True)
        class CrossTalkCorrection:
            xtalk1: ComplexNumberType = field(
                metadata={
                    "name": "Xtalk1",
                    "type": "Element",
                    "namespace": "",
                }
            )
            xtalk2: ComplexNumberType = field(
                metadata={
                    "name": "Xtalk2",
                    "type": "Element",
                    "namespace": "",
                }
            )
            xtalk3: ComplexNumberType = field(
                metadata={
                    "name": "Xtalk3",
                    "type": "Element",
                    "namespace": "",
                }
            )
            xtalk4: ComplexNumberType = field(
                metadata={
                    "name": "Xtalk4",
                    "type": "Element",
                    "namespace": "",
                }
            )
            alpha: None | ComplexNumberType = field(
                default=None,
                metadata={
                    "name": "Alpha",
                    "type": "Element",
                    "namespace": "",
                },
            )

    @dataclass(kw_only=True)
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

        internal_delay_hh: float = field(
            metadata={
                "name": "InternalDelayHH",
                "type": "Element",
                "namespace": "",
            }
        )
        internal_delay_hv: float = field(
            metadata={
                "name": "InternalDelayHV",
                "type": "Element",
                "namespace": "",
            }
        )
        internal_delay_vh: float = field(
            metadata={
                "name": "InternalDelayVH",
                "type": "Element",
                "namespace": "",
            }
        )
        internal_delay_vv: float = field(
            metadata={
                "name": "InternalDelayVV",
                "type": "Element",
                "namespace": "",
            }
        )


@dataclass(kw_only=True)
class DigitalElevationModelType:
    class Meta:
        target_namespace = "aresysConfTypes"

    type_value: DemTypes = field(
        metadata={
            "name": "Type",
            "type": "Element",
            "namespace": "",
        }
    )
    repository: None | str = field(
        default=None,
        metadata={
            "name": "Repository",
            "type": "Element",
            "namespace": "",
        },
    )
    index_file_name: None | str = field(
        default=None,
        metadata={
            "name": "IndexFileName",
            "type": "Element",
            "namespace": "",
        },
    )
    geoid_file_name: None | str = field(
        default=None,
        metadata={
            "name": "GeoidFileName",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class Fcomplex:
    class Meta:
        name = "FCOMPLEX"
        target_namespace = "aresysConfTypes"

    complex_alg: None | ComplexAlg = field(
        default=None,
        metadata={
            "name": "ComplexAlg",
            "type": "Element",
            "namespace": "",
        },
    )
    complex_pol: None | ComplexPol = field(
        default=None,
        metadata={
            "name": "ComplexPol",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    coreg_reference_polarization: None | PolarizationType = field(
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
        },
    )
    azimuth_max_shift: int = field(
        default=4,
        metadata={
            "name": "AzimuthMaxShift",
            "type": "Element",
            "namespace": "",
        },
    )
    range_block_size: int = field(
        default=0,
        metadata={
            "name": "RangeBlockSize",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_block_size: int = field(
        default=0,
        metadata={
            "name": "AzimuthBlockSize",
            "type": "Element",
            "namespace": "",
        },
    )
    coarse_input: int = field(
        default=0,
        metadata={
            "name": "CoarseInput",
            "type": "Element",
            "namespace": "",
        },
    )
    range_min_overlap: int = field(
        default=1000,
        metadata={
            "name": "RangeMinOverlap",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_min_overlap: int = field(
        default=1000,
        metadata={
            "name": "AzimuthMinOverlap",
            "type": "Element",
            "namespace": "",
        },
    )
    memory: int = field(
        default=256,
        metadata={
            "name": "Memory",
            "type": "Element",
            "namespace": "",
        },
    )
    verbose: int = field(
        default=0,
        metadata={
            "name": "Verbose",
            "type": "Element",
            "namespace": "",
        },
    )
    report_level: int = field(
        default=0,
        metadata={
            "name": "ReportLevel",
            "type": "Element",
            "namespace": "",
        },
    )
    enable_common_band_range_filter: None | bool = field(
        default=None,
        metadata={
            "name": "EnableCommonBandRangeFilter",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class IonosphericCalibrationConfType(BaseConfType):
    class Meta:
        target_namespace = "aresysConfTypes"

    perform_defocusing_on_ionospheric_height: int = field(
        metadata={
            "name": "PerformDefocusingOnIonosphericHeight",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    perform_faraday_rotation_correction: int = field(
        metadata={
            "name": "PerformFaradayRotationCorrection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    perform_phase_screen_correction: int = field(
        metadata={
            "name": "PerformPhaseScreenCorrection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    perform_group_delay_correction: int = field(
        metadata={
            "name": "PerformGroupDelayCorrection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    ionospheric_height_estimation_method: IonosphericCalibrationConfTypeIonosphericHeightEstimationMethod = field(
        metadata={
            "name": "IonosphericHeightEstimationMethod",
            "type": "Element",
            "namespace": "",
        }
    )
    squint_sensitivity: None | IonosphericCalibrationConfType.SquintSensitivity = field(
        default=None,
        metadata={
            "name": "SquintSensitivity",
            "type": "Element",
            "namespace": "",
        },
    )
    feature_tracking: None | IonosphericCalibrationConfType.FeatureTracking = field(
        default=None,
        metadata={
            "name": "FeatureTracking",
            "type": "Element",
            "namespace": "",
        },
    )
    zthreshold: float = field(
        metadata={
            "name": "ZThreshold",
            "type": "Element",
            "namespace": "",
        }
    )
    gaussian_filter_max_size_azimuth: int = field(
        metadata={
            "name": "GaussianFilterMaxSizeAzimuth",
            "type": "Element",
            "namespace": "",
        }
    )
    gaussian_filter_max_size_range: int = field(
        metadata={
            "name": "GaussianFilterMaxSizeRange",
            "type": "Element",
            "namespace": "",
        }
    )
    gaussian_filter_default_size_azimuth: int = field(
        metadata={
            "name": "GaussianFilterDefaultSizeAzimuth",
            "type": "Element",
            "namespace": "",
        }
    )
    gaussian_filter_default_size_range: int = field(
        metadata={
            "name": "GaussianFilterDefaultSizeRange",
            "type": "Element",
            "namespace": "",
        }
    )
    default_ionospheric_height: RealValueMeters = field(
        metadata={
            "name": "DefaultIonosphericHeight",
            "type": "Element",
            "namespace": "",
        }
    )
    max_ionospheric_height: RealValueMeters = field(
        metadata={
            "name": "MaxIonosphericHeight",
            "type": "Element",
            "namespace": "",
        }
    )
    min_ionospheric_height: RealValueMeters = field(
        metadata={
            "name": "MinIonosphericHeight",
            "type": "Element",
            "namespace": "",
        }
    )
    azimuth_block_overlap: int = field(
        metadata={
            "name": "AzimuthBlockOverlap",
            "type": "Element",
            "namespace": "",
        }
    )
    azimuth_block_size: int = field(
        metadata={
            "name": "AzimuthBlockSize",
            "type": "Element",
            "namespace": "",
        }
    )

    @dataclass(kw_only=True)
    class SquintSensitivity:
        number_of_looks: int = field(
            metadata={
                "name": "NumberOfLooks",
                "type": "Element",
                "namespace": "",
            }
        )
        height_step: float = field(
            metadata={
                "name": "HeightStep",
                "type": "Element",
                "namespace": "",
            }
        )
        faraday_rotation_bias: RealValueDegrees = field(
            metadata={
                "name": "FaradayRotationBias",
                "type": "Element",
                "namespace": "",
            }
        )

    @dataclass(kw_only=True)
    class FeatureTracking:
        max_offset: int = field(
            metadata={
                "name": "MaxOffset",
                "type": "Element",
                "namespace": "",
            }
        )
        profile_step: int = field(
            metadata={
                "name": "ProfileStep",
                "type": "Element",
                "namespace": "",
            }
        )
        normalized_min_value_threshold: float = field(
            metadata={
                "name": "NormalizedMinValueThreshold",
                "type": "Element",
                "namespace": "",
            }
        )


@dataclass(kw_only=True)
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

    enable_log_file: int = field(
        metadata={
            "name": "EnableLogFile",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    enable_std_output: int = field(
        metadata={
            "name": "EnableStdOutput",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    report_level: Loglevels = field(
        metadata={
            "name": "ReportLevel",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
class NoiseMapGeneratorConfType(BaseConfType):
    """
    Parameters
    ----------
    noise_normalization_constant
        Noise data normalization factor
    """

    class Meta:
        target_namespace = "aresysConfTypes"

    noise_normalization_constant: float = field(
        metadata={
            "name": "NoiseNormalizationConstant",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
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

    lpfilter_type: FilterMask = field(
        metadata={
            "name": "LPFilterType",
            "type": "Element",
            "namespace": "",
        }
    )
    order: int = field(
        metadata={
            "name": "Order",
            "type": "Element",
            "namespace": "",
        }
    )
    parameter: float = field(
        metadata={
            "name": "Parameter",
            "type": "Element",
            "namespace": "",
        }
    )
    interp_type: InterpType = field(
        metadata={
            "name": "InterpType",
            "type": "Element",
            "namespace": "",
        }
    )
    azimuth_adaptive_step_second: None | float = field(
        default=None,
        metadata={
            "name": "AzimuthAdaptiveStepSecond",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    enable_cross_talk_compensation: int = field(
        metadata={
            "name": "EnableCrossTalkCompensation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    enable_channel_imbalance_compensation: int = field(
        metadata={
            "name": "EnableChannelImbalanceCompensation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )


@dataclass(kw_only=True)
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

    remove_interferences: None | int = field(
        default=None,
        metadata={
            "name": "RemoveInterferences",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    block_size: int = field(
        metadata={
            "name": "BlockSize",
            "type": "Element",
            "namespace": "",
        }
    )
    periodgram_size: int = field(
        metadata={
            "name": "PeriodgramSize",
            "type": "Element",
            "namespace": "",
        }
    )
    persistent_rfithreshold: float = field(
        metadata={
            "name": "PersistentRFIThreshold",
            "type": "Element",
            "namespace": "",
        }
    )
    isolated_rfithreshold: float = field(
        metadata={
            "name": "IsolatedRFIThreshold",
            "type": "Element",
            "namespace": "",
        }
    )
    power_loss_threshold: None | float = field(
        default=None,
        metadata={
            "name": "PowerLossThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    threshold_std: float = field(
        metadata={
            "name": "ThresholdStd",
            "type": "Element",
            "namespace": "",
        }
    )
    percentile_low: float = field(
        metadata={
            "name": "PercentileLow",
            "type": "Element",
            "namespace": "",
        }
    )
    percentile_high: float = field(
        metadata={
            "name": "PercentileHigh",
            "type": "Element",
            "namespace": "",
        }
    )
    filtering_mode: None | RfifrequencyDomainRemovalConfTypeFilteringMode = field(
        default=None,
        metadata={
            "name": "FilteringMode",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    correction_mode: RfitimeDomainRemovalConfTypeCorrectionMode = field(
        metadata={
            "name": "CorrectionMode",
            "type": "Element",
            "namespace": "",
        }
    )
    percentile_threshold: float = field(
        metadata={
            "name": "PercentileThreshold",
            "type": "Element",
            "namespace": "",
        }
    )
    median_filter_block_lines: int = field(
        metadata={
            "name": "MedianFilterBlockLines",
            "type": "Element",
            "namespace": "",
        }
    )
    lines_in_estimate_block: int = field(
        metadata={
            "name": "LinesInEstimateBlock",
            "type": "Element",
            "namespace": "",
        }
    )
    box_filter_azimuth_dimension: int = field(
        metadata={
            "name": "BoxFilterAzimuthDimension",
            "type": "Element",
            "namespace": "",
        }
    )
    box_filter_range_dimension: int = field(
        metadata={
            "name": "BoxFilterRangeDimension",
            "type": "Element",
            "namespace": "",
        }
    )
    morph_open_line_length: int = field(
        metadata={
            "name": "MorphOpenLineLength",
            "type": "Element",
            "namespace": "",
        }
    )
    morph_close_line_length: int = field(
        metadata={
            "name": "MorphCloseLineLength",
            "type": "Element",
            "namespace": "",
        }
    )
    morph_close_before_open: None | bool = field(
        default=None,
        metadata={
            "name": "MorphCloseBeforeOpen",
            "type": "Element",
            "namespace": "",
        },
    )
    morph_open_close_iterations: None | int = field(
        default=None,
        metadata={
            "name": "MorphOpenCloseIterations",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class SarfocDigitalElevationModelType:
    class Meta:
        target_namespace = "aresysConfTypes"

    value: DemTypes = field()
    id: str = field(
        metadata={
            "name": "ID",
            "type": "Attribute",
        }
    )


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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

    ground_step: float = field(
        metadata={
            "name": "GroundStep",
            "type": "Element",
            "namespace": "",
        }
    )
    invalid_value: None | ComplexAlg = field(
        default=None,
        metadata={
            "name": "InvalidValue",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    blocks: int = field(
        metadata={
            "name": "Blocks",
            "type": "Element",
            "namespace": "",
        }
    )
    blockl: int = field(
        metadata={
            "name": "Blockl",
            "type": "Element",
            "namespace": "",
        }
    )
    undersampling_snrdcazimuth_ratio: int = field(
        metadata={
            "name": "UndersamplingSNRDCazimuthRatio",
            "type": "Element",
            "namespace": "",
        }
    )
    undersampling_snrdcrange_ratio: int = field(
        metadata={
            "name": "UndersamplingSNRDCrangeRatio",
            "type": "Element",
            "namespace": "",
        }
    )
    az_max_frequency_search_bin_number: int = field(
        metadata={
            "name": "azMaxFrequencySearchBinNumber",
            "type": "Element",
            "namespace": "",
        }
    )
    rg_max_frequency_search_bin_number: int = field(
        metadata={
            "name": "rgMaxFrequencySearchBinNumber",
            "type": "Element",
            "namespace": "",
        }
    )
    az_max_frequency_search_norm_band: float = field(
        metadata={
            "name": "azMaxFrequencySearchNormBand",
            "type": "Element",
            "namespace": "",
        }
    )
    rg_max_frequency_search_norm_band: float = field(
        metadata={
            "name": "rgMaxFrequencySearchNormBand",
            "type": "Element",
            "namespace": "",
        }
    )
    nummlbf: int = field(
        metadata={
            "name": "Nummlbf",
            "type": "Element",
            "namespace": "",
        }
    )
    nbestblocks: int = field(
        metadata={
            "name": "Nbestblocks",
            "type": "Element",
            "namespace": "",
        }
    )
    rg_band: float = field(
        metadata={
            "name": "RgBand",
            "type": "Element",
            "namespace": "",
        }
    )
    an_len: float = field(
        metadata={
            "name": "AnLen",
            "type": "Element",
            "namespace": "",
        }
    )
    lookbf: float = field(
        metadata={
            "name": "Lookbf",
            "type": "Element",
            "namespace": "",
        }
    )
    lookbt: float = field(
        metadata={
            "name": "Lookbt",
            "type": "Element",
            "namespace": "",
        }
    )
    lookrp: float = field(
        metadata={
            "name": "Lookrp",
            "type": "Element",
            "namespace": "",
        }
    )
    lookrs: float = field(
        metadata={
            "name": "Lookrs",
            "type": "Element",
            "namespace": "",
        }
    )
    decfac: int = field(
        metadata={
            "name": "Decfac",
            "type": "Element",
            "namespace": "",
        }
    )
    flength: int = field(
        metadata={
            "name": "Flength",
            "type": "Element",
            "namespace": "",
        }
    )
    dftstep: float = field(
        metadata={
            "name": "Dftstep",
            "type": "Element",
            "namespace": "",
        }
    )
    peakwid: float = field(
        metadata={
            "name": "Peakwid",
            "type": "Element",
            "namespace": "",
        }
    )
    minamb: float = field(
        metadata={
            "name": "Minamb",
            "type": "Element",
            "namespace": "",
        }
    )
    maxamb: float = field(
        metadata={
            "name": "Maxamb",
            "type": "Element",
            "namespace": "",
        }
    )
    sthr: float = field(
        metadata={
            "name": "Sthr",
            "type": "Element",
            "namespace": "",
        }
    )
    varth: float = field(
        metadata={
            "name": "Varth",
            "type": "Element",
            "namespace": "",
        }
    )
    pol_weights: StripmapDcConfType.PolWeights = field(
        metadata={
            "name": "Pol_weights",
            "type": "Element",
            "namespace": "",
        }
    )
    dc_estimation_method: DcEstimationMethodsTypes = field(
        metadata={
            "name": "DcEstimationMethod",
            "type": "Element",
            "namespace": "",
        }
    )
    attitude_fitting: AttitudeFittingTypes = field(
        metadata={
            "name": "AttitudeFitting",
            "type": "Element",
            "namespace": "",
        }
    )
    dccore_algorithm: None | DccoreAlgorithmType = field(
        default=None,
        metadata={
            "name": "DCCoreAlgorithm",
            "type": "Element",
            "namespace": "",
        },
    )
    poly_changing_freq: None | float = field(
        default=None,
        metadata={
            "name": "PolyChangingFreq",
            "type": "Element",
            "namespace": "",
        },
    )
    poly_estimation_constraint: None | PolyEstimationConstraintTypes = field(
        default=None,
        metadata={
            "name": "PolyEstimationConstraint",
            "type": "Element",
            "namespace": "",
        },
    )
    perform_joint_estimation: None | int = field(
        default=None,
        metadata={
            "name": "PerformJointEstimation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )

    @dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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

    window_type: Windows = field(
        metadata={
            "name": "WindowType",
            "type": "Element",
            "namespace": "",
        }
    )
    window_parameter: float = field(
        metadata={
            "name": "WindowParameter",
            "type": "Element",
            "namespace": "",
        }
    )
    window_look_bandwidth: WindowConfType.WindowLookBandwidth = field(
        metadata={
            "name": "WindowLookBandwidth",
            "type": "Element",
            "namespace": "",
        }
    )
    window_transition_bandwidth: WindowConfType.WindowTransitionBandwidth = field(
        metadata={
            "name": "WindowTransitionBandwidth",
            "type": "Element",
            "namespace": "",
        }
    )

    @dataclass(kw_only=True)
    class WindowLookBandwidth:
        value: float = field()
        unit: None | UnitTypes = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass(kw_only=True)
    class WindowTransitionBandwidth:
        value: float = field()
        unit: None | UnitTypes = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass(kw_only=True)
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

    lines_in_block: int = field(
        metadata={
            "name": "LinesInBlock",
            "type": "Element",
            "namespace": "",
        }
    )
    samples_in_block: int = field(
        metadata={
            "name": "SamplesInBlock",
            "type": "Element",
            "namespace": "",
        }
    )
    azimuth_overlap: int = field(
        metadata={
            "name": "AzimuthOverlap",
            "type": "Element",
            "namespace": "",
        }
    )
    range_overlap: int = field(
        metadata={
            "name": "RangeOverlap",
            "type": "Element",
            "namespace": "",
        }
    )
    perform_interpolation: int = field(
        metadata={
            "name": "PerformInterpolation",
            "type": "Element",
            "namespace": "",
        }
    )
    stolt_padding: float = field(
        metadata={
            "name": "StoltPadding",
            "type": "Element",
            "namespace": "",
        }
    )
    range_modulation: int = field(
        metadata={
            "name": "RangeModulation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    apply_azimuth_spectral_weighting_window: int = field(
        metadata={
            "name": "ApplyAzimuthSpectralWeightingWindow",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    azimuth_spectral_weighting_window: WindowConfType = field(
        metadata={
            "name": "AzimuthSpectralWeightingWindow",
            "type": "Element",
            "namespace": "",
        }
    )
    apply_rg_shift: int = field(
        metadata={
            "name": "ApplyRgShift",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    apply_az_shift: int = field(
        metadata={
            "name": "ApplyAzShift",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    whitening_flag: int = field(
        metadata={
            "name": "WhiteningFlag",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    antenna_length: float = field(
        metadata={
            "name": "AntennaLength",
            "type": "Element",
            "namespace": "",
        }
    )
    pad_result: int = field(
        metadata={
            "name": "PadResult",
            "type": "Element",
            "namespace": "",
        }
    )
    lines_to_skip_dc_fr: None | int = field(
        default=None,
        metadata={
            "name": "LinesToSkipDcFr",
            "type": "Element",
            "namespace": "",
        },
    )
    samples_to_skip_dc_fr: None | int = field(
        default=None,
        metadata={
            "name": "SamplesToSkipDcFr",
            "type": "Element",
            "namespace": "",
        },
    )
    focusing_method: None | FocusingMethodTypes = field(
        default=None,
        metadata={
            "name": "FocusingMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    az_proc_bandwidth: None | AzimuthConfType.AzProcBandwidth = field(
        default=None,
        metadata={
            "name": "AzProcBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    bistatic_delay_correction_mode: None | BistaticDelayCorrectionTypes = field(
        default=None,
        metadata={
            "name": "BistaticDelayCorrectionMode",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_time_bias: None | float = field(
        default=None,
        metadata={
            "name": "AzimuthTimeBias",
            "type": "Element",
            "namespace": "",
        },
    )
    apply_pol_channels_coregistration: None | int = field(
        default=None,
        metadata={
            "name": "ApplyPolChannelsCoregistration",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    antenna_shift_compensation_mode: None | AntennaShiftCompensationModeType = field(
        default=None,
        metadata={
            "name": "AntennaShiftCompensationMode",
            "type": "Element",
            "namespace": "",
        },
    )
    nominal_block_memory_size: None | AzimuthConfType.NominalBlockMemorySize = field(
        default=None,
        metadata={
            "name": "NominalBlockMemorySize",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class AzProcBandwidth:
        value: float = field()
        unit: None | UnitTypes = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )

    @dataclass(kw_only=True)
    class NominalBlockMemorySize:
        gpu: None | MemorySizeType = field(
            default=None,
            metadata={
                "name": "GPU",
                "type": "Element",
                "namespace": "",
            },
        )
        cpu: None | MemorySizeType = field(
            default=None,
            metadata={
                "name": "CPU",
                "type": "Element",
                "namespace": "",
            },
        )


@dataclass(kw_only=True)
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

    bpslogger_conf: BpsloggerConfType = field(
        metadata={
            "name": "BPSLoggerConf",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
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

    biomass_l0_import_conf: BiomassL0ImportConfType = field(
        metadata={
            "name": "BiomassL0ImportConf",
            "type": "Element",
            "namespace": "",
        }
    )
    biomass_raw_data_corrections_conf: BiomassRawDataCorrectionsConfType = field(
        metadata={
            "name": "BiomassRawDataCorrectionsConf",
            "type": "Element",
            "namespace": "",
        }
    )
    biomass_int_cal_conf: BiomassIntCalConfType = field(
        metadata={
            "name": "BiomassIntCalConf",
            "type": "Element",
            "namespace": "",
        }
    )
    enable_channel_delays_annotation: bool = field(
        metadata={
            "name": "EnableChannelDelaysAnnotation",
            "type": "Element",
            "namespace": "",
        }
    )
    enable_int_cal: None | bool = field(
        default=None,
        metadata={
            "name": "EnableIntCal",
            "type": "Element",
            "namespace": "",
        },
    )
    beam: None | str = field(
        default=None,
        metadata={
            "name": "Beam",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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

    non_stationary_coreg_conf: None | NonStationaryCoregConfType = field(
        default=None,
        metadata={
            "name": "NonStationaryCoregConf",
            "type": "Element",
            "namespace": "",
        },
    )
    quality_threshold_for_automatic_mode: None | float = field(
        default=None,
        metadata={
            "name": "QualityThresholdForAutomaticMode",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    residual_shift_fitting_model: FullAccuracyPostProcessingConfTypeResidualShiftFittingModel = field(
        metadata={
            "name": "ResidualShiftFittingModel",
            "type": "Element",
            "namespace": "",
        }
    )
    min_valid_blocks: int = field(
        default=50,
        metadata={
            "name": "MinValidBlocks",
            "type": "Element",
            "namespace": "",
        },
    )
    weight_threshold_refine_rg: float = field(
        metadata={
            "name": "WeightThresholdRefineRg",
            "type": "Element",
            "namespace": "",
        }
    )
    weight_threshold_refine_az: float = field(
        metadata={
            "name": "WeightThresholdRefineAz",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
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

    pfactor: int = field(
        metadata={
            "name": "PFactor",
            "type": "Element",
            "namespace": "",
        }
    )
    qfactor: int = field(
        metadata={
            "name": "QFactor",
            "type": "Element",
            "namespace": "",
        }
    )
    weighting_window: WindowConfType = field(
        metadata={
            "name": "WeightingWindow",
            "type": "Element",
            "namespace": "",
        }
    )
    central_frequency: list[MultilookDirectionConfType.CentralFrequency] = field(
        default_factory=list,
        metadata={
            "name": "CentralFrequency",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class CentralFrequency:
        value: float = field()
        unit: None | UnitTypes = field(
            default=None,
            metadata={
                "type": "Attribute",
            },
        )


@dataclass(kw_only=True)
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

    rfimitigation_method: RfimitigationMethodsType = field(
        metadata={
            "name": "RFIMitigationMethod",
            "type": "Element",
            "namespace": "",
        }
    )
    rfimask_composition_method: None | RfimaskCompositionMethodsType = field(
        default=None,
        metadata={
            "name": "RFIMaskCompositionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    rfimitigation_time_domain_conf: None | RfitimeDomainRemovalConfType = field(
        default=None,
        metadata={
            "name": "RFIMitigationTimeDomainConf",
            "type": "Element",
            "namespace": "",
        },
    )
    rfimitigation_frequency_domain_conf: None | RfifrequencyDomainRemovalConfType = field(
        default=None,
        metadata={
            "name": "RFIMitigationFrequencyDomainConf",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    rslreference_distance: float = field(
        metadata={
            "name": "RSLReferenceDistance",
            "type": "Element",
            "namespace": "",
        }
    )
    perform_rslcompensation: int = field(
        metadata={
            "name": "PerformRSLCompensation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    perform_incidence_compensation: RangeCompensatorConfType.PerformIncidenceCompensation = field(
        metadata={
            "name": "PerformIncidenceCompensation",
            "type": "Element",
            "namespace": "",
        }
    )
    perform_pattern_compensation: int = field(
        metadata={
            "name": "PerformPatternCompensation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    perform_roll_compensation: None | int = field(
        default=None,
        metadata={
            "name": "PerformRollCompensation",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    perform_line_correction: None | int = field(
        default=None,
        metadata={
            "name": "PerformLineCorrection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    fast_mode: None | int = field(
        default=None,
        metadata={
            "name": "FastMode",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    external_calibration_factor: RangeCompensatorConfType.ExternalCalibrationFactor = field(
        metadata={
            "name": "ExternalCalibrationFactor",
            "type": "Element",
            "namespace": "",
        }
    )
    processing_gain: None | Fcomplex = field(
        default=None,
        metadata={
            "name": "ProcessingGain",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class PerformIncidenceCompensation:
        value: int = field(
            metadata={
                "min_inclusive": 0,
                "max_inclusive": 1,
            }
        )
        output_quantity: None | OutputQuantityType = field(
            default=None,
            metadata={
                "name": "OutputQuantity",
                "type": "Attribute",
            },
        )

    @dataclass(kw_only=True)
    class ExternalCalibrationFactor(Fcomplex):
        apply: None | int = field(
            default=None,
            metadata={
                "name": "Apply",
                "type": "Attribute",
                "min_inclusive": 0,
                "max_inclusive": 1,
            },
        )


@dataclass(kw_only=True)
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

    flag_ortog: int = field(
        metadata={
            "name": "Flag_ortog",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    apply_range_spectral_weighting_window: int = field(
        metadata={
            "name": "ApplyRangeSpectralWeightingWindow",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        }
    )
    range_spectral_weighting_window: WindowConfType = field(
        metadata={
            "name": "RangeSpectralWeightingWindow",
            "type": "Element",
            "namespace": "",
        }
    )
    swstbias: None | float = field(
        default=None,
        metadata={
            "name": "SWSTBias",
            "type": "Element",
            "namespace": "",
        },
    )
    range_decimation_factor: None | int = field(
        default=None,
        metadata={
            "name": "RangeDecimationFactor",
            "type": "Element",
            "namespace": "",
        },
    )
    apply_rx_gain_correction: None | int = field(
        default=None,
        metadata={
            "name": "ApplyRxGainCorrection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    focusing_method: None | RangeFocusingMethodType = field(
        default=None,
        metadata={
            "name": "FocusingMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    output_border_policies: None | RangeConfType.OutputBorderPolicies = field(
        default=None,
        metadata={
            "name": "OutputBorderPolicies",
            "type": "Element",
            "namespace": "",
        },
    )
    post_processing: None | RangeConfType.PostProcessing = field(
        default=None,
        metadata={
            "name": "PostProcessing",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class OutputBorderPolicies:
        """
        Parameters
        ----------
        range
            Output range border policy
        """

        range: OutputBorderPolicyType = field(
            metadata={
                "name": "Range",
                "type": "Element",
                "namespace": "",
            }
        )

    @dataclass(kw_only=True)
    class PostProcessing:
        azimuth_resampling: None | RangeConfType.PostProcessing.AzimuthResampling = field(
            default=None,
            metadata={
                "name": "AzimuthResampling",
                "type": "Element",
                "namespace": "",
            },
        )
        enable_prfchange_data_post_processing: None | bool = field(
            default=None,
            metadata={
                "name": "EnablePRFChangeDataPostProcessing",
                "type": "Element",
                "namespace": "",
            },
        )

        @dataclass(kw_only=True)
        class AzimuthResampling:
            output_prf: RangeConfType.PostProcessing.AzimuthResampling.OutputPrf = field(
                metadata={
                    "name": "OutputPRF",
                    "type": "Element",
                    "namespace": "",
                }
            )

            @dataclass(kw_only=True)
            class OutputPrf:
                """
                Parameters
                ----------
                value
                    Output pulse repetition frequency value [Hz]
                """

                value: float = field(
                    metadata={
                        "name": "Value",
                        "type": "Element",
                        "namespace": "",
                    }
                )


@dataclass(kw_only=True)
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
    prfresampling_filter_product: None | str = field(
        default=None,
        metadata={
            "name": "PRFResamplingFilterProduct",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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
    prfchange_data_post_processing: None | bool = field(
        default=None,
        metadata={
            "name": "PRFChangeDataPostProcessing",
            "type": "Element",
            "namespace": "",
        },
    )
    rfimitigation_settings: None | SarfocProcessingSettingsType.RfimitigationSettings = field(
        default=None,
        metadata={
            "name": "RFIMitigationSettings",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class RfimitigationSettings:
        chirp_source: RfimitigationSettingsChirpSource = field(
            metadata={
                "name": "ChirpSource",
                "type": "Element",
                "namespace": "",
            }
        )
        mode: RfimitigationSettingsMode = field(
            metadata={
                "name": "Mode",
                "type": "Element",
                "namespace": "",
            }
        )


@dataclass(kw_only=True)
class Bpsl1CoreProcessingSettingsType:
    class Meta:
        name = "BPSL1CoreProcessingSettingsType"
        target_namespace = "aresysConfTypes"

    core_processing_settings: SarfocProcessingSettingsType = field(
        metadata={
            "name": "CoreProcessingSettings",
            "type": "Element",
            "namespace": "",
        }
    )
    azimuth_compression: Bpsl1CoreProcessingSettingsType.AzimuthCompression = field(
        metadata={
            "name": "AzimuthCompression",
            "type": "Element",
            "namespace": "",
        }
    )
    polarimetric_compensator: Bpsl1CoreProcessingSettingsType.PolarimetricCompensator = field(
        metadata={
            "name": "PolarimetricCompensator",
            "type": "Element",
            "namespace": "",
        }
    )
    drop_azimuth_focuser_margin: bool = field(
        metadata={
            "name": "DropAzimuthFocuserMargin",
            "type": "Element",
            "namespace": "",
        }
    )

    @dataclass(kw_only=True)
    class AzimuthCompression:
        antenna_pattern_compensation: BpsantennaPatternCompensationType = field(
            metadata={
                "name": "AntennaPatternCompensation",
                "type": "Element",
                "namespace": "",
            }
        )
        elevation_mispointing_deg: float = field(
            metadata={
                "name": "ElevationMispointingDeg",
                "type": "Element",
                "namespace": "",
            }
        )

    @dataclass(kw_only=True)
    class PolarimetricCompensator:
        enable_ionospheric_calibration: bool = field(
            metadata={
                "name": "EnableIonosphericCalibration",
                "type": "Element",
                "namespace": "",
            }
        )


@dataclass(kw_only=True)
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

    multilook_conf_name: str = field(
        metadata={
            "name": "MultilookConfName",
            "type": "Element",
            "namespace": "",
        }
    )
    apply_azimuth_time_weighting_window: None | int = field(
        default=None,
        metadata={
            "name": "ApplyAzimuthTimeWeightingWindow",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0,
            "max_inclusive": 1,
        },
    )
    azimuth_time_weighting_window: None | WindowConfType = field(
        default=None,
        metadata={
            "name": "AzimuthTimeWeightingWindow",
            "type": "Element",
            "namespace": "",
        },
    )
    normalization_factor: None | float = field(
        default=None,
        metadata={
            "name": "NormalizationFactor",
            "type": "Element",
            "namespace": "",
        },
    )
    slow_multilook: None | MultilookDirectionConfType = field(
        default=None,
        metadata={
            "name": "SlowMultilook",
            "type": "Element",
            "namespace": "",
        },
    )
    fast_multilook: None | MultilookDirectionConfType = field(
        default=None,
        metadata={
            "name": "FastMultilook",
            "type": "Element",
            "namespace": "",
        },
    )
    presum: None | MultilookConfType.Presum = field(
        default=None,
        metadata={
            "name": "Presum",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_value: None | ComplexAlg = field(
        default=None,
        metadata={
            "name": "InvalidValue",
            "type": "Element",
            "namespace": "",
        },
    )

    @dataclass(kw_only=True)
    class Presum:
        fast_factor: int = field(
            metadata={
                "name": "FastFactor",
                "type": "Element",
                "namespace": "",
            }
        )
        slow_factor: int = field(
            metadata={
                "name": "SlowFactor",
                "type": "Element",
                "namespace": "",
            }
        )


@dataclass(kw_only=True)
class StaprocessorConfType:
    class Meta:
        name = "STAProcessorConfType"
        target_namespace = "aresysConfTypes"

    full_accuracy_pre_processing_conf: FullAccuracyPreProcessingConfType = field(
        metadata={
            "name": "FullAccuracyPreProcessingConf",
            "type": "Element",
            "namespace": "",
        }
    )
    full_accuracy_post_processing_conf: None | FullAccuracyPostProcessingConfType = field(
        default=None,
        metadata={
            "name": "FullAccuracyPostProcessingConf",
            "type": "Element",
            "namespace": "",
        },
    )
    reinterpolation_conf: ReinterpolationConfType = field(
        metadata={
            "name": "ReinterpolationConf",
            "type": "Element",
            "namespace": "",
        }
    )
    coreg_mode: CoregMode = field(
        metadata={
            "name": "CoregMode",
            "type": "Element",
            "namespace": "",
        }
    )
    skip_geometry_shifts_computation: None | bool = field(
        default=None,
        metadata={
            "name": "SkipGeometryShiftsComputation",
            "type": "Element",
            "namespace": "",
        },
    )
    earth_geometry: None | EarthGeometry = field(
        default=None,
        metadata={
            "name": "EarthGeometry",
            "type": "Element",
            "namespace": "",
        },
    )
    coregistration_output_products_conf: None | CoregistrationOutputProductsConfType = field(
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
        },
    )
    memory_sargeometry: int = field(
        default=256,
        metadata={
            "name": "MemorySARGeometry",
            "type": "Element",
            "namespace": "",
        },
    )
    digital_elevation_model_repository: str = field(
        metadata={
            "name": "DigitalElevationModelRepository",
            "type": "Element",
            "namespace": "",
        }
    )
    xsd_schema_repository: str = field(
        metadata={
            "name": "XsdSchemaRepository",
            "type": "Element",
            "namespace": "",
        }
    )
    azimuth_adaptive_step_second: None | float = field(
        default=None,
        metadata={
            "name": "AzimuthAdaptiveStepSecond",
            "type": "Element",
            "namespace": "",
        },
    )
    do_slave_synthetic_compensation: None | bool = field(
        default=None,
        metadata={
            "name": "DoSlaveSyntheticCompensation",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class Bpsl1CoreProcessorConfType(BaseConfType):
    """
    BPSL1CoreProcessor configuration parameters.
    """

    class Meta:
        name = "BPSL1CoreProcessorConfType"
        target_namespace = "aresysConfTypes"

    processing_steps: SarfocProcessingStepsType = field(
        metadata={
            "name": "ProcessingSteps",
            "type": "Element",
            "namespace": "",
        }
    )
    processing_settings: Bpsl1CoreProcessingSettingsType = field(
        metadata={
            "name": "ProcessingSettings",
            "type": "Element",
            "namespace": "",
        }
    )
    output_products: SarfocOutputProductsType = field(
        metadata={
            "name": "OutputProducts",
            "type": "Element",
            "namespace": "",
        }
    )
    external_resources: SarfocExternalResourcesType = field(
        metadata={
            "name": "ExternalResources",
            "type": "Element",
            "namespace": "",
        }
    )
    interface_settings: Bpsl1CoreProcessorInterfaceSettingsType = field(
        metadata={
            "name": "InterfaceSettings",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
class AresysXmlDocType:
    number_of_channels: int = field(
        metadata={
            "name": "NumberOfChannels",
            "type": "Element",
        }
    )
    version_number: float = field(
        metadata={
            "name": "VersionNumber",
            "type": "Element",
        }
    )
    description: str = field(
        metadata={
            "name": "Description",
            "type": "Element",
        }
    )
    channel: list[AresysXmlDocType.Channel] = field(
        default_factory=list,
        metadata={
            "name": "Channel",
            "type": "Element",
        },
    )

    @dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class AresysXmlDoc(AresysXmlDocType):
    pass
