# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD PP1 models
--------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from bps.common.io.common_types import (
    AutofocusMethodType,
    AzimuthPolynomialType,
    BistaticDelayCorrectionMethodType,
    ChannelImbalanceList,
    ChannelType,
    Complex,
    ComplexArray,
    CompressionMethodType,
    CrossTalkList,
    DatumType,
    DcMethodType,
    DoubleArray,
    DoubleArrayWithUnits,
    DoubleWithUnit,
    FloatArray,
    FloatArrayWithUnits,
    FloatWithChannel,
    FloatWithPolarisation,
    FloatWithUnit,
    GeodeticReferenceFrameType,
    GroupType,
    HeightModelBaseType,
    HeightModelType,
    IntArray,
    InterferometricPairListType,
    InterferometricPairType,
    InternalCalibrationSourceType,
    IonosphereHeightEstimationMethodType,
    L1AcQlColourCodingMethodType,
    L1BQlColourCodingMethodType,
    LayerListType,
    LayerType,
    MinMaxType,
    MinMaxTypeWithUnit,
    PixelQuantityType,
    PixelRepresentationType,
    PolarisationType,
    RangeCompressionMethodType,
    RangeReferenceFunctionType,
    RfiFmmitigationMethodType,
    RfiMaskGenerationMethodType,
    RfiMaskType,
    RfiMitigationMethodType,
    SlantRangePolynomialType,
    StateType,
    SwathType,
    TimeTypeWithPolarisation,
    UnsignedIntWithGroup,
    UomType,
    WeightingWindowType,
)


class AzimuthCompressionTypeFilterType(Enum):
    FIR = "FIR"
    SINC = "SINC"


@dataclass(kw_only=True)
class L0ProductImportType:
    """
    Parameters
    ----------
    block_size
        Number of ISPs read in block during import.
    max_ispgap
        Maximum number of gaps between science data ISPs. If the number of consecutive missing packets for one APID
        exceeds this parameter, the processing will not continue.
    raw_mean_expected
        This parameter specifies the expected mean of the samples in the extracted raw data and it is used for
        verifying that the calculated mean is within the tolerated threshold.
    raw_mean_threshold
        Threshold for setting the corresponding quality parameter in the L1 product annotations. This is the value X
        such that the measured mean must fall between the rawMeanExpected-X and rawMeanExpected+X.
    raw_std_expected
        This parameter specifies the expected standard deviation of the samples in the extracted raw data and it is
        used for verifying that the calculated standard deviation is within the tolerated threshold.
    raw_std_threshold
        Threshold for setting the corresponding quality parameter in the L1 product annotations. This is the value X
        such that the measured standard deviation must fall between the rawStdExpected-X and rawStdExpected+X.
    internal_calibration_estimation_flag
        True if internal calibration estimation has to be performed, False otherwise.
    """

    class Meta:
        name = "l0ProductImportType"

    block_size: int = field(
        metadata={
            "name": "blockSize",
            "type": "Element",
            "namespace": "",
        },
    )
    max_ispgap: int = field(
        metadata={
            "name": "maxISPGap",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_mean_expected: float = field(
        metadata={
            "name": "rawMeanExpected",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_mean_threshold: float = field(
        metadata={
            "name": "rawMeanThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_std_expected: float = field(
        metadata={"name": "rawStdExpected", "type": "Element", "namespace": "", "min_inclusive": 0.0}
    )
    raw_std_threshold: float = field(
        metadata={"name": "rawStdThreshold", "type": "Element", "namespace": "", "min_inclusive": 0.0}
    )
    internal_calibration_estimation_flag: str = field(
        metadata={
            "name": "internalCalibrationEstimationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )


@dataclass(kw_only=True)
class ParcProcessingType:
    """
    Parameters
    ----------
    parc_roisamples
        Number of samples of the ROI selected around PARC position in the image. Used only in case PARC processing
        mode is activated.
    parc_roilines
        Number of lines of the ROI selected around PARC position in the image. Used only in case PARC processing
        mode is activated.
    """

    class Meta:
        name = "parcProcessingType"

    parc_roisamples: int = field(
        metadata={
            "name": "parcROISamples",
            "type": "Element",
            "namespace": "",
        },
    )
    parc_roilines: int = field(
        metadata={
            "name": "parcROILines",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class RawDataCorrectionType:
    """
    Parameters
    ----------
    raw_data_correction_flag
        True if raw data correction has to be performed, False otherwise.
    bias_correction_flag
        True if bias correction has to be performed, False otherwise.
    gain_imbalance_correction_flag
        True if gain imbalance correction has to be performed, False otherwise.
    non_orthogonality_correction_flag
        True if non-orthogonality correction has to be performed, False otherwise.
    """

    class Meta:
        name = "rawDataCorrectionType"

    raw_data_correction_flag: str = field(
        metadata={"name": "rawDataCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    bias_correction_flag: str = field(
        metadata={"name": "biasCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    gain_imbalance_correction_flag: str = field(
        metadata={
            "name": "gainImbalanceCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    non_orthogonality_correction_flag: str = field(
        metadata={
            "name": "nonOrthogonalityCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )


class RfiMitigationTypeRfiMitigationMode(Enum):
    ENABLED = "Enabled"
    DISABLED = "Disabled"
    MASK_BASED = "MaskBased"


@dataclass(kw_only=True)
class RfiTmprocessingParametersType:
    """
    Parameters
    ----------
    block_lines
        Number of lines in the processing block.
    median_filter_length
        Median filter length.
    box_samples
        Number of samples in the boxes used to compute space-variant statistics.
    box_lines
        Number of lines in the boxes used to compute space-variant statistics.
    percentile_threshold
        Normalized threshold on percentile of the signal distribution model. Lower values imply more false
        positives, higher values imply more missing detection.
    morphological_open_operator_samples
        Size in samples of the morphological open operator.
    morphological_open_operator_lines
        Size in lines of the morphological open operator.
    morphological_close_operator_samples
        Size in samples of the morphological close operator.
    morphological_close_operator_lines
        Size in lines of the morphological close operator.
    max_rfitmpercentage
        Maximum normalized percentage of image pixels impacted by time-domain RFI. If the percentage exceeds this
        parameter, the processing will not continue.
    """

    class Meta:
        name = "rfiTMProcessingParametersType"

    block_lines: int = field(
        metadata={
            "name": "blockLines",
            "type": "Element",
            "namespace": "",
        },
    )
    median_filter_length: int = field(
        metadata={
            "name": "medianFilterLength",
            "type": "Element",
            "namespace": "",
        },
    )
    box_samples: int = field(
        metadata={
            "name": "boxSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    box_lines: int = field(
        metadata={
            "name": "boxLines",
            "type": "Element",
            "namespace": "",
        },
    )
    percentile_threshold: float = field(
        metadata={
            "name": "percentileThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    morphological_open_operator_samples: int = field(
        metadata={
            "name": "morphologicalOpenOperatorSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    morphological_open_operator_lines: int = field(
        metadata={
            "name": "morphologicalOpenOperatorLines",
            "type": "Element",
            "namespace": "",
        },
    )
    morphological_close_operator_samples: int = field(
        metadata={
            "name": "morphologicalCloseOperatorSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    morphological_close_operator_lines: int = field(
        metadata={
            "name": "morphologicalCloseOperatorLines",
            "type": "Element",
            "namespace": "",
        },
    )
    max_rfitmpercentage: float = field(
        metadata={
            "name": "maxRFITMPercentage",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )


@dataclass(kw_only=True)
class AutofocusType:
    """
    Parameters
    ----------
    autofocus_flag
        True if autofocus has to be performed, False otherwise.
    autofocus_method
        Autofocus method to be used during processing (Map Drift).
    map_drift_azimuth_sub_bands
        Number of azimuth sub-bands for map-drift autofocus.
    map_drift_correlation_window_width
        Width of amplitude correlation window for map-drift autofocus.
    map_drift_correlation_window_height
        Height of amplitude correlation window for map-drift autofocus.
    map_drift_range_correlation_windows
        Number of amplitude correlation windows in range for map-drift autofocus.
    map_drift_azimuth_correlation_windows
        Number of amplitude correlation windows in azimuth for map-drift autofocus.
    max_valid_shift
        Maximum allowed shift between sub-looks (map-drift autofocus) in order to consider the measurement as valid.
        If the measurement is larger than the threshold, the estimation for that block will be considered invalid.
    valid_blocks_percentage
        Minimum normalized percentage of blocks within the image that need to have estimated an azimuth shift
        smaller than the maxValidShift threshold. If the minimum percentage is not met, no correction based on the
        azimuth shift estimates will be applied.
    """

    class Meta:
        name = "autofocusType"

    autofocus_flag: str = field(
        metadata={"name": "autofocusFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    autofocus_method: AutofocusMethodType = field(
        metadata={
            "name": "autofocusMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    map_drift_azimuth_sub_bands: int = field(
        metadata={
            "name": "mapDriftAzimuthSubBands",
            "type": "Element",
            "namespace": "",
        },
    )
    map_drift_correlation_window_width: int = field(
        metadata={
            "name": "mapDriftCorrelationWindowWidth",
            "type": "Element",
            "namespace": "",
        },
    )
    map_drift_correlation_window_height: int = field(
        metadata={
            "name": "mapDriftCorrelationWindowHeight",
            "type": "Element",
            "namespace": "",
        },
    )
    map_drift_range_correlation_windows: int = field(
        metadata={
            "name": "mapDriftRangeCorrelationWindows",
            "type": "Element",
            "namespace": "",
        },
    )
    map_drift_azimuth_correlation_windows: int = field(
        metadata={
            "name": "mapDriftAzimuthCorrelationWindows",
            "type": "Element",
            "namespace": "",
        },
    )
    max_valid_shift: float = field(
        metadata={"name": "maxValidShift", "type": "Element", "namespace": "", "min_inclusive": 0.0}
    )
    valid_blocks_percentage: float = field(
        metadata={
            "name": "validBlocksPercentage",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )


@dataclass(kw_only=True)
class RfiFmprocessingParametersType:
    """
    Parameters
    ----------
    block_lines
        Number of lines in the blocks used to compute Power Spectral Density.
    block_overlap
        Overlap between blocks.
    persistent_rfithreshold
        Threshold for the detection of persistent RFI.
    isolated_rfithreshold
        Threshold for the detection of isolated RFI.
    isolated_rfipsdstd_threshold
        Threshold on PSD standard deviation for isolated RFI.
    max_rfifmpercentage
        Maximum normalized percentage of image spectrum impacted by frequency-domain RFI. If the percentage exceeds
        this parameter, the processing will not continue.
    periodgram_size
        Size of the periodgram
    enable_power_loss_compensation
        Whether to apply power loss compensation
    power_loss_threshold
        Power loss compensation threshold
    chirp_source
        Chirp source for RFI frequency mitigation (Nominal (i.e. ideal chirp), Replica (i.e. chirp derived from
        internal calibration) or Internal (i.e. chirp taken from AUX_INS file)).
    mitigation_method
        method for RFI frequency mitigation: NOTCH_FILTER or NEAREST_NEIGHBOUR_INTERPOLATION
    """

    class Meta:
        name = "rfiFMProcessingParametersType"

    block_lines: int = field(
        metadata={
            "name": "blockLines",
            "type": "Element",
            "namespace": "",
        },
    )
    block_overlap: int = field(
        metadata={
            "name": "blockOverlap",
            "type": "Element",
            "namespace": "",
        },
    )
    persistent_rfithreshold: float = field(
        metadata={"name": "persistentRFIThreshold", "type": "Element", "namespace": "", "min_inclusive": 0.0}
    )
    isolated_rfithreshold: float = field(
        metadata={"name": "isolatedRFIThreshold", "type": "Element", "namespace": "", "min_inclusive": 0.0}
    )
    isolated_rfipsdstd_threshold: float = field(
        metadata={"name": "isolatedRFIPSDStdThreshold", "type": "Element", "namespace": "", "min_inclusive": 0.0}
    )
    max_rfifmpercentage: float = field(
        metadata={
            "name": "maxRFIFMPercentage",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    periodgram_size: int = field(
        metadata={
            "name": "periodgramSize",
            "type": "Element",
            "namespace": "",
        },
    )
    enable_power_loss_compensation: bool = field(
        metadata={
            "name": "enablePowerLossCompensation",
            "type": "Element",
            "namespace": "",
        },
    )
    power_loss_threshold: float = field(
        metadata={
            "name": "powerLossThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    chirp_source: RangeReferenceFunctionType = field(
        metadata={
            "name": "chirpSource",
            "type": "Element",
            "namespace": "",
        },
    )
    mitigation_method: RfiFmmitigationMethodType = field(
        metadata={
            "name": "mitigationMethod",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AntennaPatternCorrectionType:
    """
    Parameters
    ----------
    antenna_pattern_correction1_flag
        True if antenna pattern correction (first step, i.e., compensation for an average off-nadir profile [RD12])
        has to be applied, False otherwise. Note that in case internalCalibrationEstimationFlag is False, Transmit
        Power Tracking default values are taken from AUX_INS file.
    antenna_pattern_correction2_flag
        True if antenna pattern correction (second step, i.e., topography-adaptive compensation of radiometric bias
        [RD12]) has to be applied, False otherwise. Note that in case internalCalibrationEstimationFlag is False,
        Transmit Power Tracking default values are taken from AUX_INS file.
    antenna_cross_talk_correction_flag
        True if antenna cross-talk correction has to be applied, False otherwise. Note that in case
        internalCalibrationEstimationFlag is False, Transmit Power Tracking default values are taken from AUX_INS
        file.
    elevation_mispointing_bias
        Mis-pointing bias in elevation to be applied during antenna pattern compensation [deg].
    azimuth_mispointing_bias
        Mis-pointing bias in azimuth to be applied during antenna pattern compensation [deg].
    """

    class Meta:
        name = "antennaPatternCorrectionType"

    antenna_pattern_correction1_flag: str = field(
        metadata={
            "name": "antennaPatternCorrection1Flag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    antenna_pattern_correction2_flag: str = field(
        metadata={
            "name": "antennaPatternCorrection2Flag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    antenna_cross_talk_correction_flag: str = field(
        metadata={
            "name": "antennaCrossTalkCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    elevation_mispointing_bias: FloatWithUnit = field(
        metadata={
            "name": "elevationMispointingBias",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_mispointing_bias: FloatWithUnit = field(
        metadata={
            "name": "azimuthMispointingBias",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class CalibrationConstantListType:
    """
    Parameters
    ----------
    absolute_calibration_constant
        Absolute calibration constant applied multiplicatively to the image during processing for the current
        polarisation.
    count
    """

    class Meta:
        name = "calibrationConstantListType"

    absolute_calibration_constant: list[FloatWithPolarisation] = field(
        default_factory=list,
        metadata={
            "name": "absoluteCalibrationConstant",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
            "max_occurs": 4,
        },
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class DopplerEstimationType:
    """
    Parameters
    ----------
    dc_method
        Doppler Centroid estimation method to be used during processing (Geometry, Combined, Fixed).
    dc_value
        Doppler Centroid value to be used during processing [Hz]. Used only in case dcMethod is set to Fixed. Note
        that in any case the Doppler Centroid value used for Antenna Pattern Correction is always the estimated one
        (in this case with Combined method), while this value is used for spectrum filtering operations.
    block_samples
        Number of samples in the blocks used to estimate Doppler Centroid from data. Used only in case dcMethod is
        set to Combined.
    block_lines
        Number of lines in the blocks used to estimate Doppler Centroid from data. Used only in case dcMethod is set
        to Combined.
    polynomial_update_rate
        Estimated Doppler polynomials update rate [s]. Default value is 5s.
    dc_rmserror_threshold
        Doppler Centroid estimation root mean squared (RMS) error threshold [Hz]. If the RMS error of the Doppler
        Centroid combined estimates is above this threshold they shall not be used during processing; instead, the
        Doppler Centroid calculated from geometry shall be used.
    """

    class Meta:
        name = "dopplerEstimationType"

    dc_method: DcMethodType = field(
        metadata={
            "name": "dcMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    dc_value: DoubleWithUnit = field(
        metadata={
            "name": "dcValue",
            "type": "Element",
            "namespace": "",
        },
    )
    block_samples: int = field(
        metadata={
            "name": "blockSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    block_lines: int = field(
        metadata={
            "name": "blockLines",
            "type": "Element",
            "namespace": "",
        },
    )
    polynomial_update_rate: FloatWithUnit = field(
        metadata={
            "name": "polynomialUpdateRate",
            "type": "Element",
            "namespace": "",
        },
    )
    dc_rmserror_threshold: FloatWithUnit = field(
        metadata={
            "name": "dcRMSErrorThreshold",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class GeneralType:
    """
    Parameters
    ----------
    height_model
        Digital Elevation Model (DEM) to be used during processing.
    height_model_fallback_flag
        True to allow fallback to Ellipsoid in case selected Digital Elevation Model (DEM) is not found, False
        otherwise. Default value is False.
    height_model_margin
        Margin kept during the extraction of Digital Elevation Model (DEM) to be used during processing [deg].
    parc_processing
        PARC processing mode processing parameters.
    dual_polarisation_processing_flag
        True if processing has to be forced in case input data contain just two polarisations, False otherwise.
        Default value is False, to be set to True to manage contingency cases.
    """

    class Meta:
        name = "generalType"

    height_model: HeightModelType = field(
        metadata={
            "name": "heightModel",
            "type": "Element",
            "namespace": "",
        },
    )
    height_model_fallback_flag: str = field(
        metadata={"name": "heightModelFallbackFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    height_model_margin: FloatWithUnit = field(
        metadata={
            "name": "heightModelMargin",
            "type": "Element",
            "namespace": "",
        },
    )
    parc_processing: ParcProcessingType = field(
        metadata={
            "name": "parcProcessing",
            "type": "Element",
            "namespace": "",
        },
    )
    dual_polarisation_processing_flag: str = field(
        metadata={
            "name": "dualPolarisationProcessingFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )


@dataclass(kw_only=True)
class GroundProjectionType:
    """
    Parameters
    ----------
    ground_projection_flag
        True if slant range to ground range conversion has to be performed, False otherwise.
    range_pixel_spacing
        Pixel spacing between ground range samples [m].
    filter_type
        Type of filters used for slant range to ground range interpolation (Sinc, GLS, ...).
    filter_bandwidth
        Filter bandwidth [Hz].
    filter_length
        Filter length (in samples).
    number_of_filters
        Number of filters in the bank.
    """

    class Meta:
        name = "groundProjectionType"

    ground_projection_flag: str = field(
        metadata={"name": "groundProjectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    range_pixel_spacing: FloatWithUnit = field(
        metadata={
            "name": "rangePixelSpacing",
            "type": "Element",
            "namespace": "",
        },
    )
    filter_type: str = field(
        metadata={
            "name": "filterType",
            "type": "Element",
            "namespace": "",
        },
    )
    filter_bandwidth: FloatWithUnit = field(
        metadata={
            "name": "filterBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    filter_length: int = field(
        metadata={"name": "filterLength", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    number_of_filters: int = field(
        metadata={"name": "numberOfFilters", "type": "Element", "namespace": "", "min_inclusive": 1}
    )


@dataclass(kw_only=True)
class InternalCalibrationCorrectionType:
    """
    Parameters
    ----------
    internal_calibration_correction_flag
        True if internal calibration correction has to be performed, False otherwise. Used only in case
        internalCalibrationEstimationFlag is set to True.
    drift_correction_flag
        True if instrument drift correction has to be performed, False otherwise. Implies correction of channel
        imbalances.
    delay_correction_flag
        True if instrument delay correction has to be performed, False otherwise.
    channel_imbalance_correction_flag
        True if instrument channel imbalance correction has to be performed, False otherwise. This flag is ignored
        if driftCorrectionFlag is True.
    internal_calibration_source
        Internal calibration parameters to be used during processing (Model or Extracted). Used only in case
        internalCalibrationCorrectionFlag is set to True.
    max_drift_amplitude_std_fraction
        Maximum deviation from the mean allowed for the drift amplitude, measured as a fraction of the standard
        deviation. Relative drift validation shall fail if this value is exceeded.
    max_drift_phase_std_fraction
        Maximum deviation from the mean allowed for the drift phase, measured as a fraction of the standard
        deviation. Relative drift validation shall fail if this value is exceeded.
    max_drift_amplitude_error
        Maximum deviation allowed for a drift amplitude from the corresponding model value. Absolute drift
        validation shall fail if this value is exceeded.
    max_drift_phase_error
        Maximum deviation allowed for a drift phase from the corresponding model value [rad]. Absolute drift
        validation shall fail if this value is exceeded.
    max_invalid_drift_fraction
        Maximum number of invalid drift values allowed, expressed as a normalized fraction of the total number of
        drifts. If the percentage of the invalid drifts does not exceed this value, then the invalid drifts will be
        discarded and only the valid ones will be further used in the processing. Otherwise, all the calculated
        drift values will be discarded and replaced with the corresponding model values. Used only in case
        internalCalibrationSource is set to Extracted.
    """

    class Meta:
        name = "internalCalibrationCorrectionType"

    internal_calibration_correction_flag: str = field(
        metadata={
            "name": "internalCalibrationCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    drift_correction_flag: str = field(
        metadata={"name": "driftCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    delay_correction_flag: str = field(
        metadata={"name": "delayCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    channel_imbalance_correction_flag: str = field(
        metadata={
            "name": "channelImbalanceCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    internal_calibration_source: InternalCalibrationSourceType = field(
        metadata={
            "name": "internalCalibrationSource",
            "type": "Element",
            "namespace": "",
        },
    )
    max_drift_amplitude_std_fraction: float = field(
        metadata={"name": "maxDriftAmplitudeStdFraction", "type": "Element", "namespace": "", "min_inclusive": 0.0}
    )
    max_drift_phase_std_fraction: float = field(
        metadata={"name": "maxDriftPhaseStdFraction", "type": "Element", "namespace": "", "min_inclusive": 0.0}
    )
    max_drift_amplitude_error: float = field(
        metadata={"name": "maxDriftAmplitudeError", "type": "Element", "namespace": "", "min_inclusive": 0.0}
    )
    max_drift_phase_error: FloatWithUnit = field(
        metadata={
            "name": "maxDriftPhaseError",
            "type": "Element",
            "namespace": "",
        },
    )
    max_invalid_drift_fraction: float = field(
        metadata={
            "name": "maxInvalidDriftFraction",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )


@dataclass(kw_only=True)
class IonosphereCalibrationType:
    """
    Parameters
    ----------
    ionosphere_height_defocusing_flag
        True if defocusing at ionosphere height has to be performed, False otherwise.
    ionosphere_height_estimation_method
        Ionosphere height estimation method (Automatic, Feature Tracking, Squint Sensitivity, Model, Fixed).
    ionosphere_height_value
        Ionosphere height value to be used during processing [m]. Used only in case ionosphereHeightEstimationMethod
        is set to Fixed.
    ionosphere_height_estimation_method_latitude_threshold
        Latitude threshold for ionosphere height estimation method selection [deg]. If latitude is below this
        threshold the squint sensitivity method is used, otherwise the feature tracking one is selected.
    ionosphere_height_minimum_value
        Minimum value for ionosphere height estimation [m]. If ionosphere height is estimated and estimated value is
        lower than this value, it is discarded and input model is used instead.
    ionosphere_height_maximum_value
        Maximum value for ionosphere height estimation [m]. If ionosphere height is estimated and estimated value is
        higher than this value, it is discarded and input model is used instead.
    squint_sensitivity_number_of_looks
        Number of looks used for ionosphere height estimation through squint sensitivity method.
    squint_sensitivity_number_of_ticks
        Number of height ticks to be processed for ionosphere height estimation through squint sensitivity method.
    squint_sensitivity_fitting_error
        Fitting error for TEC and ionosphere height estimates through squint sensitivity method [(rad/nT)^2].
    gaussian_filter_maximum_major_axis_length
        Gaussian filter maximum length of major axis.
    gaussian_filter_maximum_minor_axis_length
        Gaussian filter maximum length of minor axis.
    gaussian_filter_major_axis_length
        Gaussian filter length of major axis. Used in case computed filter dimensions exceed above reported values.
    gaussian_filter_minor_axis_length
        Gaussian filter length of minor axis. Used in case computed filter dimensions exceed above reported values.
    gaussian_filter_slope
        Gaussian filter slope. Used in case computed filter dimensions exceed above reported values.
    faraday_rotation_correction_flag
        True if Faraday Rotation correction has to be performed, False otherwise.
    ionospheric_phase_screen_correction_flag
        True if ionospheric phase screen correction has to be performed, False otherwise.
    group_delay_correction_flag
        True if group delay correction has to be performed, False otherwise.
    block_lines
        Number of lines in the processing blocks.
    block_overlap_lines
        Number of lines in the overlap between azimuth processing blocks.
    """

    class Meta:
        name = "ionosphereCalibrationType"

    ionosphere_height_defocusing_flag: str = field(
        metadata={
            "name": "ionosphereHeightDefocusingFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    ionosphere_height_estimation_method: IonosphereHeightEstimationMethodType = field(
        metadata={
            "name": "ionosphereHeightEstimationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_height_value: FloatWithUnit = field(
        metadata={
            "name": "ionosphereHeightValue",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_height_estimation_method_latitude_threshold: FloatWithUnit = field(
        metadata={
            "name": "ionosphereHeightEstimationMethodLatitudeThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_height_minimum_value: FloatWithUnit = field(
        metadata={
            "name": "ionosphereHeightMinimumValue",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_height_maximum_value: FloatWithUnit = field(
        metadata={
            "name": "ionosphereHeightMaximumValue",
            "type": "Element",
            "namespace": "",
        },
    )
    squint_sensitivity_number_of_looks: int = field(
        metadata={
            "name": "squintSensitivityNumberOfLooks",
            "type": "Element",
            "namespace": "",
        },
    )
    squint_sensitivity_number_of_ticks: int = field(
        metadata={
            "name": "squintSensitivityNumberOfTicks",
            "type": "Element",
            "namespace": "",
        },
    )
    squint_sensitivity_fitting_error: FloatArrayWithUnits = field(
        metadata={
            "name": "squintSensitivityFittingError",
            "type": "Element",
            "namespace": "",
        },
    )
    gaussian_filter_maximum_major_axis_length: int = field(
        metadata={
            "name": "gaussianFilterMaximumMajorAxisLength",
            "type": "Element",
            "namespace": "",
        },
    )
    gaussian_filter_maximum_minor_axis_length: int = field(
        metadata={
            "name": "gaussianFilterMaximumMinorAxisLength",
            "type": "Element",
            "namespace": "",
        },
    )
    gaussian_filter_major_axis_length: int = field(
        metadata={
            "name": "gaussianFilterMajorAxisLength",
            "type": "Element",
            "namespace": "",
        },
    )
    gaussian_filter_minor_axis_length: int = field(
        metadata={
            "name": "gaussianFilterMinorAxisLength",
            "type": "Element",
            "namespace": "",
        },
    )
    gaussian_filter_slope: float = field(
        metadata={
            "name": "gaussianFilterSlope",
            "type": "Element",
            "namespace": "",
        },
    )
    faraday_rotation_correction_flag: str = field(
        metadata={
            "name": "faradayRotationCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    ionospheric_phase_screen_correction_flag: str = field(
        metadata={
            "name": "ionosphericPhaseScreenCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    group_delay_correction_flag: str = field(
        metadata={"name": "groupDelayCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    block_lines: int = field(
        metadata={
            "name": "blockLines",
            "type": "Element",
            "namespace": "",
        },
    )
    block_overlap_lines: int = field(
        metadata={
            "name": "blockOverlapLines",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class L1AProcessingParametersType:
    """
    Parameters
    ----------
    swath
        Swath (S1, S2, S3).
    time_bias
        Time bias to be applied on data during processing [s].
    window_type
        Name of the weighting window type to be used during processing.
    window_coefficient
        Value of the weighting window coefficient to be used during processing.
    processing_bandwidth
        Bandwidth to be used during processing [Hz].
    """

    class Meta:
        name = "l1aProcessingParametersType"

    swath: SwathType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    time_bias: DoubleWithUnit = field(
        metadata={
            "name": "timeBias",
            "type": "Element",
            "namespace": "",
        },
    )
    window_type: WeightingWindowType = field(
        metadata={
            "name": "windowType",
            "type": "Element",
            "namespace": "",
        },
    )
    window_coefficient: float = field(
        metadata={
            "name": "windowCoefficient",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    processing_bandwidth: FloatWithUnit = field(
        metadata={
            "name": "processingBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class L1BProcessingParametersType:
    """
    Parameters
    ----------
    swath
        Swath (S1, S2, S3).
    window_type
        Name of the weighting window type to be used during processing.
    window_coefficient
        Value of the weighting window coefficient to be used during processing.
    look_bandwidth
        Bandwidth for each look to be used during processing [Hz].
    number_of_looks
        Number of looks.
    look_central_frequencies
        Central frequency for each look [Hz]. Array shall contain numberOfLooks values.
    upsampling_factor
        Upsampling factor.
    downsampling_factor
        Downsampling factor.
    """

    class Meta:
        name = "l1bProcessingParametersType"

    swath: SwathType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    window_type: WeightingWindowType = field(
        metadata={
            "name": "windowType",
            "type": "Element",
            "namespace": "",
        },
    )
    window_coefficient: float = field(
        metadata={
            "name": "windowCoefficient",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    look_bandwidth: FloatWithUnit = field(
        metadata={
            "name": "lookBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    number_of_looks: int = field(
        metadata={"name": "numberOfLooks", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    look_central_frequencies: FloatArrayWithUnits = field(
        metadata={
            "name": "lookCentralFrequencies",
            "type": "Element",
            "namespace": "",
        },
    )
    upsampling_factor: int = field(
        metadata={"name": "upsamplingFactor", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    downsampling_factor: int = field(
        metadata={"name": "downsamplingFactor", "type": "Element", "namespace": "", "min_inclusive": 1}
    )


@dataclass(kw_only=True)
class LutDecimationFactorListType:
    """
    Parameters
    ----------
    lut_decimation_factor
        Look-Up Tables (LUT) ADS decimation factor w.r.t. output L1 product sampling grid for the current LUT group.
    count
    """

    class Meta:
        name = "lutDecimationFactorListType"

    lut_decimation_factor: list[UnsignedIntWithGroup] = field(
        default_factory=list,
        metadata={"name": "lutDecimationFactor", "type": "Element", "namespace": "", "min_occurs": 3, "max_occurs": 3},
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class NoiseGainListType:
    """
    Parameters
    ----------
    noise_gain
        Processing gain to be applied multiplicatively to the thermal noise level before using it during denoising
        step for the current polarisation.
    count
    """

    class Meta:
        name = "noiseGainListType"

    noise_gain: list[FloatWithPolarisation] = field(
        default_factory=list,
        metadata={"name": "noiseGain", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class PolarimetricCalibrationType:
    """
    Parameters
    ----------
    polarimetric_correction_flag
        True if polarimetric correction has to be performed, False otherwise.
    tx_distortion_matrix_correction_flag
        True if TX polarimetric distortion matrix correction has to be performed, False otherwise.
    rx_distortion_matrix_correction_flag
        True if RX polarimetric distortion matrix correction has to be performed, False otherwise.
    cross_talk_correction_flag
        True if cross-talk correction has to be performed, False otherwise.
    cross_talk_list
        Cross-talk values to be used during processing if crossTalkCorrectionFlag is True.
    channel_imbalance_correction_flag
        True if channel imbalance correction has to be performed, False otherwise.
    channel_imbalance_list
        Channel imbalance values to be used during processing if channelImbalanceCorrectionFlag is True.
    """

    class Meta:
        name = "polarimetricCalibrationType"

    polarimetric_correction_flag: str = field(
        metadata={
            "name": "polarimetricCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    tx_distortion_matrix_correction_flag: str = field(
        metadata={
            "name": "txDistortionMatrixCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    rx_distortion_matrix_correction_flag: str = field(
        metadata={
            "name": "rxDistortionMatrixCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    cross_talk_correction_flag: str = field(
        metadata={"name": "crossTalkCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    cross_talk_list: CrossTalkList = field(
        metadata={
            "name": "crossTalkList",
            "type": "Element",
            "namespace": "",
        },
    )
    channel_imbalance_correction_flag: str = field(
        metadata={
            "name": "channelImbalanceCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    channel_imbalance_list: ChannelImbalanceList = field(
        metadata={
            "name": "channelImbalanceList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class ProcessingGainListType:
    """
    Parameters
    ----------
    processing_gain
        Processing gain to be applied multiplicatively to the image during processing for the current polarisation.
    count
    """

    class Meta:
        name = "processingGainListType"

    processing_gain: list[FloatWithPolarisation] = field(
        default_factory=list,
        metadata={"name": "processingGain", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class QlAbsoluteScalingFactorListType:
    """
    Parameters
    ----------
    ql_absolute_scaling_factor
        Absolute scaling factor to be applied to quick-look ADS for the current RGB channel.
    count
    """

    class Meta:
        name = "qlAbsoluteScalingFactorListType"

    ql_absolute_scaling_factor: list[FloatWithChannel] = field(
        default_factory=list,
        metadata={
            "name": "qlAbsoluteScalingFactor",
            "type": "Element",
            "namespace": "",
            "min_occurs": 3,
            "max_occurs": 3,
        },
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class RfiMitigationType:
    """
    Parameters
    ----------
    rfi_detection_flag
        True if RFI detection has to be performed, False otherwise.
    rfi_mitigation_mode
        Enabled/Disabled to activate RFI mitigation. MaskBased decides if to activate the mitigation using the
        activation mask.
    rfi_activation_mask_threshold
        Threshold on the overlap between L0 Footprint and activation mask active region above which RFI mitigation
        is activated (if rfiMitigationMode is MaskBased)
    rfi_mitigation_method
        Domain where the RFI mitigation step has to be performed (Time, Frequency, Time and Frequency, Frequency and
        Time).
    rfi_mask
        RFI mask to be used for mitigation. Valid values are: "Single", to use the same mask for all the
        polarizations; "Multiple", to use a dedicated mask for each polarization.
    rfi_mask_generation_method
        Polarization-dependent RFI masks combination method (AND, OR). Used only in case rfiMask is set to Single.
    rfi_tmprocessing_parameters
        Time-domain RFI mitigation processing parameters.
    rfi_fmprocessing_parameters
        Frequency-domain RFI mitigation processing parameters.
    """

    class Meta:
        name = "rfiMitigationType"

    rfi_detection_flag: str = field(
        metadata={"name": "rfiDetectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    rfi_mitigation_mode: RfiMitigationTypeRfiMitigationMode = field(
        metadata={
            "name": "rfiMitigationMode",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_activation_mask_threshold: float = field(
        metadata={
            "name": "rfiActivationMaskThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    rfi_mitigation_method: RfiMitigationMethodType = field(
        metadata={
            "name": "rfiMitigationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_mask: RfiMaskType = field(
        metadata={
            "name": "rfiMask",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_mask_generation_method: RfiMaskGenerationMethodType = field(
        metadata={
            "name": "rfiMaskGenerationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_tmprocessing_parameters: RfiTmprocessingParametersType = field(
        metadata={
            "name": "rfiTMProcessingParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_fmprocessing_parameters: RfiFmprocessingParametersType = field(
        metadata={
            "name": "rfiFMProcessingParameters",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class L1ProductExportType:
    """
    Parameters
    ----------
    l1a_product_doi
        Digital Object Identifier (DOI) to be written in output L1a (SCS) products.
    l1b_product_doi
        Digital Object Identifier (DOI) to be written in output L1b (DGM) products.
    pixel_representation
        Representation of the image pixels within the image MDS (I Q, Abs Phase, Abs, Pow Phase, Pow).
    pixel_quantity
        Physical quantity stored in output data (Beta-Nought, Sigma-Nought or Gamma-Nought).
    abs_compression_method
        Abs values TIFF compression method.
    abs_max_zerror
        Maximum error threshold on abs values. Used only in case absCompressionMethod is set to LERC, LERC_DEFLATE
        or LERC_ZSTD. If set to 0, compression is lossless. If set to -1, value is computed dynamically w.r.t. a
        percentile value defined by absMaxZErrorPercentile.
    abs_max_zerror_percentile
        Percentile value to compute dynamically absMaxZError. Used only in case absMaxZError is set to -1.
    phase_compression_method
        Phase values TIFF compression method.
    phase_max_zerror
        Maximum error threshold on phase values [rad]. Used only in case phaseCompressionMethod is set to LERC,
        LERC_DEFLATE or LERC_ZSTD. If set to 0, compression is lossless. If set to -1, value is computed dynamically
        w.r.t. a percentile value defined by phaseMaxZErrorPercentile.
    phase_max_zerror_percentile
        Percentile value to compute dynamically phaseMaxZError. Used only in case phaseMaxZError is set to -1.
    no_pixel_value
        Pixel value in case of invalid data.
    block_size
        Size of block written during TIFF files export.
    lut_range_decimation_factor_list
        Look-Up Tables (LUT) ADS range decimation factors (one value per LUT group) w.r.t. output L1 product
        sampling grid.
    lut_azimuth_decimation_factor_list
        Look-Up Tables (LUT) ADS azimuth decimation factors (one value per LUT group) w.r.t. output L1 product
        sampling grid.
    lut_block_size
        Size of block written during NetCDF files export.
    lut_layers_completeness_flag
        True if all the layers of the Look-Up Tables (LUT) ADS have to be included in output L1M product too, False
        if RFI masks and ionosphere binary layers shall be removed. Default value is True.
    l1a_ql_colour_coding_method
        Colour coding method to be used for output L1a (SCS) products quick-look. Default value is Pauli.
    l1b_ql_colour_coding_method
        Colour coding method to be used for output L1b (DGM) products quick-look. Default value is Lexicographic.
    ql_range_decimation_factor
        Quick-look ADS range decimation factor w.r.t. output L1 product sampling grid.
    ql_range_averaging_factor
        Quick-look ADS range averaging factor.
    ql_azimuth_decimation_factor
        Quick-look ADS azimuth decimation factor w.r.t. output L1 product sampling grid.
    ql_azimuth_averaging_factor
        Quick-look ADS azimuth averaging factor.
    ql_absolute_scaling_factor_list
        Absolute scaling factors (one value per RGB channel) to be applied to quick-look ADS.
    """

    class Meta:
        name = "l1ProductExportType"

    l1a_product_doi: str = field(
        metadata={
            "name": "l1aProductDOI",
            "type": "Element",
            "namespace": "",
        },
    )
    l1b_product_doi: str = field(
        metadata={
            "name": "l1bProductDOI",
            "type": "Element",
            "namespace": "",
        },
    )
    pixel_representation: PixelRepresentationType = field(
        metadata={
            "name": "pixelRepresentation",
            "type": "Element",
            "namespace": "",
        },
    )
    pixel_quantity: PixelQuantityType = field(
        metadata={
            "name": "pixelQuantity",
            "type": "Element",
            "namespace": "",
        },
    )
    abs_compression_method: CompressionMethodType = field(
        metadata={
            "name": "absCompressionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    abs_max_zerror: float = field(
        metadata={"name": "absMaxZError", "type": "Element", "namespace": "", "min_inclusive": -1.0}
    )
    abs_max_zerror_percentile: float = field(
        metadata={
            "name": "absMaxZErrorPercentile",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    phase_compression_method: CompressionMethodType = field(
        metadata={
            "name": "phaseCompressionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    phase_max_zerror: FloatWithUnit = field(
        metadata={
            "name": "phaseMaxZError",
            "type": "Element",
            "namespace": "",
        },
    )
    phase_max_zerror_percentile: float = field(
        metadata={
            "name": "phaseMaxZErrorPercentile",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    no_pixel_value: float = field(
        metadata={
            "name": "noPixelValue",
            "type": "Element",
            "namespace": "",
        },
    )
    block_size: int = field(metadata={"name": "blockSize", "type": "Element", "namespace": "", "min_inclusive": 1})
    lut_range_decimation_factor_list: LutDecimationFactorListType = field(
        metadata={
            "name": "lutRangeDecimationFactorList",
            "type": "Element",
            "namespace": "",
        },
    )
    lut_azimuth_decimation_factor_list: LutDecimationFactorListType = field(
        metadata={
            "name": "lutAzimuthDecimationFactorList",
            "type": "Element",
            "namespace": "",
        },
    )
    lut_block_size: int = field(
        metadata={"name": "lutBlockSize", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    lut_layers_completeness_flag: str = field(
        metadata={"name": "lutLayersCompletenessFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    l1a_ql_colour_coding_method: L1AcQlColourCodingMethodType = field(
        metadata={
            "name": "l1aQlColourCodingMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    l1b_ql_colour_coding_method: L1BQlColourCodingMethodType = field(
        metadata={
            "name": "l1bQlColourCodingMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    ql_range_decimation_factor: int = field(
        metadata={"name": "qlRangeDecimationFactor", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    ql_range_averaging_factor: int = field(
        metadata={"name": "qlRangeAveragingFactor", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    ql_azimuth_decimation_factor: int = field(
        metadata={"name": "qlAzimuthDecimationFactor", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    ql_azimuth_averaging_factor: int = field(
        metadata={"name": "qlAzimuthAveragingFactor", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    ql_absolute_scaling_factor_list: QlAbsoluteScalingFactorListType = field(
        metadata={
            "name": "qlAbsoluteScalingFactorList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class L1AProcessingParametersListType:
    """
    Parameters
    ----------
    processing_parameters
        Parameters to be used during processing for a given swath.
    count
    """

    class Meta:
        name = "l1aProcessingParametersListType"

    processing_parameters: list[L1AProcessingParametersType] = field(
        default_factory=list,
        metadata={"name": "processingParameters", "type": "Element", "namespace": "", "min_occurs": 3, "max_occurs": 3},
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class L1BProcessingParametersListType:
    """
    Parameters
    ----------
    processing_parameters
        Parameters to be used during processing for a given swath.
    count
    """

    class Meta:
        name = "l1bProcessingParametersListType"

    processing_parameters: list[L1BProcessingParametersType] = field(
        default_factory=list,
        metadata={"name": "processingParameters", "type": "Element", "namespace": "", "min_occurs": 3, "max_occurs": 3},
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class RadiometricCalibrationType:
    """
    Parameters
    ----------
    range_spreading_loss_compensation_flag
        True if range spreading loss compensation has to be performed, False otherwise.
    reference_range
        Range spreading loss reference slant range [m]. The range spreading loss is compensated by amplitude scaling
        each range sample by 1/Grsl(R) where: Grsl(R) = cuberoot(rRef/R); and, R = slant range of sample.
    processing_gain_list
        Processing gain to be applied multiplicatively to the image during processing for all the polarisations.
        Note that these gains are applied on top of those automatically computed and applied by the processor.
    absolute_calibration_constant_list
        Absolute calibration constant to be applied to the image during processing for all the polarisations.
    """

    class Meta:
        name = "radiometricCalibrationType"

    range_spreading_loss_compensation_flag: str = field(
        metadata={
            "name": "rangeSpreadingLossCompensationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    reference_range: FloatWithUnit = field(
        metadata={
            "name": "referenceRange",
            "type": "Element",
            "namespace": "",
        },
    )
    processing_gain_list: ProcessingGainListType = field(
        metadata={
            "name": "processingGainList",
            "type": "Element",
            "namespace": "",
        },
    )
    absolute_calibration_constant_list: CalibrationConstantListType = field(
        metadata={
            "name": "absoluteCalibrationConstantList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class ThermalDenoisingType:
    """
    Parameters
    ----------
    thermal_denoising_flag
        True if thermal denoising has to be performed, False otherwise.
    noise_parameters_source
        Noise parameters to be used during processing (Model or Extracted). Used only in case thermalDenoisingFlag
        is set to True.
    noise_equivalent_echoes_flag
        True if noise-equivalent echoes have to be used to derive noise parameters, False otherwise.
    noise_gain_list
        Processing gain to be applied multiplicatively to the thermal noise level before using it during denoising
        step for all the polarisations. Note that these gains are applied on top of those automatically computed and
        applied by the processor.
    """

    class Meta:
        name = "thermalDenoisingType"

    thermal_denoising_flag: str = field(
        metadata={"name": "thermalDenoisingFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    noise_parameters_source: InternalCalibrationSourceType = field(
        metadata={
            "name": "noiseParametersSource",
            "type": "Element",
            "namespace": "",
        },
    )
    noise_equivalent_echoes_flag: str = field(
        metadata={"name": "noiseEquivalentEchoesFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    noise_gain_list: NoiseGainListType = field(
        metadata={
            "name": "noiseGainList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AzimuthCompressionType:
    """
    Parameters
    ----------
    block_samples
        Number of samples in the processing blocks.
    block_lines
        Number of lines in the processing blocks.
    block_overlap_samples
        Number of samples in the overlap between range processing blocks.
    block_overlap_lines
        Number of lines in the overlap between azimuth processing blocks.
    azimuth_processing_parameters_list
        Swath-dependent parameters to be used during azimuth processing. The list includes one entry per swath.
    bistatic_delay_correction_flag
        True if bistatic delay correction has to be applied, False otherwise.
    bistatic_delay_correction_method
        Method used for bistatic delay correction (Bulk or Full).
    azimuth_resampling_flag
        True if resampling along azimuth direction has to be applied, False otherwise. Note that PRF changes are
        *always* managed: if this flag is False and there is a PRF change, data are resampled to the higher PRF. If
        this flag is true, data are resampled to azimuthResamplingFrequency set by user.
    azimuth_resampling_frequency
        Azimuth resampling frequency [Hz]. Used only in case azimuthResamplingFlag is set to True.
    azimuth_focusing_margins_removal_flag
        True if azimuth focusing margins have to be removed, False otherwise. Default value is True.
    azimuth_coregistration_flag
        True if all the four data polarisations have to be coregistered over the same azimuth sampling grid, False
        otherwise. Default value is True.
    filter_type
        Type of filters used for azimuth resampling (SINC, FIR).
    filter_bandwidth
        Filter bandwidth [Hz].
    filter_length
        Filter length (in samples).
    number_of_filters
        Number of filters in the bank.
    """

    class Meta:
        name = "azimuthCompressionType"

    block_samples: int = field(
        metadata={
            "name": "blockSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    block_lines: int = field(
        metadata={
            "name": "blockLines",
            "type": "Element",
            "namespace": "",
        },
    )
    block_overlap_samples: int = field(
        metadata={
            "name": "blockOverlapSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    block_overlap_lines: int = field(
        metadata={
            "name": "blockOverlapLines",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_processing_parameters_list: L1AProcessingParametersListType = field(
        metadata={
            "name": "azimuthProcessingParametersList",
            "type": "Element",
            "namespace": "",
        },
    )
    bistatic_delay_correction_flag: str = field(
        metadata={
            "name": "bistaticDelayCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    bistatic_delay_correction_method: BistaticDelayCorrectionMethodType = field(
        metadata={
            "name": "bistaticDelayCorrectionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_resampling_flag: str = field(
        metadata={"name": "azimuthResamplingFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    azimuth_resampling_frequency: FloatWithUnit = field(
        metadata={
            "name": "azimuthResamplingFrequency",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_focusing_margins_removal_flag: str = field(
        metadata={
            "name": "azimuthFocusingMarginsRemovalFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    azimuth_coregistration_flag: str = field(
        metadata={"name": "azimuthCoregistrationFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    filter_type: AzimuthCompressionTypeFilterType = field(
        metadata={
            "name": "filterType",
            "type": "Element",
            "namespace": "",
        },
    )
    filter_bandwidth: FloatWithUnit = field(
        metadata={
            "name": "filterBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    filter_length: int = field(
        metadata={
            "name": "filterLength",
            "type": "Element",
            "namespace": "",
        },
    )
    number_of_filters: int = field(
        metadata={
            "name": "numberOfFilters",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class MultilookType:
    """
    Parameters
    ----------
    range_processing_parameters_list
        List of parameters to be used during range processing (one entry per swath).
    azimuth_processing_parameters_list
        List of parameters to be used during azimuth processing (one entry per swath).
    detection_flag
        True if detection has to be performed, False otherwise.
    """

    class Meta:
        name = "multilookType"

    range_processing_parameters_list: L1BProcessingParametersListType = field(
        metadata={
            "name": "rangeProcessingParametersList",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_processing_parameters_list: L1BProcessingParametersListType = field(
        metadata={
            "name": "azimuthProcessingParametersList",
            "type": "Element",
            "namespace": "",
        },
    )
    detection_flag: str = field(
        metadata={"name": "detectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )


@dataclass(kw_only=True)
class RangeCompressionType:
    """
    Parameters
    ----------
    range_reference_function_source
        Chirp source to be used for range compression (Nominal (i.e. ideal chirp), Replica (i.e. chirp derived from
        internal calibration) or Internal (i.e. chirp taken from AUX_INS file)).
    range_compression_method
        Range compression method to be used during processing (Matched Filter or Inverse Filter).
    extended_swath_processing_flag
        True if processing has to be extended in the range direction including samples not having the full phase
        history, False otherwise.
    range_processing_parameters_list
        Swath-dependent parameters to be used during range processing. The list includes one entry per swath.
    """

    class Meta:
        name = "rangeCompressionType"

    range_reference_function_source: RangeReferenceFunctionType = field(
        metadata={
            "name": "rangeReferenceFunctionSource",
            "type": "Element",
            "namespace": "",
        },
    )
    range_compression_method: RangeCompressionMethodType = field(
        metadata={
            "name": "rangeCompressionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    extended_swath_processing_flag: str = field(
        metadata={
            "name": "extendedSwathProcessingFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    range_processing_parameters_list: L1AProcessingParametersListType = field(
        metadata={
            "name": "rangeProcessingParametersList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class L1ProductType:
    """
    Parameters
    ----------
    product_id
        Product identifier (SM_SCS__1S, SM_DGM__1S or RO_SCS__1S).
    general
        General processing parameters.
    l0_product_import
        L0 product import processing parameters.
    raw_data_correction
        Raw data correction processing parameters.
    rfi_mitigation
        RFI mitigation processing parameters.
    internal_calibration_correction
        Internal calibration corrections processing parameters.
    range_compression
        Range compression processing parameters.
    doppler_estimation
        Doppler Centroid and Doppler Rate estimation processing parameters.
    antenna_pattern_correction
        Antenna pattern correction processing parameters.
    azimuth_compression
        Azimuth compression processing parameters.
    radiometric_calibration
        RAdiometric calibration processing parameters.
    polarimetric_calibration
        Polarimetric calibration processing parameters.
    ionosphere_calibration
        Ionosphere calibration processing parameters.
    autofocus
        Autofocus processing parameters.
    multilook
        Multilook processing parameters.
    thermal_denoising
        Thermal denoising processing parameters.
    ground_projection
        Ground projection processing parameters.
    l1_product_export
        L1 product export processing parameters.
    """

    class Meta:
        name = "l1ProductType"

    product_id: str = field(
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
        },
    )
    general: GeneralType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    l0_product_import: L0ProductImportType = field(
        metadata={
            "name": "l0ProductImport",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_data_correction: RawDataCorrectionType = field(
        metadata={
            "name": "rawDataCorrection",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_mitigation: RfiMitigationType = field(
        metadata={
            "name": "rfiMitigation",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_calibration_correction: InternalCalibrationCorrectionType = field(
        metadata={
            "name": "internalCalibrationCorrection",
            "type": "Element",
            "namespace": "",
        },
    )
    range_compression: RangeCompressionType = field(
        metadata={
            "name": "rangeCompression",
            "type": "Element",
            "namespace": "",
        },
    )
    doppler_estimation: DopplerEstimationType = field(
        metadata={
            "name": "dopplerEstimation",
            "type": "Element",
            "namespace": "",
        },
    )
    antenna_pattern_correction: AntennaPatternCorrectionType = field(
        metadata={
            "name": "antennaPatternCorrection",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_compression: AzimuthCompressionType = field(
        metadata={
            "name": "azimuthCompression",
            "type": "Element",
            "namespace": "",
        },
    )
    radiometric_calibration: RadiometricCalibrationType = field(
        metadata={
            "name": "radiometricCalibration",
            "type": "Element",
            "namespace": "",
        },
    )
    polarimetric_calibration: PolarimetricCalibrationType = field(
        metadata={
            "name": "polarimetricCalibration",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_calibration: IonosphereCalibrationType = field(
        metadata={
            "name": "ionosphereCalibration",
            "type": "Element",
            "namespace": "",
        },
    )
    autofocus: AutofocusType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    multilook: MultilookType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    thermal_denoising: ThermalDenoisingType = field(
        metadata={
            "name": "thermalDenoising",
            "type": "Element",
            "namespace": "",
        },
    )
    ground_projection: GroundProjectionType = field(
        metadata={
            "name": "groundProjection",
            "type": "Element",
            "namespace": "",
        },
    )
    l1_product_export: L1ProductExportType = field(
        metadata={
            "name": "l1ProductExport",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class L1ProductListType:
    """
    Parameters
    ----------
    l1_product
        L1 processing parameters for a given product ID.
    count
    """

    class Meta:
        name = "l1ProductListType"

    l1_product: L1ProductType = field(
        metadata={
            "name": "l1Product",
            "type": "Element",
            "namespace": "",
        },
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryL1ProcessingParametersType:
    """
    Parameters
    ----------
    l1_product_list
        List of L1 processing parameters for each product the L1 Processor is capable of generating (i.e.,
        SM_SCS__1S, SM_DGM__1S, RO_SCS__1S, ...).
    """

    class Meta:
        name = "auxiliaryL1ProcessingParametersType"

    l1_product_list: L1ProductListType = field(
        metadata={
            "name": "l1ProductList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryL1ProcessingParameters(AuxiliaryL1ProcessingParametersType):
    """
    BIOMASS auxiliary L1 processing parameters element.
    """

    class Meta:
        name = "auxiliaryL1ProcessingParameters"
