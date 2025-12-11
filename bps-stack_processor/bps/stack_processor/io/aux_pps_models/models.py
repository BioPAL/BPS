# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD PPS models
--------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from bps.common.io.common_types import (
    AzimuthPolynomialType,
    BaselineMethodType,
    ChannelImbalanceList,
    ChannelType,
    Complex,
    ComplexArray,
    CompressionMethodType,
    CoregistrationExecutionPolicyType,
    CoregistrationMethodType,
    CrossTalkList,
    DatumType,
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
    LayerListType,
    LayerType,
    MinMaxType,
    MinMaxTypeWithUnit,
    PixelQuantityType,
    PixelRepresentationType,
    PolarisationCombinationMethodType,
    PolarisationType,
    PrimaryImageSelectionInformationType,
    SkpPhaseCorrectionType,
    SlantRangePolynomialType,
    StateType,
    TimeTypeWithPolarisation,
    UnsignedIntWithGroup,
    UomType,
    WeightingWindowType,
)


class CoregistrationTypeLowPassFilterType(Enum):
    AVERAGE = "Average"
    GAUSSIAN = "Gaussian"


@dataclass
class RfiDegradationEstimationType:
    """
    Parameters
    ----------
    rfi_degradation_estimation_flag
        True if estimation of degradation due to RFI filtering has to be performed, False otherwise.
    """

    class Meta:
        name = "rfiDegradationEstimationType"

    rfi_degradation_estimation_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "rfiDegradationEstimationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
class AzimuthSpectralFilteringType:
    """
    Parameters
    ----------
    azimuth_spectral_filtering_flag
        True if azimuth spectral filtering has to be performed, False otherwise.
    use_primary_weighting_window_flag
        True if azimuth spectral filtering should use the azimuth spectral window of the primary as common azimuth
        window (recommended configuration). Otherwise, use the selected window.
    spectral_weighting_window
        The user provided common spectral window (default: Hamming).
    spectral_weighting_window_parameter
        The parameter of the spectral weighting window (to be interpreted depending on the window).
    use32bit_flag
        True if 32-bit precision (aka complex64 and float32) has to be used for model estimations instead of 64-bit
        precision (aka complex128 and float64), False otherwise.
    """

    class Meta:
        name = "azimuthSpectralFilteringType"

    azimuth_spectral_filtering_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "azimuthSpectralFilteringFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    use_primary_weighting_window_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "usePrimaryWeightingWindowFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    spectral_weighting_window: WeightingWindowType = field(
        default=WeightingWindowType.HAMMING,
        metadata={
            "name": "spectralWeightingWindow",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    spectral_weighting_window_parameter: Optional[float] = field(
        default=None,
        metadata={
            "name": "spectralWeightingWindowParameter",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_exclusive": 0.0,
        },
    )
    use32bit_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "use32bitFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
class CoregistrationType:
    """
    Parameters
    ----------
    coregistration_method
        Coregistration method (Geometry, Geometry and Data, Automatic).
    range_spectral_filtering_flag
        True if range spectral filtering has to be performed (during coregistration), False otherwise.
    residual_shifts_quality_threshold
        Threshold on residual coregistration shifts quality (between 0 and 1). Residual shifts with a quality lower
        than this threshold are not used for processing. Used only in case coregistrationMethod is set to Geometry
        and Data.
    polarisation_used
        Polarisation used as primary for coregistration.
    block_quality_threshold
        Threshold on the cross-correlation, as a measure of coregistration quality (between 0 and 1).
    fitting_quality_threshold
        Threshold on quality for the estimated shifts to be accepted as valid (between 0 and 1).
    min_valid_blocks
        Minimum percentage of valid blocks to accept a coregistration result (negative to disable, maximum 100).
    azimuth_max_shift
        Maximum pixel shift that can be estimated in azimuth direction (in pixel). Recommended value: 5.
    azimuth_block_size
        Size of the estimation blocks used when computing cross-correlation in azimuth direction (in pixel).
        Recommended value: 101.
    azimuth_min_overlap
        Minimum azimuth overlap required for shifts estimated using cross-correlation (in pixel). Recommended value:
        0.
    range_max_shift
        Maximum pixel shift that can be estimated in range direction (in pixel). Recommended value: 5.
    range_block_size
        Size of the estimation blocks used for speckle tracking in range direction (in pixel). Recommended value:
        51.
    range_min_overlap
        Minimum range overlap required for shifts estimated using cross-correlation (in pixel). Recommended value:
        0.
    model_based_fit_flag
        True if model-based fit should be performed after estimating shifts via cross-correlation. Defaults to
        false.
    low_pass_filter_type
        Type of low-pass filter applied to shifts estimated via cross-correlation. This parameter is ignored/unused
        when 'modelBasedFitFlag' is set to 'true'. Supported values are "Average" and "Gaussian". Recommended value
        is "Average".
    low_pass_filter_order
        Low-pass filter order. Used only if "Average" is selected. Recommended value is 1.
    low_pass_filter_std_dev
        Standard deviation of the Gaussian low-pass filter. Recommended value is 0.84089642.
    export_debug_products_flag
        True if all products for debugging needs to be exported. Defaults to False.
    coregistration_execution_policy
        As to whether the coregistration should be run nominally (i.e. shift estimation and image warping), shift
        estimation only, or image warping only (which requires user provided coregistration shifts). Accepted
        policies are Nominal, Shift Estimation Only, and Warping Only. Defaults to Nominal.
    """

    class Meta:
        name = "coregistrationType"

    coregistration_method: Optional[CoregistrationMethodType] = field(
        default=None,
        metadata={
            "name": "coregistrationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_spectral_filtering_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "rangeSpectralFilteringFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    residual_shifts_quality_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "residualShiftsQualityThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    polarisation_used: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "name": "polarisationUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    block_quality_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "blockQualityThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    fitting_quality_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "fittingQualityThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    min_valid_blocks: Optional[int] = field(
        default=None,
        metadata={
            "name": "minValidBlocks",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": -1,
            "max_inclusive": 100,
        },
    )
    azimuth_max_shift: Optional[int] = field(
        default=None,
        metadata={"name": "azimuthMaxShift", "type": "Element", "namespace": "", "required": True, "min_inclusive": 1},
    )
    azimuth_block_size: Optional[int] = field(
        default=None,
        metadata={"name": "azimuthBlockSize", "type": "Element", "namespace": "", "required": True, "min_inclusive": 1},
    )
    azimuth_min_overlap: Optional[int] = field(
        default=None,
        metadata={
            "name": "azimuthMinOverlap",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0,
        },
    )
    range_max_shift: Optional[int] = field(
        default=None,
        metadata={"name": "rangeMaxShift", "type": "Element", "namespace": "", "required": True, "min_inclusive": 1},
    )
    range_block_size: Optional[int] = field(
        default=None,
        metadata={"name": "rangeBlockSize", "type": "Element", "namespace": "", "required": True, "min_inclusive": 1},
    )
    range_min_overlap: Optional[int] = field(
        default=None,
        metadata={"name": "rangeMinOverlap", "type": "Element", "namespace": "", "required": True, "min_inclusive": 0},
    )
    model_based_fit_flag: str = field(
        default="false",
        metadata={
            "name": "modelBasedFitFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    low_pass_filter_type: Optional[CoregistrationTypeLowPassFilterType] = field(
        default=None,
        metadata={
            "name": "lowPassFilterType",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    low_pass_filter_order: Optional[int] = field(
        default=None,
        metadata={
            "name": "lowPassFilterOrder",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 1,
        },
    )
    low_pass_filter_std_dev: Optional[float] = field(
        default=None,
        metadata={
            "name": "lowPassFilterStdDev",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_exclusive": 0.0,
        },
    )
    export_debug_products_flag: str = field(
        default="false",
        metadata={
            "name": "exportDebugProductsFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    coregistration_execution_policy: Optional[CoregistrationExecutionPolicyType] = field(
        default=None,
        metadata={
            "name": "coregistrationExecutionPolicy",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class GeneralType:
    """
    Parameters
    ----------
    polarisation_combination_method
        Polarisations combination method: “HV” or “VH”, if just one of the two is selected (in addition to HH and VV
        ones); “Average”, if the average of HV and VH is computed and used; “None” if no combination is performed
        (all the four polarisations are used).
    outer_parallelization_max_cores
        Maximum number of parallel instances (outer parallelization). This number shall be lower or equal to the
        total number of cores specified in the Job Order. It allows to control memory consumption. Default value is
        0, i.e., the number of cores set in the Job Order is used.
    allow_duplicate_images_flag
        Allow duplicated products in job-order. When set to false, the processor exits with an error if the input
        stack contains duplicated images.
    flattening_phase_bias_compensation_flag
        If true, biases due to DEM upsampling residuals are compensated. Defaults to true.
    """

    class Meta:
        name = "generalType"

    polarisation_combination_method: Optional[PolarisationCombinationMethodType] = field(
        default=None,
        metadata={
            "name": "polarisationCombinationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    outer_parallelization_max_cores: int = field(
        default=0,
        metadata={
            "name": "outerParallelizationMaxCores",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    allow_duplicate_images_flag: str = field(
        default="false",
        metadata={
            "name": "allowDuplicateImagesFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    flattening_phase_bias_compensation_flag: str = field(
        default="true",
        metadata={
            "name": "flatteningPhaseBiasCompensationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
class InSarcalibrationType:
    """
    Parameters
    ----------
    in_sarcalibration_flag
        True if the InSAR calibration step is active. False otherwise
    polarisation_used
        Polarisation used in the estimation of the InSAR calibration's phase removal step. Recommended value is HH.
    fft2_zero_padding_upsampling_factor
        Multiplicative factor that controls zero-padding before FFT2 to increasing frequency-domain sampling
        density. It must be at least 1.0. Recommended value is 1.2.
    fft2_peak_window_size
        Size of the local window used for parabolic interpolation of the FFT2 around the max/peak. It must be at
        least 3 (and odd). Recommended value is 3.
    use32bit_flag
        True if 32-bit precision (aka complex64 and float32) has to be used for model estimations instead of 64-bit
        precision (aka complex128 and float64), False otherwise.
    """

    class Meta:
        name = "inSARCalibrationType"

    in_sarcalibration_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "inSARCalibrationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    polarisation_used: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "name": "polarisationUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    fft2_zero_padding_upsampling_factor: Optional[float] = field(
        default=None,
        metadata={
            "name": "fft2ZeroPaddingUpsamplingFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 1.0,
        },
    )
    fft2_peak_window_size: Optional[int] = field(
        default=None,
        metadata={
            "name": "fft2PeakWindowSize",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 3,
        },
    )
    use32bit_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "use32bitFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
class PrimaryImageSelectionType:
    """
    Parameters
    ----------
    primary_image_selection_information
        Information to be used to select coregistration primary image (Geometry, Geometry and RFI Correction,
        Geometry and FR Correction, Geometry and RFI+FR Corrections).
    rfi_decorrelation_threshold
        Maximum decorrelation factor due to RFI strength admissible to use the image as the primary image.
    faraday_decorrelation_threshold
        Maximum decorrelation factor due to Faraday residual admissible to use the image as the primary image.
    """

    class Meta:
        name = "primaryImageSelectionType"

    primary_image_selection_information: Optional[PrimaryImageSelectionInformationType] = field(
        default=None,
        metadata={
            "name": "primaryImageSelectionInformation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_decorrelation_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "rfiDecorrelationThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    faraday_decorrelation_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "faradayDecorrelationThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )


@dataclass
class L1CProductExportType:
    """
    Parameters
    ----------
    l1_product_doi
        Digital Object Identifier (DOI) to be written in output products.
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
    ql_range_decimation_factor
        Quick-look ADS range decimation factor w.r.t. output L1 product sampling grid.
    ql_range_averaging_factor
        Quick-look ADS range averaging factor.
    ql_azimuth_decimation_factor
        Quick-look ADS azimuth decimation factor w.r.t. output L1 product sampling grid.
    ql_azimuth_averaging_factor
        Quick-look ADS azimuth averaging factor.
    ql_absolute_scaling_factor
        Absolute scaling factor to be applied to quick-look ADS.
    """

    class Meta:
        name = "l1cProductExportType"

    l1_product_doi: Optional[str] = field(
        default=None,
        metadata={
            "name": "l1ProductDOI",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pixel_representation: Optional[PixelRepresentationType] = field(
        default=None,
        metadata={
            "name": "pixelRepresentation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    pixel_quantity: Optional[PixelQuantityType] = field(
        default=None,
        metadata={
            "name": "pixelQuantity",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    abs_compression_method: Optional[CompressionMethodType] = field(
        default=None,
        metadata={
            "name": "absCompressionMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    abs_max_zerror: Optional[float] = field(
        default=None,
        metadata={
            "name": "absMaxZError",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    abs_max_zerror_percentile: Optional[float] = field(
        default=None,
        metadata={
            "name": "absMaxZErrorPercentile",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    phase_compression_method: Optional[CompressionMethodType] = field(
        default=None,
        metadata={
            "name": "phaseCompressionMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    phase_max_zerror: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "phaseMaxZError",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    phase_max_zerror_percentile: Optional[float] = field(
        default=None,
        metadata={
            "name": "phaseMaxZErrorPercentile",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    no_pixel_value: Optional[float] = field(
        default=None,
        metadata={
            "name": "noPixelValue",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ql_range_decimation_factor: Optional[int] = field(
        default=None,
        metadata={
            "name": "qlRangeDecimationFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 1,
        },
    )
    ql_range_averaging_factor: Optional[int] = field(
        default=None,
        metadata={
            "name": "qlRangeAveragingFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 1,
        },
    )
    ql_azimuth_decimation_factor: Optional[int] = field(
        default=None,
        metadata={
            "name": "qlAzimuthDecimationFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 1,
        },
    )
    ql_azimuth_averaging_factor: Optional[int] = field(
        default=None,
        metadata={
            "name": "qlAzimuthAveragingFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 1,
        },
    )
    ql_absolute_scaling_factor: Optional[float] = field(
        default=None,
        metadata={
            "name": "qlAbsoluteScalingFactor",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_exclusive": 0.0,
        },
    )


@dataclass
class SkpPhaseCalibrationType:
    """
    Parameters
    ----------
    skp_phase_estimation_flag
        True if ground steering phases estimation has to be performed, False otherwise.
    phase_correction
        Type of phase correction executed by the SKP. None: Disable the SKP Phase correction. flatteningPhaseScreen:
        Correct DSI only. groundPhaseScreen: Correct DSI and forest disturbance.
    estimation_window_size
        Estimation window size [m].
    skp_calibration_phase_screen_quality_threshold
        Threshold on SKP phase estimation quality (between 0 and 1). Estimates with a quality lower than this
        threshold are not corrected. Used only in case skpPhaseCorrectionFlag is set to True.
    overall_product_quality_threshold
        Threshold on the percentage of ground phase estimations to consider the SKP valid (between 0 and 1).
    median_filter_flag
        True if post-processing of the SKP calibration phase screen via median filter has to be performed, False
        otherwise.
    median_filter_window_size
        Median filter window size [m].
    exclude_mpmbpolarization_cross_covariance_flag
        True if the MPMB coherence matrix should only contain autocovariances (i.e. HH vs HH and not HH vs. VH
        etc.), False otherwise.
    use32bit_flag
        True if 32-bit precision (aka complex64 and float32) has to be used for model estimations instead of 64-bit
        precision (aka complex128 and float64), False otherwise.
    """

    class Meta:
        name = "skpPhaseCalibrationType"

    skp_phase_estimation_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "skpPhaseEstimationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    phase_correction: Optional[SkpPhaseCorrectionType] = field(
        default=None,
        metadata={
            "name": "phaseCorrection",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    estimation_window_size: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "estimationWindowSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    skp_calibration_phase_screen_quality_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "skpCalibrationPhaseScreenQualityThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    overall_product_quality_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "overallProductQualityThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    median_filter_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "medianFilterFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    median_filter_window_size: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "medianFilterWindowSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    exclude_mpmbpolarization_cross_covariance_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "excludeMPMBPolarizationCrossCovarianceFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    use32bit_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "use32bitFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
class SlowIonosphereRemovalType:
    """
    Parameters
    ----------
    slow_ionosphere_removal_flag
        True if slow-varying ionospheric phase screen estimation and removal has to be performed, False otherwise.
    primary_image_flag
        True if image to be used as primary for multi-baseline calibration steps is the same as the one to be used
        for coregistration, False otherwise.
    polarisation_used
        Polarisation used in the estimation of the slow-varying ionosphere removal step. Recommended value is HH.
    compensate_l1_iono_phase_screen_flag
        True if the ionospheric phase screen from L1 should be compensated during split-spectrum. Defaults to True.
    range_look_bandwidth
        Range looks bandwidth percentage (from 0 to 1).
    range_look_frequency
        Range looks center frequency (from 0 to 0.5).
    phase_unwrapping_flag
        True if the single-baseline phase slope estimation should perform a prelimimnary phase unwrapping, False
        otherwise. Defaulted to True.
    latitude_threshold
        Latitude threshold above/below (North/South) which the IOB will be skipped [deg] (even if the IOB is enabled
        by flag)
    baseline_method
        The baseline method, i.e. Single-Baseline or Multi-Baseline. Defaulted to Multi-Baseline.
    unweighted_multi_baseline_estimation
        True if the Multi-Baseline estimation should use uniform weighting instead of weights from Single-Baseline
        estimation. Defaulted to false.
    slow_ionosphere_quality_threshold
        Ionosphere estimations below this threshold will be ignored. Values between 0 and 1.
    sublook_window_azimuth_size
        Window size in along-track direction to generate the sub-looks interferograms [px]. Recommended value is
        501, i.e., approximately 200m.
    sublook_window_range_size
        Window size in slant-range direction to generate the sub-looks interferograms [px]. Recommended value is 41,
        i.e., approximately 200m.
    multi_baseline_critical_baseline_threshold
        Maximum allowed percentage of CB displacement for multi-baseline pairs. Suggested value 0.45 (i.e. 45% CB).
    min_coherence_threshold
        Threshold on the coherence to consider a pixel usable for fitting the phase plane. Recommended value is 0.0.
    min_usable_pixel_ratio
        Minimum ratio of pixels that are usable for fitting the phase plane according to the minimum coherence and
        phase unwrap test. Recommended value is 0.05 (5%).
    max_delta_phase_unwrap_test
        Pixelwise threshold applied to the phase difference between high and low interferogram downstream of phase
        unwrapping [rad]. If the phase difference exceeds this threshold value, the pixel is discarded during phase
        screen estiamation. See stack ATBD. Recommended value is 0.5 radians.
    use32bit_flag
        True if 32-bit precision (aka complex64 and float32) has to be used for model estimations instead of 64-bit
        precision (aka complex128 and float64), False otherwise.
    """

    class Meta:
        name = "slowIonosphereRemovalType"

    slow_ionosphere_removal_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "slowIonosphereRemovalFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    primary_image_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "primaryImageFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    polarisation_used: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "name": "polarisationUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    compensate_l1_iono_phase_screen_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "compensateL1IonoPhaseScreenFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    range_look_bandwidth: Optional[float] = field(
        default=None,
        metadata={
            "name": "rangeLookBandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    range_look_frequency: Optional[float] = field(
        default=None,
        metadata={
            "name": "rangeLookFrequency",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 0.5,
        },
    )
    phase_unwrapping_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "phaseUnwrappingFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    latitude_threshold: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "latitudeThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    baseline_method: BaselineMethodType = field(
        default=BaselineMethodType.MULTI_BASELINE,
        metadata={
            "name": "baselineMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    unweighted_multi_baseline_estimation: str = field(
        default="false",
        metadata={
            "name": "unweightedMultiBaselineEstimation",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    slow_ionosphere_quality_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "slowIonosphereQualityThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    sublook_window_azimuth_size: Optional[int] = field(
        default=None,
        metadata={
            "name": "sublookWindowAzimuthSize",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 1,
        },
    )
    sublook_window_range_size: Optional[int] = field(
        default=None,
        metadata={
            "name": "sublookWindowRangeSize",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 1,
        },
    )
    multi_baseline_critical_baseline_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "multiBaselineCriticalBaselineThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
        },
    )
    min_coherence_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "minCoherenceThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    min_usable_pixel_ratio: Optional[float] = field(
        default=None,
        metadata={
            "name": "minUsablePixelRatio",
            "type": "Element",
            "namespace": "",
            "required": True,
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        },
    )
    max_delta_phase_unwrap_test: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "maxDeltaPhaseUnwrapTest",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    use32bit_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "use32bitFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
class StaProductType:
    """
    Parameters
    ----------
    product_id
        Product identifier.
    general
        General processing parameters.
    primary_image_selection
        Primary image selection processing parameters.
    coregistration
        Coregistration processing parameters.
    rfi_degradation_estimation
        RFI degradation estimation processing parameters.
    azimuth_spectral_filtering
        Processing parameters of the Azimuth Spectral Filtering (AZF).
    slow_ionosphere_removal
        Processing parameters of the Background Ionosphere Removal (IOB).
    in_sarcalibration
        Processing parameters of the InSAR calibration step.
    skp_phase_calibration
        Processing parameters of the Sum-of-Kronecker-Products (SKP).
    l1c_product_export
        L1c product export processing parameters.
    """

    class Meta:
        name = "staProductType"

    product_id: Optional[str] = field(
        default=None,
        metadata={
            "name": "productID",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    general: Optional[GeneralType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    primary_image_selection: Optional[PrimaryImageSelectionType] = field(
        default=None,
        metadata={
            "name": "primaryImageSelection",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    coregistration: Optional[CoregistrationType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_degradation_estimation: Optional[RfiDegradationEstimationType] = field(
        default=None,
        metadata={
            "name": "rfiDegradationEstimation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_spectral_filtering: Optional[AzimuthSpectralFilteringType] = field(
        default=None,
        metadata={
            "name": "azimuthSpectralFiltering",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    slow_ionosphere_removal: Optional[SlowIonosphereRemovalType] = field(
        default=None,
        metadata={
            "name": "slowIonosphereRemoval",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    in_sarcalibration: Optional[InSarcalibrationType] = field(
        default=None,
        metadata={
            "name": "inSARCalibration",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    skp_phase_calibration: Optional[SkpPhaseCalibrationType] = field(
        default=None,
        metadata={
            "name": "skpPhaseCalibration",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    l1c_product_export: Optional[L1CProductExportType] = field(
        default=None,
        metadata={
            "name": "l1cProductExport",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class StaProductListType:
    """
    Parameters
    ----------
    sta_product
        Stack processing parameters for a given product ID.
    count
    """

    class Meta:
        name = "staProductListType"

    sta_product: Optional[StaProductType] = field(
        default=None,
        metadata={
            "name": "staProduct",
            "type": "Element",
            "namespace": "",
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
class AuxiliaryStaprocessingParametersType:
    """
    Parameters
    ----------
    sta_product_list
        List of stack processing parameters for each product the Stack Processor is capable of generating (i.e.,
        SM_STA__1S, ...).
    """

    class Meta:
        name = "auxiliarySTAProcessingParametersType"

    sta_product_list: Optional[StaProductListType] = field(
        default=None,
        metadata={
            "name": "staProductList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class AuxiliaryStaprocessingParameters(AuxiliaryStaprocessingParametersType):
    """
    BIOMASS auxiliary stack processing parameters element.
    """

    class Meta:
        name = "auxiliarySTAProcessingParameters"
