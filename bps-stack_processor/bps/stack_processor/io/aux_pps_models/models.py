# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD PPS models
--------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
    L1AcQlColourCodingMethodType,
    LayerListType,
    LayerType,
    MinMaxType,
    MinMaxTypeWithUnit,
    PixelQuantityType,
    PixelRepresentationType,
    PolarisationCombinationMethodType,
    PolarisationType,
    PrimaryImageSelectionInformationType,
    SkpPhaseCalibrationPostprocessingType,
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


class MultiSquintCalibrationTypePropagationStepInversionMethod(Enum):
    PROJECTION = "Projection"
    DERIVATIVE = "Derivative"


@dataclass(kw_only=True)
class RfiDegradationEstimationType:
    """
    Parameters
    ----------
    rfi_degradation_estimation_flag
        True if estimation of degradation due to RFI filtering has to be performed, False otherwise.
    """

    class Meta:
        name = "rfiDegradationEstimationType"

    rfi_degradation_estimation_flag: str = field(
        metadata={
            "name": "rfiDegradationEstimationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )


@dataclass(kw_only=True)
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

    azimuth_spectral_filtering_flag: str = field(
        metadata={
            "name": "azimuthSpectralFilteringFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    use_primary_weighting_window_flag: str = field(
        metadata={
            "name": "usePrimaryWeightingWindowFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    spectral_weighting_window: WeightingWindowType = field(
        default=WeightingWindowType.HAMMING,
        metadata={
            "name": "spectralWeightingWindow",
            "type": "Element",
            "namespace": "",
        },
    )
    spectral_weighting_window_parameter: float = field(
        metadata={"name": "spectralWeightingWindowParameter", "type": "Element", "namespace": "", "min_exclusive": 0.0}
    )
    use32bit_flag: str = field(
        metadata={"name": "use32bitFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )


@dataclass(kw_only=True)
class CoregistrationType:
    """
    Parameters
    ----------
    coregistration_method
        Coregistration method (Geometry, Geometry and Data, Automatic).
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

    coregistration_method: CoregistrationMethodType = field(
        metadata={
            "name": "coregistrationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    residual_shifts_quality_threshold: float = field(
        metadata={
            "name": "residualShiftsQualityThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    polarisation_used: PolarisationType = field(
        metadata={
            "name": "polarisationUsed",
            "type": "Element",
            "namespace": "",
        },
    )
    block_quality_threshold: float = field(
        metadata={
            "name": "blockQualityThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    fitting_quality_threshold: float = field(
        metadata={
            "name": "fittingQualityThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    min_valid_blocks: int = field(
        metadata={
            "name": "minValidBlocks",
            "type": "Element",
            "namespace": "",
            "min_inclusive": -1,
            "max_inclusive": 100,
        }
    )
    azimuth_max_shift: int = field(
        metadata={"name": "azimuthMaxShift", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    azimuth_block_size: int = field(
        metadata={"name": "azimuthBlockSize", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    azimuth_min_overlap: int = field(
        metadata={"name": "azimuthMinOverlap", "type": "Element", "namespace": "", "min_inclusive": 0}
    )
    range_max_shift: int = field(
        metadata={"name": "rangeMaxShift", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    range_block_size: int = field(
        metadata={"name": "rangeBlockSize", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    range_min_overlap: int = field(
        metadata={"name": "rangeMinOverlap", "type": "Element", "namespace": "", "min_inclusive": 0}
    )
    model_based_fit_flag: str = field(
        default="false",
        metadata={"name": "modelBasedFitFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"},
    )
    low_pass_filter_type: CoregistrationTypeLowPassFilterType = field(
        metadata={
            "name": "lowPassFilterType",
            "type": "Element",
            "namespace": "",
        },
    )
    low_pass_filter_order: int = field(
        metadata={"name": "lowPassFilterOrder", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    low_pass_filter_std_dev: float = field(
        metadata={"name": "lowPassFilterStdDev", "type": "Element", "namespace": "", "min_exclusive": 0.0}
    )
    export_debug_products_flag: str = field(
        default="false",
        metadata={"name": "exportDebugProductsFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"},
    )
    coregistration_execution_policy: CoregistrationExecutionPolicyType = field(
        metadata={
            "name": "coregistrationExecutionPolicy",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    polarisation_combination_method: PolarisationCombinationMethodType = field(
        metadata={
            "name": "polarisationCombinationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    outer_parallelization_max_cores: int = field(
        default=0,
        metadata={
            "name": "outerParallelizationMaxCores",
            "type": "Element",
            "namespace": "",
        },
    )
    allow_duplicate_images_flag: str = field(
        default="false",
        metadata={"name": "allowDuplicateImagesFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"},
    )
    flattening_phase_bias_compensation_flag: str = field(
        default="true",
        metadata={
            "name": "flatteningPhaseBiasCompensationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        },
    )


@dataclass(kw_only=True)
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

    primary_image_selection_information: PrimaryImageSelectionInformationType = field(
        metadata={
            "name": "primaryImageSelectionInformation",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_decorrelation_threshold: float = field(
        metadata={
            "name": "rfiDecorrelationThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    faraday_decorrelation_threshold: float = field(
        metadata={
            "name": "faradayDecorrelationThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )


@dataclass(kw_only=True)
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
    l1c_ql_colour_coding_method
        Colour coding method to be used for output products quick-look. Default value is Pauli.
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
    ql_export_coherence_flag
        If true, export the coherence quick-look ADS for secondary frames.
    ql_export_interferogram_flag
        If true, export the interferogram quick-look ADS for secondary frames.
    """

    class Meta:
        name = "l1cProductExportType"

    l1_product_doi: str = field(
        metadata={
            "name": "l1ProductDOI",
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
        metadata={
            "name": "absMaxZError",
            "type": "Element",
            "namespace": "",
        },
    )
    abs_max_zerror_percentile: float = field(
        metadata={
            "name": "absMaxZErrorPercentile",
            "type": "Element",
            "namespace": "",
        },
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
        },
    )
    no_pixel_value: float = field(
        metadata={
            "name": "noPixelValue",
            "type": "Element",
            "namespace": "",
        },
    )
    l1c_ql_colour_coding_method: L1AcQlColourCodingMethodType = field(
        metadata={
            "name": "l1cQlColourCodingMethod",
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
    ql_absolute_scaling_factor: float = field(
        metadata={"name": "qlAbsoluteScalingFactor", "type": "Element", "namespace": "", "min_exclusive": 0.0}
    )
    ql_export_coherence_flag: str = field(
        default="true",
        metadata={"name": "qlExportCoherenceFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"},
    )
    ql_export_interferogram_flag: str = field(
        default="true",
        metadata={
            "name": "qlExportInterferogramFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        },
    )


@dataclass(kw_only=True)
class MultiSquintCalibrationType:
    """
    Parameters
    ----------
    multi_squint_calibration_flag
        True if the Multi-Squint calibration step is active. False otherwise
    polarisation_used
        Polarisation used in the estimation of the InSAR calibration's phase removal step. Recommended value is HH.
    seed_range_coherence_threshold
        Coherence threshold used for range seeding during the ionosphere model inversion. Recommended value is 0.3.
    edge_guard
        Ignore border pixels (in along-track direction). Recommended value is 11.
    valid_azimuth_threshold
        Ratio of valid lines/azimuth/rows used to fallback to central range duing range seeding. Recommended value
        is 0.1.
    propagation_step_inversion_method
        Inversion method used during the ionosphere range propagation step. Supported values are "Propagation" and
        "Derivative". Recommended value is "Projection".
    min_iono_range_candidate
        Minimum ionospheric range candidate used for estimating the ionosphere layers [m]. Recommended value is
        200,000 meters.
    max_iono_range_candidate
        Maximum ionospheric range candidate used for estimating the ionosphere layers [m]. Recommended value is
        700,000 meters.
    num_iono_range_candidates
        Number of ionosphere range candidates used for estimating the ionosphere layers. Recommended value is 25.
    robust_layer_estimation_high_res_num_slices
        Number of range slices for high resolution ionosphere layer estimation. Recommended value is 5.
    robust_layer_estimation_low_res_num_slices
        Number of range slices for low resolution ionosphere layer estimation. Recommended value is 3.
    layer_addition_threshold
        Minimum required coherence gain to accept a new ionospheric layer candidate. Recommended value is 5e-3.
    max_num_ionosphere_layers
        Cap the number of ionosphere layers used for calibration. Use 0 to disable capping. Recommended value is 5.
    max_iono_range_offset
        Global buffer for ionosphere (celestial) axis in along-track direction [m]. Recommended value is 50,000
        meters.
    iono_range_estimation_max_iterations_derivative
        Max iterations for the ionosphere range estimation's composite iterative step ("Derivative"). Recommended
        value is 100.
    iono_range_estimation_max_iterations_projection
        Max iterations for the ionosphere range estimation's composite iterative step ("Projection"). Recommended
        value is 100.
    iono_range_estimation_convergence_threshold_derivative
        Convergence threshold for the ionosphere range estimation's composite iterative step ("Derivative").
        Recommended value is 5e-4.
    iono_range_estimation_convergence_threshold_projection
        Convergence threshold for the ionosphere range estimation's composite iterative step ("Projection").
        Recommended value is 5e-4.
    iono_range_seeding_max_iterations_derivative
        Max iterations for the ionosphere range seeding's composite iterative step ("Derivative"). Recommended value
        is 100.
    iono_range_seeding_max_iterations_projection
        Max iterations for the ionosphere range seeding's composite iterative step ("Projection"). Recommended value
        is 100.
    iono_range_seeding_convergence_threshold_derivative
        Convergence threshold for the ionosphere range seeding's composite iterative step ("Derivative").
        Recommended value is 5e-4.
    iono_range_seeding_convergence_threshold_projection
        Convergence threshold for the ionosphere range seeding's composite iterative step ("Projection").
        Recommended value is 5e-4.
    iono_range_propagation_max_iterations_derivative
        Max iterations for the ionosphere range propagation's composite iterative step ("Derivative"). Recommended
        value is 100.
    iono_range_propagation_max_iterations_projection
        Max iterations for the ionosphere range propagation's composite iterative step ("Projection"). Recommended
        value is 100.
    iono_range_propagation_convergence_threshold_derivative
        Convergence threshold for the ionosphere range propagation's composite iterative step ("Derivative").
        Recommended value is 5e-4.
    iono_range_propagation_convergence_threshold_projection
        Convergence threshold for the ionosphere range propagation's composite iterative step ("Projection").
        Recommended value is 5e-4.
    force_multi_squint_iono_correction_flag
        If true, force the Multi-Squint ionosphere estimation irrespectively of the coherence gain.
    disable_multi_squint_iono_correction_flag
        If true, disable the Multi-Squint ionosphere estimation irrespectively of the coherence gain.
    coherence_azimuth_window_size
        The multi-looking window in along-track direction used for estimating the coherence (coherence test) [m].
        Recommended value is 100 meters.
    coherence_range_window_size
        The multi-looking window in slant-range direction used for estimating the coherence (coherence test) [m].
        Recommended value is 100 meters.
    multi_squint_coherence_subband_resolution
        Sub-band resolution of the Multi-Squint coherence matrix [m]. Recommended value is 300 meters.
    multi_squint_coherence_high_res_azimuth_window_size
        The high resolution multi-look window in along-track direction for the Multi-Squint cohrence matrix [m].
        Recommended value is 0 meters (full resolution).
    multi_squint_coherence_high_res_range_window_size
        The high resolution multi-look window in slant-range direction for the Multi-Squint cohrence matrix [m].
        Recommended value is 250 meters.
    multi_squint_coherence_low_res_azimuth_window_size
        The low resolution multi-look window in along-track direction for the Multi-Squint cohrence matrix [m].
        Recommended value is 250 meters (full resolution).
    multi_squint_coherence_low_res_range_window_size
        The low resolution multi-look window in slant-range direction for the Multi-Squint cohrence matrix [m].
        Recommended value is 300 meters.
    multi_squint_coherence_validity_threshold
        Restrict the area for the coherence gain check. Recommended value is 0.3.
    multi_squint_coherence_improvement_threshold
        Threshold on the coherence gian to trigger the Multi-Squint ionosphere calibration. Recommended value is
        0.3.
    multi_squint_coherence_low_res_threshold
        Threshold on the average Multi-Squint coherence to trigger a lowering of the resolution. Recommended value
        is 0.1.
    use32bit_estimation_flag
        True if 32-bit precision (aka complex64 and float32) has to be used for model estimations instead of 64-bit
        precision (aka complex128 and float64), False otherwise.
    use32bit_correction_flag
        True if 32-bit precision (aka complex64 and float32) has to be used ionosphere correction instead of 64-bit
        precision (aka complex128 and float64), False otherwise.
    """

    class Meta:
        name = "multiSquintCalibrationType"

    multi_squint_calibration_flag: str = field(
        metadata={
            "name": "multiSquintCalibrationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    polarisation_used: PolarisationType = field(
        metadata={
            "name": "polarisationUsed",
            "type": "Element",
            "namespace": "",
        },
    )
    seed_range_coherence_threshold: float = field(
        metadata={
            "name": "seedRangeCoherenceThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_exclusive": 1.0,
        }
    )
    edge_guard: int = field(
        metadata={
            "name": "edgeGuard",
            "type": "Element",
            "namespace": "",
        },
    )
    valid_azimuth_threshold: float = field(
        metadata={
            "name": "validAzimuthThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_exclusive": 1.0,
        }
    )
    propagation_step_inversion_method: MultiSquintCalibrationTypePropagationStepInversionMethod = field(
        metadata={
            "name": "propagationStepInversionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    min_iono_range_candidate: FloatWithUnit = field(
        metadata={
            "name": "minIonoRangeCandidate",
            "type": "Element",
            "namespace": "",
        },
    )
    max_iono_range_candidate: FloatWithUnit = field(
        metadata={
            "name": "maxIonoRangeCandidate",
            "type": "Element",
            "namespace": "",
        },
    )
    num_iono_range_candidates: int = field(
        metadata={"name": "numIonoRangeCandidates", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    robust_layer_estimation_high_res_num_slices: int = field(
        metadata={
            "name": "robustLayerEstimationHighResNumSlices",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 1,
        }
    )
    robust_layer_estimation_low_res_num_slices: int = field(
        metadata={
            "name": "robustLayerEstimationLowResNumSlices",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 1,
        }
    )
    layer_addition_threshold: float = field(
        metadata={
            "name": "layerAdditionThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_exclusive": 1.0,
        }
    )
    max_num_ionosphere_layers: int = field(
        metadata={
            "name": "maxNumIonosphereLayers",
            "type": "Element",
            "namespace": "",
        },
    )
    max_iono_range_offset: FloatWithUnit = field(
        metadata={
            "name": "maxIonoRangeOffset",
            "type": "Element",
            "namespace": "",
        },
    )
    iono_range_estimation_max_iterations_derivative: int = field(
        metadata={
            "name": "ionoRangeEstimationMaxIterationsDerivative",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 1,
        }
    )
    iono_range_estimation_max_iterations_projection: int = field(
        metadata={
            "name": "ionoRangeEstimationMaxIterationsProjection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 1,
        }
    )
    iono_range_estimation_convergence_threshold_derivative: float = field(
        metadata={
            "name": "ionoRangeEstimationConvergenceThresholdDerivative",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
        }
    )
    iono_range_estimation_convergence_threshold_projection: float = field(
        metadata={
            "name": "ionoRangeEstimationConvergenceThresholdProjection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
        }
    )
    iono_range_seeding_max_iterations_derivative: int = field(
        metadata={
            "name": "ionoRangeSeedingMaxIterationsDerivative",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 1,
        }
    )
    iono_range_seeding_max_iterations_projection: int = field(
        metadata={
            "name": "ionoRangeSeedingMaxIterationsProjection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 1,
        }
    )
    iono_range_seeding_convergence_threshold_derivative: float = field(
        metadata={
            "name": "ionoRangeSeedingConvergenceThresholdDerivative",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
        }
    )
    iono_range_seeding_convergence_threshold_projection: float = field(
        metadata={
            "name": "ionoRangeSeedingConvergenceThresholdProjection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
        }
    )
    iono_range_propagation_max_iterations_derivative: int = field(
        metadata={
            "name": "ionoRangePropagationMaxIterationsDerivative",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 1,
        }
    )
    iono_range_propagation_max_iterations_projection: int = field(
        metadata={
            "name": "ionoRangePropagationMaxIterationsProjection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 1,
        }
    )
    iono_range_propagation_convergence_threshold_derivative: float = field(
        metadata={
            "name": "ionoRangePropagationConvergenceThresholdDerivative",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
        }
    )
    iono_range_propagation_convergence_threshold_projection: float = field(
        metadata={
            "name": "ionoRangePropagationConvergenceThresholdProjection",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
        }
    )
    force_multi_squint_iono_correction_flag: str = field(
        metadata={
            "name": "forceMultiSquintIonoCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    disable_multi_squint_iono_correction_flag: str = field(
        metadata={
            "name": "disableMultiSquintIonoCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    coherence_azimuth_window_size: FloatWithUnit = field(
        metadata={
            "name": "coherenceAzimuthWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    coherence_range_window_size: FloatWithUnit = field(
        metadata={
            "name": "coherenceRangeWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_coherence_subband_resolution: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCoherenceSubbandResolution",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_coherence_high_res_azimuth_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCoherenceHighResAzimuthWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_coherence_high_res_range_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCoherenceHighResRangeWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_coherence_low_res_azimuth_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCoherenceLowResAzimuthWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_coherence_low_res_range_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCoherenceLowResRangeWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_coherence_validity_threshold: float = field(
        metadata={
            "name": "multiSquintCoherenceValidityThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_exclusive": 1.0,
        }
    )
    multi_squint_coherence_improvement_threshold: float = field(
        metadata={
            "name": "multiSquintCoherenceImprovementThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_exclusive": 1.0,
        }
    )
    multi_squint_coherence_low_res_threshold: float = field(
        metadata={
            "name": "multiSquintCoherenceLowResThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_exclusive": 1.0,
        }
    )
    use32bit_estimation_flag: str = field(
        metadata={"name": "use32bitEstimationFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    use32bit_correction_flag: str = field(
        metadata={"name": "use32bitCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )


@dataclass(kw_only=True)
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
    calibration_phase_postprocessing
        Whether to post-process the SKP calibration phase screen via "Goldstein" filter, "Boxcar" filter, "Median"
        filter, or no filtering ("None"). It defaults to no filtering.
    postprocessing_filter_window_size
        Size of the filtering window to be used for post-processing the SKP phase screen [m] (Only "Boxcar" and
        "Median"). It defaults to 1000 meters.
    goldstein_filter_window_size
        Smoothing parameter of the Golstein weights (only "Goldstein"). It defaults to 5 pixels.
    goldstein_filter_alpha
        Filtering parameter of the Goldstein filter (only "Goldstein"). It defaults to 0.5.
    exclude_mpmbpolarization_cross_covariance_flag
        True if the MPMB coherence matrix should only contain autocovariances (i.e. HH vs HH and not HH vs. VH
        etc.), False otherwise.
    cross_pol_merging_flag
        True if the SKP phase screen must be computed with merged cross-polarization. Use the stack polarizations
        otherwise.
    use32bit_flag
        True if 32-bit precision (aka complex64 and float32) has to be used for model estimations instead of 64-bit
        precision (aka complex128 and float64), False otherwise.
    """

    class Meta:
        name = "skpPhaseCalibrationType"

    skp_phase_estimation_flag: str = field(
        metadata={"name": "skpPhaseEstimationFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    phase_correction: SkpPhaseCorrectionType = field(
        metadata={
            "name": "phaseCorrection",
            "type": "Element",
            "namespace": "",
        },
    )
    estimation_window_size: FloatWithUnit = field(
        metadata={
            "name": "estimationWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_calibration_phase_screen_quality_threshold: float = field(
        metadata={
            "name": "skpCalibrationPhaseScreenQualityThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    overall_product_quality_threshold: float = field(
        metadata={
            "name": "overallProductQualityThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    calibration_phase_postprocessing: SkpPhaseCalibrationPostprocessingType = field(
        metadata={
            "name": "calibrationPhasePostprocessing",
            "type": "Element",
            "namespace": "",
        },
    )
    postprocessing_filter_window_size: FloatWithUnit = field(
        metadata={
            "name": "postprocessingFilterWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    goldstein_filter_window_size: int = field(
        metadata={
            "name": "goldsteinFilterWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    goldstein_filter_alpha: float = field(
        metadata={
            "name": "goldsteinFilterAlpha",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    exclude_mpmbpolarization_cross_covariance_flag: str = field(
        metadata={
            "name": "excludeMPMBPolarizationCrossCovarianceFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    cross_pol_merging_flag: str = field(
        metadata={"name": "crossPolMergingFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    use32bit_flag: str = field(
        metadata={"name": "use32bitFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )


@dataclass(kw_only=True)
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

    slow_ionosphere_removal_flag: str = field(
        metadata={"name": "slowIonosphereRemovalFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    primary_image_flag: str = field(
        metadata={"name": "primaryImageFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    polarisation_used: PolarisationType = field(
        metadata={
            "name": "polarisationUsed",
            "type": "Element",
            "namespace": "",
        },
    )
    compensate_l1_iono_phase_screen_flag: str = field(
        metadata={
            "name": "compensateL1IonoPhaseScreenFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    range_look_bandwidth: float = field(
        metadata={
            "name": "rangeLookBandwidth",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    range_look_frequency: float = field(
        metadata={
            "name": "rangeLookFrequency",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 0.5,
        }
    )
    phase_unwrapping_flag: str = field(
        metadata={"name": "phaseUnwrappingFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    latitude_threshold: FloatWithUnit = field(
        metadata={
            "name": "latitudeThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    baseline_method: BaselineMethodType = field(
        default=BaselineMethodType.MULTI_BASELINE,
        metadata={
            "name": "baselineMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    unweighted_multi_baseline_estimation: str = field(
        default="false",
        metadata={
            "name": "unweightedMultiBaselineEstimation",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        },
    )
    slow_ionosphere_quality_threshold: float = field(
        metadata={
            "name": "slowIonosphereQualityThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    sublook_window_azimuth_size: int = field(
        metadata={"name": "sublookWindowAzimuthSize", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    sublook_window_range_size: int = field(
        metadata={"name": "sublookWindowRangeSize", "type": "Element", "namespace": "", "min_inclusive": 1}
    )
    multi_baseline_critical_baseline_threshold: float = field(
        metadata={
            "name": "multiBaselineCriticalBaselineThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
        }
    )
    min_coherence_threshold: float = field(
        metadata={
            "name": "minCoherenceThreshold",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    min_usable_pixel_ratio: float = field(
        metadata={
            "name": "minUsablePixelRatio",
            "type": "Element",
            "namespace": "",
            "min_inclusive": 0.0,
            "max_inclusive": 1.0,
        }
    )
    max_delta_phase_unwrap_test: FloatWithUnit = field(
        metadata={
            "name": "maxDeltaPhaseUnwrapTest",
            "type": "Element",
            "namespace": "",
        },
    )
    use32bit_flag: str = field(
        metadata={"name": "use32bitFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )


@dataclass(kw_only=True)
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
    multi_squint_calibration
        Processing parameters of the Multi-Squint calibration step (MSC).
    skp_phase_calibration
        Processing parameters of the Sum-of-Kronecker-Products (SKP).
    l1c_product_export
        L1c product export processing parameters.
    """

    class Meta:
        name = "staProductType"

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
    primary_image_selection: PrimaryImageSelectionType = field(
        metadata={
            "name": "primaryImageSelection",
            "type": "Element",
            "namespace": "",
        },
    )
    coregistration: CoregistrationType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_degradation_estimation: RfiDegradationEstimationType = field(
        metadata={
            "name": "rfiDegradationEstimation",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_spectral_filtering: AzimuthSpectralFilteringType = field(
        metadata={
            "name": "azimuthSpectralFiltering",
            "type": "Element",
            "namespace": "",
        },
    )
    slow_ionosphere_removal: SlowIonosphereRemovalType = field(
        metadata={
            "name": "slowIonosphereRemoval",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration: MultiSquintCalibrationType = field(
        metadata={
            "name": "multiSquintCalibration",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_phase_calibration: SkpPhaseCalibrationType = field(
        metadata={
            "name": "skpPhaseCalibration",
            "type": "Element",
            "namespace": "",
        },
    )
    l1c_product_export: L1CProductExportType = field(
        metadata={
            "name": "l1cProductExport",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    sta_product: StaProductType = field(
        metadata={
            "name": "staProduct",
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

    sta_product_list: StaProductListType = field(
        metadata={
            "name": "staProductList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryStaprocessingParameters(AuxiliaryStaprocessingParametersType):
    """
    BIOMASS auxiliary stack processing parameters element.
    """

    class Meta:
        name = "auxiliarySTAProcessingParameters"
