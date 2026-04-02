# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Main annotation models l1c
------------------------------
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
    CoregistrationMethodType,
    CrossTalkList,
    DataFormatModeType,
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
    LayerListType,
    LayerType,
    MinMaxType,
    MinMaxTypeWithUnit,
    MissionPhaseIdtype,
    MissionType,
    OrbitAttitudeSourceType,
    OrbitPassType,
    PixelQuantityType,
    PixelRepresentationType,
    PixelTypeType,
    PolarisationCombinationMethodType,
    PolarisationType,
    PrimaryImageSelectionInformationType,
    PrimaryImageSelectionMethodType,
    ProcessingModeType,
    ProductCompositionType,
    ProductType,
    ProjectionType,
    RangeCompressionMethodType,
    RangeReferenceFunctionType,
    RfiFmmitigationMethodType,
    RfiMaskGenerationMethodType,
    RfiMaskType,
    RfiMitigationMethodType,
    SensorModeType,
    SkpPhaseCalibrationPostprocessingType,
    SlantRangePolynomialType,
    StateType,
    SwathType,
    TimeTypeWithPolarisation,
    UnsignedIntWithGroup,
    UomType,
    WeightingWindowType,
)
from bps.transcoder.io.common_annotation_models_l1 import (
    AcquisitionInformationType,
    CalibrationConstantListType,
    CoordinateConversionListType,
    CoordinateConversionType,
    DataFormatType,
    DcEstimateListType,
    DcEstimateType,
    DopplerParametersType,
    ErrorCountersType,
    FirstLineSensingTimeListType,
    FmRateEstimatesListType,
    GeometryType,
    InOutBandPowerRatioListType,
    InstrumentParametersType,
    InternalCalibrationParametersListType,
    InternalCalibrationSequenceListType,
    InternalCalibrationSequenceType,
    InternalCalibrationType,
    IonosphereCorrectionType,
    LastLineSensingTimeListType,
    NoiseGainListType,
    NoiseListType,
    NoiseSequenceListType,
    NoiseSequenceType,
    PolarimetricDistortionType,
    PolarisationListType,
    PowerRatioListType,
    PowerRatioType,
    PrfListType,
    ProcessingGainListType,
    ProcessingParametersType,
    QualityParametersListType,
    QualityParametersType,
    QualityType,
    RadiometricCalibrationType,
    RawDataAnalysisType,
    RawDataStatisticsListType,
    RawDataStatisticsType,
    RfiIsolatedFmreportListType,
    RfiIsolatedFmreportType,
    RfiMitigationType,
    RfiPersistentFmreportListType,
    RfiPersistentFmreportType,
    RfiTmreportListType,
    RfiTmreportType,
    RxGainListType,
    SarImageType,
    SpectrumProcessingParametersType,
    SwlListType,
    SwpListType,
    TxPulseListType,
    TxPulseType,
)


@dataclass(kw_only=True)
class StaQualityParametersType:
    """
    Parameters
    ----------
    invalid_l1a_data_samples
        Number of invalid L1a data samples (according to their mean and standard deviation), expressed as a
        normalized ratio of the total number of samples.
    rfi_decorrelation
        Decorrelation factor due to RFI strength.
    rfi_decorrelation_threshold
        Maximum decorrelation factor due to RFI strength admissible to use the image as the primary image.
    faraday_decorrelation
        Decorrelation factor due to Faraday residual.
    faraday_decorrelation_threshold
        Maximum decorrelation factor due to Faraday residual admissible to use the image as the primary image.
    invalid_residual_shifts_ratio
        Ratio of invalid residual coregistration shifts with respect to the number of pixels of the coregistration
        reference image.
    residual_shifts_quality_threshold
        Threshold on residual coregistration shifts quality (between 0 and 1). Residual shifts with a quality lower
        than this threshold are not used for processing.
    invalid_skp_calibration_phase_screen_ratio
        Ratio between invalid SKP calibration phase estimations and the size of the SKP phase screen.
    skp_calibration_phase_screen_quality_threshold
        Threshold on the SKP estimation quality (between 0 and 1). Estimates with a quality lower than this
        threshold are not corrected. Used only in case skpPhaseCorrectionFlag is set to True and
        flatteningPhaseCorrectionFlag is set to False.
    skp_decomposition_index
        Error code for the SKP decomposition. Positive if the SKP decomposition hit a nonblocking contingency case
        (e.g. SVD decomposition failure etc.) and the exported SKP-related LUTs contain values obtained from a
        contingency handling procedure. 0 otherwise.
    coherence_mean
        Mean coherence (amplitude) of the calibrated L1c frame with respect to the coregistration primary.
    coherence_std
        Standard deviation of the coherence (amplitude) of the calibrated L1c frame with respect to the
        coregistration primary.
    coherence_min
        Minimum coherence (amplitude) of the calibrated L1c frame with respect to the coregistration primary.
    coherence_max
        Maximum coherence (amplitude) of the calibrated L1c frame with respect to the coregistration primary.
    coherence_median
        Median coherence (amplitude) of the calibrated L1c frame with respect to the coregistration primary.
    coherence_histogram
        Coherence distribution of the calibrated L1c frame with respect to the coregistration primary.
    polarisation
    """

    class Meta:
        name = "staQualityParametersType"

    invalid_l1a_data_samples: float = field(
        metadata={
            "name": "invalidL1aDataSamples",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_decorrelation: float = field(
        metadata={
            "name": "rfiDecorrelation",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_decorrelation_threshold: float = field(
        metadata={
            "name": "rfiDecorrelationThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    faraday_decorrelation: float = field(
        metadata={
            "name": "faradayDecorrelation",
            "type": "Element",
            "namespace": "",
        },
    )
    faraday_decorrelation_threshold: float = field(
        metadata={
            "name": "faradayDecorrelationThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_residual_shifts_ratio: float = field(
        metadata={
            "name": "invalidResidualShiftsRatio",
            "type": "Element",
            "namespace": "",
        },
    )
    residual_shifts_quality_threshold: float = field(
        metadata={
            "name": "residualShiftsQualityThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    invalid_skp_calibration_phase_screen_ratio: float = field(
        metadata={
            "name": "invalidSkpCalibrationPhaseScreenRatio",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_calibration_phase_screen_quality_threshold: float = field(
        metadata={
            "name": "skpCalibrationPhaseScreenQualityThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_decomposition_index: int = field(
        metadata={
            "name": "skpDecompositionIndex",
            "type": "Element",
            "namespace": "",
        },
    )
    coherence_mean: float = field(
        metadata={
            "name": "coherenceMean",
            "type": "Element",
            "namespace": "",
        },
    )
    coherence_std: float = field(
        metadata={
            "name": "coherenceStd",
            "type": "Element",
            "namespace": "",
        },
    )
    coherence_min: float = field(
        metadata={
            "name": "coherenceMin",
            "type": "Element",
            "namespace": "",
        },
    )
    coherence_max: float = field(
        metadata={
            "name": "coherenceMax",
            "type": "Element",
            "namespace": "",
        },
    )
    coherence_median: float = field(
        metadata={
            "name": "coherenceMedian",
            "type": "Element",
            "namespace": "",
        },
    )
    coherence_histogram: StaQualityParametersType.CoherenceHistogram = field(
        metadata={
            "name": "coherenceHistogram",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation: PolarisationType = field(
        metadata={
            "type": "Attribute",
        },
    )

    @dataclass(kw_only=True)
    class CoherenceHistogram:
        """
        Parameters
        ----------
        bins
            The bins of the coherence histrogram.
        pdf_values
            The PDF values of the coherence histrogram.
        """

        bins: FloatArray = field(
            metadata={
                "type": "Element",
                "namespace": "",
            },
        )
        pdf_values: FloatArray = field(
            metadata={
                "name": "pdfValues",
                "type": "Element",
                "namespace": "",
            },
        )


@dataclass(kw_only=True)
class StaCoregistrationParametersType:
    """
    Parameters
    ----------
    datum
        Datum used during coregistration.
    primary_image
        Product name of coregistration primary image.
    secondary_image
        Product name of coregistration secondary image.
    primary_image_selection_information
        Information used to select coregistration primary image (Geometry, Geometry and RFI Correction, Geometry and
        FR Correction, Geometry and RFI+FR Corrections).
    normal_baseline
        Normal baseline between primary and secondary images [m].
    average_range_coregistration_shift
        Average coregistration shift along range direction [m].
    average_azimuth_coregistration_shift
        Average coregistration shift along azimuth direction [m].
    polarisation_used
        Polarisation used for shift estimation.
    """

    class Meta:
        name = "staCoregistrationParametersType"

    datum: DatumType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    primary_image: str = field(
        metadata={
            "name": "primaryImage",
            "type": "Element",
            "namespace": "",
        },
    )
    secondary_image: str = field(
        metadata={
            "name": "secondaryImage",
            "type": "Element",
            "namespace": "",
        },
    )
    primary_image_selection_information: PrimaryImageSelectionInformationType = field(
        metadata={
            "name": "primaryImageSelectionInformation",
            "type": "Element",
            "namespace": "",
        },
    )
    normal_baseline: FloatWithUnit = field(
        metadata={
            "name": "normalBaseline",
            "type": "Element",
            "namespace": "",
        },
    )
    average_range_coregistration_shift: FloatWithUnit = field(
        metadata={
            "name": "averageRangeCoregistrationShift",
            "type": "Element",
            "namespace": "",
        },
    )
    average_azimuth_coregistration_shift: FloatWithUnit = field(
        metadata={
            "name": "averageAzimuthCoregistrationShift",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation_used: PolarisationType = field(
        metadata={
            "name": "polarisationUsed",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class StaInSarparametersType:
    """
    Parameters
    ----------
    calibration_primary_image
        Product name of multi-baseline calibration steps primary image (or, simply, calibration primary image).
    azimuth_common_bandwidth
        Azimuth bandwidth in common with primary image after azimuth spectral filtering step [Hz].
    azimuth_central_frequency
        Azimuth central frequency after azimuth spectral filtering step [Hz].
    slow_ionosphere_range_phase_screen
        Estimated slope of background ionosphere in range direction [rad/s].
    slow_ionosphere_azimuth_phase_screen
        Estimated slope of background ionosphere in azimuth direction [rad/s].
    slow_ionosphere_quality
        Quality of the background ionosphere estimation, between 0 (bad quality) and 1 (good quality).
    slow_ionosphere_removal_interferometric_pairs
        Interferometric pairs used to estimate the slow-varying ionosphere calibration (multi-baseline and single-
        baseline).
    multi_squint_calibration_low_resolution_flag
        True if Multi-Squint calibration was executed in low resolution modality, False otherwise.
    multi_squint_calibration_range_phase_slope
        Phase plane slope estimated by the Multi-Squint Calibration module, in slant-range direction [rad/s].
    multi_squint_calibration_azimuth_phase_slope
        Phase plane slope estimated by Multi-Squint Calibration moduke, in along-track direction [rad/s].
    multi_squint_calibration_azimuth_shift
        Constant shift in along-track direction estimated from spectral analysis on the Multi-Squint coherence [m].
    multi_squint_calibration_azimuth_frequency_centroid
        Spatial (along-track) frequency centroid from spectral analysis on the Multi-Squint coherence [m].
    multi_squint_calibration_ionospheric_layer_altitudes
        Altitude of the ionospheric layers estimated by the Multi-Squint Calibration module.
    multi_squint_calibration_coherence_check_value
        Coherence improvement used during the coherence check of the Multi-Squint Calibration module.
    multi_squint_calibration_ionosphere_correction_flag
        True if the Multi-Squint Calibration module executed the ionosphere correction. False otherwise.
    baseline_ordering_index
        Index in of the product with respect to the baseline-based ordering.
    skp_calibration_phase_screen_mean
        Circular mean of SKP calibration phase screen [rad].
    skp_calibration_phase_screen_std
        Circular standard deviation of SKP calibration phase screen.
    skp_calibration_phase_screen_var
        Circular dispersion of SKP calibration phase screen.
    skp_calibration_phase_screen_mad
        Mean absolute deviation of SKP calibration phase screen [rad].
    """

    class Meta:
        name = "staInSARParametersType"

    calibration_primary_image: str = field(
        metadata={
            "name": "calibrationPrimaryImage",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_common_bandwidth: FloatWithUnit = field(
        metadata={
            "name": "azimuthCommonBandwidth",
            "type": "Element",
            "namespace": "",
        },
    )
    azimuth_central_frequency: FloatWithUnit = field(
        metadata={
            "name": "azimuthCentralFrequency",
            "type": "Element",
            "namespace": "",
        },
    )
    slow_ionosphere_range_phase_screen: FloatWithUnit = field(
        metadata={
            "name": "slowIonosphereRangePhaseScreen",
            "type": "Element",
            "namespace": "",
        },
    )
    slow_ionosphere_azimuth_phase_screen: FloatWithUnit = field(
        metadata={
            "name": "slowIonosphereAzimuthPhaseScreen",
            "type": "Element",
            "namespace": "",
        },
    )
    slow_ionosphere_quality: float = field(
        metadata={
            "name": "slowIonosphereQuality",
            "type": "Element",
            "namespace": "",
        },
    )
    slow_ionosphere_removal_interferometric_pairs: InterferometricPairListType = field(
        metadata={
            "name": "slowIonosphereRemovalInterferometricPairs",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_low_resolution_flag: str = field(
        metadata={
            "name": "multiSquintCalibrationLowResolutionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    multi_squint_calibration_range_phase_slope: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationRangePhaseSlope",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_azimuth_phase_slope: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationAzimuthPhaseSlope",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_azimuth_shift: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationAzimuthShift",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_azimuth_frequency_centroid: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationAzimuthFrequencyCentroid",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_ionospheric_layer_altitudes: FloatArrayWithUnits = field(
        metadata={
            "name": "multiSquintCalibrationIonosphericLayerAltitudes",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_coherence_check_value: float = field(
        metadata={
            "name": "multiSquintCalibrationCoherenceCheckValue",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_ionosphere_correction_flag: str = field(
        metadata={
            "name": "multiSquintCalibrationIonosphereCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    baseline_ordering_index: int = field(
        metadata={
            "name": "baselineOrderingIndex",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_calibration_phase_screen_mean: FloatWithUnit = field(
        metadata={
            "name": "skpCalibrationPhaseScreenMean",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_calibration_phase_screen_std: float = field(
        metadata={
            "name": "skpCalibrationPhaseScreenStd",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_calibration_phase_screen_var: float = field(
        metadata={
            "name": "skpCalibrationPhaseScreenVar",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_calibration_phase_screen_mad: FloatWithUnit = field(
        metadata={
            "name": "skpCalibrationPhaseScreenMAD",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class StaProcessingParametersType:
    """
    Parameters
    ----------
    processor_version
        Version of the processor used to generate the product.
    product_generation_time
        Product generation time [UTC].
    polarisations_used
        Number of polarisations used for processing (3 or 4).
    polarisation_combination_method
        Polarisations combination method: “HV” or “VH”, if just one of the two is selected (in addition to HH and VV
        ones); “Average”, if the average of HV and VH is computed and used; “None” if no combination is performed
        (all the four polarisations are used).
    primary_image_selection_method
        Coregistration primary image selection method (Geometry, Geometry and Quality).
    coregistration_method
        Coregistration method (Geometry, Geometry and Data).
    height_model
        Digital Elevation Model (DEM) used during coregistration.
    rfi_degradation_estimation_flag
        True if estimation of degradation due to RFI filtering was performed, False otherwise.
    azimuth_spectral_filtering_flag
        True if azimuth spectral filtering was performed, False otherwise.
    polarisation_used_for_slow_ionosphere_removal
        Polarisation used for the Slow-Ionosphere Remova (IOB) step.
    polarisation_used_for_multi_squint_calibration
        Polarisation used for estimation in the Multi-Squint Calibration (MSC) step.
    calibration_primary_image_flag
        True if image used as primary for multi-baseline calibration steps is the same as the one used for
        coregistration, False otherwise.
    slow_ionosphere_removal_flag
        True if slow-varying ionospheric phase screen estimation and removal was performed, False otherwise.
    multi_squint_calibration_flag
        True if Multi-Squint calibration module was enabled, False otherwise.
    multi_squint_calibration_force_ionosphere_correction_flag
        True if Multi-Squint ionosphere calibration step was forcefully executed by user, False otherwise.
    multi_squint_calibration_disable_ionosphere_correction_flag
        True if Multi-Squint ionosphere calibration step was forcefully executed by user, False otherwise.
    multi_squint_calibration_coherence_improvement_threshold
        Threshold on the coherence gian that was used to trigger the Multi-Squint ionosphere calibration step.
    multi_squint_calibration_coherence_azimuth_window_size
        The multi-looking window in along-track direction used for estimating the standard coherence (coherence
        test) [m].
    multi_squint_calibration_coherence_range_window_size
        The multi-looking window in slant-range direction used for estimating the standard coherence (coherence
        test) [m].
    multi_squint_calibration_multi_squint_coherence_sub_band_resolution
        The squint axis resolution of the Multi-Squint cohrence matrix [m].
    multi_squint_calibration_multi_squint_high_res_coherence_azimuth_window_size
        The high resolution multi-look window in along-track direction for the Multi-Squint cohrence matrix [m].
    multi_squint_calibration_multi_squint_high_res_coherence_range_window_size
        The high resolution multi-look window in slant-range direction for the Multi-Squint cohrence matrix [m].
    multi_squint_calibration_multi_squint_low_res_coherence_azimuth_window_size
        The low resolution multi-look window in along-track direction for the Multi-Squint cohrence matrix [m].
    multi_squint_calibration_multi_squint_low_res_coherence_range_window_size
        The low resolution multi-look window in slant-range direction for the Multi-Squint cohrence matrix [m].
    skp_phase_calibration_flag
        True if SKP estimation was performed, False otherwise.
    skp_phase_correction_flag
        True if skpPhaseCorrectionFlag was enabled, False otherwise.
    skp_phase_correction_flattening_only_flag
        True if only the flattening phases was corrected by the SKP module, False otherwise.
    skp_estimation_window_size
        SKP estimation window size [m].
    skp_calibration_phase_postprocessing
        Postprocessing applied to the SKP calibration phase screens.
    skp_postprocessing_filter_window_size
        Filter window size used in SKP postprocessing in case of "Boxcar" or "Median" filter [m].
    skp_goldstein_filter_window_size
        Filter window size used in SKP postprocessing in case of "Goldstein" filter [px].
    slow_ionosphere_removal_multi_baseline_threshold
        Threshold used for selecting the interferometric pair for slow-ionosphere calibration (IOB) as ratio of the
        critical baseline.
    slow_ionosphere_removal_use32_bit_flag
        True if IOB was run with 32-bit precision.
    azimuth_spectral_filtering_use32_bit_flag
        True if AZF was run with 32-bit precision.
    multi_squint_estimation_use32_bit_flag
        True if the estimation step of the MSC was run with 32-bit precision.
    multi_squint_correction_use32_bit_flag
        True if the ionosphere correction step of the MSC was run with 32-bit precision.
    skp_phase_calibration_use32_bit_flag
        True if SKP was run with 32-bit precision.
    """

    class Meta:
        name = "staProcessingParametersType"

    processor_version: str = field(
        metadata={
            "name": "processorVersion",
            "type": "Element",
            "namespace": "",
        },
    )
    product_generation_time: str = field(
        metadata={
            "name": "productGenerationTime",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    polarisations_used: int = field(
        metadata={
            "name": "polarisationsUsed",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation_combination_method: PolarisationCombinationMethodType = field(
        metadata={
            "name": "polarisationCombinationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    primary_image_selection_method: PrimaryImageSelectionMethodType = field(
        metadata={
            "name": "primaryImageSelectionMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    coregistration_method: CoregistrationMethodType = field(
        metadata={
            "name": "coregistrationMethod",
            "type": "Element",
            "namespace": "",
        },
    )
    height_model: HeightModelType = field(
        metadata={
            "name": "heightModel",
            "type": "Element",
            "namespace": "",
        },
    )
    rfi_degradation_estimation_flag: str = field(
        metadata={
            "name": "rfiDegradationEstimationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    azimuth_spectral_filtering_flag: str = field(
        metadata={
            "name": "azimuthSpectralFilteringFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    polarisation_used_for_slow_ionosphere_removal: PolarisationType = field(
        metadata={
            "name": "polarisationUsedForSlowIonosphereRemoval",
            "type": "Element",
            "namespace": "",
        },
    )
    polarisation_used_for_multi_squint_calibration: PolarisationType = field(
        metadata={
            "name": "polarisationUsedForMultiSquintCalibration",
            "type": "Element",
            "namespace": "",
        },
    )
    calibration_primary_image_flag: str = field(
        metadata={
            "name": "calibrationPrimaryImageFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    slow_ionosphere_removal_flag: str = field(
        metadata={"name": "slowIonosphereRemovalFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    multi_squint_calibration_flag: str = field(
        metadata={
            "name": "multiSquintCalibrationFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    multi_squint_calibration_force_ionosphere_correction_flag: str = field(
        metadata={
            "name": "multiSquintCalibrationForceIonosphereCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    multi_squint_calibration_disable_ionosphere_correction_flag: str = field(
        metadata={
            "name": "multiSquintCalibrationDisableIonosphereCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    multi_squint_calibration_coherence_improvement_threshold: float = field(
        metadata={
            "name": "multiSquintCalibrationCoherenceImprovementThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_coherence_azimuth_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationCoherenceAzimuthWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_coherence_range_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationCoherenceRangeWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_multi_squint_coherence_sub_band_resolution: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationMultiSquintCoherenceSubBandResolution",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_multi_squint_high_res_coherence_azimuth_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationMultiSquintHighResCoherenceAzimuthWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_multi_squint_high_res_coherence_range_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationMultiSquintHighResCoherenceRangeWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_multi_squint_low_res_coherence_azimuth_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationMultiSquintLowResCoherenceAzimuthWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    multi_squint_calibration_multi_squint_low_res_coherence_range_window_size: FloatWithUnit = field(
        metadata={
            "name": "multiSquintCalibrationMultiSquintLowResCoherenceRangeWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_phase_calibration_flag: str = field(
        metadata={"name": "skpPhaseCalibrationFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    skp_phase_correction_flag: str = field(
        metadata={"name": "skpPhaseCorrectionFlag", "type": "Element", "namespace": "", "pattern": r"(false)|(true)"}
    )
    skp_phase_correction_flattening_only_flag: str = field(
        metadata={
            "name": "skpPhaseCorrectionFlatteningOnlyFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    skp_estimation_window_size: FloatWithUnit = field(
        metadata={
            "name": "skpEstimationWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_calibration_phase_postprocessing: SkpPhaseCalibrationPostprocessingType = field(
        metadata={
            "name": "skpCalibrationPhasePostprocessing",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_postprocessing_filter_window_size: FloatWithUnit = field(
        metadata={
            "name": "skpPostprocessingFilterWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    skp_goldstein_filter_window_size: int = field(
        metadata={
            "name": "skpGoldsteinFilterWindowSize",
            "type": "Element",
            "namespace": "",
        },
    )
    slow_ionosphere_removal_multi_baseline_threshold: float = field(
        metadata={
            "name": "slowIonosphereRemovalMultiBaselineThreshold",
            "type": "Element",
            "namespace": "",
        },
    )
    slow_ionosphere_removal_use32_bit_flag: str = field(
        metadata={
            "name": "slowIonosphereRemovalUse32BitFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    azimuth_spectral_filtering_use32_bit_flag: str = field(
        metadata={
            "name": "azimuthSpectralFilteringUse32BitFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    multi_squint_estimation_use32_bit_flag: str = field(
        metadata={
            "name": "multiSquintEstimationUse32BitFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    multi_squint_correction_use32_bit_flag: str = field(
        metadata={
            "name": "multiSquintCorrectionUse32BitFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )
    skp_phase_calibration_use32_bit_flag: str = field(
        metadata={
            "name": "skpPhaseCalibrationUse32BitFlag",
            "type": "Element",
            "namespace": "",
            "pattern": r"(false)|(true)",
        }
    )


@dataclass(kw_only=True)
class StaQualityParametersListType:
    """
    Parameters
    ----------
    sta_quality_parameters
        Quality parameters for the current polarisation.
    count
    """

    class Meta:
        name = "staQualityParametersListType"

    sta_quality_parameters: list[StaQualityParametersType] = field(
        default_factory=list,
        metadata={"name": "staQualityParameters", "type": "Element", "namespace": "", "min_occurs": 1, "max_occurs": 4},
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class StaQualityType:
    """
    Parameters
    ----------
    overall_product_quality_index
        Overall product quality index. This annotation is calculated based on specific quality parameters and gives
        an overall quality value to the product. Equal to 0 for valid products and to 1 for invalid ones.
    sta_quality_parameters_list
        Quality parameters list. The list contains an entry for each polarisation.
    """

    class Meta:
        name = "staQualityType"

    overall_product_quality_index: int = field(
        metadata={
            "name": "overallProductQualityIndex",
            "type": "Element",
            "namespace": "",
        },
    )
    sta_quality_parameters_list: StaQualityParametersListType = field(
        metadata={
            "name": "staQualityParametersList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class StaMainAnnotationType:
    """
    Parameters
    ----------
    acquisition_information
        Acquisition information DSR. This DSR contains information that applies to the entire data set.
    sar_image
        SAR image DSR. This DSR contains all the necessary information to exploit the measurement data set (i.e. SAR
        images).
    instrument_parameters
        Instrument parameters DSR. This DSR contains the main instrument settings at the time of imaging.
    raw_data_analysis
        RAW data analysis DSR. This DSR contains the main elements related to the RAW data consolidation and
        analysis performed by the processor.
    processing_parameters
        Processing parameters DSR. This DSR contains the exhaustive list of static SAR processing parameters and of
        corrections applied.
    internal_calibration
        Internal calibration DSR. This DSR contains the results of the internal calibration analysis performed by
        the processor.
    rfi_mitigation
        Radio Frequency Interference mitigation DSR. This DSR contains information on detected RFI and on their
        mitigation.
    doppler_parameters
        Doppler parameters DSR. This DSR contains the Doppler Centroid (DC) and Frequency Modulation rate (FM)
        parameters estimated and used during processing.
    radiometric_calibration
        Radiometric calibration DSR. This DSR contains all the necessary information to absolutely calibrate the
        data pixels.
    polarimetric_distortion
        Polarimetric distortion DSR. This DSR contains the necessary information e.g. receive and transmit
        polarisation distortion matrix, allowing users to correct for them if not applied at processing level.
    ionosphere_correction
        Ionosphere correction DSR. This DSR contains the results of the ionosphere correction estimated and applied
        by the processor.
    geometry
        Geometry DSR. This DSR contains all the necessary information to understand the Earth model/geometry used in
        the image and its geolocation.
    quality
        Quality DSR. This DSR contains in a single DSR quality flags and thresholds for basic quality assessment
        done at processing level.
    annotation_lut
        Annotation LUT DSR. This DSR contains the list of Look-Up Tables (LUTs) complementing product main
        annotations.
    sta_processing_parameters
        Stack processing parameters DSR. This DSR contains the exhaustive list of static stack processing parameters
        and of corrections applied.
    sta_coregistration_parameters
        Stack coregistration parameters DSR. This DSR contains high-level information about data coregistration
        step.
    sta_in_sarparameters
        Stack InSAR parameters DSR. This DSR contains high-level information about calibration steps.
    sta_quality
        Stack quality DSR. This DSR contains in a single DSR quality flags and thresholds for basic quality
        assessment done at processing level.
    """

    class Meta:
        name = "staMainAnnotationType"

    acquisition_information: AcquisitionInformationType = field(
        metadata={
            "name": "acquisitionInformation",
            "type": "Element",
            "namespace": "",
        },
    )
    sar_image: SarImageType = field(
        metadata={
            "name": "sarImage",
            "type": "Element",
            "namespace": "",
        },
    )
    instrument_parameters: InstrumentParametersType = field(
        metadata={
            "name": "instrumentParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    raw_data_analysis: RawDataAnalysisType = field(
        metadata={
            "name": "rawDataAnalysis",
            "type": "Element",
            "namespace": "",
        },
    )
    processing_parameters: ProcessingParametersType = field(
        metadata={
            "name": "processingParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    internal_calibration: InternalCalibrationType = field(
        metadata={
            "name": "internalCalibration",
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
    doppler_parameters: DopplerParametersType = field(
        metadata={
            "name": "dopplerParameters",
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
    polarimetric_distortion: PolarimetricDistortionType = field(
        metadata={
            "name": "polarimetricDistortion",
            "type": "Element",
            "namespace": "",
        },
    )
    ionosphere_correction: IonosphereCorrectionType = field(
        metadata={
            "name": "ionosphereCorrection",
            "type": "Element",
            "namespace": "",
        },
    )
    geometry: GeometryType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    quality: QualityType = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    annotation_lut: LayerListType = field(
        metadata={
            "name": "annotationLUT",
            "type": "Element",
            "namespace": "",
        },
    )
    sta_processing_parameters: StaProcessingParametersType = field(
        metadata={
            "name": "staProcessingParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    sta_coregistration_parameters: StaCoregistrationParametersType = field(
        metadata={
            "name": "staCoregistrationParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    sta_in_sarparameters: StaInSarparametersType = field(
        metadata={
            "name": "staInSARParameters",
            "type": "Element",
            "namespace": "",
        },
    )
    sta_quality: StaQualityType = field(
        metadata={
            "name": "staQuality",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class MainAnnotation(StaMainAnnotationType):
    """
    BIOMASS L1c product main annotation element.
    """

    class Meta:
        name = "mainAnnotation"
