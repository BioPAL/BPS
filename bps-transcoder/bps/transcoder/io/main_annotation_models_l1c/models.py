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

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

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


@dataclass
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
    polarisation
    """

    class Meta:
        name = "staQualityParametersType"

    invalid_l1a_data_samples: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidL1aDataSamples",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_decorrelation: Optional[float] = field(
        default=None,
        metadata={
            "name": "rfiDecorrelation",
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
        },
    )
    faraday_decorrelation: Optional[float] = field(
        default=None,
        metadata={
            "name": "faradayDecorrelation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    faraday_decorrelation_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "faradayDecorrelationThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_residual_shifts_ratio: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidResidualShiftsRatio",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    residual_shifts_quality_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "residualShiftsQualityThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    invalid_skp_calibration_phase_screen_ratio: Optional[float] = field(
        default=None,
        metadata={
            "name": "invalidSkpCalibrationPhaseScreenRatio",
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
        },
    )
    skp_decomposition_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "skpDecompositionIndex",
            "type": "Element",
            "namespace": "",
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
    range_spectral_filtering_flag
        True if range spectral filtering was performed during the coregistration step, False otherwise.
    polarisation_used
        Polarisation used for shift estimation.
    """

    class Meta:
        name = "staCoregistrationParametersType"

    datum: Optional[DatumType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    primary_image: Optional[str] = field(
        default=None,
        metadata={
            "name": "primaryImage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    secondary_image: Optional[str] = field(
        default=None,
        metadata={
            "name": "secondaryImage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    primary_image_selection_information: Optional[PrimaryImageSelectionInformationType] = field(
        default=None,
        metadata={
            "name": "primaryImageSelectionInformation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    normal_baseline: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "normalBaseline",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    average_range_coregistration_shift: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "averageRangeCoregistrationShift",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    average_azimuth_coregistration_shift: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "averageAzimuthCoregistrationShift",
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
    polarisation_used: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "name": "polarisationUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
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
    range_phase_slope
        Phase plane slope estimated by PPR, in slant-range direction [rad/s].
    azimuth_phase_slope
        Phase plane slope estimated by PPR, in along-track direction [rad/s].
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

    calibration_primary_image: Optional[str] = field(
        default=None,
        metadata={
            "name": "calibrationPrimaryImage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_common_bandwidth: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "azimuthCommonBandwidth",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_central_frequency: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "azimuthCentralFrequency",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    slow_ionosphere_range_phase_screen: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "slowIonosphereRangePhaseScreen",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    slow_ionosphere_azimuth_phase_screen: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "slowIonosphereAzimuthPhaseScreen",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    slow_ionosphere_quality: Optional[float] = field(
        default=None,
        metadata={
            "name": "slowIonosphereQuality",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    slow_ionosphere_removal_interferometric_pairs: Optional[InterferometricPairListType] = field(
        default=None,
        metadata={
            "name": "slowIonosphereRemovalInterferometricPairs",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    range_phase_slope: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "rangePhaseSlope",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    azimuth_phase_slope: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "azimuthPhaseSlope",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    baseline_ordering_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "baselineOrderingIndex",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    skp_calibration_phase_screen_mean: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "skpCalibrationPhaseScreenMean",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    skp_calibration_phase_screen_std: Optional[float] = field(
        default=None,
        metadata={
            "name": "skpCalibrationPhaseScreenStd",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    skp_calibration_phase_screen_var: Optional[float] = field(
        default=None,
        metadata={
            "name": "skpCalibrationPhaseScreenVar",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    skp_calibration_phase_screen_mad: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "skpCalibrationPhaseScreenMAD",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
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
    polarisation_used_for_phase_plane_removal
        Polarisation used for estimation in the Phase-Plane-Removal (PPR) step.
    calibration_primary_image_flag
        True if image used as primary for multi-baseline calibration steps is the same as the one used for
        coregistration, False otherwise.
    slow_ionosphere_removal_flag
        True if slow-varying ionospheric phase screen estimation and removal was performed, False otherwise.
    in_sarcalibration_flag
        True if InSAR calibration was enabled, False otherwise.
    skp_phase_calibration_flag
        True if SKP estimation was performed, False otherwise.
    skp_phase_correction_flag
        True if skpPhaseCorrectionFlag was enabled, False otherwise.
    skp_phase_correction_flattening_only_flag
        True if only the flattening phases was corrected by the SKP module, False otherwise.
    skp_estimation_window_size
        SKP estimation window size [m].
    skp_median_filter_flag
        True if median filter was applied to the SKP calibration phase screen.
    skp_median_filter_window_size
        SKP median filter window size [m].
    slow_ionosphere_removal_multi_baseline_threshold
        Threshold used for selecting the interferometric pair for slow-ionosphere calibration (IOB) as ratio of the
        critical baseline.
    slow_ionosphere_removal_use32_bit_flag
        True if IOB was run with 32-bit precision.
    azimuth_spectral_filtering_use32_bit_flag
        True if AZF was run with 32-bit precision.
    in_sarcalibration_use32_bit_flag
        True if PPR was run with 32-bit precision.
    skp_phase_calibration_use32_bit_flag
        True if SKP was run with 32-bit precision.
    """

    class Meta:
        name = "staProcessingParametersType"

    processor_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "processorVersion",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    product_generation_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "productGenerationTime",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        },
    )
    polarisations_used: Optional[int] = field(
        default=None,
        metadata={
            "name": "polarisationsUsed",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation_combination_method: Optional[PolarisationCombinationMethodType] = field(
        default=None,
        metadata={
            "name": "polarisationCombinationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    primary_image_selection_method: Optional[PrimaryImageSelectionMethodType] = field(
        default=None,
        metadata={
            "name": "primaryImageSelectionMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    coregistration_method: Optional[CoregistrationMethodType] = field(
        default=None,
        metadata={
            "name": "coregistrationMethod",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    height_model: Optional[HeightModelType] = field(
        default=None,
        metadata={
            "name": "heightModel",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
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
    polarisation_used_for_slow_ionosphere_removal: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "name": "polarisationUsedForSlowIonosphereRemoval",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarisation_used_for_phase_plane_removal: Optional[PolarisationType] = field(
        default=None,
        metadata={
            "name": "polarisationUsedForPhasePlaneRemoval",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    calibration_primary_image_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "calibrationPrimaryImageFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
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
    skp_phase_calibration_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "skpPhaseCalibrationFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    skp_phase_correction_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "skpPhaseCorrectionFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    skp_phase_correction_flattening_only_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "skpPhaseCorrectionFlatteningOnlyFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    skp_estimation_window_size: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "skpEstimationWindowSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    skp_median_filter_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "skpMedianFilterFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    skp_median_filter_window_size: Optional[FloatWithUnit] = field(
        default=None,
        metadata={
            "name": "skpMedianFilterWindowSize",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    slow_ionosphere_removal_multi_baseline_threshold: Optional[float] = field(
        default=None,
        metadata={
            "name": "slowIonosphereRemovalMultiBaselineThreshold",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    slow_ionosphere_removal_use32_bit_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "slowIonosphereRemovalUse32BitFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    azimuth_spectral_filtering_use32_bit_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "azimuthSpectralFilteringUse32BitFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    in_sarcalibration_use32_bit_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "inSARCalibrationUse32BitFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )
    skp_phase_calibration_use32_bit_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "skpPhaseCalibrationUse32BitFlag",
            "type": "Element",
            "namespace": "",
            "required": True,
            "pattern": r"(false)|(true)",
        },
    )


@dataclass
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
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
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

    overall_product_quality_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "overallProductQualityIndex",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sta_quality_parameters_list: Optional[StaQualityParametersListType] = field(
        default=None,
        metadata={
            "name": "staQualityParametersList",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
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

    acquisition_information: Optional[AcquisitionInformationType] = field(
        default=None,
        metadata={
            "name": "acquisitionInformation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sar_image: Optional[SarImageType] = field(
        default=None,
        metadata={
            "name": "sarImage",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    instrument_parameters: Optional[InstrumentParametersType] = field(
        default=None,
        metadata={
            "name": "instrumentParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    raw_data_analysis: Optional[RawDataAnalysisType] = field(
        default=None,
        metadata={
            "name": "rawDataAnalysis",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    processing_parameters: Optional[ProcessingParametersType] = field(
        default=None,
        metadata={
            "name": "processingParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    internal_calibration: Optional[InternalCalibrationType] = field(
        default=None,
        metadata={
            "name": "internalCalibration",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rfi_mitigation: Optional[RfiMitigationType] = field(
        default=None,
        metadata={
            "name": "rfiMitigation",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    doppler_parameters: Optional[DopplerParametersType] = field(
        default=None,
        metadata={
            "name": "dopplerParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    radiometric_calibration: Optional[RadiometricCalibrationType] = field(
        default=None,
        metadata={
            "name": "radiometricCalibration",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polarimetric_distortion: Optional[PolarimetricDistortionType] = field(
        default=None,
        metadata={
            "name": "polarimetricDistortion",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ionosphere_correction: Optional[IonosphereCorrectionType] = field(
        default=None,
        metadata={
            "name": "ionosphereCorrection",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    geometry: Optional[GeometryType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    quality: Optional[QualityType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    annotation_lut: Optional[LayerListType] = field(
        default=None,
        metadata={
            "name": "annotationLUT",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sta_processing_parameters: Optional[StaProcessingParametersType] = field(
        default=None,
        metadata={
            "name": "staProcessingParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sta_coregistration_parameters: Optional[StaCoregistrationParametersType] = field(
        default=None,
        metadata={
            "name": "staCoregistrationParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sta_in_sarparameters: Optional[StaInSarparametersType] = field(
        default=None,
        metadata={
            "name": "staInSARParameters",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    sta_quality: Optional[StaQualityType] = field(
        default=None,
        metadata={
            "name": "staQuality",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class MainAnnotation(StaMainAnnotationType):
    """
    BIOMASS L1c product main annotation element.
    """

    class Meta:
        name = "mainAnnotation"
