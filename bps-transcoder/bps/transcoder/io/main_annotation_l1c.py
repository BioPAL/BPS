# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Main annotation L1c"""

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common
from bps.transcoder.io import common_annotation_l1


@dataclass
class BIOMASSStackProcessingParameters:
    """Data struct for the BIOMASS Stack Processor parameters."""

    processor_version: str
    product_generation_time: PreciseDateTime | str
    polarizations_used: int
    polarization_combination_method: common.PolarisationCombinationMethodType | None
    primary_image_selection_method: common.PrimaryImageSelectionMethodType
    coregistration_method: common.CoregistrationMethodType
    height_model: common.HeightModelType
    rfi_degradation_estimation_flag: bool
    azimuth_spectral_filtering_flag: bool
    polarization_used_for_slow_ionosphere_removal: common.PolarisationType
    polarization_used_for_multi_squint_calibration: common.PolarisationType
    calibration_primary_image_flag: bool
    slow_ionosphere_removal_flag: bool
    multi_squint_calibration_flag: bool
    multi_squint_calibration_force_ionosphere_correction_flag: bool
    multi_squint_calibration_disable_ionosphere_correction_flag: bool
    multi_squint_calibration_coherence_improvement_threshold: float
    multi_squint_calibration_coherence_azimuth_window_size: float
    multi_squint_calibration_coherence_range_window_size: float
    multi_squint_calibration_multi_squint_coherence_subband_resolution: float
    multi_squint_calibration_multi_squint_high_res_coherence_azimuth_window_size: float
    multi_squint_calibration_multi_squint_high_res_coherence_range_window_size: float
    multi_squint_calibration_multi_squint_low_res_coherence_azimuth_window_size: float
    multi_squint_calibration_multi_squint_low_res_coherence_range_window_size: float
    skp_phase_calibration_flag: bool
    skp_phase_correction_flag: bool
    skp_phase_correction_flattening_only_flag: bool
    skp_estimation_window_size: float
    skp_calibration_phase_postprocessing: common.SkpPhaseCalibrationPostprocessingType
    skp_postprocessing_filter_window_size: float
    skp_goldstein_filter_window_size: int
    azimuth_spectral_filtering_use_32bit_flag: bool
    slow_ionosphere_removal_multi_baseline_threshold: float
    slow_ionosphere_removal_use_32bit_flag: bool
    multi_squint_estimation_use_32bit_flag: bool
    multi_squint_correction_use_32bit_flag: bool
    skp_phase_calibration_use_32bit_flag: bool


@dataclass
class BIOMASSStackCoregistrationParameters:
    """Data struct for the BIOMASS Coregistration Processor parameters."""

    datum: common.DatumType
    primary_image: str
    secondary_image: str
    primary_image_selection_information: common.PrimaryImageSelectionInformationType
    average_azimuth_coregistration_shift: float
    average_range_coregistration_shift: float
    normal_baseline: float
    polarization_used: common.PolarisationType


@dataclass
class BIOMASSStackInSARParameters:
    """Data struct for the BIOMASS Stack InSAR parameters."""

    calibration_primary_image: str
    azimuth_common_bandwidth: float
    azimuth_central_frequency: float
    slow_ionosphere_azimuth_phase_screen: float
    slow_ionosphere_range_phase_screen: float
    slow_ionosphere_quality: float
    slow_ionosphere_removal_interferometric_pairs: list[tuple[int, int]]
    multi_squint_calibration_low_resolution_flag: bool
    multi_squint_calibration_azimuth_phase_slope: float
    multi_squint_calibration_range_phase_slope: float
    multi_squint_calibration_azimuth_shift: float
    multi_squint_calibration_azimuth_frequency_centroid: float
    multi_squint_calibration_ionospheric_layer_altitudes: list[float]
    multi_squint_calibration_coherence_check_value: float
    multi_squint_calibration_ionosphere_correction_flag: bool
    baseline_ordering_index: int
    skp_calibration_phase_screen_mean: float
    skp_calibration_phase_screen_std: float
    skp_calibration_phase_screen_var: float
    skp_calibration_phase_screen_mad: float


@dataclass
class CoherenceHistogram:
    bins: np.ndarray
    pdf_values: np.ndarray

    def __eq__(self, other):
        if not isinstance(other, CoherenceHistogram):
            return NotImplemented

        return np.array_equal(self.bins, other.bins) and np.array_equal(self.pdf_values, other.pdf_values)

    @classmethod
    def from_coherence_abs(cls, coherence: np.ndarray) -> Self:
        """Compute bins and pdf of the coherence"""
        hist, bins = np.histogram(
            coherence[np.logical_and(coherence > 0, coherence <= 1.0)],  # It also excludes nan, +inf, and -inf.
            bins=np.linspace(0, 1, 51),
            density=True,
        )
        return cls(
            bins=0.5 * (bins[:-1] + bins[1:]),
            pdf_values=hist,
        )


@dataclass
class BIOMASSStackQualityParameters:
    """Data struct for the BIOMASS Stack Quality parameters."""

    polarization: common.PolarisationType
    invalid_l1a_data_samples: float
    rfi_decorrelation: float
    rfi_decorrelation_threshold: float
    faraday_decorrelation: float
    faraday_decorrelation_threshold: float
    invalid_residual_shifts_ratio: float
    residual_shifts_quality_threshold: float
    invalid_skp_calibration_phase_screen_ratio: float
    skp_calibration_phase_screen_quality_threshold: float
    skp_decomposition_index: int
    coherence_mean: float
    coherence_std: float
    coherence_min: float
    coherence_max: float
    coherence_median: float
    coherence_histogram: CoherenceHistogram


@dataclass
class BIOMASSStackQuality:
    """Data struct for the BIOMASS Stack quality parameters."""

    overall_product_quality_index: int
    sta_quality_parameters_list: list[BIOMASSStackQualityParameters] = field(default_factory=list)


@dataclass
class MainAnnotationL1c:
    """L1c main annotation"""

    acquisition_information: common_annotation_l1.AcquisitionInformationType
    sar_image: common_annotation_l1.SarImageType
    instrument_parameters: common_annotation_l1.InstrumentParametersType
    raw_data_analysis: common_annotation_l1.RawDataAnalysisType
    processing_parameters: common_annotation_l1.ProcessingParameters
    internal_calibration: common_annotation_l1.InternalCalibrationType
    rfi_mitigation: common_annotation_l1.RfiMitigationType
    doppler_parameters: common_annotation_l1.DopplerParametersType
    radiometric_calibration: common_annotation_l1.RadiometricCalibrationType
    polarimetric_distortion: common_annotation_l1.PolarimetricDistortionType
    ionosphere_correction: common_annotation_l1.IonosphereCorrection
    geometry: common_annotation_l1.GeometryType
    quality: common_annotation_l1.QualityType
    annotation_lut: list[common.LayerType]

    # l1c specific fields
    sta_processing_parameters: BIOMASSStackProcessingParameters
    sta_coregistration_parameters: BIOMASSStackCoregistrationParameters
    sta_in_sarparameters: BIOMASSStackInSARParameters
    sta_quality: BIOMASSStackQuality
