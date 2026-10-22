# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Main annotation L1ab translation"""

from pathlib import Path

import numpy as np
from bps.common.io import common, translate_common
from bps.common.io.parsing import parse, serialize
from bps.common.io.translate_common import (
    translate_bool,
    translate_bool_to_model,
    translate_coregistration_method,
    translate_coregistration_method_to_model,
    translate_datetime,
    translate_datetime_to_model,
    translate_datum,
    translate_datum_to_model,
    translate_float_array,
    translate_float_array_to_model,
    translate_float_array_with_units,
    translate_float_array_with_units_to_model,
    translate_float_with_unit,
    translate_float_with_unit_to_model,
    translate_height_model,
    translate_height_model_to_model,
    translate_interferometric_pair_list,
    translate_interferometric_pair_list_to_model,
    translate_polarisation_combination_method,
    translate_polarisation_combination_method_to_model,
    translate_polarisation_type,
    translate_polarisation_type_to_model,
    translate_primary_image_selection_information,
    translate_primary_image_selection_information_to_model,
    translate_primary_image_selection_method,
    translate_primary_image_selection_method_to_model,
    translate_skp_phase_calibration_postprocessing_type,
    translate_skp_phase_calibration_postprocessing_type_to_model,
)
from bps.transcoder.io import (
    main_annotation_l1c,
    main_annotation_models_l1c,
    translate_common_annotation_l1,
)


def translate_sta_processing_parameters(
    params: main_annotation_models_l1c.StaProcessingParametersType,
) -> main_annotation_l1c.BIOMASSStackProcessingParameters:
    """Translate from a L1c main annotation StaProcessingParametersType object."""
    return main_annotation_l1c.BIOMASSStackProcessingParameters(
        processor_version=params.processor_version,
        product_generation_time=translate_datetime(params.product_generation_time),
        polarizations_used=params.polarisations_used,
        polarization_combination_method=translate_polarisation_combination_method(
            params.polarisation_combination_method
        ),
        primary_image_selection_method=translate_primary_image_selection_method(params.primary_image_selection_method),
        coregistration_method=translate_coregistration_method(params.coregistration_method),
        height_model=translate_height_model(params.height_model),
        rfi_degradation_estimation_flag=translate_bool(params.rfi_degradation_estimation_flag),
        azimuth_spectral_filtering_flag=translate_bool(params.azimuth_spectral_filtering_flag),
        polarization_used_for_slow_ionosphere_removal=translate_polarisation_type(
            params.polarisation_used_for_slow_ionosphere_removal
        ),
        polarization_used_for_multi_squint_calibration=translate_polarisation_type(
            params.polarisation_used_for_multi_squint_calibration
        ),
        calibration_primary_image_flag=translate_bool(params.calibration_primary_image_flag),
        slow_ionosphere_removal_flag=translate_bool(params.slow_ionosphere_removal_flag),
        multi_squint_calibration_flag=translate_bool(params.multi_squint_calibration_flag),
        multi_squint_calibration_force_ionosphere_correction_flag=translate_bool(
            params.multi_squint_calibration_force_ionosphere_correction_flag
        ),
        multi_squint_calibration_disable_ionosphere_correction_flag=translate_bool(
            params.multi_squint_calibration_disable_ionosphere_correction_flag
        ),
        multi_squint_calibration_coherence_improvement_threshold=params.multi_squint_calibration_coherence_improvement_threshold,
        multi_squint_calibration_coherence_azimuth_window_size=translate_float_with_unit(
            params.multi_squint_calibration_coherence_azimuth_window_size
        ),
        multi_squint_calibration_coherence_range_window_size=translate_float_with_unit(
            params.multi_squint_calibration_coherence_range_window_size
        ),
        multi_squint_calibration_multi_squint_coherence_subband_resolution=translate_float_with_unit(
            params.multi_squint_calibration_multi_squint_coherence_sub_band_resolution
        ),
        multi_squint_calibration_multi_squint_high_res_coherence_azimuth_window_size=translate_float_with_unit(
            params.multi_squint_calibration_multi_squint_high_res_coherence_azimuth_window_size
        ),
        multi_squint_calibration_multi_squint_high_res_coherence_range_window_size=translate_float_with_unit(
            params.multi_squint_calibration_multi_squint_high_res_coherence_range_window_size
        ),
        multi_squint_calibration_multi_squint_low_res_coherence_azimuth_window_size=translate_float_with_unit(
            params.multi_squint_calibration_multi_squint_low_res_coherence_azimuth_window_size
        ),
        multi_squint_calibration_multi_squint_low_res_coherence_range_window_size=translate_float_with_unit(
            params.multi_squint_calibration_multi_squint_low_res_coherence_range_window_size
        ),
        skp_phase_calibration_flag=translate_bool(params.skp_phase_calibration_flag),
        skp_phase_correction_flag=translate_bool(params.skp_phase_correction_flag),
        skp_phase_correction_flattening_only_flag=translate_bool(params.skp_phase_correction_flattening_only_flag),
        skp_estimation_window_size=translate_float_with_unit(params.skp_estimation_window_size),
        skp_calibration_phase_postprocessing=translate_skp_phase_calibration_postprocessing_type(
            params.skp_calibration_phase_postprocessing
        ),
        skp_postprocessing_filter_window_size=translate_float_with_unit(params.skp_postprocessing_filter_window_size),
        skp_goldstein_filter_window_size=params.skp_goldstein_filter_window_size,
        slow_ionosphere_removal_multi_baseline_threshold=params.slow_ionosphere_removal_multi_baseline_threshold,
        azimuth_spectral_filtering_use_32bit_flag=translate_bool(params.azimuth_spectral_filtering_use32_bit_flag),
        slow_ionosphere_removal_use_32bit_flag=translate_bool(params.slow_ionosphere_removal_use32_bit_flag),
        multi_squint_estimation_use_32bit_flag=translate_bool(params.multi_squint_estimation_use32_bit_flag),
        multi_squint_correction_use_32bit_flag=translate_bool(params.multi_squint_correction_use32_bit_flag),
        skp_phase_calibration_use_32bit_flag=translate_bool(params.skp_phase_calibration_use32_bit_flag),
    )


def translate_sta_coregistration_parameters(
    params: main_annotation_models_l1c.StaCoregistrationParametersType,
) -> main_annotation_l1c.BIOMASSStackCoregistrationParameters:
    """Translate L1c StaCoregistrationParametersType to BIOMASSStackCoregistrationParameters."""
    return main_annotation_l1c.BIOMASSStackCoregistrationParameters(
        datum=translate_datum(params.datum),
        primary_image=params.primary_image,
        secondary_image=params.secondary_image,
        primary_image_selection_information=translate_primary_image_selection_information(
            params.primary_image_selection_information
        ),
        average_azimuth_coregistration_shift=translate_float_with_unit(params.average_azimuth_coregistration_shift),
        average_range_coregistration_shift=translate_float_with_unit(params.average_range_coregistration_shift),
        normal_baseline=translate_float_with_unit(params.normal_baseline),
        polarization_used=translate_polarisation_type(params.polarisation_used),
    )


def translate_sta_coregistration_parameters_to_model(
    annot: main_annotation_l1c.BIOMASSStackCoregistrationParameters,
) -> main_annotation_models_l1c.StaCoregistrationParametersType:
    """Translate BIOMASSStackCoregistrationParameters to L1c StaCoregistrationParametersType."""
    return main_annotation_models_l1c.StaCoregistrationParametersType(
        datum=translate_datum_to_model(annot.datum),
        primary_image=annot.primary_image,
        secondary_image=annot.secondary_image,
        primary_image_selection_information=translate_primary_image_selection_information_to_model(
            annot.primary_image_selection_information
        ),
        average_azimuth_coregistration_shift=translate_float_with_unit_to_model(
            float(annot.average_azimuth_coregistration_shift), common.UomType.M
        ),
        average_range_coregistration_shift=translate_float_with_unit_to_model(
            float(annot.average_range_coregistration_shift), common.UomType.M
        ),
        normal_baseline=translate_float_with_unit_to_model(float(annot.normal_baseline), common.UomType.M),
        polarisation_used=translate_polarisation_type_to_model(annot.polarization_used),
    )


def translate_sta_processing_parameters_to_model(
    params: main_annotation_l1c.BIOMASSStackProcessingParameters,
) -> main_annotation_models_l1c.StaProcessingParametersType:
    """Marshall to related L1c annotation."""
    return main_annotation_models_l1c.StaProcessingParametersType(
        processor_version=params.processor_version,
        product_generation_time=translate_datetime_to_model(params.product_generation_time),
        polarisations_used=params.polarizations_used,
        polarisation_combination_method=translate_polarisation_combination_method_to_model(
            params.polarization_combination_method
        ),
        polarisation_used_for_slow_ionosphere_removal=translate_polarisation_type_to_model(
            params.polarization_used_for_slow_ionosphere_removal
        ),
        polarisation_used_for_multi_squint_calibration=translate_polarisation_type_to_model(
            params.polarization_used_for_multi_squint_calibration
        ),
        primary_image_selection_method=translate_primary_image_selection_method_to_model(
            params.primary_image_selection_method
        ),
        coregistration_method=translate_coregistration_method_to_model(params.coregistration_method),
        calibration_primary_image_flag=translate_bool_to_model(params.calibration_primary_image_flag),
        height_model=translate_height_model_to_model(params.height_model),
        rfi_degradation_estimation_flag=translate_bool_to_model(params.rfi_degradation_estimation_flag),
        azimuth_spectral_filtering_flag=translate_bool_to_model(params.azimuth_spectral_filtering_flag),
        slow_ionosphere_removal_flag=translate_bool_to_model(params.slow_ionosphere_removal_flag),
        multi_squint_calibration_flag=translate_bool_to_model(params.multi_squint_calibration_flag),
        multi_squint_calibration_force_ionosphere_correction_flag=translate_bool_to_model(
            params.multi_squint_calibration_force_ionosphere_correction_flag
        ),
        multi_squint_calibration_disable_ionosphere_correction_flag=translate_bool_to_model(
            params.multi_squint_calibration_disable_ionosphere_correction_flag
        ),
        multi_squint_calibration_coherence_improvement_threshold=params.multi_squint_calibration_coherence_improvement_threshold,
        multi_squint_calibration_multi_squint_coherence_sub_band_resolution=translate_float_with_unit_to_model(
            params.multi_squint_calibration_multi_squint_coherence_subband_resolution, common.UomType.M
        ),
        multi_squint_calibration_coherence_azimuth_window_size=translate_float_with_unit_to_model(
            params.multi_squint_calibration_coherence_azimuth_window_size, common.UomType.M
        ),
        multi_squint_calibration_coherence_range_window_size=translate_float_with_unit_to_model(
            params.multi_squint_calibration_coherence_range_window_size, common.UomType.M
        ),
        multi_squint_calibration_multi_squint_high_res_coherence_azimuth_window_size=translate_float_with_unit_to_model(
            params.multi_squint_calibration_multi_squint_high_res_coherence_azimuth_window_size, common.UomType.M
        ),
        multi_squint_calibration_multi_squint_high_res_coherence_range_window_size=translate_float_with_unit_to_model(
            params.multi_squint_calibration_multi_squint_high_res_coherence_range_window_size, common.UomType.M
        ),
        multi_squint_calibration_multi_squint_low_res_coherence_azimuth_window_size=translate_float_with_unit_to_model(
            params.multi_squint_calibration_multi_squint_low_res_coherence_azimuth_window_size, common.UomType.M
        ),
        multi_squint_calibration_multi_squint_low_res_coherence_range_window_size=translate_float_with_unit_to_model(
            params.multi_squint_calibration_multi_squint_low_res_coherence_range_window_size, common.UomType.M
        ),
        skp_phase_calibration_flag=translate_bool_to_model(params.skp_phase_calibration_flag),
        skp_phase_correction_flag=translate_bool_to_model(params.skp_phase_correction_flag),
        skp_phase_correction_flattening_only_flag=translate_bool_to_model(
            params.skp_phase_correction_flattening_only_flag
        ),
        skp_estimation_window_size=translate_float_with_unit_to_model(
            float(params.skp_estimation_window_size), common.UomType.M
        ),
        skp_calibration_phase_postprocessing=translate_skp_phase_calibration_postprocessing_type_to_model(
            params.skp_calibration_phase_postprocessing
        ),
        skp_postprocessing_filter_window_size=translate_float_with_unit_to_model(
            float(params.skp_postprocessing_filter_window_size), common.UomType.M
        ),
        skp_goldstein_filter_window_size=params.skp_goldstein_filter_window_size,
        slow_ionosphere_removal_multi_baseline_threshold=params.slow_ionosphere_removal_multi_baseline_threshold,
        azimuth_spectral_filtering_use32_bit_flag=translate_bool_to_model(
            params.azimuth_spectral_filtering_use_32bit_flag
        ),
        slow_ionosphere_removal_use32_bit_flag=translate_bool_to_model(params.slow_ionosphere_removal_use_32bit_flag),
        multi_squint_estimation_use32_bit_flag=translate_bool_to_model(params.multi_squint_estimation_use_32bit_flag),
        multi_squint_correction_use32_bit_flag=translate_bool_to_model(params.multi_squint_correction_use_32bit_flag),
        skp_phase_calibration_use32_bit_flag=translate_bool_to_model(params.skp_phase_calibration_use_32bit_flag),
    )


def translate_sta_in_sarparameters(
    params: main_annotation_models_l1c.StaInSarparametersType,
) -> main_annotation_l1c.BIOMASSStackInSARParameters:
    """Translate from L1c StaInSarparametersType to BIOMASSStackInSARParameters."""
    return main_annotation_l1c.BIOMASSStackInSARParameters(
        calibration_primary_image=params.calibration_primary_image,
        azimuth_common_bandwidth=translate_float_with_unit(params.azimuth_common_bandwidth),
        azimuth_central_frequency=translate_float_with_unit(params.azimuth_central_frequency),
        slow_ionosphere_azimuth_phase_screen=translate_float_with_unit(params.slow_ionosphere_azimuth_phase_screen),
        slow_ionosphere_range_phase_screen=translate_float_with_unit(params.slow_ionosphere_range_phase_screen),
        slow_ionosphere_quality=params.slow_ionosphere_quality,
        slow_ionosphere_removal_interferometric_pairs=translate_interferometric_pair_list(
            params.slow_ionosphere_removal_interferometric_pairs
        ),
        multi_squint_calibration_low_resolution_flag=translate_bool(
            params.multi_squint_calibration_low_resolution_flag,
        ),
        multi_squint_calibration_azimuth_phase_slope=translate_float_with_unit(
            params.multi_squint_calibration_azimuth_phase_slope
        ),
        multi_squint_calibration_range_phase_slope=translate_float_with_unit(
            params.multi_squint_calibration_range_phase_slope
        ),
        multi_squint_calibration_azimuth_shift=translate_float_with_unit(params.multi_squint_calibration_azimuth_shift),
        multi_squint_calibration_azimuth_frequency_centroid=translate_float_with_unit(
            params.multi_squint_calibration_azimuth_frequency_centroid
        ),
        multi_squint_calibration_ionospheric_layer_altitudes=translate_float_array_with_units(
            params.multi_squint_calibration_ionospheric_layer_altitudes,
        ),
        multi_squint_calibration_coherence_check_value=params.multi_squint_calibration_coherence_check_value,
        multi_squint_calibration_ionosphere_correction_flag=translate_bool(
            params.multi_squint_calibration_ionosphere_correction_flag,
        ),
        baseline_ordering_index=params.baseline_ordering_index,
        skp_calibration_phase_screen_mean=translate_float_with_unit(params.skp_calibration_phase_screen_mean),
        skp_calibration_phase_screen_std=params.skp_calibration_phase_screen_std,
        skp_calibration_phase_screen_var=params.skp_calibration_phase_screen_var,
        skp_calibration_phase_screen_mad=translate_float_with_unit(params.skp_calibration_phase_screen_mad),
    )


def translate_sta_in_sarparameters_to_model(
    params: main_annotation_l1c.BIOMASSStackInSARParameters,
) -> main_annotation_models_l1c.StaInSarparametersType:
    """Translate to L1c StaInSarparametersType annotation."""
    return main_annotation_models_l1c.StaInSarparametersType(
        calibration_primary_image=params.calibration_primary_image,
        azimuth_common_bandwidth=translate_float_with_unit_to_model(
            float(params.azimuth_common_bandwidth), common.UomType.HZ
        ),
        azimuth_central_frequency=translate_float_with_unit_to_model(
            float(params.azimuth_central_frequency), common.UomType.HZ
        ),
        slow_ionosphere_azimuth_phase_screen=translate_float_with_unit_to_model(
            float(params.slow_ionosphere_azimuth_phase_screen), common.UomType.RAD_S
        ),
        slow_ionosphere_range_phase_screen=translate_float_with_unit_to_model(
            float(params.slow_ionosphere_range_phase_screen), common.UomType.RAD_S
        ),
        slow_ionosphere_quality=float(params.slow_ionosphere_quality),
        slow_ionosphere_removal_interferometric_pairs=translate_interferometric_pair_list_to_model(
            params.slow_ionosphere_removal_interferometric_pairs
        ),
        multi_squint_calibration_low_resolution_flag=translate_bool_to_model(
            params.multi_squint_calibration_low_resolution_flag
        ),
        multi_squint_calibration_azimuth_phase_slope=translate_float_with_unit_to_model(
            float(params.multi_squint_calibration_azimuth_phase_slope), common.UomType.RAD_S
        ),
        multi_squint_calibration_range_phase_slope=translate_float_with_unit_to_model(
            float(params.multi_squint_calibration_range_phase_slope), common.UomType.RAD_S
        ),
        multi_squint_calibration_azimuth_shift=translate_float_with_unit_to_model(
            float(params.multi_squint_calibration_azimuth_shift), common.UomType.M
        ),
        multi_squint_calibration_azimuth_frequency_centroid=translate_float_with_unit_to_model(
            float(params.multi_squint_calibration_azimuth_frequency_centroid), common.UomType.VALUE_1_M
        ),
        multi_squint_calibration_ionospheric_layer_altitudes=translate_float_array_with_units_to_model(
            [float(h) for h in params.multi_squint_calibration_ionospheric_layer_altitudes],
            units=common.UomType.M,
        ),
        multi_squint_calibration_coherence_check_value=float(params.multi_squint_calibration_coherence_check_value),
        multi_squint_calibration_ionosphere_correction_flag=translate_bool_to_model(
            params.multi_squint_calibration_ionosphere_correction_flag
        ),
        baseline_ordering_index=int(params.baseline_ordering_index),
        skp_calibration_phase_screen_mean=translate_float_with_unit_to_model(
            float(params.skp_calibration_phase_screen_mean), common.UomType.RAD
        ),
        skp_calibration_phase_screen_std=float(params.skp_calibration_phase_screen_std),
        skp_calibration_phase_screen_var=float(params.skp_calibration_phase_screen_var),
        skp_calibration_phase_screen_mad=translate_float_with_unit_to_model(
            float(params.skp_calibration_phase_screen_mad), common.UomType.RAD
        ),
    )


def translate_coherence_histogram(
    histogram: main_annotation_models_l1c.StaQualityParametersType.CoherenceHistogram,
) -> main_annotation_l1c.CoherenceHistogram:
    """Translate a coherence histogram from the L1c annotation model."""
    return main_annotation_l1c.CoherenceHistogram(
        bins=np.array(translate_float_array(histogram.bins)),
        pdf_values=np.array(translate_float_array(histogram.pdf_values)),
    )


def translate_coherence_histogram_to_model(
    histogram: main_annotation_l1c.CoherenceHistogram,
) -> main_annotation_models_l1c.StaQualityParametersType.CoherenceHistogram:
    """Translate a coherence histogram to the L1c annotation model."""
    return main_annotation_models_l1c.StaQualityParametersType.CoherenceHistogram(
        bins=translate_float_array_to_model(histogram.bins.tolist()),
        pdf_values=translate_float_array_to_model(histogram.pdf_values.tolist()),
    )


def translate_sta_quality_parameters(
    params: main_annotation_models_l1c.StaQualityParametersType,
) -> main_annotation_l1c.BIOMASSStackQualityParameters:
    """Translate stack-quality parameters from the L1c annotation model."""
    return main_annotation_l1c.BIOMASSStackQualityParameters(
        polarization=translate_polarisation_type(params.polarisation),
        invalid_l1a_data_samples=params.invalid_l1a_data_samples,
        rfi_decorrelation=params.rfi_decorrelation,
        rfi_decorrelation_threshold=params.rfi_decorrelation_threshold,
        faraday_decorrelation=params.faraday_decorrelation,
        faraday_decorrelation_threshold=params.faraday_decorrelation_threshold,
        invalid_residual_shifts_ratio=params.invalid_residual_shifts_ratio,
        residual_shifts_quality_threshold=params.residual_shifts_quality_threshold,
        invalid_skp_calibration_phase_screen_ratio=params.invalid_skp_calibration_phase_screen_ratio,
        skp_calibration_phase_screen_quality_threshold=params.skp_calibration_phase_screen_quality_threshold,
        skp_decomposition_index=params.skp_decomposition_index,
        coherence_mean=params.coherence_mean,
        coherence_std=params.coherence_std,
        coherence_min=params.coherence_min,
        coherence_max=params.coherence_max,
        coherence_median=params.coherence_median,
        coherence_histogram=translate_coherence_histogram(params.coherence_histogram),
    )


def translate_sta_quality_parameters_to_model(
    params: main_annotation_l1c.BIOMASSStackQualityParameters,
) -> main_annotation_models_l1c.StaQualityParametersType:
    """Translate stack-quality parameters to the L1c annotation model."""
    return main_annotation_models_l1c.StaQualityParametersType(
        polarisation=translate_polarisation_type_to_model(params.polarization),
        invalid_l1a_data_samples=float(params.invalid_l1a_data_samples),
        rfi_decorrelation=float(params.rfi_decorrelation),
        rfi_decorrelation_threshold=float(params.rfi_decorrelation_threshold),
        faraday_decorrelation=float(params.faraday_decorrelation),
        faraday_decorrelation_threshold=float(params.faraday_decorrelation_threshold),
        invalid_residual_shifts_ratio=float(params.invalid_residual_shifts_ratio),
        residual_shifts_quality_threshold=float(params.residual_shifts_quality_threshold),
        invalid_skp_calibration_phase_screen_ratio=float(params.invalid_skp_calibration_phase_screen_ratio),
        skp_calibration_phase_screen_quality_threshold=float(params.skp_calibration_phase_screen_quality_threshold),
        skp_decomposition_index=int(params.skp_decomposition_index),
        coherence_mean=float(params.coherence_mean),
        coherence_std=float(params.coherence_std),
        coherence_min=float(params.coherence_min),
        coherence_max=float(params.coherence_max),
        coherence_median=float(params.coherence_median),
        coherence_histogram=translate_coherence_histogram_to_model(params.coherence_histogram),
    )


def translate_sta_quality(
    quality: main_annotation_models_l1c.StaQualityType,
) -> main_annotation_l1c.BIOMASSStackQuality:
    """Translate stack quality from the L1c annotation model."""
    return main_annotation_l1c.BIOMASSStackQuality(
        overall_product_quality_index=quality.overall_product_quality_index,
        sta_quality_parameters_list=[
            translate_sta_quality_parameters(params)
            for params in quality.sta_quality_parameters_list.sta_quality_parameters
        ],
    )


def translate_sta_quality_to_model(
    quality: main_annotation_l1c.BIOMASSStackQuality,
) -> main_annotation_models_l1c.StaQualityType:
    """Translate stack quality to the L1c annotation model."""
    return main_annotation_models_l1c.StaQualityType(
        overall_product_quality_index=int(quality.overall_product_quality_index),
        sta_quality_parameters_list=main_annotation_models_l1c.StaQualityParametersListType(
            sta_quality_parameters=[
                translate_sta_quality_parameters_to_model(params) for params in quality.sta_quality_parameters_list
            ],
            count=len(quality.sta_quality_parameters_list),
        ),
    )


def translate_main_annotation_l1c(
    annotation: main_annotation_models_l1c.MainAnnotation,
) -> main_annotation_l1c.MainAnnotationL1c:
    """Translate L1c main annotation file"""
    return main_annotation_l1c.MainAnnotationL1c(
        acquisition_information=translate_common_annotation_l1.translate_acquisition_information(
            annotation.acquisition_information
        ),
        sar_image=translate_common_annotation_l1.translate_sar_image(annotation.sar_image),
        instrument_parameters=translate_common_annotation_l1.translate_instrument_parameters(
            annotation.instrument_parameters
        ),
        raw_data_analysis=translate_common_annotation_l1.translate_raw_data_analysis(annotation.raw_data_analysis),
        processing_parameters=translate_common_annotation_l1.translate_processing_parameters(
            annotation.processing_parameters
        ),
        internal_calibration=translate_common_annotation_l1.translate_internal_calibration(
            annotation.internal_calibration
        ),
        rfi_mitigation=translate_common_annotation_l1.translate_rfi_mitigation(annotation.rfi_mitigation),
        doppler_parameters=translate_common_annotation_l1.translate_doppler_parameters(annotation.doppler_parameters),
        radiometric_calibration=translate_common_annotation_l1.translate_radiometric_calibration(
            annotation.radiometric_calibration
        ),
        polarimetric_distortion=translate_common_annotation_l1.translate_polarimetric_distortion(
            annotation.polarimetric_distortion
        ),
        ionosphere_correction=translate_common_annotation_l1.translate_ionosphere_correction(
            annotation.ionosphere_correction
        ),
        geometry=translate_common_annotation_l1.translate_geometry(annotation.geometry),
        quality=translate_common_annotation_l1.translate_quality(annotation.quality),
        annotation_lut=translate_common.translate_layer_list(annotation.annotation_lut),
        sta_processing_parameters=translate_sta_processing_parameters(annotation.sta_processing_parameters),
        sta_coregistration_parameters=translate_sta_coregistration_parameters(annotation.sta_coregistration_parameters),
        sta_in_sarparameters=translate_sta_in_sarparameters(annotation.sta_in_sarparameters),
        sta_quality=translate_sta_quality(annotation.sta_quality),
    )


def translate_main_annotation_l1c_to_model(
    annotation: main_annotation_l1c.MainAnnotationL1c,
) -> main_annotation_models_l1c.MainAnnotation:
    """Translate L1c main annotation file"""
    return main_annotation_models_l1c.MainAnnotation(
        acquisition_information=translate_common_annotation_l1.translate_acquisition_information_to_model(
            annotation.acquisition_information
        ),
        sar_image=translate_common_annotation_l1.translate_sar_image_to_model(annotation.sar_image),
        instrument_parameters=translate_common_annotation_l1.translate_instrument_parameters_to_model(
            annotation.instrument_parameters
        ),
        raw_data_analysis=translate_common_annotation_l1.translate_raw_data_analysis_to_model(
            annotation.raw_data_analysis
        ),
        processing_parameters=translate_common_annotation_l1.translate_processing_parameters_to_model(
            annotation.processing_parameters
        ),
        internal_calibration=translate_common_annotation_l1.translate_internal_calibration_to_model(
            annotation.internal_calibration
        ),
        rfi_mitigation=translate_common_annotation_l1.translate_rfi_mitigation_to_model(annotation.rfi_mitigation),
        doppler_parameters=translate_common_annotation_l1.translate_doppler_parameters_to_model(
            annotation.doppler_parameters
        ),
        radiometric_calibration=translate_common_annotation_l1.translate_radiometric_calibration_to_model(
            annotation.radiometric_calibration
        ),
        polarimetric_distortion=translate_common_annotation_l1.translate_polarimetric_distortion_to_model(
            annotation.polarimetric_distortion
        ),
        ionosphere_correction=translate_common_annotation_l1.translate_ionosphere_correction_to_model(
            annotation.ionosphere_correction
        ),
        geometry=translate_common_annotation_l1.translate_geometry_to_model(annotation.geometry),
        quality=translate_common_annotation_l1.translate_quality_to_model(annotation.quality),
        annotation_lut=translate_common.translate_layer_list_to_model(annotation.annotation_lut),
        sta_processing_parameters=translate_sta_processing_parameters_to_model(annotation.sta_processing_parameters),
        sta_coregistration_parameters=translate_sta_coregistration_parameters_to_model(
            annotation.sta_coregistration_parameters
        ),
        sta_in_sarparameters=translate_sta_in_sarparameters_to_model(annotation.sta_in_sarparameters),
        sta_quality=translate_sta_quality_to_model(annotation.sta_quality),
    )


def read_main_annotation_l1c(
    file: Path,
) -> main_annotation_l1c.MainAnnotationL1c:
    """Read main annotation of l1c products"""
    main_annotation_model: main_annotation_models_l1c.MainAnnotation = parse(
        file.read_text(encoding="utf-8"),
        main_annotation_models_l1c.MainAnnotation,
    )

    return translate_main_annotation_l1c(main_annotation_model)


def write_main_annotation_l1c(
    file: Path,
    annotation: main_annotation_l1c.MainAnnotationL1c,
):
    """Write main annotation of l1c products"""
    main_annotation_model = translate_main_annotation_l1c_to_model(annotation)
    main_annotation_text = serialize(main_annotation_model)
    file.write_text(main_annotation_text, encoding="utf-8")
