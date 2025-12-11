# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PPS Translation
-------------------
"""

import bps.common.io.common_types.models as common_models
import numpy as np
from bps.common.io import translate_common
from bps.stack_cal_processor.configuration import BaselineMethodType
from bps.stack_cal_processor.core.filtering import ConvolutionWindowType
from bps.stack_pre_processor.configuration import PrimaryImageSelectionConf
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
    AzimuthSpectralFilteringConf,
    CoregistrationConf,
    GeneralConf,
    InSarCalibrationConf,
    L1cProductExportConf,
    RfiDegradationEstimationConf,
    SkpPhaseCalibrationConf,
    SlowIonosphereRemovalConf,
)
from bps.stack_processor.io import aux_pps_models
from bps.transcoder.utils.polarization_conversions import translate_polarization_tag


class AuxPPSParsingError(RuntimeError):
    """Raised when AUX-PPS does not parse."""


class UnsupportedAuxPPSConfError(ValueError):
    """Handle unsupported AUX-PPS configurations."""


class InvalidBoolTagContent(RuntimeError):
    """Raised when input bool tag content is different from true or false."""


def translate_model_to_general_conf(
    conf: aux_pps_models.GeneralType,
) -> GeneralConf:
    """Translate general configuration section to the corresponding conf."""
    return GeneralConf(
        polarization_combination_method=translate_common.translate_polarisation_combination_method(
            conf.polarisation_combination_method,
        ),
        outer_parallelization_max_cores=conf.outer_parallelization_max_cores,
        allow_duplicate_images_flag=translate_common.translate_bool(conf.allow_duplicate_images_flag),
        flattening_phase_bias_compensation_flag=translate_common.translate_bool(
            conf.flattening_phase_bias_compensation_flag
        ),
    )


def translate_model_to_primary_image_selection_conf(
    conf: aux_pps_models.PrimaryImageSelectionType,
) -> PrimaryImageSelectionConf:
    """Translate primary image selection section to the corresponding conf."""
    return PrimaryImageSelectionConf(
        primary_image_selection_information=translate_common.translate_primary_image_selection_information(
            conf.primary_image_selection_information,
        ),
        rfi_decorrelation_threshold=conf.rfi_decorrelation_threshold,
        faraday_decorrelation_threshold=conf.faraday_decorrelation_threshold,
    )


def translate_model_to_coregistration_conf(
    conf: aux_pps_models.CoregistrationType,
) -> CoregistrationConf:
    """Translate internal calibration section to the corresponding conf."""
    return CoregistrationConf(
        coregistration_method=translate_common.translate_coregistration_method(conf.coregistration_method),
        range_spectral_filtering_flag=translate_common.translate_bool(conf.range_spectral_filtering_flag),
        residual_shift_quality_threshold=conf.residual_shifts_quality_threshold,
        polarization_used=translate_polarization_tag(
            conf.polarisation_used.value,
            poltype="arepytools",
        ),
        block_quality_threshold=conf.block_quality_threshold,
        fitting_quality_threshold=conf.fitting_quality_threshold,
        min_valid_blocks=conf.min_valid_blocks,
        azimuth_max_shift=conf.azimuth_max_shift,
        azimuth_block_size=conf.azimuth_block_size,
        azimuth_min_overlap=conf.azimuth_min_overlap,
        range_max_shift=conf.range_max_shift,
        range_block_size=conf.range_block_size,
        range_min_overlap=conf.range_min_overlap,
        model_based_fit_flag=translate_common.translate_bool(conf.model_based_fit_flag),
        low_pass_filter_type=conf.low_pass_filter_type.value,
        low_pass_filter_order=conf.low_pass_filter_order,
        low_pass_filter_std_dev=conf.low_pass_filter_std_dev,
        export_debug_products_flag=translate_common.translate_bool(conf.export_debug_products_flag),
        coregistration_execution_policy=translate_common.translate_coregistration_execution_policy(
            conf.coregistration_execution_policy
        ),
    )


def translate_model_to_rfi_degradation_estimation_conf(
    conf: aux_pps_models.RfiDegradationEstimationType,
) -> RfiDegradationEstimationConf:
    """Translate rfi degradation section to the corresponding conf."""
    return RfiDegradationEstimationConf(
        rfi_degradation_estimation_flag=translate_common.translate_bool(conf.rfi_degradation_estimation_flag),
    )


def translate_weighting_window(
    weighting_window: common_models.WeightingWindowType,
) -> ConvolutionWindowType:
    """Translate the convolution window type."""
    return ConvolutionWindowType(weighting_window.value.upper())


def translate_model_to_azimuth_spectral_filtering_conf(
    conf: aux_pps_models.AzimuthSpectralFilteringType,
) -> AzimuthSpectralFilteringConf:
    """Translate azimuth_spectral_filtering to the corresponding conf."""
    return AzimuthSpectralFilteringConf(
        azimuth_spectral_filtering_flag=translate_common.translate_bool(conf.azimuth_spectral_filtering_flag),
        use_primary_weighting_window_flag=translate_common.translate_bool(
            conf.use_primary_weighting_window_flag,
        ),
        spectral_weighting_window=translate_weighting_window(
            conf.spectral_weighting_window,
        ),
        spectral_weighting_window_parameter=conf.spectral_weighting_window_parameter,
        use_32bit_flag=translate_common.translate_bool(conf.use32bit_flag),
    )


def translate_baseline_method(
    baseline_method: common_models.BaselineMethodType,
) -> BaselineMethodType:
    """Translate AUX-PPS interface model to internal AUX-PPS model."""
    return BaselineMethodType(baseline_method.value.upper().replace("-", "_"))


def translate_model_to_slow_ionosphere_removal_conf(
    conf: aux_pps_models.SlowIonosphereRemovalType,
) -> SlowIonosphereRemovalConf:
    """Translate slow ionosphere removal section to the corresponding conf."""
    return SlowIonosphereRemovalConf(
        slow_ionosphere_removal_flag=translate_common.translate_bool(conf.slow_ionosphere_removal_flag),
        primary_image_flag=translate_common.translate_bool(conf.primary_image_flag),
        polarization_used=translate_polarization_tag(conf.polarisation_used.value, poltype="arepytools"),
        range_look_bandwidth=conf.range_look_bandwidth,
        range_look_frequency=conf.range_look_frequency,
        phase_unwrapping_flag=translate_common.translate_bool(conf.phase_unwrapping_flag),
        latitude_threshold=np.deg2rad(translate_common.translate_float_with_unit(conf.latitude_threshold)),
        baseline_method=translate_baseline_method(conf.baseline_method),
        multi_baseline_critical_baseline_threshold=conf.multi_baseline_critical_baseline_threshold,
        sublook_window_azimuth_size=conf.sublook_window_azimuth_size,
        sublook_window_range_size=conf.sublook_window_range_size,
        unweighted_multi_baseline_estimation=translate_common.translate_bool(conf.unweighted_multi_baseline_estimation),
        compensate_l1_iono_phase_screen_flag=translate_common.translate_bool(conf.compensate_l1_iono_phase_screen_flag),
        slow_ionosphere_quality_threshold=conf.slow_ionosphere_quality_threshold,
        min_coherence_threshold=conf.min_coherence_threshold,
        max_lh_phase_delta=translate_common.translate_float_with_unit(conf.max_delta_phase_unwrap_test),
        min_usable_pixel_ratio=conf.min_usable_pixel_ratio,
        use_32bit_flag=translate_common.translate_bool(conf.use32bit_flag),
    )


def translate_skp_phase_correction_type(
    phase_correction: aux_pps_models.SkpPhaseCorrectionType,
) -> SkpPhaseCalibrationConf.SkpPhaseCorrectionType:
    """Translate AUX-PPS interface model to internal AUX-PPS model."""
    return SkpPhaseCalibrationConf.SkpPhaseCorrectionType(phase_correction.value.upper())


def translate_model_to_skp_calibration_conf(
    conf: aux_pps_models.SkpPhaseCalibrationType,
) -> SkpPhaseCalibrationConf:
    """Translate skp calibration to the corresponding conf."""
    return SkpPhaseCalibrationConf(
        skp_phase_estimation_flag=translate_common.translate_bool(conf.skp_phase_estimation_flag),
        phase_correction=translate_skp_phase_correction_type(conf.phase_correction),
        estimation_window_size=translate_common.translate_float_with_unit(conf.estimation_window_size),
        skp_calibration_phase_screen_quality_threshold=conf.skp_calibration_phase_screen_quality_threshold,
        overall_product_quality_threshold=conf.overall_product_quality_threshold,
        median_filter_flag=translate_common.translate_bool(conf.median_filter_flag),
        median_filter_window_size=translate_common.translate_float_with_unit(conf.median_filter_window_size),
        exclude_mpmb_polarization_cross_covariance_flag=translate_common.translate_bool(
            conf.exclude_mpmbpolarization_cross_covariance_flag
        ),
        use_32bit_flag=translate_common.translate_bool(conf.use32bit_flag),
    )


def translate_compression_method(
    pixel_compression: common_models.CompressionMethodType,
) -> L1cProductExportConf.CompressionMethodType:
    return L1cProductExportConf.CompressionMethodType(pixel_compression.value)


def translate_model_to_l1c_product_export_conf(
    conf: aux_pps_models.L1CProductExportType,
) -> L1cProductExportConf:
    """Translate l1c product export section to the corresponding conf."""
    return L1cProductExportConf(
        l1_product_doi=conf.l1_product_doi,
        pixel_representation=translate_common.translate_pixel_representation_type(conf.pixel_representation),
        pixel_quantity=translate_common.translate_pixel_quantity_type(conf.pixel_quantity),
        abs_compression_method=translate_compression_method(conf.abs_compression_method),
        abs_max_z_error=conf.abs_max_zerror,
        phase_compression_method=translate_compression_method(conf.phase_compression_method),
        phase_max_z_error=conf.phase_max_zerror.value,
        no_pixel_value=conf.no_pixel_value,
        ql_range_decimation_factor=conf.ql_range_decimation_factor,
        ql_range_averaging_factor=conf.ql_range_averaging_factor,
        ql_azimuth_decimation_factor=conf.ql_azimuth_decimation_factor,
        ql_azimuth_averaging_factor=conf.ql_azimuth_averaging_factor,
        ql_absolute_scaling_factor=conf.ql_absolute_scaling_factor,
    )


def translate_model_to_in_sar_calibration_conf(
    conf: aux_pps_models.InSarcalibrationType,
) -> InSarCalibrationConf:
    """Translate InSAR calibration configuration to corresponding conf."""
    return InSarCalibrationConf(
        in_sar_calibration_flag=translate_common.translate_bool(
            conf.in_sarcalibration_flag,
        ),
        polarization_used=translate_polarization_tag(
            conf.polarisation_used.value,
            poltype="arepytools",
        ),
        fft2_zero_padding_upsampling_factor=conf.fft2_zero_padding_upsampling_factor,
        fft2_peak_window_size=conf.fft2_peak_window_size,
        use_32bit_flag=translate_common.translate_bool(
            conf.use32bit_flag,
        ),
    )


def translate_model_to_sta_product_conf(
    model: aux_pps_models.AuxiliaryStaprocessingParameters,
) -> AuxiliaryStaprocessingParameters:
    """Translate aux pps to the corresponding structure."""
    return AuxiliaryStaprocessingParameters(
        product_id=model.sta_product_list.sta_product.product_id,
        general=translate_model_to_general_conf(model.sta_product_list.sta_product.general),
        primary_image_selection=translate_model_to_primary_image_selection_conf(
            model.sta_product_list.sta_product.primary_image_selection
        ),
        coregistration=translate_model_to_coregistration_conf(model.sta_product_list.sta_product.coregistration),
        rfi_degradation_estimation=translate_model_to_rfi_degradation_estimation_conf(
            model.sta_product_list.sta_product.rfi_degradation_estimation
        ),
        azimuth_spectral_filtering=translate_model_to_azimuth_spectral_filtering_conf(
            model.sta_product_list.sta_product.azimuth_spectral_filtering,
        ),
        slow_ionosphere_removal=translate_model_to_slow_ionosphere_removal_conf(
            model.sta_product_list.sta_product.slow_ionosphere_removal
        ),
        in_sar_calibration=translate_model_to_in_sar_calibration_conf(
            model.sta_product_list.sta_product.in_sarcalibration,
        ),
        skp_phase_calibration=translate_model_to_skp_calibration_conf(
            model.sta_product_list.sta_product.skp_phase_calibration
        ),
        l1c_product_export=translate_model_to_l1c_product_export_conf(
            model.sta_product_list.sta_product.l1c_product_export
        ),
    )
