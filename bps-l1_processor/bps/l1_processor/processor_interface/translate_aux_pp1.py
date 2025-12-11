# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP1 Translation
-------------------
"""

from collections.abc import Callable
from typing import Any

from bps.common import Polarization, Swath
from bps.common.io import translate_common
from bps.l1_processor.io import aux_pp1_models
from bps.l1_processor.processor_interface.aux_pp1 import (
    AntennaPatternCorrectionConf,
    AutofocusConf,
    AuxProcessingParametersL1,
    AzimuthCompressionConf,
    ChirpSource,
    DopplerEstimationConf,
    GeneralConf,
    GroundProjectionConf,
    InternalCalibrationConf,
    IonosphereCalibrationConf,
    L0ProductImportConf,
    L1ProductExportConf,
    MultilookConf,
    PolarimetricCalibrationConf,
    RadiometricCalibrationConf,
    RangeCompressionConf,
    RawDataCorrectionConf,
    RFIMitigationConf,
    RGBChannel,
    ThermalDenoisingConf,
    WindowType,
)


class InvalidAuxPP1(RuntimeError):
    """Raised when input aux pp1 is invalid"""


class InvalidBoolTagContent(RuntimeError):
    """Raised when input bool tag content is different from true or false"""


def str_to_bool(tag: str) -> bool:
    """Safe string to bool tag content conversion"""
    tag = tag.lower()
    if tag == "true":
        return True
    if tag == "false":
        return False
    raise InvalidBoolTagContent(tag)


def translate_parameters_per_swath(swath_based_params: Any, translator: Callable) -> dict[Swath, Any]:
    """Translate swath-based parameter section"""
    assert swath_based_params.processing_parameters is not None
    swath_to_params = {}
    for parameters in swath_based_params.processing_parameters:
        assert parameters.swath is not None
        swath = Swath(parameters.swath.name)
        if swath in swath_to_params:
            raise InvalidAuxPP1(f"Duplicated parameters for swath {swath}")

        swath_to_params[swath] = translator(parameters)
    return swath_to_params


def translate_model_to_general_conf(
    conf: aux_pp1_models.GeneralType,
) -> GeneralConf:
    """Translate general configuration section to the corresponding conf"""
    assert conf.height_model is not None
    assert conf.height_model.value is not None
    assert conf.height_model.version is not None
    assert conf.height_model_margin is not None
    assert conf.height_model_margin.value is not None
    assert conf.parc_processing is not None
    assert conf.parc_processing.parc_roisamples is not None
    assert conf.parc_processing.parc_roilines is not None
    assert conf.dual_polarisation_processing_flag is not None
    return GeneralConf(
        requested_height_model=GeneralConf.EarthModel(conf.height_model.value.name),
        requested_height_model_version=conf.height_model.version,
        height_model=GeneralConf.EarthModel(conf.height_model.value.name),
        height_model_version=conf.height_model.version,
        height_model_margin=conf.height_model_margin.value,
        parc_roi_samples=conf.parc_processing.parc_roisamples,
        parc_roi_lines=conf.parc_processing.parc_roilines,
        dual_polarisation_processing_flag=str_to_bool(conf.dual_polarisation_processing_flag),
    )


def translate_model_to_l0_import_conf(
    conf: aux_pp1_models.L0ProductImportType,
) -> L0ProductImportConf:
    """Translate l0 import section to the corresponding conf"""
    assert conf.block_size is not None
    assert conf.max_ispgap is not None
    assert conf.raw_mean_expected is not None
    assert conf.raw_mean_threshold is not None
    assert conf.raw_std_expected is not None
    assert conf.raw_std_threshold is not None
    assert conf.internal_calibration_estimation_flag is not None
    return L0ProductImportConf(
        block_size=conf.block_size,
        max_isp_gap=conf.max_ispgap,
        raw_mean_expected=conf.raw_mean_expected,
        raw_mean_threshold=conf.raw_mean_threshold,
        raw_std_expected=conf.raw_std_expected,
        raw_std_threshold=conf.raw_std_threshold,
        internal_calibration_estimation_flag=str_to_bool(conf.internal_calibration_estimation_flag),
    )


def translate_model_to_raw_data_correction_conf(
    conf: aux_pp1_models.RawDataCorrectionType,
) -> RawDataCorrectionConf:
    """Translate raw data correction section to the corresponding conf"""
    assert conf.bias_correction_flag is not None
    assert conf.gain_imbalance_correction_flag is not None
    assert conf.non_orthogonality_correction_flag is not None
    assert conf.raw_data_correction_flag is not None
    return RawDataCorrectionConf(
        bias_correction_flag=str_to_bool(conf.bias_correction_flag),
        gain_imbalance_correction_flag=str_to_bool(conf.gain_imbalance_correction_flag),
        non_orthogonality_correction_flag=str_to_bool(conf.non_orthogonality_correction_flag),
        raw_data_correction_flag=str_to_bool(conf.raw_data_correction_flag),
    )


def translate_model_to_internal_calibration_conf(
    conf: aux_pp1_models.InternalCalibrationCorrectionType,
) -> InternalCalibrationConf:
    """Translate internal calibration section to the corresponding conf"""
    assert conf.internal_calibration_correction_flag is not None
    assert conf.drift_correction_flag is not None
    assert conf.delay_correction_flag is not None
    assert conf.channel_imbalance_correction_flag is not None
    assert conf.internal_calibration_source is not None
    assert conf.max_drift_amplitude_std_fraction is not None
    assert conf.max_drift_phase_std_fraction is not None
    assert conf.max_drift_amplitude_error is not None
    assert conf.max_drift_phase_error is not None
    assert conf.max_drift_phase_error.value is not None
    assert conf.max_invalid_drift_fraction is not None
    return InternalCalibrationConf(
        internal_calibration_correction_flag=str_to_bool(conf.internal_calibration_correction_flag),
        drift_correction_flag=str_to_bool(conf.drift_correction_flag),
        delay_correction_flag=str_to_bool(conf.delay_correction_flag),
        channel_imbalance_correction_flag=str_to_bool(conf.channel_imbalance_correction_flag),
        internal_calibration_source=InternalCalibrationConf.Source(conf.internal_calibration_source.name),
        max_drift_amplitude_std_fraction=conf.max_drift_amplitude_std_fraction,
        max_drift_phase_std_fraction=conf.max_drift_phase_std_fraction,
        max_drift_amplitude_error=conf.max_drift_amplitude_error,
        max_drift_phase_error=conf.max_drift_phase_error.value,
        max_invalid_drift_fraction=conf.max_invalid_drift_fraction,
    )


def translate_model_to_rfi_mitigation_conf(
    conf: aux_pp1_models.RfiMitigationType,
) -> RFIMitigationConf:
    """Translate rfi mitigation section to the corresponding conf"""
    assert conf.rfi_detection_flag is not None
    assert conf.rfi_mitigation_mode is not None
    assert conf.rfi_activation_mask_threshold is not None
    assert conf.rfi_mitigation_method is not None
    assert conf.rfi_mask is not None
    assert conf.rfi_mask_generation_method is not None
    assert conf.rfi_tmprocessing_parameters is not None
    assert conf.rfi_tmprocessing_parameters.block_lines is not None
    assert conf.rfi_tmprocessing_parameters.median_filter_length is not None
    assert conf.rfi_tmprocessing_parameters.box_samples is not None
    assert conf.rfi_tmprocessing_parameters.box_lines is not None
    assert conf.rfi_tmprocessing_parameters.percentile_threshold is not None
    assert conf.rfi_tmprocessing_parameters.morphological_open_operator_samples is not None
    assert conf.rfi_tmprocessing_parameters.morphological_open_operator_lines is not None
    assert conf.rfi_tmprocessing_parameters.morphological_close_operator_samples is not None
    assert conf.rfi_tmprocessing_parameters.morphological_close_operator_lines is not None
    assert conf.rfi_tmprocessing_parameters.max_rfitmpercentage is not None
    assert conf.rfi_fmprocessing_parameters is not None
    assert conf.rfi_fmprocessing_parameters.block_lines is not None
    assert conf.rfi_fmprocessing_parameters.block_overlap is not None
    assert conf.rfi_fmprocessing_parameters.persistent_rfithreshold is not None
    assert conf.rfi_fmprocessing_parameters.isolated_rfithreshold is not None
    assert conf.rfi_fmprocessing_parameters.isolated_rfipsdstd_threshold is not None
    assert conf.rfi_fmprocessing_parameters.max_rfifmpercentage is not None
    assert conf.rfi_fmprocessing_parameters.periodgram_size is not None
    assert conf.rfi_fmprocessing_parameters.enable_power_loss_compensation is not None
    assert conf.rfi_fmprocessing_parameters.power_loss_threshold is not None
    assert conf.rfi_fmprocessing_parameters.chirp_source is not None
    assert conf.rfi_fmprocessing_parameters.mitigation_method is not None

    return RFIMitigationConf(
        detection_flag=str_to_bool(conf.rfi_detection_flag),
        activation_mode=conf.rfi_mitigation_mode.value,
        activation_mask_threshold=conf.rfi_activation_mask_threshold,
        mitigation_method=RFIMitigationConf.MitigationMethod(conf.rfi_mitigation_method.name),
        mask=RFIMitigationConf.MaskType(conf.rfi_mask.name),
        mask_generation_method=RFIMitigationConf.MaskGenerationMethod(conf.rfi_mask_generation_method.value),
        time_domain_processing_parameters=RFIMitigationConf.TimeDomainParams(
            block_lines=conf.rfi_tmprocessing_parameters.block_lines,
            median_filter_length=conf.rfi_tmprocessing_parameters.median_filter_length,
            box_samples=conf.rfi_tmprocessing_parameters.box_samples,
            box_lines=conf.rfi_tmprocessing_parameters.box_lines,
            percentile_threshold=conf.rfi_tmprocessing_parameters.percentile_threshold,
            morphological_open_operator_samples=conf.rfi_tmprocessing_parameters.morphological_open_operator_samples,
            morphological_open_operator_lines=conf.rfi_tmprocessing_parameters.morphological_open_operator_lines,
            morphological_close_operator_samples=conf.rfi_tmprocessing_parameters.morphological_close_operator_samples,
            morphological_close_operator_lines=conf.rfi_tmprocessing_parameters.morphological_close_operator_lines,
            max_rfi_percentage=conf.rfi_tmprocessing_parameters.max_rfitmpercentage,
        ),
        freq_domain_processing_parameters=RFIMitigationConf.FreqDomainParams(
            block_lines=conf.rfi_fmprocessing_parameters.block_lines,
            block_overlap=conf.rfi_fmprocessing_parameters.block_overlap,
            persistent_rfi_threshold=conf.rfi_fmprocessing_parameters.persistent_rfithreshold,
            isolated_rfi_threshold=conf.rfi_fmprocessing_parameters.isolated_rfithreshold,
            isolated_rfi_psd_std_threshold=conf.rfi_fmprocessing_parameters.isolated_rfipsdstd_threshold,
            max_rfi_percentage=conf.rfi_fmprocessing_parameters.max_rfifmpercentage,
            periodgram_size=conf.rfi_fmprocessing_parameters.periodgram_size,
            enable_power_loss_compensation=conf.rfi_fmprocessing_parameters.enable_power_loss_compensation,
            power_loss_threshold=conf.rfi_fmprocessing_parameters.power_loss_threshold,
            chirp_source=ChirpSource(conf.rfi_fmprocessing_parameters.chirp_source.name),
            mitigation_method=conf.rfi_fmprocessing_parameters.mitigation_method.value,
        ),
    )


def translate_model_to_range_compression_params(
    params: aux_pp1_models.L1AProcessingParametersType,
) -> RangeCompressionConf.Parameters:
    """Translate range focuser parameters section to the corresponding conf"""
    assert params.time_bias is not None
    assert params.time_bias.value is not None
    assert params.window_type is not None
    assert params.window_coefficient is not None
    assert params.processing_bandwidth is not None
    assert params.processing_bandwidth.value is not None

    return RangeCompressionConf.Parameters(
        time_bias=params.time_bias.value,
        window_type=WindowType(params.window_type.name),
        window_coefficient=params.window_coefficient,
        processing_bandwidth=params.processing_bandwidth.value,
    )


def translate_model_to_range_compression_conf(
    conf: aux_pp1_models.RangeCompressionType,
) -> RangeCompressionConf:
    """Translate range focuser section to the corresponding conf"""
    assert conf.range_reference_function_source is not None
    assert conf.range_compression_method is not None
    assert conf.extended_swath_processing_flag is not None
    assert conf.range_processing_parameters_list is not None

    swath_to_parameters = translate_parameters_per_swath(
        conf.range_processing_parameters_list,
        translate_model_to_range_compression_params,
    )

    return RangeCompressionConf(
        range_reference_function_source=ChirpSource(conf.range_reference_function_source.name),
        range_compression_method=RangeCompressionConf.Method(conf.range_compression_method.name),
        extended_swath_processing=str_to_bool(conf.extended_swath_processing_flag),
        parameters=swath_to_parameters,
    )


def translate_model_to_doppler_estimation_conf(
    conf: aux_pp1_models.DopplerEstimationType,
) -> DopplerEstimationConf:
    """Translate doppler estimation section to the corresponding conf"""
    assert conf.dc_method is not None
    assert conf.dc_method.value is not None
    assert conf.dc_value is not None
    assert conf.dc_value.value is not None
    assert conf.block_samples is not None
    assert conf.block_lines is not None
    assert conf.polynomial_update_rate is not None
    assert conf.polynomial_update_rate.value is not None
    assert conf.dc_rmserror_threshold is not None
    assert conf.dc_rmserror_threshold.value is not None
    return DopplerEstimationConf(
        method=DopplerEstimationConf.Method(conf.dc_method.name),
        value=conf.dc_value.value,
        block_samples=conf.block_samples,
        block_lines=conf.block_lines,
        polynomial_update_rate=conf.polynomial_update_rate.value,
        rms_error_threshold=conf.dc_rmserror_threshold.value,
    )


def translate_model_to_antenna_pattern_correction_conf(
    conf: aux_pp1_models.AntennaPatternCorrectionType,
) -> AntennaPatternCorrectionConf:
    """Translate antenna pattern compensation section to the corresponding conf"""
    assert conf.antenna_cross_talk_correction_flag is not None
    assert conf.antenna_pattern_correction1_flag is not None
    assert conf.antenna_pattern_correction2_flag is not None
    assert conf.elevation_mispointing_bias is not None
    assert conf.elevation_mispointing_bias.value is not None
    assert conf.azimuth_mispointing_bias is not None
    assert conf.azimuth_mispointing_bias.value is not None
    return AntennaPatternCorrectionConf(
        antenna_pattern_correction1_flag=str_to_bool(conf.antenna_pattern_correction1_flag),
        antenna_pattern_correction2_flag=str_to_bool(conf.antenna_pattern_correction2_flag),
        antenna_cross_talk_correction_flag=str_to_bool(conf.antenna_cross_talk_correction_flag),
        elevation_mispointing_bias=conf.elevation_mispointing_bias.value,
        azimuth_mispointing_bias=conf.azimuth_mispointing_bias.value,
    )


def translate_model_to_azimuth_compression_params(
    params: aux_pp1_models.L1AProcessingParametersType,
) -> AzimuthCompressionConf.Parameters:
    """Translate azimuth parameters section to the corresponding conf"""
    assert params.time_bias is not None
    assert params.time_bias.value is not None
    assert params.window_type is not None
    assert params.window_coefficient is not None
    assert params.processing_bandwidth is not None
    assert params.processing_bandwidth.value is not None
    return AzimuthCompressionConf.Parameters(
        time_bias=params.time_bias.value,
        window_type=WindowType(params.window_type.name),
        window_coefficient=params.window_coefficient,
        processing_bandwidth=params.processing_bandwidth.value,
    )


def translate_model_to_azimuth_compression_conf(
    conf: aux_pp1_models.AzimuthCompressionType,
) -> AzimuthCompressionConf:
    """Translate azimuth configuration section to the corresponding conf"""
    assert conf.block_samples is not None
    assert conf.block_lines is not None
    assert conf.block_overlap_samples is not None
    assert conf.block_overlap_lines is not None
    assert conf.azimuth_processing_parameters_list is not None
    assert conf.bistatic_delay_correction_flag is not None
    assert conf.bistatic_delay_correction_method is not None
    assert conf.azimuth_resampling_flag is not None
    assert conf.azimuth_resampling_frequency is not None
    assert conf.azimuth_resampling_frequency.value is not None
    assert conf.azimuth_focusing_margins_removal_flag is not None
    assert conf.azimuth_coregistration_flag is not None
    assert conf.filter_type is not None
    assert conf.filter_bandwidth is not None
    assert conf.filter_bandwidth.value is not None
    assert conf.filter_length is not None
    assert conf.number_of_filters is not None

    swath_to_parameters = translate_parameters_per_swath(
        conf.azimuth_processing_parameters_list,
        translate_model_to_azimuth_compression_params,
    )

    return AzimuthCompressionConf(
        block_samples=conf.block_samples,
        block_lines=conf.block_lines,
        block_overlap_samples=conf.block_overlap_samples,
        block_overlap_lines=conf.block_overlap_lines,
        parameters=swath_to_parameters,
        bistatic_delay_correction=str_to_bool(conf.bistatic_delay_correction_flag),
        bistatic_delay_correction_method=AzimuthCompressionConf.Method(conf.bistatic_delay_correction_method.name),
        azimuth_resampling=str_to_bool(conf.azimuth_resampling_flag),
        azimuth_resampling_frequency=conf.azimuth_resampling_frequency.value,
        azimuth_focusing_margins_removal_flag=str_to_bool(conf.azimuth_focusing_margins_removal_flag),
        azimuth_coregistration_flag=str_to_bool(conf.azimuth_coregistration_flag),
        filter_type=conf.filter_type.name,
        filter_bandwidth=conf.filter_bandwidth.value,
        filter_length=conf.filter_length,
        number_of_filters=conf.number_of_filters,
    )


def translate_model_to_radiometric_calibration_conf(
    conf: aux_pp1_models.RadiometricCalibrationType,
) -> RadiometricCalibrationConf:
    """Translate radiometric calibration section to the corresponding conf"""
    assert conf.absolute_calibration_constant_list is not None
    assert conf.processing_gain_list is not None
    assert conf.reference_range is not None
    assert conf.reference_range.value is not None
    assert conf.range_spreading_loss_compensation_flag is not None

    polarization_to_cal_const = {}

    for absolute_calibration_constant in conf.absolute_calibration_constant_list.absolute_calibration_constant:
        assert absolute_calibration_constant.value is not None
        assert absolute_calibration_constant.polarisation is not None
        polarization = Polarization(absolute_calibration_constant.polarisation.name)

        if polarization in polarization_to_cal_const:
            raise InvalidAuxPP1(
                "Absolute calibration constant for "
                + f"polarization {absolute_calibration_constant.polarisation} defined multiple times"
            )
        polarization_to_cal_const[polarization] = absolute_calibration_constant.value

    polarization_to_gain = {}

    for processing_gain in conf.processing_gain_list.processing_gain:
        assert processing_gain.value is not None
        assert processing_gain.polarisation is not None
        polarization = Polarization(processing_gain.polarisation.name)

        if polarization in polarization_to_gain:
            raise InvalidAuxPP1(
                f"Processing gain for polarization {processing_gain.polarisation} defined multiple times"
            )
        polarization_to_gain[polarization] = processing_gain.value

    return RadiometricCalibrationConf(
        absolute_calibration_constant=polarization_to_cal_const,
        processing_gain=polarization_to_gain,
        reference_range=conf.reference_range.value,
        range_spreading_loss_compensation_enabled=str_to_bool(conf.range_spreading_loss_compensation_flag),
    )


def translate_model_to_multilook_params(
    params: aux_pp1_models.L1BProcessingParametersType,
) -> MultilookConf.Parameters:
    """Translate multilooker parameters section to the corresponding conf"""
    assert params.window_type is not None
    assert params.window_coefficient is not None
    assert params.look_bandwidth is not None
    assert params.look_bandwidth.value is not None
    assert params.number_of_looks is not None
    assert params.look_central_frequencies is not None
    assert params.look_central_frequencies.value is not None
    assert params.upsampling_factor is not None
    assert params.downsampling_factor is not None
    return MultilookConf.Parameters(
        window_type=WindowType(params.window_type.name),
        window_coefficient=params.window_coefficient,
        look_bandwidth=params.look_bandwidth.value,
        number_of_looks=params.number_of_looks,
        look_central_frequencies=[float(p) for p in params.look_central_frequencies.value.split()],
        upsampling_factor=params.upsampling_factor,
        downsampling_factor=params.downsampling_factor,
    )


def translate_model_to_multilook_conf(
    conf: aux_pp1_models.MultilookType,
) -> MultilookConf:
    """Translate multilooker section to the corresponding conf"""
    assert conf.detection_flag is not None
    assert conf.azimuth_processing_parameters_list is not None
    assert conf.range_processing_parameters_list is not None

    az_params = translate_parameters_per_swath(
        conf.azimuth_processing_parameters_list,
        translate_model_to_multilook_params,
    )

    rg_params = translate_parameters_per_swath(
        conf.range_processing_parameters_list,
        translate_model_to_multilook_params,
    )

    return MultilookConf(
        apply_detection=str_to_bool(conf.detection_flag),
        range_parameters=rg_params,
        azimuth_parameters=az_params,
    )


def translate_model_to_polarimetric_calibration_conf(
    conf: aux_pp1_models.PolarimetricCalibrationType,
) -> PolarimetricCalibrationConf:
    """Translate aux pp1 polarimetric calibration to the corresponding conf"""
    assert conf.polarimetric_correction_flag is not None
    assert conf.tx_distortion_matrix_correction_flag is not None

    assert conf.rx_distortion_matrix_correction_flag is not None
    assert conf.cross_talk_correction_flag is not None
    assert conf.cross_talk_list is not None
    assert conf.channel_imbalance_correction_flag is not None
    assert conf.channel_imbalance_list is not None

    return PolarimetricCalibrationConf(
        polarimetric_correction_flag=str_to_bool(conf.polarimetric_correction_flag),
        tx_distortion_matrix_correction_flag=str_to_bool(conf.tx_distortion_matrix_correction_flag),
        rx_distortion_matrix_correction_flag=str_to_bool(conf.rx_distortion_matrix_correction_flag),
        cross_talk_correction_flag=str_to_bool(conf.cross_talk_correction_flag),
        cross_talk=translate_common.translate_cross_talk_list(conf.cross_talk_list),
        channel_imbalance_correction_flag=str_to_bool(conf.channel_imbalance_correction_flag),
        channel_imbalance=translate_common.translate_channel_imbalance_list(conf.channel_imbalance_list),
    )


def translate_model_to_ionospheric_calibration_conf(
    conf: aux_pp1_models.IonosphereCalibrationType,
) -> IonosphereCalibrationConf:
    """Translate ionosphere calibration to the corresponding conf"""
    assert conf.block_lines is not None
    assert conf.block_overlap_lines is not None
    assert conf.ionosphere_height_defocusing_flag is not None
    assert conf.ionosphere_height_estimation_method is not None
    assert conf.ionosphere_height_value is not None
    assert conf.ionosphere_height_value.value is not None
    assert conf.ionosphere_height_estimation_method_latitude_threshold is not None
    assert conf.ionosphere_height_estimation_method_latitude_threshold.value is not None
    assert conf.ionosphere_height_minimum_value is not None
    assert conf.ionosphere_height_minimum_value.value is not None
    assert conf.ionosphere_height_maximum_value is not None
    assert conf.ionosphere_height_maximum_value.value is not None
    assert conf.squint_sensitivity_number_of_looks is not None
    assert conf.squint_sensitivity_number_of_ticks is not None
    assert conf.squint_sensitivity_fitting_error is not None
    assert conf.gaussian_filter_maximum_major_axis_length is not None
    assert conf.gaussian_filter_maximum_minor_axis_length is not None
    assert conf.gaussian_filter_major_axis_length is not None
    assert conf.gaussian_filter_minor_axis_length is not None
    assert conf.gaussian_filter_slope is not None
    assert conf.faraday_rotation_correction_flag is not None
    assert conf.ionospheric_phase_screen_correction_flag is not None
    assert conf.group_delay_correction_flag is not None

    ionosphere_height_estimation_method_latitude_threshold = (
        conf.ionosphere_height_estimation_method_latitude_threshold.value
    )
    return IonosphereCalibrationConf(
        block_lines=conf.block_lines,
        block_overlap_lines=conf.block_overlap_lines,
        ionosphere_height_defocusing_flag=str_to_bool(conf.ionosphere_height_defocusing_flag),
        ionosphere_height_estimation_method=IonosphereCalibrationConf.Method(
            conf.ionosphere_height_estimation_method.name
        ),
        ionosphere_height_value=conf.ionosphere_height_value.value,
        ionosphere_height_estimation_method_latitude_threshold=ionosphere_height_estimation_method_latitude_threshold,
        ionosphere_height_minimum_value=conf.ionosphere_height_minimum_value.value,
        ionosphere_height_maximum_value=conf.ionosphere_height_maximum_value.value,
        squint_sensitivity_number_of_looks=conf.squint_sensitivity_number_of_looks,
        squint_sensitivity_number_of_ticks=conf.squint_sensitivity_number_of_ticks,
        squint_sensitivity_fitting_error=[float(p) for p in conf.squint_sensitivity_fitting_error.value.split()],
        gaussian_filter_maximum_major_axis_length=conf.gaussian_filter_maximum_major_axis_length,
        gaussian_filter_maximum_minor_axis_length=conf.gaussian_filter_maximum_minor_axis_length,
        gaussian_filter_major_axis_length=conf.gaussian_filter_major_axis_length,
        gaussian_filter_minor_axis_length=conf.gaussian_filter_minor_axis_length,
        gaussian_filter_slope=conf.gaussian_filter_slope,
        faraday_rotation_correction_flag=str_to_bool(conf.faraday_rotation_correction_flag),
        ionospheric_phase_screen_correction_flag=str_to_bool(conf.ionospheric_phase_screen_correction_flag),
        group_delay_correction_flag=str_to_bool(conf.group_delay_correction_flag),
    )


def translate_model_to_autofocus_conf(
    conf: aux_pp1_models.AutofocusType,
) -> AutofocusConf:
    """Translate autofocus configuration to the corresponding conf"""
    assert conf.autofocus_flag is not None
    assert conf.autofocus_method is not None
    assert conf.autofocus_method.name is not None
    assert conf.map_drift_azimuth_sub_bands is not None
    assert conf.map_drift_correlation_window_width is not None
    assert conf.map_drift_correlation_window_height is not None
    assert conf.map_drift_range_correlation_windows is not None
    assert conf.map_drift_azimuth_correlation_windows is not None
    assert conf.max_valid_shift is not None
    assert conf.valid_blocks_percentage is not None

    return AutofocusConf(
        autofocus_flag=str_to_bool(conf.autofocus_flag),
        autofocus_method=AutofocusConf.Method(conf.autofocus_method.name),
        map_drift_azimuth_sub_bands=conf.map_drift_azimuth_sub_bands,
        map_drift_correlation_window_width=conf.map_drift_correlation_window_width,
        map_drift_correlation_window_height=conf.map_drift_correlation_window_height,
        map_drift_range_correlation_windows=conf.map_drift_range_correlation_windows,
        map_drift_azimuth_correlation_windows=conf.map_drift_azimuth_correlation_windows,
        max_valid_shift=conf.max_valid_shift,
        valid_blocks_percentage=conf.valid_blocks_percentage,
    )


def translate_model_to_thermal_denoising_conf(
    conf: aux_pp1_models.ThermalDenoisingType,
) -> ThermalDenoisingConf:
    """Translate thermal denoising configuration to the corresponding conf"""
    assert conf.thermal_denoising_flag is not None
    assert conf.noise_parameters_source is not None
    assert conf.noise_equivalent_echoes_flag is not None
    assert conf.noise_gain_list is not None

    polarization_to_gain = {}

    for processing_gain in conf.noise_gain_list.noise_gain:
        assert processing_gain.value is not None
        assert processing_gain.polarisation is not None
        polarization = Polarization(processing_gain.polarisation.name)

        if polarization in polarization_to_gain:
            raise InvalidAuxPP1(
                f"Processing gains for polarizaiton {processing_gain.polarisation} defined multiple times"
            )
        polarization_to_gain[polarization] = processing_gain.value

    return ThermalDenoisingConf(
        thermal_denoising_flag=str_to_bool(conf.thermal_denoising_flag),
        noise_parameters_source=ThermalDenoisingConf.Source(conf.noise_parameters_source.name),
        noise_equivalent_echoes_flag=str_to_bool(conf.noise_equivalent_echoes_flag),
        noise_gain_list=polarization_to_gain,
    )


def translate_model_to_ground_projection(
    conf: aux_pp1_models.GroundProjectionType,
) -> GroundProjectionConf:
    """Translate ground projection configuration to the corresponding conf"""
    assert conf.ground_projection_flag is not None
    assert conf.range_pixel_spacing is not None
    assert conf.range_pixel_spacing.value is not None
    assert conf.filter_type is not None
    assert conf.filter_bandwidth is not None
    assert conf.filter_bandwidth.value is not None
    assert conf.filter_length is not None
    assert conf.number_of_filters is not None
    return GroundProjectionConf(
        ground_projection_flag=str_to_bool(conf.ground_projection_flag),
        range_pixel_spacing=conf.range_pixel_spacing.value,
        filter_type=GroundProjectionConf.FilterType(conf.filter_type.upper()),
        filter_bandwidth=conf.filter_bandwidth.value,
        filter_length=conf.filter_length,
        number_of_filters=conf.number_of_filters,
    )


def translate_lut_decimation_factors(
    factors: aux_pp1_models.LutDecimationFactorListType,
) -> L1ProductExportConf.LutDecimationFactors:
    """Translate the LUT decimation factors"""

    factors_by_group: dict[aux_pp1_models.GroupType, int] = {}
    for factor in factors.lut_decimation_factor:
        assert factor.group is not None and factor.value is not None
        factors_by_group[factor.group] = factor.value

    return L1ProductExportConf.LutDecimationFactors(
        dem_based_quantity=factors_by_group[aux_pp1_models.GroupType.DEM_BASED_LUT],
        rfi_based_quantity=factors_by_group[aux_pp1_models.GroupType.RFI_BASED_LUT],
        image_based_quantity=factors_by_group[aux_pp1_models.GroupType.IMAGE_BASED_LUT],
    )


def translate_model_to_l1_product_export_conf(
    conf: aux_pp1_models.L1ProductExportType,
) -> L1ProductExportConf:
    """Translate l1 product export configuration to the corresponding conf"""
    assert conf.l1a_product_doi is not None
    assert conf.l1b_product_doi is not None
    assert conf.pixel_representation is not None
    assert conf.pixel_quantity is not None
    assert conf.abs_compression_method is not None
    assert conf.abs_max_zerror is not None
    assert conf.abs_max_zerror_percentile is not None
    assert conf.phase_compression_method is not None
    assert conf.phase_max_zerror is not None
    assert conf.phase_max_zerror_percentile is not None
    assert conf.no_pixel_value is not None
    assert conf.block_size is not None
    assert conf.lut_range_decimation_factor_list is not None
    assert conf.lut_azimuth_decimation_factor_list is not None
    assert conf.lut_block_size is not None
    assert conf.lut_layers_completeness_flag is not None
    assert conf.ql_range_averaging_factor is not None
    assert conf.ql_range_decimation_factor is not None
    assert conf.ql_azimuth_averaging_factor is not None
    assert conf.ql_azimuth_decimation_factor is not None
    assert conf.ql_absolute_scaling_factor_list is not None

    rgbchannel_to_gain = {}
    for rgb_gain in conf.ql_absolute_scaling_factor_list.ql_absolute_scaling_factor:
        assert rgb_gain.value is not None
        assert rgb_gain.channel is not None
        channel = RGBChannel(rgb_gain.channel.name)

        if channel in rgbchannel_to_gain:
            raise InvalidAuxPP1(f"Scaling factors for RGB channel {rgb_gain.channel} defined multiple times")
        rgbchannel_to_gain[channel] = rgb_gain.value

    return L1ProductExportConf(
        l1a_product_doi=conf.l1a_product_doi,
        l1b_product_doi=conf.l1b_product_doi,
        pixel_representation=translate_common.translate_pixel_representation_type(conf.pixel_representation),
        pixel_quantity=translate_common.translate_pixel_quantity_type(conf.pixel_quantity),
        abs_compression_method=L1ProductExportConf.CompressionMethodType(conf.abs_compression_method.name),
        abs_max_zerror=conf.abs_max_zerror,
        abs_max_zerror_percentile=conf.abs_max_zerror_percentile,
        phase_compression_method=L1ProductExportConf.CompressionMethodType(conf.phase_compression_method.name),
        phase_max_zerror=conf.phase_max_zerror.value,
        phase_max_zerror_percentile=conf.phase_max_zerror_percentile,
        no_pixel_value=conf.no_pixel_value,
        block_size=conf.block_size,
        lut_range_decimation_factor=translate_lut_decimation_factors(conf.lut_range_decimation_factor_list),
        lut_azimuth_decimation_factor=translate_lut_decimation_factors(conf.lut_azimuth_decimation_factor_list),
        lut_block_size=conf.lut_block_size,
        lut_layers_completeness_flag=str_to_bool(conf.lut_layers_completeness_flag),
        ql_range_averaging_factor=conf.ql_range_averaging_factor,
        ql_range_decimation_factor=conf.ql_range_decimation_factor,
        ql_azimuth_averaging_factor=conf.ql_azimuth_averaging_factor,
        ql_azimuth_decimation_factor=conf.ql_azimuth_decimation_factor,
        ql_absolute_scaling_factor_list=rgbchannel_to_gain,
    )


def translate_model_to_aux_processing_parameters_l1(
    model: aux_pp1_models.AuxiliaryL1ProcessingParameters,
) -> AuxProcessingParametersL1:
    """Translate aux pp1 to the corresponding structure"""

    assert model.l1_product_list is not None
    params = model.l1_product_list.l1_product

    assert params is not None
    assert params.product_id is not None
    assert params.general is not None
    assert params.l0_product_import is not None
    assert params.raw_data_correction is not None
    assert params.internal_calibration_correction is not None
    assert params.rfi_mitigation is not None
    assert params.range_compression is not None
    assert params.doppler_estimation is not None
    assert params.antenna_pattern_correction is not None
    assert params.azimuth_compression is not None
    assert params.radiometric_calibration is not None
    assert params.polarimetric_calibration is not None
    assert params.ionosphere_calibration is not None
    assert params.autofocus is not None
    assert params.multilook is not None
    assert params.thermal_denoising is not None
    assert params.ground_projection is not None
    assert params.l1_product_export is not None

    return AuxProcessingParametersL1(
        product_id=params.product_id,
        general=translate_model_to_general_conf(params.general),
        l0_product_import=translate_model_to_l0_import_conf(params.l0_product_import),
        raw_data_correction=translate_model_to_raw_data_correction_conf(params.raw_data_correction),
        rfi_mitigation=translate_model_to_rfi_mitigation_conf(params.rfi_mitigation),
        internal_calibration_correction=translate_model_to_internal_calibration_conf(
            params.internal_calibration_correction
        ),
        range_compression=translate_model_to_range_compression_conf(params.range_compression),
        doppler_estimation=translate_model_to_doppler_estimation_conf(params.doppler_estimation),
        antenna_pattern_correction=translate_model_to_antenna_pattern_correction_conf(
            params.antenna_pattern_correction
        ),
        azimuth_compression=translate_model_to_azimuth_compression_conf(params.azimuth_compression),
        radiometric_calibration=translate_model_to_radiometric_calibration_conf(params.radiometric_calibration),
        polarimetric_calibration=translate_model_to_polarimetric_calibration_conf(params.polarimetric_calibration),
        ionosphere_calibration=translate_model_to_ionospheric_calibration_conf(params.ionosphere_calibration),
        autofocus=translate_model_to_autofocus_conf(params.autofocus),
        multilook=translate_model_to_multilook_conf(params.multilook),
        thermal_denoising=translate_model_to_thermal_denoising_conf(params.thermal_denoising),
        ground_projection=translate_model_to_ground_projection(params.ground_projection),
        l1_product_export=translate_model_to_l1_product_export_conf(params.l1_product_export),
    )
