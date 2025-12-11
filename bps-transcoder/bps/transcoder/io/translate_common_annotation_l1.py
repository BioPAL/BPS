# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Common annotations l1 translate
-------------------------------
"""

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common, common_types, translate_common
from bps.transcoder.io import common_annotation_l1, common_annotation_models_l1
from bps.transcoder.utils.production_model_utils import (
    translate_global_coverage_id,
    translate_major_cycle_id,
    translate_repeat_cycle_id,
)


def translate_noise_gain_list(
    noise_gain_list: common_annotation_models_l1.NoiseGainListType,
) -> common_annotation_l1.NoiseGainList:
    """Translate noise gain list"""
    assert noise_gain_list.count == len(noise_gain_list.noise_gain)
    return [translate_common.translate_float_with_polarisation(gain) for gain in noise_gain_list.noise_gain]


def translate_noise_gain_list_to_model(
    noise_gain_list: common_annotation_l1.NoiseGainList,
) -> common_annotation_models_l1.NoiseGainListType:
    """Translate noise gain list"""
    return common_annotation_models_l1.NoiseGainListType(
        noise_gain=[translate_common.translate_float_with_polarisation_to_model(gain) for gain in noise_gain_list],
        count=len(noise_gain_list),
    )


def translate_processing_gain_list(
    processing_gain_list: common_annotation_models_l1.ProcessingGainListType,
) -> common_annotation_l1.ProcessingGainList:
    """Translate processing gain list"""
    assert processing_gain_list.count == len(processing_gain_list.processing_gain)
    return [translate_common.translate_float_with_polarisation(gain) for gain in processing_gain_list.processing_gain]


def translate_processing_gain_list_to_model(
    processing_gain_list: common_annotation_l1.ProcessingGainList,
) -> common_annotation_models_l1.ProcessingGainListType:
    """Translate processing gain list"""
    return common_annotation_models_l1.ProcessingGainListType(
        processing_gain=[
            translate_common.translate_float_with_polarisation_to_model(gain) for gain in processing_gain_list
        ],
        count=len(processing_gain_list),
    )


def translate_spectrum_processing_parameters_type(
    params: common_annotation_models_l1.SpectrumProcessingParametersType,
) -> common_annotation_l1.SpectrumProcessingParametersType:
    """Translate spectrum proc params"""
    assert params.window_type is not None
    assert params.window_coefficient is not None
    assert params.total_bandwidth is not None
    assert params.processing_bandwidth is not None
    assert params.look_bandwidth is not None
    assert params.number_of_looks is not None
    assert params.look_overlap is not None
    return common_annotation_l1.SpectrumProcessingParametersType(
        window_type=translate_common.translate_weighting_window_type(params.window_type),
        window_coefficient=params.window_coefficient,
        total_bandwidth=translate_common.translate_double_with_unit(params.total_bandwidth),
        processing_bandwidth=translate_common.translate_double_with_unit(params.processing_bandwidth),
        look_bandwidth=translate_common.translate_double_with_unit(params.look_bandwidth),
        number_of_looks=params.number_of_looks,
        look_overlap=translate_common.translate_double_with_unit(params.look_overlap),
    )


def translate_spectrum_processing_parameters_type_to_model(
    params: common_annotation_l1.SpectrumProcessingParametersType,
) -> common_annotation_models_l1.SpectrumProcessingParametersType:
    """Translate spectrum proc params"""

    return common_annotation_models_l1.SpectrumProcessingParametersType(
        window_type=translate_common.translate_weighting_window_type_to_model(params.window_type),
        window_coefficient=float(params.window_coefficient),
        total_bandwidth=translate_common.translate_double_with_unit_to_model(
            params.total_bandwidth, units=common.UomType.HZ
        ),
        processing_bandwidth=translate_common.translate_double_with_unit_to_model(
            params.processing_bandwidth, units=common.UomType.HZ
        ),
        look_bandwidth=translate_common.translate_double_with_unit_to_model(
            params.look_bandwidth, units=common.UomType.HZ
        ),
        number_of_looks=params.number_of_looks,
        look_overlap=translate_common.translate_double_with_unit_to_model(params.look_overlap, units=common.UomType.HZ),
    )


def translate_processing_parameters(
    params: common_annotation_models_l1.ProcessingParametersType,
) -> common_annotation_l1.ProcessingParameters:
    assert params.processor_version is not None
    assert params.product_generation_time is not None
    assert params.processing_mode is not None
    assert params.orbit_source is not None
    assert params.attitude_source is not None
    assert params.raw_data_correction_flag is not None
    assert params.rfi_detection_flag is not None
    assert params.rfi_correction_flag is not None
    assert params.rfi_mitigation_method is not None
    assert params.rfi_mask is not None
    assert params.rfi_mask_generation_method is not None
    assert params.rfi_fmmitigation_method is not None
    assert params.rfi_fmchirp_source is not None
    assert params.internal_calibration_estimation_flag is not None
    assert params.internal_calibration_correction_flag is not None
    assert params.range_reference_function_source is not None
    assert params.range_compression_method is not None
    assert params.extended_swath_processing_flag is not None
    assert params.dc_method is not None
    assert params.dc_value is not None
    assert params.antenna_pattern_correction1_flag is not None
    assert params.antenna_pattern_correction2_flag is not None
    assert params.antenna_cross_talk_correction_flag is not None
    assert params.range_processing_parameters is not None
    assert params.azimuth_processing_parameters is not None
    assert params.bistatic_delay_correction_flag is not None
    assert params.bistatic_delay_correction_method is not None
    assert params.range_spreading_loss_compensation_flag is not None
    assert params.reference_range is not None
    assert params.processing_gain_list is not None
    assert params.polarimetric_correction_flag is not None
    assert params.ionosphere_height_defocusing_flag is not None
    assert params.ionosphere_height_estimation_method is not None
    assert params.faraday_rotation_correction_flag is not None
    assert params.ionospheric_phase_screen_correction_flag is not None
    assert params.group_delay_correction_flag is not None
    assert params.autofocus_flag is not None
    assert params.autofocus_method is not None
    assert params.detection_flag is not None
    assert params.thermal_denoising_flag is not None
    assert params.noise_gain_list is not None
    assert params.ground_projection_flag is not None

    return common_annotation_l1.ProcessingParameters(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime(params.product_generation_time),
        processing_mode=translate_common.translate_processing_mode(params.processing_mode),
        orbit_source=translate_common.translate_orbit_attitude_source(params.orbit_source),
        attitude_source=translate_common.translate_orbit_attitude_source(params.attitude_source),
        raw_data_correction_flag=translate_common.translate_bool(params.raw_data_correction_flag),
        rfi_detection_flag=translate_common.translate_bool(params.rfi_detection_flag),
        rfi_correction_flag=translate_common.translate_bool(params.rfi_correction_flag),
        rfi_mitigation_method=translate_common.translate_rfi_mitigation_method_type(params.rfi_mitigation_method),
        rfi_mask=translate_common.translate_rfi_mask_type(params.rfi_mask),
        rfi_mask_generation_method=translate_common.translate_rfi_mask_generation_method_type(
            params.rfi_mask_generation_method
        ),
        rfi_fm_mitigation_method=params.rfi_fmmitigation_method.value,
        rfi_fm_chirp_source=translate_common.translate_range_reference_function_type(params.rfi_fmchirp_source),
        internal_calibration_estimation_flag=translate_common.translate_bool(
            params.internal_calibration_estimation_flag
        ),
        internal_calibration_correction_flag=translate_common.translate_bool(
            params.internal_calibration_correction_flag
        ),
        range_reference_function_source=translate_common.translate_range_reference_function_type(
            params.range_reference_function_source
        ),
        range_compression_method=translate_common.translate_range_compression_method_type(
            params.range_compression_method
        ),
        extended_swath_processing_flag=translate_common.translate_bool(params.extended_swath_processing_flag),
        dc_method=translate_common.translate_dc_method_type(params.dc_method),
        dc_value=translate_common.translate_double_with_unit(params.dc_value),
        antenna_pattern_correction1_flag=translate_common.translate_bool(params.antenna_pattern_correction1_flag),
        antenna_pattern_correction2_flag=translate_common.translate_bool(params.antenna_pattern_correction2_flag),
        antenna_cross_talk_correction_flag=translate_common.translate_bool(params.antenna_cross_talk_correction_flag),
        range_processing_parameters=translate_spectrum_processing_parameters_type(params.range_processing_parameters),
        azimuth_processing_parameters=translate_spectrum_processing_parameters_type(
            params.azimuth_processing_parameters
        ),
        bistatic_delay_correction_flag=translate_common.translate_bool(params.bistatic_delay_correction_flag),
        bistatic_delay_correction_method=translate_common.translate_bistatic_delay_correction_method_type(
            params.bistatic_delay_correction_method
        ),
        range_spreading_loss_compensation_flag=translate_common.translate_bool(
            params.range_spreading_loss_compensation_flag
        ),
        reference_range=translate_common.translate_double_with_unit(params.reference_range),
        processing_gain_list=translate_processing_gain_list(params.processing_gain_list),
        polarimetric_correction_flag=translate_common.translate_bool(params.polarimetric_correction_flag),
        ionosphere_height_defocusing_flag=translate_common.translate_bool(params.ionosphere_height_defocusing_flag),
        ionosphere_height_estimation_method=translate_common.translate_ionosphere_height_estimation_method_type(
            params.ionosphere_height_estimation_method
        ),
        faraday_rotation_correction_flag=translate_common.translate_bool(params.faraday_rotation_correction_flag),
        ionospheric_phase_screen_correction_flag=translate_common.translate_bool(
            params.ionospheric_phase_screen_correction_flag
        ),
        group_delay_correction_flag=translate_common.translate_bool(params.group_delay_correction_flag),
        autofocus_flag=translate_common.translate_bool(params.autofocus_flag),
        autofocus_method=translate_common.translate_autofocus_method_type(params.autofocus_method),
        detection_flag=translate_common.translate_bool(params.detection_flag),
        thermal_denoising_flag=translate_common.translate_bool(params.thermal_denoising_flag),
        noise_gain_list=translate_noise_gain_list(params.noise_gain_list),
        ground_projection_flag=translate_common.translate_bool(params.ground_projection_flag),
    )


def translate_processing_parameters_to_model(
    params: common_annotation_l1.ProcessingParameters,
) -> common_annotation_models_l1.ProcessingParametersType:
    """Translate processing parameters"""
    return common_annotation_models_l1.ProcessingParametersType(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime_to_model(params.product_generation_time),
        processing_mode=translate_common.translate_processing_mode_to_model(params.processing_mode),
        orbit_source=translate_common.translate_orbit_attitude_source_to_model(params.orbit_source),
        attitude_source=translate_common.translate_orbit_attitude_source_to_model(params.attitude_source),
        raw_data_correction_flag=translate_common.translate_bool_to_model(params.raw_data_correction_flag),
        rfi_detection_flag=translate_common.translate_bool_to_model(params.rfi_detection_flag),
        rfi_correction_flag=translate_common.translate_bool_to_model(params.rfi_correction_flag),
        rfi_mitigation_method=translate_common.translate_rfi_mitigation_method_type_to_model(
            params.rfi_mitigation_method
        ),
        rfi_mask=translate_common.translate_rfi_mask_type_to_model(params.rfi_mask),
        rfi_mask_generation_method=translate_common.translate_rfi_mask_generation_method_type_to_model(
            params.rfi_mask_generation_method
        ),
        rfi_fmchirp_source=translate_common.translate_range_reference_function_type_to_model(
            params.rfi_fm_chirp_source
        ),
        rfi_fmmitigation_method=common_types.RfiFmmitigationMethodType[params.rfi_fm_mitigation_method],
        internal_calibration_estimation_flag=translate_common.translate_bool_to_model(
            params.internal_calibration_estimation_flag
        ),
        internal_calibration_correction_flag=translate_common.translate_bool_to_model(
            params.internal_calibration_correction_flag
        ),
        range_reference_function_source=translate_common.translate_range_reference_function_type_to_model(
            params.range_reference_function_source
        ),
        range_compression_method=translate_common.translate_range_compression_method_type_to_model(
            params.range_compression_method
        ),
        extended_swath_processing_flag=translate_common.translate_bool_to_model(params.extended_swath_processing_flag),
        dc_method=translate_common.translate_dc_method_type_to_model(params.dc_method),
        dc_value=translate_common.translate_double_with_unit_to_model(params.dc_value, units=common.UomType.HZ),
        antenna_pattern_correction1_flag=translate_common.translate_bool_to_model(
            params.antenna_pattern_correction1_flag
        ),
        antenna_pattern_correction2_flag=translate_common.translate_bool_to_model(
            params.antenna_pattern_correction2_flag
        ),
        antenna_cross_talk_correction_flag=translate_common.translate_bool_to_model(
            params.antenna_cross_talk_correction_flag
        ),
        range_processing_parameters=translate_spectrum_processing_parameters_type_to_model(
            params.range_processing_parameters
        ),
        azimuth_processing_parameters=translate_spectrum_processing_parameters_type_to_model(
            params.azimuth_processing_parameters
        ),
        bistatic_delay_correction_flag=translate_common.translate_bool_to_model(params.bistatic_delay_correction_flag),
        bistatic_delay_correction_method=translate_common.translate_bistatic_delay_correction_method_type_to_model(
            params.bistatic_delay_correction_method
        ),
        range_spreading_loss_compensation_flag=translate_common.translate_bool_to_model(
            params.range_spreading_loss_compensation_flag
        ),
        reference_range=translate_common.translate_double_with_unit_to_model(
            params.reference_range, units=common.UomType.M
        ),
        processing_gain_list=translate_processing_gain_list_to_model(params.processing_gain_list),
        polarimetric_correction_flag=translate_common.translate_bool_to_model(params.polarimetric_correction_flag),
        ionosphere_height_defocusing_flag=translate_common.translate_bool_to_model(
            params.ionosphere_height_defocusing_flag
        ),
        ionosphere_height_estimation_method=translate_common.translate_ionosphere_height_estimation_method_type_to_model(
            params.ionosphere_height_estimation_method
        ),
        faraday_rotation_correction_flag=translate_common.translate_bool_to_model(
            params.faraday_rotation_correction_flag
        ),
        ionospheric_phase_screen_correction_flag=translate_common.translate_bool_to_model(
            params.ionospheric_phase_screen_correction_flag
        ),
        group_delay_correction_flag=translate_common.translate_bool_to_model(params.group_delay_correction_flag),
        autofocus_flag=translate_common.translate_bool_to_model(params.autofocus_flag),
        autofocus_method=translate_common.translate_autofocus_method_type_to_model(params.autofocus_method),
        detection_flag=translate_common.translate_bool_to_model(params.detection_flag),
        thermal_denoising_flag=translate_common.translate_bool_to_model(params.thermal_denoising_flag),
        noise_gain_list=translate_noise_gain_list_to_model(params.noise_gain_list),
        ground_projection_flag=translate_common.translate_bool_to_model(params.ground_projection_flag),
    )


def translate_dc_estimate_type(
    section: common_annotation_models_l1.DcEstimateType,
) -> common_annotation_l1.DcEstimateType:
    """Translate dc annotatoin section"""
    assert section.azimuth_time is not None
    assert section.t0 is not None
    assert section.geometry_dcpolynomial is not None
    assert section.combined_dcpolynomial is not None
    assert section.combined_dcvalues is not None
    assert section.combined_dcslant_range_times is not None
    assert section.combined_dcrmserror is not None
    assert section.combined_dcrmserror_above_threshold is not None
    return common_annotation_l1.DcEstimateType(
        azimuth_time=translate_common.translate_datetime(section.azimuth_time),
        t0=translate_common.translate_double_with_unit(section.t0),
        geometry_dcpolynomial=translate_common.translate_float_array(section.geometry_dcpolynomial),
        combined_dcpolynomial=translate_common.translate_float_array(section.combined_dcpolynomial),
        combined_dcvalues=translate_common.translate_double_array_with_units(section.combined_dcvalues),
        combined_dcslant_range_times=translate_common.translate_double_array_with_units(
            section.combined_dcslant_range_times
        ),
        combined_dcrmserror=translate_common.translate_double_with_unit(section.combined_dcrmserror),
        combined_dcrmserror_above_threshold=translate_common.translate_bool(
            section.combined_dcrmserror_above_threshold
        ),
    )


def translate_dc_estimate_type_to_model(
    section: common_annotation_l1.DcEstimateType,
) -> common_annotation_models_l1.DcEstimateType:
    """Translate dc annotatoin section"""
    return common_annotation_models_l1.DcEstimateType(
        azimuth_time=translate_common.translate_datetime_to_model(section.azimuth_time),
        t0=translate_common.translate_double_with_unit_to_model(section.t0, units=common.UomType.S),
        geometry_dcpolynomial=translate_common.translate_float_array_to_model(section.geometry_dcpolynomial),
        combined_dcpolynomial=translate_common.translate_float_array_to_model(section.combined_dcpolynomial),
        combined_dcvalues=translate_common.translate_double_array_with_units_to_model(
            section.combined_dcvalues, units=common.UomType.HZ
        ),
        combined_dcslant_range_times=translate_common.translate_double_array_with_units_to_model(
            section.combined_dcslant_range_times, units=common.UomType.S
        ),
        combined_dcrmserror=translate_common.translate_double_with_unit_to_model(
            section.combined_dcrmserror, units=common.UomType.HZ
        ),
        combined_dcrmserror_above_threshold=translate_common.translate_bool_to_model(
            section.combined_dcrmserror_above_threshold
        ),
    )


def translate_dc_estimate_list_type(
    dc_estimates: common_annotation_models_l1.DcEstimateListType,
) -> list[common_annotation_l1.DcEstimateType]:
    """Translate list of dc estimate"""
    assert dc_estimates.count is not None
    if len(dc_estimates.dc_estimate) != dc_estimates.count:
        raise RuntimeError(
            "Inconsistency in dc estimate list: "
            + f"{len(dc_estimates.dc_estimate)} length and count: {dc_estimates.count} do not match"
        )

    return [translate_dc_estimate_type(dc_estimate) for dc_estimate in dc_estimates.dc_estimate]


def translate_dc_estimate_list_type_to_model(
    dc_estimates: list[common_annotation_l1.DcEstimateType],
) -> common_annotation_models_l1.DcEstimateListType:
    """Translate list of dc estimate"""
    return common_annotation_models_l1.DcEstimateListType(
        [translate_dc_estimate_type_to_model(dc_estimate) for dc_estimate in dc_estimates],
        count=len(dc_estimates),
    )


def translate_coordinate_conversion_type(
    conversion: common_annotation_models_l1.CoordinateConversionType,
) -> common_annotation_l1.CoordinateConversionType:
    """Translate coordinate conversion section"""
    assert conversion.azimuth_time is not None
    assert conversion.t0 is not None
    assert conversion.sr0 is not None
    assert conversion.slant_to_ground_coefficients is not None
    assert conversion.gr0 is not None
    assert conversion.ground_to_slant_coefficients is not None
    return common_annotation_l1.CoordinateConversionType(
        azimuth_time=translate_common.translate_datetime(conversion.azimuth_time),
        t0=translate_common.translate_double_with_unit(conversion.t0),
        sr0=translate_common.translate_double_with_unit(conversion.sr0),
        slant_to_ground_coefficients=translate_common.translate_double_array(conversion.slant_to_ground_coefficients),
        gr0=translate_common.translate_double_with_unit(conversion.gr0),
        ground_to_slant_coefficients=translate_common.translate_double_array(conversion.ground_to_slant_coefficients),
    )


def translate_coordinate_conversion_type_to_model(
    conversion: common_annotation_l1.CoordinateConversionType,
) -> common_annotation_models_l1.CoordinateConversionType:
    """Translate coordinate conversion section"""
    return common_annotation_models_l1.CoordinateConversionType(
        azimuth_time=translate_common.translate_datetime_to_model(conversion.azimuth_time),
        t0=translate_common.translate_double_with_unit_to_model(conversion.t0, common.UomType.S),
        sr0=translate_common.translate_double_with_unit_to_model(conversion.sr0, common.UomType.M),
        slant_to_ground_coefficients=translate_common.translate_double_array_to_model(
            conversion.slant_to_ground_coefficients
        ),
        gr0=translate_common.translate_double_with_unit_to_model(conversion.gr0, common.UomType.M),
        ground_to_slant_coefficients=translate_common.translate_double_array_to_model(
            conversion.ground_to_slant_coefficients
        ),
    )


def translate_coordinate_conversion_list_type(
    conversions: common_annotation_models_l1.CoordinateConversionListType,
) -> list[common_annotation_l1.CoordinateConversionType]:
    """Translate list of coordinate conversions"""
    assert conversions.count is not None
    if len(conversions.coordinate_conversion) != conversions.count:
        raise RuntimeError(
            "Inconsistency in coordinate conversions list: "
            + f"{len(conversions.coordinate_conversion)} length and count: {conversions.count} do not match"
        )

    return [translate_coordinate_conversion_type(conversion) for conversion in conversions.coordinate_conversion]


def translate_coordinate_conversion_list_type_to_model(
    conversions: list[common_annotation_l1.CoordinateConversionType],
) -> common_annotation_models_l1.CoordinateConversionListType:
    """Translate list of coordinate conversions"""
    return common_annotation_models_l1.CoordinateConversionListType(
        [translate_coordinate_conversion_type_to_model(conversion) for conversion in conversions],
        count=len(conversions),
    )


def translate_sar_image(
    image: common_annotation_models_l1.SarImageType,
) -> common_annotation_l1.SarImageType:
    """Translate SAR Image section"""
    assert image.first_sample_slant_range_time is not None
    assert image.last_sample_slant_range_time is not None
    assert image.first_line_azimuth_time is not None
    assert image.last_line_azimuth_time is not None
    assert image.range_time_interval is not None
    assert image.azimuth_time_interval is not None
    assert image.range_pixel_spacing is not None
    assert image.azimuth_pixel_spacing is not None
    assert image.number_of_samples is not None
    assert image.number_of_lines is not None
    assert image.projection is not None
    assert image.range_coordinate_conversion is not None
    assert image.datum is not None
    assert image.footprint is not None
    assert image.pixel_representation is not None
    assert image.pixel_type is not None
    assert image.pixel_quantity is not None
    assert image.no_data_value is not None

    return common_annotation_l1.SarImageType(
        first_sample_slant_range_time=translate_common.translate_double_with_unit(image.first_sample_slant_range_time),
        last_sample_slant_range_time=translate_common.translate_double_with_unit(image.last_sample_slant_range_time),
        first_line_azimuth_time=translate_common.translate_datetime(image.first_line_azimuth_time),
        last_line_azimuth_time=translate_common.translate_datetime(image.last_line_azimuth_time),
        range_time_interval=translate_common.translate_double_with_unit(image.range_time_interval),
        azimuth_time_interval=translate_common.translate_double_with_unit(image.azimuth_time_interval),
        range_pixel_spacing=translate_common.translate_float_with_unit(image.range_pixel_spacing),
        azimuth_pixel_spacing=translate_common.translate_float_with_unit(image.azimuth_pixel_spacing),
        number_of_samples=image.number_of_samples,
        number_of_lines=image.number_of_lines,
        projection=translate_common.translate_projection_type(image.projection),
        range_coordinate_conversion=translate_coordinate_conversion_list_type(image.range_coordinate_conversion),
        datum=translate_common.translate_datum(image.datum),
        footprint=translate_common.translate_float_array_with_units(image.footprint),
        pixel_representation=translate_common.translate_pixel_representation_type(image.pixel_representation),
        pixel_type=translate_common.translate_pixel_type_type(image.pixel_type),
        pixel_quantity=translate_common.translate_pixel_quantity_type(image.pixel_quantity),
        no_data_value=image.no_data_value,
    )


def translate_sar_image_to_model(
    image: common_annotation_l1.SarImageType,
) -> common_annotation_models_l1.SarImageType:
    """Translate SAR Image section"""
    return common_annotation_models_l1.SarImageType(
        first_sample_slant_range_time=translate_common.translate_double_with_unit_to_model(
            image.first_sample_slant_range_time, common.UomType.S
        ),
        last_sample_slant_range_time=translate_common.translate_double_with_unit_to_model(
            image.last_sample_slant_range_time, common.UomType.S
        ),
        first_line_azimuth_time=translate_common.translate_datetime_to_model(image.first_line_azimuth_time),
        last_line_azimuth_time=translate_common.translate_datetime_to_model(image.last_line_azimuth_time),
        range_time_interval=translate_common.translate_double_with_unit_to_model(
            image.range_time_interval, common.UomType.S
        ),
        azimuth_time_interval=translate_common.translate_double_with_unit_to_model(
            image.azimuth_time_interval, common.UomType.S
        ),
        range_pixel_spacing=translate_common.translate_float_with_unit_to_model(
            image.range_pixel_spacing, common.UomType.M
        ),
        azimuth_pixel_spacing=translate_common.translate_float_with_unit_to_model(
            image.azimuth_pixel_spacing, common.UomType.M
        ),
        number_of_samples=image.number_of_samples,
        number_of_lines=image.number_of_lines,
        projection=translate_common.translate_projection_type_to_model(image.projection),
        range_coordinate_conversion=translate_coordinate_conversion_list_type_to_model(
            image.range_coordinate_conversion
        ),
        datum=translate_common.translate_datum_to_model(image.datum),
        footprint=translate_common.translate_float_array_with_units_to_model(image.footprint, common.UomType.DEG),
        pixel_representation=translate_common.translate_pixel_representation_type_to_model(image.pixel_representation),
        pixel_type=translate_common.translate_pixel_type_type_to_model(image.pixel_type),
        pixel_quantity=translate_common.translate_pixel_quantity_type_to_model(image.pixel_quantity),
        no_data_value=float(image.no_data_value),
    )


def translate_polarisation_list(
    polarisations: common_annotation_models_l1.PolarisationListType,
) -> list[common.PolarisationType]:
    """Translate polarisation list"""
    assert polarisations.count is not None
    if len(polarisations.polarisation) != polarisations.count:
        raise RuntimeError(
            "Inconsistency in polarisation list: "
            + f"{len(polarisations.polarisation)} length and count: {polarisations.count} do not match"
        )

    return [translate_common.translate_polarisation_type(pol) for pol in polarisations.polarisation]


def translate_polarisation_list_to_model(
    polarisations: list[common.PolarisationType],
) -> common_annotation_models_l1.PolarisationListType:
    """Translate polarisation list"""
    return common_annotation_models_l1.PolarisationListType(
        polarisation=[translate_common.translate_polarisation_type_to_model(pol) for pol in polarisations],
        count=len(polarisations),
    )


def translate_acquisition_information_to_model(
    info: common_annotation_l1.AcquisitionInformationType,
) -> common_annotation_models_l1.AcquisitionInformationType:
    """Translate acquisition information"""
    return common_annotation_models_l1.AcquisitionInformationType(
        mission=translate_common.translate_mission_type_to_model(info.mission),
        swath=translate_common.translate_swath_type_to_model(info.swath),
        product_type=translate_common.translate_product_type_to_model(info.product_type),
        polarisation_list=translate_polarisation_list_to_model(info.polarisation_list),
        start_time=translate_common.translate_datetime_to_model(info.start_time),
        stop_time=translate_common.translate_datetime_to_model(info.stop_time),
        mission_phase_id=translate_common.translate_mission_phase_id_to_model(info.mission_phase_id),
        drift_phase_flag=translate_common.translate_bool_to_model(info.drift_phase_flag),
        sensor_mode=translate_common.translate_sensor_mode_to_model(info.sensor_mode),
        global_coverage_id=translate_global_coverage_id(info.global_coverage_id),
        major_cycle_id=translate_major_cycle_id(info.major_cycle_id),
        repeat_cycle_id=translate_repeat_cycle_id(info.repeat_cycle_id),
        absolute_orbit_number=info.absolute_orbit_number,
        relative_orbit_number=info.relative_orbit_number,
        orbit_pass=translate_common.translate_orbit_pass_to_model(info.orbit_pass),
        platform_heading=translate_common.translate_double_with_unit_to_model(
            info.platform_heading, units=common.UomType.DEG
        ),
        data_take_id=info.data_take_id,
        frame=info.frame,
        product_composition=translate_common.translate_product_composition_to_model(info.product_composition),
    )


def translate_acquisition_information(
    info: common_annotation_models_l1.AcquisitionInformationType,
) -> common_annotation_l1.AcquisitionInformationType:
    """Translate acquisition information"""
    assert info.mission is not None
    assert info.swath is not None
    assert info.product_type is not None
    assert info.polarisation_list is not None
    assert info.start_time is not None
    assert info.stop_time is not None
    assert info.mission_phase_id is not None
    assert info.drift_phase_flag is not None
    assert info.sensor_mode is not None
    assert info.global_coverage_id is not None
    assert info.major_cycle_id is not None
    assert info.repeat_cycle_id is not None
    assert info.absolute_orbit_number is not None
    assert info.relative_orbit_number is not None
    assert info.orbit_pass is not None
    assert info.platform_heading is not None
    assert info.data_take_id is not None
    assert info.frame is not None
    assert info.product_composition is not None

    return common_annotation_l1.AcquisitionInformationType(
        mission=translate_common.translate_mission_type(info.mission),
        swath=translate_common.translate_swath_type(info.swath),
        product_type=translate_common.translate_product_type(info.product_type),
        polarisation_list=translate_polarisation_list(info.polarisation_list),
        start_time=translate_common.translate_datetime(info.start_time),
        stop_time=translate_common.translate_datetime(info.stop_time),
        mission_phase_id=translate_common.translate_mission_phase_id(info.mission_phase_id),
        drift_phase_flag=translate_common.translate_bool(info.drift_phase_flag),
        sensor_mode=translate_common.translate_sensor_mode(info.sensor_mode),
        global_coverage_id=info.global_coverage_id,
        major_cycle_id=info.major_cycle_id,
        repeat_cycle_id=info.repeat_cycle_id,
        absolute_orbit_number=info.absolute_orbit_number,
        relative_orbit_number=info.relative_orbit_number,
        orbit_pass=translate_common.translate_orbit_pass(info.orbit_pass),
        platform_heading=translate_common.translate_double_with_unit(info.platform_heading),
        data_take_id=info.data_take_id,
        frame=info.frame,
        product_composition=translate_common.translate_product_composition(info.product_composition),
    )


def translate_tx_pulse(
    pulse: common_annotation_models_l1.TxPulseType,
) -> common_annotation_l1.TxPulseType:
    """Translate TX pulse"""
    assert pulse.azimuth_time is not None
    assert pulse.tx_pulse_length is not None
    assert pulse.tx_pulse_start_frequency is not None
    assert pulse.tx_pulse_start_phase is not None
    assert pulse.tx_pulse_ramp_rate is not None
    return common_annotation_l1.TxPulseType(
        azimuth_time=translate_common.translate_datetime(pulse.azimuth_time),
        tx_pulse_length=translate_common.translate_double_with_unit(pulse.tx_pulse_length),
        tx_pulse_start_frequency=translate_common.translate_double_with_unit(pulse.tx_pulse_start_frequency),
        tx_pulse_start_phase=translate_common.translate_double_with_unit(pulse.tx_pulse_start_phase),
        tx_pulse_ramp_rate=translate_common.translate_double_with_unit(pulse.tx_pulse_ramp_rate),
    )


def translate_tx_pulse_to_model(
    pulse: common_annotation_l1.TxPulseType,
) -> common_annotation_models_l1.TxPulseType:
    """Translate TX pulse"""
    return common_annotation_models_l1.TxPulseType(
        azimuth_time=translate_common.translate_datetime_to_model(pulse.azimuth_time),
        tx_pulse_length=translate_common.translate_double_with_unit_to_model(
            pulse.tx_pulse_length, units=common.UomType.S
        ),
        tx_pulse_start_frequency=translate_common.translate_double_with_unit_to_model(
            pulse.tx_pulse_start_frequency, units=common.UomType.HZ
        ),
        tx_pulse_start_phase=translate_common.translate_double_with_unit_to_model(
            pulse.tx_pulse_start_phase, units=common.UomType.RAD
        ),
        tx_pulse_ramp_rate=translate_common.translate_double_with_unit_to_model(
            pulse.tx_pulse_ramp_rate, units=common.UomType.HZ_S
        ),
    )


def translate_data_format_type(
    data_format: common_annotation_models_l1.DataFormatType,
) -> common_annotation_l1.DataFormatType:
    """Translate data format"""
    assert data_format.echo_format is not None
    assert data_format.calibration_format is not None
    assert data_format.noise_format is not None
    assert data_format.mean_bit_rate is not None
    return common_annotation_l1.DataFormatType(
        echo_format=translate_common.translate_data_format_mode(data_format.echo_format),
        calibration_format=translate_common.translate_data_format_mode(data_format.calibration_format),
        noise_format=translate_common.translate_data_format_mode(data_format.noise_format),
        mean_bit_rate=translate_common.translate_double_with_unit(data_format.mean_bit_rate),
    )


def translate_data_format_type_to_model(
    data_format: common_annotation_l1.DataFormatType,
) -> common_annotation_models_l1.DataFormatType:
    """Translate data format"""
    return common_annotation_models_l1.DataFormatType(
        echo_format=translate_common.translate_data_format_mode_to_model(data_format.echo_format),
        calibration_format=translate_common.translate_data_format_mode_to_model(data_format.calibration_format),
        noise_format=translate_common.translate_data_format_mode_to_model(data_format.noise_format),
        mean_bit_rate=translate_common.translate_double_with_unit_to_model(
            data_format.mean_bit_rate, units=common.UomType.MBPS
        ),
    )


def _translate_list_time_with_pol(
    pol_list: list[common_types.TimeTypeWithPolarisation],
) -> dict[common.PolarisationType, PreciseDateTime]:
    output = {}
    for item in pol_list:
        pol, time = translate_common.translate_time_with_pol(item)
        output[pol] = time
    return output


def _translate_list_time_with_pol_to_model(
    pol_list: dict[common.PolarisationType, PreciseDateTime],
) -> list[common_types.TimeTypeWithPolarisation]:
    return [translate_common.translate_time_with_pol_to_model(item) for item in pol_list.items()]


def _translate_list_float_with_pol(
    pol_list: list[common_types.FloatWithPolarisation],
) -> dict[common.PolarisationType, float]:
    output = {}
    for item in pol_list:
        pol, value = translate_common.translate_float_with_pol(item)
        output[pol] = value
    return output


def _translate_list_float_with_pol_to_model(
    pol_list: dict[common.PolarisationType, float],
) -> list[common_types.FloatWithPolarisation]:
    return [translate_common.translate_float_with_pol_to_model(item) for item in pol_list.items()]


def _translate_list_state(
    states: list[common_types.StateType],
) -> list[tuple[PreciseDateTime, float]]:
    return [translate_common.translate_state_type(state) for state in states]


def _translate_list_state_to_model(
    states: list[tuple[PreciseDateTime, float]], unit: common.UomType
) -> list[common_types.StateType]:
    return [translate_common.translate_state_type_to_model(state, unit) for state in states]


def _translate_tx_pulse_list(
    pulses: common_annotation_models_l1.TxPulseListType,
) -> list[common_annotation_l1.TxPulseType]:
    return [translate_tx_pulse(pulse) for pulse in pulses.tx_pulse]


def _translate_tx_pulse_list_to_model(
    pulses: list[common_annotation_l1.TxPulseType],
) -> common_annotation_models_l1.TxPulseListType:
    return common_annotation_models_l1.TxPulseListType(
        tx_pulse=[translate_tx_pulse_to_model(pulse) for pulse in pulses],
        count=len(pulses),
    )


def translate_instrument_parameters(
    params: common_annotation_models_l1.InstrumentParametersType,
) -> common_annotation_l1.InstrumentParametersType:
    """Translate instrument parameters"""

    assert params.first_line_sensing_time_list is not None
    assert params.first_line_sensing_time_list.first_line_sensing_time is not None
    assert params.last_line_sensing_time_list is not None
    assert params.last_line_sensing_time_list.last_line_sensing_time is not None
    assert params.number_of_input_samples is not None
    assert params.number_of_input_lines is not None
    assert params.swp_list is not None
    assert params.swl_list is not None
    assert params.prf_list is not None
    assert params.rank is not None
    assert params.tx_pulse_list is not None
    assert params.instrument_configuration_id is not None
    assert params.radar_carrier_frequency is not None
    assert params.rx_gain_list is not None
    assert params.preamble_flag is not None
    assert params.postamble_flag is not None
    assert params.interleaved_calibration_flag is not None
    assert params.data_format is not None
    return common_annotation_l1.InstrumentParametersType(
        first_line_sensing_time_list=_translate_list_time_with_pol(
            params.first_line_sensing_time_list.first_line_sensing_time
        ),
        last_line_sensing_time_list=_translate_list_time_with_pol(
            params.last_line_sensing_time_list.last_line_sensing_time
        ),
        number_of_input_samples=params.number_of_input_samples,
        number_of_input_lines=params.number_of_input_lines,
        swp_list=_translate_list_state(params.swp_list.swp),
        swl_list=_translate_list_state(params.swl_list.swl),
        prf_list=_translate_list_state(params.prf_list.prf),
        rank=params.rank,
        tx_pulse_list=_translate_tx_pulse_list(params.tx_pulse_list),
        instrument_configuration_id=params.instrument_configuration_id,
        radar_carrier_frequency=translate_common.translate_double_with_unit(params.radar_carrier_frequency),
        rx_gain_list=_translate_list_float_with_pol(params.rx_gain_list.rx_gain),
        preamble_flag=translate_common.translate_bool(params.preamble_flag),
        postamble_flag=translate_common.translate_bool(params.postamble_flag),
        interleaved_calibration_flag=translate_common.translate_bool(params.interleaved_calibration_flag),
        data_format=translate_data_format_type(params.data_format),
    )


def translate_instrument_parameters_to_model(
    params: common_annotation_l1.InstrumentParametersType,
) -> common_annotation_models_l1.InstrumentParametersType:
    """Translate instrument parameters"""
    first_line_sensing_time_list = _translate_list_time_with_pol_to_model(params.first_line_sensing_time_list)
    last_line_sensing_time_list = _translate_list_time_with_pol_to_model(params.last_line_sensing_time_list)
    swp_list = _translate_list_state_to_model(params.swp_list, unit=common.UomType.S)
    swl_list = _translate_list_state_to_model(params.swl_list, unit=common.UomType.S)
    prf_list = _translate_list_state_to_model(params.prf_list, unit=common.UomType.HZ)
    rx_gain_list = _translate_list_float_with_pol_to_model(params.rx_gain_list)

    return common_annotation_models_l1.InstrumentParametersType(
        first_line_sensing_time_list=common_annotation_models_l1.FirstLineSensingTimeListType(
            first_line_sensing_time_list, len(first_line_sensing_time_list)
        ),
        last_line_sensing_time_list=common_annotation_models_l1.LastLineSensingTimeListType(
            last_line_sensing_time_list, len(last_line_sensing_time_list)
        ),
        number_of_input_samples=params.number_of_input_samples,
        number_of_input_lines=params.number_of_input_lines,
        swp_list=common_annotation_models_l1.SwpListType(swp_list, len(swp_list)),
        swl_list=common_annotation_models_l1.SwlListType(swl_list, len(swl_list)),
        prf_list=common_annotation_models_l1.PrfListType(prf_list, len(prf_list)),
        rank=params.rank,
        tx_pulse_list=_translate_tx_pulse_list_to_model(params.tx_pulse_list),
        instrument_configuration_id=params.instrument_configuration_id,
        radar_carrier_frequency=translate_common.translate_double_with_unit_to_model(
            params.radar_carrier_frequency, units=common.UomType.HZ
        ),
        rx_gain_list=common_annotation_models_l1.RxGainListType(rx_gain_list, len(rx_gain_list)),
        preamble_flag=translate_common.translate_bool_to_model(params.preamble_flag),
        postamble_flag=translate_common.translate_bool_to_model(params.postamble_flag),
        interleaved_calibration_flag=translate_common.translate_bool_to_model(params.interleaved_calibration_flag),
        data_format=translate_data_format_type_to_model(params.data_format),
    )


def translate_ionosphere_correction(
    info: common_annotation_models_l1.IonosphereCorrectionType,
) -> common_annotation_l1.IonosphereCorrection:
    """Translate ionosphere correction section"""

    assert info.ionosphere_height_used is not None
    assert info.ionosphere_height_estimated is not None
    assert info.ionosphere_height_estimation_method_selected is not None
    assert info.ionosphere_height_estimation_latitude_value is not None
    assert info.ionosphere_height_estimation_flag is not None
    assert info.ionosphere_height_estimation_method_used is not None
    assert info.gaussian_filter_computation_flag is not None
    assert info.faraday_rotation_correction_applied is not None
    assert info.autofocus_shifts_applied is not None

    return common_annotation_l1.IonosphereCorrection(
        ionosphere_height_used=translate_common.translate_float_with_unit(info.ionosphere_height_used),
        ionosphere_height_estimated=translate_common.translate_float_with_unit(info.ionosphere_height_estimated),
        ionosphere_height_estimation_method_selected=translate_common.translate_ionosphere_height_estimation_method_type(
            info.ionosphere_height_estimation_method_selected
        ),
        ionosphere_height_estimation_latitude_value=translate_common.translate_float_with_unit(
            info.ionosphere_height_estimation_latitude_value
        ),
        ionosphere_height_estimation_flag=translate_common.translate_bool(info.ionosphere_height_estimation_flag),
        ionosphere_height_estimation_method_used=translate_common.translate_ionosphere_height_estimation_method_type(
            info.ionosphere_height_estimation_method_used
        ),
        gaussian_filter_computation_flag=translate_common.translate_bool(info.gaussian_filter_computation_flag),
        faraday_rotation_correction_applied=translate_common.translate_bool(info.faraday_rotation_correction_applied),
        autofocus_shifts_applied=translate_common.translate_bool(info.autofocus_shifts_applied),
    )


def translate_ionosphere_correction_to_model(
    info: common_annotation_l1.IonosphereCorrection,
) -> common_annotation_models_l1.IonosphereCorrectionType:
    """Translate ionosphere correction section"""

    return common_annotation_models_l1.IonosphereCorrectionType(
        ionosphere_height_used=translate_common.translate_float_with_unit_to_model(
            info.ionosphere_height_used, units=common.UomType.M
        ),
        ionosphere_height_estimated=translate_common.translate_float_with_unit_to_model(
            info.ionosphere_height_estimated, units=common.UomType.M
        ),
        ionosphere_height_estimation_method_selected=translate_common.translate_ionosphere_height_estimation_method_type_to_model(
            info.ionosphere_height_estimation_method_selected
        ),
        ionosphere_height_estimation_latitude_value=translate_common.translate_float_with_unit_to_model(
            info.ionosphere_height_estimation_latitude_value, units=common.UomType.DEG
        ),
        ionosphere_height_estimation_flag=translate_common.translate_bool_to_model(
            info.ionosphere_height_estimation_flag
        ),
        ionosphere_height_estimation_method_used=translate_common.translate_ionosphere_height_estimation_method_type_to_model(
            info.ionosphere_height_estimation_method_used
        ),
        gaussian_filter_computation_flag=translate_common.translate_bool_to_model(
            info.gaussian_filter_computation_flag
        ),
        faraday_rotation_correction_applied=translate_common.translate_bool_to_model(
            info.faraday_rotation_correction_applied
        ),
        autofocus_shifts_applied=translate_common.translate_bool_to_model(info.autofocus_shifts_applied),
    )


def translate_fm_rate_estimate_list(
    fm_rate_estimates: common_annotation_models_l1.FmRateEstimatesListType,
) -> list[common.SlantRangePolynomialType]:
    """Translate list of fm rate estimate"""
    assert fm_rate_estimates.count is not None
    if len(fm_rate_estimates.fm_rate_estimate) != fm_rate_estimates.count:
        raise RuntimeError(
            "Inconsistency in fm rate estimate list: "
            + f"{len(fm_rate_estimates.fm_rate_estimate)} length and count: {fm_rate_estimates.count} do not match"
        )

    return [
        translate_common.translate_slant_range_polynomial(fm_rate_estimate)
        for fm_rate_estimate in fm_rate_estimates.fm_rate_estimate
    ]


def translate_fm_rate_estimate_list_to_model(
    fm_rate_estimates: list[common.SlantRangePolynomialType],
) -> common_annotation_models_l1.FmRateEstimatesListType:
    """Translate list of fm rate estimate"""
    return common_annotation_models_l1.FmRateEstimatesListType(
        [
            translate_common.translate_slant_range_polynomial_to_model(fm_rate_estimate)
            for fm_rate_estimate in fm_rate_estimates
        ],
        count=len(fm_rate_estimates),
    )


def translate_doppler_parameters(
    params: common_annotation_models_l1.DopplerParametersType,
) -> common_annotation_l1.DopplerParametersType:
    """Translate doppler parameters section"""
    assert params.dc_estimate_list is not None
    assert params.fm_rate_estimate_list is not None
    return common_annotation_l1.DopplerParametersType(
        dc_estimate_list=translate_dc_estimate_list_type(params.dc_estimate_list),
        fm_rate_estimate_list=translate_fm_rate_estimate_list(params.fm_rate_estimate_list),
    )


def translate_doppler_parameters_to_model(
    params: common_annotation_l1.DopplerParametersType,
) -> common_annotation_models_l1.DopplerParametersType:
    """Translate doppler parameters section"""
    return common_annotation_models_l1.DopplerParametersType(
        dc_estimate_list=translate_dc_estimate_list_type_to_model(params.dc_estimate_list),
        fm_rate_estimate_list=translate_fm_rate_estimate_list_to_model(params.fm_rate_estimate_list),
    )


def translate_error_counters(
    counters: common_annotation_models_l1.ErrorCountersType,
) -> common_annotation_l1.ErrorCountersType:
    """Translate error counters"""
    assert counters.num_isp_header_errors is not None
    assert counters.num_isp_missing is not None

    return common_annotation_l1.ErrorCountersType(
        num_isp_header_errors=counters.num_isp_header_errors,
        num_isp_missing=counters.num_isp_missing,
    )


def translate_error_counters_to_model(
    counters: common_annotation_l1.ErrorCountersType,
) -> common_annotation_models_l1.ErrorCountersType:
    """Translate error counters"""
    return common_annotation_models_l1.ErrorCountersType(
        num_isp_header_errors=counters.num_isp_header_errors,
        num_isp_missing=counters.num_isp_missing,
    )


def translate_raw_data_statistics(
    stats: common_annotation_models_l1.RawDataStatisticsType,
) -> common_annotation_l1.RawDataStatisticsType:
    """Translate raw data statistics"""
    assert stats.i_bias is not None
    assert stats.q_bias is not None
    assert stats.iq_quadrature_departure is not None
    assert stats.iq_gain_imbalance is not None
    assert stats.polarisation is not None
    return common_annotation_l1.RawDataStatisticsType(
        i_bias=stats.i_bias,
        q_bias=stats.q_bias,
        iq_quadrature_departure=stats.iq_quadrature_departure,
        iq_gain_imbalance=stats.iq_gain_imbalance,
        polarisation=translate_common.translate_polarisation_type(stats.polarisation),
    )


def translate_raw_data_statistics_to_model(
    stats: common_annotation_l1.RawDataStatisticsType,
) -> common_annotation_models_l1.RawDataStatisticsType:
    """Translate raw data statistics"""
    return common_annotation_models_l1.RawDataStatisticsType(
        i_bias=float(stats.i_bias),
        q_bias=float(stats.q_bias),
        iq_quadrature_departure=float(stats.iq_quadrature_departure),
        iq_gain_imbalance=float(stats.iq_gain_imbalance),
        polarisation=translate_common.translate_polarisation_type_to_model(stats.polarisation),
    )


def translate_raw_data_statistics_list(
    stat_list: common_annotation_models_l1.RawDataStatisticsListType,
) -> list[common_annotation_l1.RawDataStatisticsType]:
    """Translate raw data statistics list"""
    assert stat_list.count is not None
    if len(stat_list.raw_data_statistics) != stat_list.count:
        raise RuntimeError(
            "Inconsistency in raw data statistics list: "
            + f"{len(stat_list.raw_data_statistics)} length and count: {stat_list.count} do not match"
        )

    return [translate_raw_data_statistics(stat) for stat in stat_list.raw_data_statistics]


def translate_raw_data_statistics_list_to_model(
    stat_list: list[common_annotation_l1.RawDataStatisticsType],
) -> common_annotation_models_l1.RawDataStatisticsListType:
    """Translate raw data statistics list"""
    return common_annotation_models_l1.RawDataStatisticsListType(
        raw_data_statistics=[translate_raw_data_statistics_to_model(stat) for stat in stat_list],
        count=len(stat_list),
    )


def translate_raw_data_analysis(
    analysis: common_annotation_models_l1.RawDataAnalysisType,
) -> common_annotation_l1.RawDataAnalysisType:
    """Translate raw data analysis"""
    assert analysis.error_counters is not None
    assert analysis.raw_data_statistics_list is not None
    return common_annotation_l1.RawDataAnalysisType(
        error_counters=translate_error_counters(analysis.error_counters),
        raw_data_statistics_list=translate_raw_data_statistics_list(analysis.raw_data_statistics_list),
    )


def translate_raw_data_analysis_to_model(
    analysis: common_annotation_l1.RawDataAnalysisType,
) -> common_annotation_models_l1.RawDataAnalysisType:
    """Translate raw data analysis"""
    return common_annotation_models_l1.RawDataAnalysisType(
        error_counters=translate_error_counters_to_model(analysis.error_counters),
        raw_data_statistics_list=translate_raw_data_statistics_list_to_model(analysis.raw_data_statistics_list),
    )


def translate_internal_calibration_sequence(
    sequence: common_annotation_models_l1.InternalCalibrationSequenceType,
) -> common_annotation_l1.InternalCalibrationSequenceType:
    """Translate internal calibration sequence"""
    assert sequence.azimuth_time is not None
    assert sequence.drift_amplitude is not None
    assert sequence.drift_phase is not None
    assert sequence.model_drift_amplitude is not None
    assert sequence.model_drift_phase is not None
    assert sequence.relative_drift_valid_flag is not None
    assert sequence.absolute_drift_valid_flag is not None
    assert sequence.cross_correlation_bandwidth is not None
    assert sequence.cross_correlation_pslr is not None
    assert sequence.cross_correlation_islr is not None
    assert sequence.cross_correlation_peak_location is not None
    assert sequence.reconstructed_replica_valid_flag is not None
    assert sequence.internal_time_delay is not None
    assert sequence.internal_tx_channel_imbalance_amplitude is not None
    assert sequence.internal_tx_channel_imbalance_phase is not None
    assert sequence.internal_rx_channel_imbalance_amplitude is not None
    assert sequence.internal_rx_channel_imbalance_phase is not None
    assert sequence.transmit_power_tracking_d1_amplitude is not None
    assert sequence.transmit_power_tracking_d1_phase is not None
    assert sequence.receive_power_tracking_d1_amplitude is not None
    assert sequence.receive_power_tracking_d1_phase is not None
    assert sequence.transmit_power_tracking_d2_amplitude is not None
    assert sequence.transmit_power_tracking_d2_phase is not None
    assert sequence.receive_power_tracking_d2_amplitude is not None
    assert sequence.receive_power_tracking_d2_phase is not None

    return common_annotation_l1.InternalCalibrationSequenceType(
        azimuth_time=translate_common.translate_datetime(sequence.azimuth_time),
        drift_amplitude=sequence.drift_amplitude,
        drift_phase=translate_common.translate_float_with_unit(sequence.drift_phase),
        model_drift_amplitude=sequence.model_drift_amplitude,
        model_drift_phase=translate_common.translate_float_with_unit(sequence.model_drift_phase),
        relative_drift_valid_flag=translate_common.translate_bool(sequence.relative_drift_valid_flag),
        absolute_drift_valid_flag=translate_common.translate_bool(sequence.absolute_drift_valid_flag),
        cross_correlation_bandwidth=translate_common.translate_float_with_unit(sequence.cross_correlation_bandwidth),
        cross_correlation_pslr=translate_common.translate_float_with_unit(sequence.cross_correlation_pslr),
        cross_correlation_islr=translate_common.translate_float_with_unit(sequence.cross_correlation_islr),
        cross_correlation_peak_location=translate_common.translate_float_with_unit(
            sequence.cross_correlation_peak_location
        ),
        reconstructed_replica_valid_flag=translate_common.translate_bool(sequence.reconstructed_replica_valid_flag),
        internal_time_delay=translate_common.translate_float_with_unit(sequence.internal_time_delay),
        internal_tx_channel_imbalance_amplitude=sequence.internal_tx_channel_imbalance_amplitude,
        internal_tx_channel_imbalance_phase=translate_common.translate_float_with_unit(
            sequence.internal_tx_channel_imbalance_phase
        ),
        internal_rx_channel_imbalance_amplitude=sequence.internal_rx_channel_imbalance_amplitude,
        internal_rx_channel_imbalance_phase=translate_common.translate_float_with_unit(
            sequence.internal_rx_channel_imbalance_phase
        ),
        transmit_power_tracking_d1_amplitude=sequence.transmit_power_tracking_d1_amplitude,
        transmit_power_tracking_d1_phase=translate_common.translate_float_with_unit(
            sequence.transmit_power_tracking_d1_phase
        ),
        receive_power_tracking_d1_amplitude=sequence.receive_power_tracking_d1_amplitude,
        receive_power_tracking_d1_phase=translate_common.translate_float_with_unit(
            sequence.receive_power_tracking_d1_phase
        ),
        transmit_power_tracking_d2_amplitude=sequence.transmit_power_tracking_d2_amplitude,
        transmit_power_tracking_d2_phase=translate_common.translate_float_with_unit(
            sequence.transmit_power_tracking_d2_phase
        ),
        receive_power_tracking_d2_amplitude=sequence.receive_power_tracking_d2_amplitude,
        receive_power_tracking_d2_phase=translate_common.translate_float_with_unit(
            sequence.receive_power_tracking_d2_phase
        ),
    )


def translate_internal_calibration_sequence_to_model(
    sequence: common_annotation_l1.InternalCalibrationSequenceType,
) -> common_annotation_models_l1.InternalCalibrationSequenceType:
    """Translate internal calibration sequence"""
    return common_annotation_models_l1.InternalCalibrationSequenceType(
        azimuth_time=translate_common.translate_datetime_to_model(sequence.azimuth_time),
        drift_amplitude=sequence.drift_amplitude,
        drift_phase=translate_common.translate_float_with_unit_to_model(sequence.drift_phase, units=common.UomType.RAD),
        model_drift_amplitude=sequence.model_drift_amplitude,
        model_drift_phase=translate_common.translate_float_with_unit_to_model(
            sequence.model_drift_phase, units=common.UomType.RAD
        ),
        relative_drift_valid_flag=translate_common.translate_bool_to_model(sequence.relative_drift_valid_flag),
        absolute_drift_valid_flag=translate_common.translate_bool_to_model(sequence.absolute_drift_valid_flag),
        cross_correlation_bandwidth=translate_common.translate_float_with_unit_to_model(
            sequence.cross_correlation_bandwidth, units=common.UomType.HZ
        ),
        cross_correlation_pslr=translate_common.translate_float_with_unit_to_model(
            sequence.cross_correlation_pslr, units=common.UomType.D_B
        ),
        cross_correlation_islr=translate_common.translate_float_with_unit_to_model(
            sequence.cross_correlation_islr, units=common.UomType.D_B
        ),
        cross_correlation_peak_location=translate_common.translate_float_with_unit_to_model(
            sequence.cross_correlation_peak_location, units=common.UomType.SAMPLES
        ),
        reconstructed_replica_valid_flag=translate_common.translate_bool_to_model(
            sequence.reconstructed_replica_valid_flag
        ),
        internal_time_delay=translate_common.translate_float_with_unit_to_model(
            sequence.internal_time_delay, units=common.UomType.S
        ),
        internal_tx_channel_imbalance_amplitude=sequence.internal_tx_channel_imbalance_amplitude,
        internal_tx_channel_imbalance_phase=translate_common.translate_float_with_unit_to_model(
            sequence.internal_tx_channel_imbalance_phase, units=common.UomType.RAD
        ),
        internal_rx_channel_imbalance_amplitude=sequence.internal_rx_channel_imbalance_amplitude,
        internal_rx_channel_imbalance_phase=translate_common.translate_float_with_unit_to_model(
            sequence.internal_rx_channel_imbalance_phase, units=common.UomType.RAD
        ),
        transmit_power_tracking_d1_amplitude=sequence.transmit_power_tracking_d1_amplitude,
        transmit_power_tracking_d1_phase=translate_common.translate_float_with_unit_to_model(
            sequence.transmit_power_tracking_d1_phase, units=common.UomType.RAD
        ),
        receive_power_tracking_d1_amplitude=sequence.receive_power_tracking_d1_amplitude,
        receive_power_tracking_d1_phase=translate_common.translate_float_with_unit_to_model(
            sequence.receive_power_tracking_d1_phase, units=common.UomType.RAD
        ),
        transmit_power_tracking_d2_amplitude=sequence.transmit_power_tracking_d2_amplitude,
        transmit_power_tracking_d2_phase=translate_common.translate_float_with_unit_to_model(
            sequence.transmit_power_tracking_d2_phase, units=common.UomType.RAD
        ),
        receive_power_tracking_d2_amplitude=sequence.receive_power_tracking_d2_amplitude,
        receive_power_tracking_d2_phase=translate_common.translate_float_with_unit_to_model(
            sequence.receive_power_tracking_d2_phase, units=common.UomType.RAD
        ),
    )


def translate_internal_calibration_sequence_list(
    sequences: common_annotation_models_l1.InternalCalibrationSequenceListType,
) -> tuple[common.PolarisationType, list[common_annotation_l1.InternalCalibrationSequenceType]]:
    """Translate internal calibration sequence list"""
    assert sequences.polarisation is not None
    assert sequences.count is not None
    if len(sequences.internal_calibration_sequence) != sequences.count:
        raise RuntimeError(
            "Inconsistency in internal calibration sequence list: "
            + f"{len(sequences.internal_calibration_sequence)} length and count: {sequences.count} do not match"
        )

    return translate_common.translate_polarisation_type(sequences.polarisation), [
        translate_internal_calibration_sequence(sequence) for sequence in sequences.internal_calibration_sequence
    ]


def translate_internal_calibration_sequence_list_to_model(
    polarisation: common.PolarisationType,
    sequences: list[common_annotation_l1.InternalCalibrationSequenceType],
) -> common_annotation_models_l1.InternalCalibrationSequenceListType:
    """Translate raw data statistics list"""
    return common_annotation_models_l1.InternalCalibrationSequenceListType(
        internal_calibration_sequence=[
            translate_internal_calibration_sequence_to_model(sequence) for sequence in sequences
        ],
        count=len(sequences),
        polarisation=translate_common.translate_polarisation_type_to_model(polarisation),
    )


def translate_noise_sequence(
    sequence: common_annotation_models_l1.NoiseSequenceType,
) -> common_annotation_l1.NoiseSequenceType:
    """Translate noise sequence"""
    assert sequence.azimuth_time is not None
    assert sequence.noise_power_correction_factor is not None
    assert sequence.number_of_noise_lines is not None
    return common_annotation_l1.NoiseSequenceType(
        azimuth_time=translate_common.translate_datetime(sequence.azimuth_time),
        noise_power_correction_factor=sequence.noise_power_correction_factor,
        number_of_noise_lines=sequence.number_of_noise_lines,
    )


def translate_noise_sequence_to_model(
    sequence: common_annotation_l1.NoiseSequenceType,
) -> common_annotation_models_l1.NoiseSequenceType:
    """Translate noise sequence"""
    return common_annotation_models_l1.NoiseSequenceType(
        azimuth_time=translate_common.translate_datetime_to_model(sequence.azimuth_time),
        noise_power_correction_factor=sequence.noise_power_correction_factor,
        number_of_noise_lines=sequence.number_of_noise_lines,
    )


def translate_noise_sequence_list(
    sequences: common_annotation_models_l1.NoiseSequenceListType,
) -> tuple[common.PolarisationType, list[common_annotation_l1.NoiseSequenceType]]:
    """Translate internal calibration sequence list"""
    assert sequences.polarisation is not None
    assert sequences.count is not None
    if len(sequences.noise_sequence) != sequences.count:
        raise RuntimeError(
            "Inconsistency in noise sequence list: "
            + f"{len(sequences.noise_sequence)} length and count: {sequences.count} do not match"
        )

    return translate_common.translate_polarisation_type(sequences.polarisation), [
        translate_noise_sequence(sequence) for sequence in sequences.noise_sequence
    ]


def translate_noise_sequence_list_to_model(
    polarisation: common.PolarisationType,
    sequences: list[common_annotation_l1.NoiseSequenceType],
) -> common_annotation_models_l1.NoiseSequenceListType:
    """Translate raw data statistics list"""
    return common_annotation_models_l1.NoiseSequenceListType(
        noise_sequence=[translate_noise_sequence_to_model(sequence) for sequence in sequences],
        count=len(sequences),
        polarisation=translate_common.translate_polarisation_type_to_model(polarisation),
    )


def translate_internal_calibration(
    cal: common_annotation_models_l1.InternalCalibrationType,
) -> common_annotation_l1.InternalCalibrationType:
    """Translate internal calibration"""
    assert cal.internal_calibration_parameters_used is not None
    assert cal.range_reference_function_used is not None
    assert cal.noise_parameters_used is not None
    assert cal.internal_calibration_parameters_list is not None
    assert cal.noise_list is not None

    int_cal_sequences = {}
    for sequence in cal.internal_calibration_parameters_list.internal_calibration_sequence_list:
        pol, sequence_list = translate_internal_calibration_sequence_list(sequence)
        int_cal_sequences[pol] = sequence_list

    noise_sequences = {}
    for sequence in cal.noise_list.noise_sequence_list:
        pol, sequence_list = translate_noise_sequence_list(sequence)
        noise_sequences[pol] = sequence_list

    return common_annotation_l1.InternalCalibrationType(
        internal_calibration_parameters_used=translate_common.translate_internal_calibration_source(
            cal.internal_calibration_parameters_used
        ),
        range_reference_function_used=translate_common.translate_range_reference_function_type(
            cal.range_reference_function_used
        ),
        noise_parameters_used=translate_common.translate_internal_calibration_source(cal.noise_parameters_used),
        internal_calibration_parameters_list=int_cal_sequences,
        noise_list=noise_sequences,
    )


def translate_internal_calibration_to_model(
    cal: common_annotation_l1.InternalCalibrationType,
) -> common_annotation_models_l1.InternalCalibrationType:
    """Translate internal calibration"""

    int_cal_sequences = common_annotation_models_l1.InternalCalibrationParametersListType(
        [
            translate_internal_calibration_sequence_list_to_model(pol, sequence)
            for pol, sequence in cal.internal_calibration_parameters_list.items()
        ],
        count=len(cal.internal_calibration_parameters_list),
    )
    noise_sequences = common_annotation_models_l1.NoiseListType(
        [translate_noise_sequence_list_to_model(pol, sequence) for pol, sequence in cal.noise_list.items()],
        count=len(cal.noise_list),
    )

    return common_annotation_models_l1.InternalCalibrationType(
        internal_calibration_parameters_used=translate_common.translate_internal_calibration_source_to_model(
            cal.internal_calibration_parameters_used
        ),
        range_reference_function_used=translate_common.translate_range_reference_function_type_to_model(
            cal.range_reference_function_used
        ),
        noise_parameters_used=translate_common.translate_internal_calibration_source_to_model(
            cal.noise_parameters_used
        ),
        internal_calibration_parameters_list=int_cal_sequences,
        noise_list=noise_sequences,
    )


def translate_rfi_tmreport_type(
    report: common_annotation_models_l1.RfiTmreportType,
) -> common_annotation_l1.RfiTmreportType:
    """Translate time domain report item"""
    assert report.polarisation is not None
    assert report.percentage_affected_lines is not None
    assert report.avg_percentage_affected_samples is not None
    assert report.max_percentage_affected_samples is not None
    return common_annotation_l1.RfiTmreportType(
        polarisation=translate_common.translate_polarisation_type(report.polarisation),
        percentage_affected_lines=report.percentage_affected_lines,
        avg_percentage_affected_samples=report.avg_percentage_affected_samples,
        max_percentage_affected_samples=report.max_percentage_affected_samples,
    )


def translate_rfi_tmreport_type_to_model(
    report: common_annotation_l1.RfiTmreportType,
) -> common_annotation_models_l1.RfiTmreportType:
    """Translate time domain report item"""
    return common_annotation_models_l1.RfiTmreportType(
        polarisation=translate_common.translate_polarisation_type_to_model(report.polarisation),
        percentage_affected_lines=float(report.percentage_affected_lines),
        avg_percentage_affected_samples=float(report.avg_percentage_affected_samples),
        max_percentage_affected_samples=float(report.max_percentage_affected_samples),
    )


def translate_rfi_tmreport_list_type(
    reports: common_annotation_models_l1.RfiTmreportListType,
) -> list[common_annotation_l1.RfiTmreportType]:
    """Translate list of rfi tm report"""
    assert reports.count is not None
    if len(reports.rfi_tmreport) != reports.count:
        raise RuntimeError(
            "Inconsistency in rfi tm report list: "
            + f"{len(reports.rfi_tmreport)} length and count: {reports.count} do not match"
        )

    return [translate_rfi_tmreport_type(report) for report in reports.rfi_tmreport]


def translate_rfi_tmreport_list_type_to_model(
    reports: list[common_annotation_l1.RfiTmreportType],
) -> common_annotation_models_l1.RfiTmreportListType:
    """Translate list of rfi tm report"""
    return common_annotation_models_l1.RfiTmreportListType(
        [translate_rfi_tmreport_type_to_model(report) for report in reports],
        count=len(reports),
    )


def translate_rfi_isolated_fmreport_type(
    report: common_annotation_models_l1.RfiIsolatedFmreportType,
) -> common_annotation_l1.RfiIsolatedFmreportType:
    """Translate isolated RI report item"""
    assert report.polarisation is not None
    assert report.percentage_affected_lines is not None
    assert report.max_percentage_affected_bw is not None
    assert report.avg_percentage_affected_bw is not None
    return common_annotation_l1.RfiIsolatedFmreportType(
        polarisation=translate_common.translate_polarisation_type(report.polarisation),
        percentage_affected_lines=report.percentage_affected_lines,
        max_percentage_affected_bw=report.max_percentage_affected_bw,
        avg_percentage_affected_bw=report.avg_percentage_affected_bw,
    )


def translate_rfi_isolated_fmreport_type_to_model(
    report: common_annotation_l1.RfiIsolatedFmreportType,
) -> common_annotation_models_l1.RfiIsolatedFmreportType:
    """Translate isolated RI report item"""
    return common_annotation_models_l1.RfiIsolatedFmreportType(
        polarisation=translate_common.translate_polarisation_type_to_model(report.polarisation),
        percentage_affected_lines=float(report.percentage_affected_lines),
        max_percentage_affected_bw=float(report.max_percentage_affected_bw),
        avg_percentage_affected_bw=float(report.avg_percentage_affected_bw),
    )


def translate_rfi_isolated_fmreport_list_type(
    reports: common_annotation_models_l1.RfiIsolatedFmreportListType,
) -> list[common_annotation_l1.RfiIsolatedFmreportType]:
    """Translate list of  rfi isolated fm report"""
    assert reports.count is not None
    if len(reports.rfi_isolated_fmreport) != reports.count:
        raise RuntimeError(
            "Inconsistency in rfi isolated fm report list: "
            + f"{len(reports.rfi_isolated_fmreport)} length and count: {reports.count} do not match"
        )

    return [translate_rfi_isolated_fmreport_type(report) for report in reports.rfi_isolated_fmreport]


def translate_rfi_isolated_fmreport_list_type_to_model(
    reports: list[common_annotation_l1.RfiIsolatedFmreportType],
) -> common_annotation_models_l1.RfiIsolatedFmreportListType:
    """Translate list of  rfi isolated fm report"""
    return common_annotation_models_l1.RfiIsolatedFmreportListType(
        [translate_rfi_isolated_fmreport_type_to_model(report) for report in reports],
        count=len(reports),
    )


def translate_rfi_persistent_fmreport_type(
    report: common_annotation_models_l1.RfiPersistentFmreportType,
) -> common_annotation_l1.RfiPersistentFmreportType:
    """Translate persistent frequency domain report item"""
    assert report.polarisation is not None
    assert report.percentage_affected_lines is not None
    assert report.max_percentage_affected_bw is not None
    assert report.avg_percentage_affected_bw is not None
    return common_annotation_l1.RfiPersistentFmreportType(
        polarisation=translate_common.translate_polarisation_type(report.polarisation),
        percentage_affected_lines=report.percentage_affected_lines,
        max_percentage_affected_bw=report.max_percentage_affected_bw,
        avg_percentage_affected_bw=report.avg_percentage_affected_bw,
    )


def translate_rfi_persistent_fmreport_type_to_model(
    report: common_annotation_l1.RfiPersistentFmreportType,
) -> common_annotation_models_l1.RfiPersistentFmreportType:
    """Translate persistent frequency domain report item"""
    return common_annotation_models_l1.RfiPersistentFmreportType(
        polarisation=translate_common.translate_polarisation_type_to_model(report.polarisation),
        percentage_affected_lines=float(report.percentage_affected_lines),
        max_percentage_affected_bw=float(report.max_percentage_affected_bw),
        avg_percentage_affected_bw=float(report.avg_percentage_affected_bw),
    )


def translate_rfi_persistent_fmreport_list(
    reports: common_annotation_models_l1.RfiPersistentFmreportListType,
) -> list[common_annotation_l1.RfiPersistentFmreportType]:
    """Translate persistent frequency domain report item list"""
    assert reports.count is not None
    if len(reports.rfi_persistent_fmreport) != reports.count:
        raise RuntimeError(
            "Inconsistency in rfi persistent fm report list: "
            + f"{len(reports.rfi_persistent_fmreport)} length and count: {reports.count} do not match"
        )

    return [translate_rfi_persistent_fmreport_type(report) for report in reports.rfi_persistent_fmreport]


def translate_rfi_persistent_fmreport_list_type_to_model(
    reports: list[common_annotation_l1.RfiPersistentFmreportType],
) -> common_annotation_models_l1.RfiPersistentFmreportListType:
    """Translate persistent frequency domain report item list"""
    return common_annotation_models_l1.RfiPersistentFmreportListType(
        [translate_rfi_persistent_fmreport_type_to_model(report) for report in reports],
        count=len(reports),
    )


def translate_rfi_mitigation(
    info: common_annotation_models_l1.RfiMitigationType,
) -> common_annotation_l1.RfiMitigationType:
    """Translate RFI mitigation"""
    return common_annotation_l1.RfiMitigationType(
        rfi_tmreport_list=(translate_rfi_tmreport_list_type(info.rfi_tmreport_list) if info.rfi_tmreport_list else []),
        rfi_isolated_fmreport_list=(
            translate_rfi_isolated_fmreport_list_type(info.rfi_isolated_fmreport_list)
            if info.rfi_isolated_fmreport_list
            else []
        ),
        rfi_persistent_fmreport_list=(
            translate_rfi_persistent_fmreport_list(info.rfi_persistent_fmreport_list)
            if info.rfi_persistent_fmreport_list
            else []
        ),
    )


def translate_rfi_mitigation_to_model(
    info: common_annotation_l1.RfiMitigationType,
) -> common_annotation_models_l1.RfiMitigationType:
    """Translate RFI mitigation"""
    return common_annotation_models_l1.RfiMitigationType(
        rfi_tmreport_list=(
            translate_rfi_tmreport_list_type_to_model(info.rfi_tmreport_list) if info.rfi_tmreport_list else None
        ),
        rfi_isolated_fmreport_list=(
            translate_rfi_isolated_fmreport_list_type_to_model(info.rfi_isolated_fmreport_list)
            if info.rfi_isolated_fmreport_list
            else None
        ),
        rfi_persistent_fmreport_list=(
            translate_rfi_persistent_fmreport_list_type_to_model(info.rfi_persistent_fmreport_list)
            if info.rfi_persistent_fmreport_list
            else None
        ),
    )


def translate_radiometric_calibration(
    info: common_annotation_models_l1.RadiometricCalibrationType,
) -> common_annotation_l1.RadiometricCalibrationType:
    """Translate radiometric calibration section"""

    assert info.absolute_calibration_constant_list is not None

    calibrations = [
        translate_common.translate_float_with_polarisation(values)
        for values in info.absolute_calibration_constant_list.absolute_calibration_constant
    ]

    calibrations_per_pol = {pol: value for value, pol in calibrations}

    return common_annotation_l1.RadiometricCalibrationType(absolute_calibration_constant_list=calibrations_per_pol)


def translate_radiometric_calibration_to_model(
    info: common_annotation_l1.RadiometricCalibrationType,
) -> common_annotation_models_l1.RadiometricCalibrationType:
    """Translate radiometric calibration section"""

    calibrations = [
        translate_common.translate_float_with_polarisation_to_model((value, pol))
        for pol, value in info.absolute_calibration_constant_list.items()
    ]

    return common_annotation_models_l1.RadiometricCalibrationType(
        absolute_calibration_constant_list=common_annotation_models_l1.CalibrationConstantListType(
            calibrations, len(calibrations)
        ),
    )


def translate_geometry(
    geometry: common_annotation_models_l1.GeometryType,
) -> common_annotation_l1.GeometryType:
    """Translate geometry"""
    assert geometry.height_model is not None
    assert geometry.height_model_used_flag is not None
    assert geometry.roll_bias is not None
    return common_annotation_l1.GeometryType(
        height_model=translate_common.translate_height_model(geometry.height_model),
        height_model_used_flag=translate_common.translate_bool(geometry.height_model_used_flag),
        roll_bias=translate_common.translate_float_with_unit(geometry.roll_bias),
    )


def translate_geometry_to_model(
    geometry: common_annotation_l1.GeometryType,
) -> common_annotation_models_l1.GeometryType:
    """Translate geometry"""
    return common_annotation_models_l1.GeometryType(
        height_model=translate_common.translate_height_model_to_model(geometry.height_model),
        height_model_used_flag=translate_common.translate_bool_to_model(geometry.height_model_used_flag),
        roll_bias=translate_common.translate_float_with_unit_to_model(geometry.roll_bias, units=common.UomType.DEG),
    )


def translate_quality_parameters(
    params: common_annotation_models_l1.QualityParametersType,
) -> common_annotation_l1.QualityParametersType:
    """Translate quality parameters"""
    assert params.missing_ispfraction is not None
    assert params.max_ispgap is not None
    assert params.max_ispgap_threshold is not None
    assert params.invalid_raw_data_samples is not None
    assert params.raw_mean_expected is not None
    assert params.raw_mean_threshold is not None
    assert params.raw_std_expected is not None
    assert params.raw_std_threshold is not None
    assert params.rfi_tmfraction is not None
    assert params.max_rfitmpercentage is not None
    assert params.rfi_fmfraction is not None
    assert params.max_rfifmpercentage is not None
    assert params.invalid_drift_fraction is not None
    assert params.max_invalid_drift_fraction is not None
    assert params.invalid_replica_fraction is not None
    assert params.invalid_dcestimates_fraction is not None
    assert params.dc_rmserror_threshold is not None
    assert params.residual_ionospheric_phase_screen_std is not None
    assert params.invalid_blocks_percentage is not None
    assert params.invalid_blocks_percentage_threshold is not None
    assert params.polarisation is not None

    return common_annotation_l1.QualityParametersType(
        missing_ispfraction=params.missing_ispfraction,
        max_ispgap=params.max_ispgap,
        max_ispgap_threshold=params.max_ispgap_threshold,
        invalid_raw_data_samples=params.invalid_raw_data_samples,
        raw_mean_expected=params.raw_mean_expected,
        raw_mean_threshold=params.raw_mean_threshold,
        raw_std_expected=params.raw_std_expected,
        raw_std_threshold=params.raw_std_threshold,
        rfi_tmfraction=params.rfi_tmfraction,
        max_rfitmpercentage=params.max_rfitmpercentage,
        rfi_fmfraction=params.rfi_fmfraction,
        max_rfifmpercentage=params.max_rfifmpercentage,
        invalid_drift_fraction=params.invalid_drift_fraction,
        max_invalid_drift_fraction=params.max_invalid_drift_fraction,
        invalid_replica_fraction=params.invalid_replica_fraction,
        invalid_dcestimates_fraction=params.invalid_dcestimates_fraction,
        dc_rmserror_threshold=translate_common.translate_float_with_unit(params.dc_rmserror_threshold),
        residual_ionospheric_phase_screen_std=translate_common.translate_float_with_unit(
            params.residual_ionospheric_phase_screen_std
        ),
        invalid_blocks_percentage=params.invalid_blocks_percentage,
        invalid_blocks_percentage_threshold=params.invalid_blocks_percentage_threshold,
        polarisation=translate_common.translate_polarisation_type(params.polarisation),
    )


def translate_quality_parameters_to_model(
    params: common_annotation_l1.QualityParametersType,
) -> common_annotation_models_l1.QualityParametersType:
    """Translate quality parameters"""

    return common_annotation_models_l1.QualityParametersType(
        missing_ispfraction=float(params.missing_ispfraction),
        max_ispgap=params.max_ispgap,
        max_ispgap_threshold=params.max_ispgap_threshold,
        invalid_raw_data_samples=float(params.invalid_raw_data_samples),
        raw_mean_expected=float(params.raw_mean_expected),
        raw_mean_threshold=float(params.raw_mean_threshold),
        raw_std_expected=float(params.raw_std_expected),
        raw_std_threshold=float(params.raw_std_threshold),
        rfi_tmfraction=float(params.rfi_tmfraction),
        max_rfitmpercentage=float(params.max_rfitmpercentage),
        rfi_fmfraction=float(params.rfi_fmfraction),
        max_rfifmpercentage=float(params.max_rfifmpercentage),
        invalid_drift_fraction=float(params.invalid_drift_fraction),
        max_invalid_drift_fraction=float(params.max_invalid_drift_fraction),
        invalid_replica_fraction=float(params.invalid_replica_fraction),
        invalid_dcestimates_fraction=float(params.invalid_dcestimates_fraction),
        dc_rmserror_threshold=translate_common.translate_float_with_unit_to_model(
            params.dc_rmserror_threshold, units=common.UomType.HZ
        ),
        residual_ionospheric_phase_screen_std=translate_common.translate_float_with_unit_to_model(
            params.residual_ionospheric_phase_screen_std, units=common.UomType.RAD
        ),
        invalid_blocks_percentage=float(params.invalid_blocks_percentage),
        invalid_blocks_percentage_threshold=float(params.invalid_blocks_percentage_threshold),
        polarisation=translate_common.translate_polarisation_type_to_model(params.polarisation),
    )


def translate_quality(
    quality: common_annotation_models_l1.QualityType,
) -> common_annotation_l1.QualityType:
    """Translate quality"""
    assert quality.overall_product_quality_index is not None
    assert quality.quality_parameters_list is not None
    return common_annotation_l1.QualityType(
        overall_product_quality_index=quality.overall_product_quality_index,
        quality_parameters_list=[
            translate_quality_parameters(params) for params in quality.quality_parameters_list.quality_parameters
        ],
    )


def translate_quality_to_model(
    quality: common_annotation_l1.QualityType,
) -> common_annotation_models_l1.QualityType:
    """Translate quality"""
    params_list = [translate_quality_parameters_to_model(params) for params in quality.quality_parameters_list]
    return common_annotation_models_l1.QualityType(
        overall_product_quality_index=quality.overall_product_quality_index,
        quality_parameters_list=common_annotation_models_l1.QualityParametersListType(
            quality_parameters=params_list, count=len(params_list)
        ),
    )


def translate_polarimetric_distortion(
    distortion: common_annotation_models_l1.PolarimetricDistortionType,
) -> common_annotation_l1.PolarimetricDistortionType:
    """Polarimetric distortion type"""
    assert distortion.cross_talk_list is not None
    assert distortion.channel_imbalance_list is not None

    return common_annotation_l1.PolarimetricDistortionType(
        cross_talk=translate_common.translate_cross_talk_list(distortion.cross_talk_list),
        channel_imbalance=translate_common.translate_channel_imbalance_list(distortion.channel_imbalance_list),
    )


def translate_polarimetric_distortion_to_model(
    distortion: common_annotation_l1.PolarimetricDistortionType,
) -> common_annotation_models_l1.PolarimetricDistortionType:
    """Polarimetric distortion type"""
    return common_annotation_models_l1.PolarimetricDistortionType(
        cross_talk_list=translate_common.translate_cross_talk_list_to_model(distortion.cross_talk),
        channel_imbalance_list=translate_common.translate_channel_imbalance_list_to_model(distortion.channel_imbalance),
    )
