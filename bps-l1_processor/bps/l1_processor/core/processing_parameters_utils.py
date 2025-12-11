# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to fill the processing parameters file
------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from arepytools.geometry.conversions import llh2xyz
from bps.common import STRIPMAP_SWATHS, Polarization, Swath
from bps.l1_core_processor import processing_parameters as configuration
from bps.l1_processor.core.channel_imbalance import (
    ChannelImbalanceProcessingParametersL1,
)
from bps.l1_processor.processor_interface import aux_pp1


class MissingSwathParams(RuntimeError):
    """Raised when swath-specific parameters ares not found in the aux product"""


def _translate_filtering_method(
    method: Literal["NOTCH_FILTER", "NEAREST_NEIGHBOUR_INTERPOLATION"],
) -> Literal["NOTCH", "NEAREST_NEIGHBOUR"]:
    """Translate filtering method from aux pp1 to core processor"""
    if method == "NOTCH_FILTER":
        return "NOTCH"
    if method == "NEAREST_NEIGHBOUR_INTERPOLATION":
        return "NEAREST_NEIGHBOUR"

    raise RuntimeError(f"Unknown aux pp1 filtering method: {method}")


def fill_rfi_mitigation_conf(conf: aux_pp1.RFIMitigationConf, swath: str) -> configuration.RFIMitigationConf:
    """Fill RFI mitigation configuration"""
    time_domain_conf = configuration.RFIMitigationConf.TimeDomainConf(
        correction_mode=configuration.RFIMitigationConf.TimeDomainConf.CorrectionMode.NEAREST,
        percentile_threshold=conf.time_domain_processing_parameters.percentile_threshold,
        median_filter_block_lines=conf.time_domain_processing_parameters.median_filter_length,
        lines_in_estimate_block=conf.time_domain_processing_parameters.block_lines,
        box_filter_azimuth_dimension=conf.time_domain_processing_parameters.box_lines,
        box_filter_range_dimension=conf.time_domain_processing_parameters.box_samples,
        morph_open_line_length=conf.time_domain_processing_parameters.morphological_open_operator_lines,
        morph_close_line_length=conf.time_domain_processing_parameters.morphological_close_operator_lines,
        morph_close_before_open=None,
        morph_open_close_iterations=None,
    )
    freq_domain_conf = configuration.RFIMitigationConf.FrequencyDomainConf(
        block_size=conf.freq_domain_processing_parameters.block_lines,
        periodgram_size=conf.freq_domain_processing_parameters.periodgram_size,
        persistent_rfi_threshold=conf.freq_domain_processing_parameters.persistent_rfi_threshold,
        isolated_rfi_threshold=conf.freq_domain_processing_parameters.isolated_rfi_threshold,
        threshold_std=conf.freq_domain_processing_parameters.isolated_rfi_psd_std_threshold,
        percentile_low=0.25,
        percentile_high=0.25,
        power_loss_threshold=conf.freq_domain_processing_parameters.power_loss_threshold
        if conf.freq_domain_processing_parameters.enable_power_loss_compensation
        else None,
        remove_interferences=True,
        filtering_mode=_translate_filtering_method(conf.freq_domain_processing_parameters.mitigation_method),
    )

    rfi_composition_method = (
        configuration.RFIMitigationConf.MaskCompositionMethod(conf.mask_generation_method.name)
        if conf.mask is aux_pp1.RFIMitigationConf.MaskType.SINGLE
        else configuration.RFIMitigationConf.MaskCompositionMethod.NONE
    )

    return configuration.RFIMitigationConf(
        rfi_mitigation_method=configuration.RFIMitigationConf.Method(conf.mitigation_method.name),
        rfi_mask_composition_method=rfi_composition_method,
        time_domain_conf=time_domain_conf,
        frequency_domain_conf=freq_domain_conf,
        swath=swath,
    )


def fill_range_focuser_conf(
    conf: aux_pp1.RangeCompressionConf, output_prf_value: float | None, instrument_swst_bias: float, swath: str
) -> configuration.RangeFocuserConf:
    """Fill range focuser configuration"""
    params = conf.parameters.get(Swath(swath))

    if params is None:
        raise MissingSwathParams(f"Range focuser parameters missing for swath {swath}")

    apply_window = params.window_type != aux_pp1.WindowType.NONE
    window = configuration.WindowConf(
        window_type=configuration.WindowConf.Type(params.window_type.name),
        window_parameter=params.window_coefficient,
        window_look_bandwidth=configuration.Quantity(
            value=params.processing_bandwidth,
            unit=configuration.Quantity.Unit.HZ,
        ),
        window_transition_bandwidth=configuration.Quantity(
            value=params.processing_bandwidth,
            unit=configuration.Quantity.Unit.HZ,
        ),
    )

    return configuration.RangeFocuserConf(
        flag_ortog=False,
        apply_range_spectral_weighting_window=apply_window,
        range_spectral_weighting_window=window,
        swst_bias=params.time_bias + instrument_swst_bias,
        range_decimation_factor=None,
        apply_rx_gain_correction=None,
        focusing_method=configuration.RangeFocuserConf.Method(conf.range_compression_method.name),
        output_prf_value=output_prf_value,
        output_range_border_policy=(
            configuration.BorderPolicy.DATA if conf.extended_swath_processing else configuration.BorderPolicy.CUT
        ),
        swath=swath,
    )


def fill_doppler_estimator_conf(
    conf: aux_pp1.DopplerEstimationConf, swath: str
) -> configuration.DopplerEstimatorStripmapConf:
    """Fill doppler estimator configuration"""
    dc_method_map = {
        aux_pp1.DopplerEstimationConf.Method.GEOMETRY: configuration.DopplerEstimatorStripmapConf.Method.GEOMETRICAL,
        aux_pp1.DopplerEstimationConf.Method.COMBINED: configuration.DopplerEstimatorStripmapConf.Method.COMBINED,
        aux_pp1.DopplerEstimationConf.Method.FIXED: configuration.DopplerEstimatorStripmapConf.Method.COMBINED,
    }

    if conf.method not in dc_method_map:
        raise RuntimeError(f"Unknown doppler estimation method: {conf.method}")
    dc_estimation_method = dc_method_map.get(conf.method)
    assert dc_estimation_method is not None

    return configuration.DopplerEstimatorStripmapConf(
        blocks=conf.block_samples,
        blockl=conf.block_lines,
        undersampling_snrd_cazimuth_ratio=5,
        undersampling_snrd_crange_ratio=1,
        az_max_frequency_search_bin_number=4096,
        rg_max_frequency_search_bin_number=1024,
        az_max_frequency_search_norm_band=0.02,
        rg_max_frequency_search_norm_band=0.1,
        nummlbf=10,
        nbestblocks=5,
        rg_band=0.793103447364249,
        an_len=12,
        lookbf=0.25,
        lookbt=0.100000001490116,
        lookrp=0.0500000007450581,
        lookrs=0.01,
        decfac=3,
        flength=21,
        dftstep=3.9999998989515e-05,
        peakwid=0.0056,
        minamb=-20,
        maxamb=20,
        sthr=20,
        varth=0.5,
        pol_weights=[1, 1, 0, 0, 1, 1, 1],
        dc_estimation_method=dc_estimation_method,
        attitude_fitting=configuration.DopplerEstimatorStripmapConf.AttitudeFitting.AVERAGE,
        poly_changing_freq=conf.polynomial_update_rate,
        poly_estimation_constraint=configuration.DopplerEstimatorStripmapConf.PolyEstimationConstraint.UNCONSTRAINED,
        dc_core_algorithm=None,
        swath=swath,
    )


def fill_azimuth_conf(
    conf: aux_pp1.AzimuthCompressionConf,
    swath: str,
) -> configuration.AzimuthConf:
    """Fill azimuth configuration"""
    params = conf.parameters.get(Swath(swath))

    if params is None:
        raise MissingSwathParams(f"Missing azimuth configuration parameters for swath {swath}")

    processing_bandwidth = configuration.Quantity(
        params.processing_bandwidth,
        unit=configuration.Quantity.Unit.HZ,
    )

    apply_window = params.window_type != aux_pp1.WindowType.NONE
    window = configuration.WindowConf(
        window_type=configuration.WindowConf.Type(params.window_type.name),
        window_parameter=params.window_coefficient,
        window_look_bandwidth=processing_bandwidth,
        window_transition_bandwidth=processing_bandwidth,
    )

    bistatic_delay_correction_mode = configuration.AzimuthConf.BistaticDelayCorrectionMode.BIAS_ONLY
    if conf.bistatic_delay_correction:
        if conf.bistatic_delay_correction_method == aux_pp1.AzimuthCompressionConf.Method.FULL:
            bistatic_delay_correction_mode = configuration.AzimuthConf.BistaticDelayCorrectionMode.RANGE_DEPENDENT
        if conf.bistatic_delay_correction_method == aux_pp1.AzimuthCompressionConf.Method.BULK:
            bistatic_delay_correction_mode = configuration.AzimuthConf.BistaticDelayCorrectionMode.SCENE_CENTER

    pad_result_keep_margin = 3
    pad_result_remove_margin = 0
    pad_result = pad_result_remove_margin if conf.azimuth_focusing_margins_removal_flag else pad_result_keep_margin

    return configuration.AzimuthConf(
        lines_in_block=conf.block_lines,
        samples_in_block=conf.block_samples,
        azimuth_overlap=conf.block_overlap_lines,
        range_overlap=conf.block_overlap_samples,
        perform_interpolation=0,
        stolt_padding=0.4,
        range_modulation=False,
        apply_azimuth_spectral_weighting_window=apply_window,
        azimuth_spectral_weighting_window=window,
        apply_rg_shift=True,
        apply_az_shift=True,
        whitening_flag=False,
        antenna_length=0.0,
        pad_result=pad_result,
        lines_to_skip_dc_fr=None,
        samples_to_skip_dc_fr=None,
        focusing_method=configuration.AzimuthConf.Method.WK,
        az_proc_bandwidth=processing_bandwidth,
        bistatic_delay_correction_mode=bistatic_delay_correction_mode,
        azimuth_time_bias=params.time_bias,
        antenna_shift_compensation_mode=None,
        apply_pol_channels_coregistration=conf.azimuth_coregistration_flag,
        nominal_block_memory_size_cpu=None,
        nominal_block_memory_size_gpu=None,
        swath=swath,
    )


def fill_radiometric_calibration_conf(
    conf: aux_pp1.RadiometricCalibrationConf, swath: str
) -> configuration.RadiometricCalibrationConf:
    """Fill radiometric calibration configuration"""
    return configuration.RadiometricCalibrationConf(
        rsl_reference_distance=conf.reference_range,
        perform_rsl_compensation=conf.range_spreading_loss_compensation_enabled,
        perform_pattern_compensation=False,
        external_calibration_factor=conf.absolute_calibration_constant[Polarization.HH],
        apply_external_calibration_factor=True,
        output_quantity=None,
        perform_line_correction=None,
        fast_mode=True,
        processing_gain=None,
        swath=swath,
    )


def fill_polarimetric_processor_conf(
    conf: aux_pp1.PolarimetricCalibrationConf, swath: str
) -> configuration.PolarimetricProcessorConf:
    """Fill Polarimetric step configuration"""
    return configuration.PolarimetricProcessorConf(
        enable_cross_talk_compensation=conf.cross_talk_correction_flag,
        enable_channel_imbalance_compensation=conf.channel_imbalance_correction_flag,
        swath=swath,
    )


def translate_ionospheric_height_estimation_method(
    method: aux_pp1.IonosphereCalibrationConf.Method,
) -> configuration.IonosphericHeightEstimationMethod:
    """Translate ionospheric height estimation method"""
    if method == aux_pp1.IonosphereCalibrationConf.Method.AUTOMATIC:
        return configuration.IonosphericHeightEstimationMethod.AUTO
    if method == aux_pp1.IonosphereCalibrationConf.Method.FEATURE_TRACKING:
        return configuration.IonosphericHeightEstimationMethod.FEATURE_TRACKING
    if method == aux_pp1.IonosphereCalibrationConf.Method.SQUINT_SENSITIVITY:
        return configuration.IonosphericHeightEstimationMethod.SQUINT_SENSITIVITY
    if method == aux_pp1.IonosphereCalibrationConf.Method.MODEL:
        return configuration.IonosphericHeightEstimationMethod.MODEL
    if method == aux_pp1.IonosphereCalibrationConf.Method.FIXED:
        return configuration.IonosphericHeightEstimationMethod.NONE

    raise RuntimeError(f"Unknown aux pp1 method: {method}")


def _convert_latitude_threshold_to_z_threshold(latitude_deg: float) -> float:
    return float(llh2xyz([np.deg2rad(latitude_deg), 0.0, 0.0]).squeeze()[2])


def fill_ionospheric_calibration_conf(
    conf: aux_pp1.IonosphereCalibrationConf, swath: str
) -> configuration.IonosphericCalibrationConf:
    """Fill Polarimetric step configuration"""

    return configuration.IonosphericCalibrationConf(
        perform_defocusing_on_ionospheric_height=conf.ionosphere_height_defocusing_flag,
        perform_faraday_rotation_correction=conf.faraday_rotation_correction_flag,
        perform_phase_screen_correction=conf.ionospheric_phase_screen_correction_flag,
        perform_group_delay_correction=conf.group_delay_correction_flag,
        ionospheric_height_estimation_method=translate_ionospheric_height_estimation_method(
            conf.ionosphere_height_estimation_method
        ),
        squint_sensitivity=configuration.IonosphericSquintSensitivity(
            number_of_looks=conf.squint_sensitivity_number_of_looks,
            height_step=10_000.0,
            faraday_rotation_bias=0.3,
        ),
        feature_tracking=configuration.IonosphericFeatureTracking(
            max_offset=2_000, profile_step=10, normalized_min_value_threshold=0.2
        ),
        z_threshold=_convert_latitude_threshold_to_z_threshold(
            conf.ionosphere_height_estimation_method_latitude_threshold
        ),
        gaussian_filter_max_size_azimuth=conf.gaussian_filter_maximum_major_axis_length,
        gaussian_filter_max_size_range=conf.gaussian_filter_maximum_minor_axis_length,
        gaussian_filter_default_size_azimuth=conf.gaussian_filter_major_axis_length,
        gaussian_filter_default_size_range=conf.gaussian_filter_minor_axis_length,
        default_ionospheric_height=conf.ionosphere_height_value,
        max_ionospheric_height=conf.ionosphere_height_maximum_value,
        min_ionospheric_height=conf.ionosphere_height_minimum_value,
        azimuth_block_size=conf.block_lines,
        azimuth_block_overlap=conf.block_overlap_lines,
        swath=swath,
    )


def fill_calibration_constants_conf(
    conf: aux_pp1.PolarimetricCalibrationConf,
    channel_imbalance: ChannelImbalanceProcessingParametersL1 | None,
) -> configuration.CalibrationConstantsConf:
    """Fill Polarimetric step configuration"""
    channel_imbalance = (
        channel_imbalance
        if channel_imbalance is not None
        else ChannelImbalanceProcessingParametersL1(tx=conf.channel_imbalance.hv_tx, rx=conf.channel_imbalance.hv_rx)
    )

    assert channel_imbalance.tx is not None
    assert channel_imbalance.rx is not None

    internal_delay_hh = 0.0
    internal_delay_hv = 0.0
    internal_delay_vh = 0.0
    internal_delay_vv = 0.0

    return configuration.CalibrationConstantsConf(
        channel_imbalance_tx=channel_imbalance.tx,
        channel_imbalance_rx=channel_imbalance.rx,
        cross_talk_hv_rx=conf.cross_talk.hv_rx,
        cross_talk_vh_rx=conf.cross_talk.vh_rx,
        cross_talk_vh_tx=conf.cross_talk.vh_tx,
        cross_talk_hv_tx=conf.cross_talk.hv_tx,
        internal_delay_hh=internal_delay_hh,
        internal_delay_hv=internal_delay_hv,
        internal_delay_vh=internal_delay_vh,
        internal_delay_vv=internal_delay_vv,
    )


def fill_multilooker_conf(conf: aux_pp1.MultilookConf, swath: str) -> configuration.MultilookerConf:
    """Fill multilooker configuration"""
    range_params = conf.range_parameters.get(Swath(swath))
    if range_params is None:
        raise MissingSwathParams(f"Multilooker range parameters missing for swath {swath}")

    azimuth_params = conf.azimuth_parameters.get(Swath(swath))
    if azimuth_params is None:
        raise MissingSwathParams(f"Multilooker azimuth parameters missing for swath {swath}")

    def fill_multilooker_single_direction(
        params: aux_pp1.MultilookConf.Parameters,
    ) -> configuration.MultilookerConf.MultilookerDoubleDirectionConf.MultilookerSingleDirectionConf:
        """Fill multilooker direction based configuration"""
        window = configuration.WindowConf(
            window_type=configuration.WindowConf.Type(params.window_type.name),
            window_parameter=params.window_coefficient,
            window_look_bandwidth=configuration.Quantity(
                value=params.look_bandwidth,
                unit=configuration.Quantity.Unit.HZ,
            ),
            window_transition_bandwidth=configuration.Quantity(
                value=params.look_bandwidth,
                unit=configuration.Quantity.Unit.HZ,
            ),
        )
        return configuration.MultilookerConf.MultilookerDoubleDirectionConf.MultilookerSingleDirectionConf(
            p_factor=params.upsampling_factor,
            q_factor=params.downsampling_factor,
            weighting_window=window,
            central_frequency=[
                configuration.Quantity(
                    value=frequency,
                    unit=configuration.Quantity.Unit.HZ,
                )
                for frequency in params.look_central_frequencies
            ],
        )

    return configuration.MultilookerConf(
        multilook_conf_name=swath,
        azimuth_time_weighting_window_info=None,
        normalization_factor=1.0,
        multilook_conf=configuration.MultilookerConf.MultilookerDoubleDirectionConf(
            slow_multilook=fill_multilooker_single_direction(azimuth_params),
            fast_multilook=fill_multilooker_single_direction(range_params),
        ),
        invalid_value=None,
        swath=swath,
    )


def fill_noise_map_conf(conf: aux_pp1.ThermalDenoisingConf, swath: str) -> configuration.NoiseMapConf:
    """Fill noise map generator configuration"""
    return configuration.NoiseMapConf(noise_normalization_constant=conf.noise_gain_list[Polarization.HH], swath=swath)


def fill_slant_to_ground_conf(conf: aux_pp1.GroundProjectionConf, swath: str) -> configuration.SlantToGroundConf:
    """Fill slant to ground configuration"""
    return configuration.SlantToGroundConf(ground_step=conf.range_pixel_spacing, swath=swath)


def fill_sarfoc_processing_parameters(
    pp1: aux_pp1.AuxProcessingParametersL1,
    channel_imbalance: ChannelImbalanceProcessingParametersL1 | None,
    instrument_swst_bias: float,
) -> configuration.SarfocProcessingParameters:
    """Fill sarfoc processing parameters structure

    Parameters
    ----------
    pp1 : aux_pp1.AuxPP1
        BPS L1 Processing parameters structure
    channel_imbalance:
        Parameters generated by BPSL1PreProcessor
    instrument_swst_bias : float
        instrument SWST bias value

    Returns
    -------
    configuration.SarfocProcessingParameters
        SARFOC processing parameters object
    """

    output_prf_value = (
        pp1.azimuth_compression.azimuth_resampling_frequency
        if pp1.azimuth_compression.azimuth_resampling and pp1.azimuth_compression.azimuth_resampling_frequency
        else None
    )

    return configuration.SarfocProcessingParameters(
        rfi_mitigation_conf=[fill_rfi_mitigation_conf(pp1.rfi_mitigation, swath.name) for swath in STRIPMAP_SWATHS],
        range_focuser_conf=[
            fill_range_focuser_conf(
                pp1.range_compression,
                output_prf_value,
                instrument_swst_bias,
                swath.name,
            )
            for swath in STRIPMAP_SWATHS
        ],
        doppler_estimator_conf=[
            fill_doppler_estimator_conf(pp1.doppler_estimation, swath.name) for swath in STRIPMAP_SWATHS
        ],
        azimuth_conf=[
            fill_azimuth_conf(
                pp1.azimuth_compression,
                swath.name,
            )
            for swath in STRIPMAP_SWATHS
        ],
        radiometric_calibration_conf=[
            fill_radiometric_calibration_conf(pp1.radiometric_calibration, swath.name) for swath in STRIPMAP_SWATHS
        ],
        polarimetric_processor_conf=[
            fill_polarimetric_processor_conf(pp1.polarimetric_calibration, swath.name) for swath in STRIPMAP_SWATHS
        ],
        ionospheric_calibration_conf=[
            fill_ionospheric_calibration_conf(pp1.ionosphere_calibration, swath.name) for swath in STRIPMAP_SWATHS
        ],
        calibration_constants_conf=fill_calibration_constants_conf(pp1.polarimetric_calibration, channel_imbalance),
        multilooker_conf=[
            fill_multilooker_conf(
                pp1.multilook,
                swath.name,
            )
            for swath in STRIPMAP_SWATHS
        ],
        noise_map_conf=[fill_noise_map_conf(pp1.thermal_denoising, swath.name) for swath in STRIPMAP_SWATHS],
        slant_to_ground_conf=[
            fill_slant_to_ground_conf(pp1.ground_projection, swath.name) for swath in STRIPMAP_SWATHS
        ],
    )
