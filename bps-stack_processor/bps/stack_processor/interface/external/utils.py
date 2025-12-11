# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
External Interface Utilities
----------------------------
"""

import logging

import numpy as np
from arepytools.io.metadata import EPolarization
from bps.common.joborder import DeviceResources, ProcessorConfiguration
from bps.stack_cal_processor.configuration import StackCalConf
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
    SkpPhaseCalibrationConf,
)
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder


def get_bps_logger_level(
    stdout_log_level: ProcessorConfiguration.LogLevel,
    stderr_log_level: ProcessorConfiguration.LogLevel,
) -> int:
    """Convert the BPS logging levels to the internal logger's levels."""
    sta_log_levels_to_bps_logger_levels = {
        ProcessorConfiguration.LogLevel.ERROR: logging.ERROR,
        ProcessorConfiguration.LogLevel.WARNING: logging.WARNING,
        ProcessorConfiguration.LogLevel.PROGRESS: logging.INFO,
        ProcessorConfiguration.LogLevel.INFO: logging.INFO,
        ProcessorConfiguration.LogLevel.DEBUG: logging.DEBUG,
    }
    stdout_level = sta_log_levels_to_bps_logger_levels[stdout_log_level]
    stderr_level = sta_log_levels_to_bps_logger_levels[stderr_log_level]
    return max(stdout_level, stderr_level)


def fill_stack_cal_conf_from_aux_pps(
    aux_pps: AuxiliaryStaprocessingParameters,
    polarizations: tuple[EPolarization, ...] = (
        EPolarization.xx,
        EPolarization.hh,
        EPolarization.vv,
    ),
    skp_lut_azimuth_decimation_factor: int | None = None,
    skp_lut_range_decimation_factor: int | None = None,
) -> StackCalConf:
    """
    Initialize a StackCalConf object from AUX-PPS.

    Parameters
    ----------
    aux_pps: AuxiliaryStaprocessingParameters
        The user provided AUX-PPS.

    polarizations: tuple[EPolarization]
        The image polarizations that we will calibrate.

    skp_lut_azimuth_decimation_factor: int | None = None
        The decimation factor for the SKP LUTs (azimuth).

    skp_lut_range_decimation_factor: int | None = None
        The decimation factor for the SKP LUTs (range).

    Raises
    ------
    AzfValueError, and StackCalConfError

    Return
    ------
    StackCalConf
        The calibration configuration object.

    """
    if len(polarizations) == 0:
        raise StackCalConf.StackCalConfError("The stack has no usable polarizations")

    # Search for the user selected reference calibration polarization. If not
    # available, stop here.

    # Fill the parameters of Azimuth Spectral Filtering (AZF) module.
    azf_conf = None
    if aux_pps.azimuth_spectral_filtering.azimuth_spectral_filtering_flag:
        azf_conf = StackCalConf.AzfConf(
            use_primary_spectral_weighting_window_flag=aux_pps.azimuth_spectral_filtering.use_primary_weighting_window_flag,
            window_type=aux_pps.azimuth_spectral_filtering.spectral_weighting_window,
            window_parameter=aux_pps.azimuth_spectral_filtering.spectral_weighting_window_parameter,
            use_32bit_precision=aux_pps.azimuth_spectral_filtering.use_32bit_flag,
        )

    # Fill parameters of the Background Ionosphere Removal (IOB) module.
    iob_conf = None
    if aux_pps.slow_ionosphere_removal.slow_ionosphere_removal_flag:
        aux_pps_iob_conf = aux_pps.slow_ionosphere_removal
        try:
            calib_polarization_index = polarizations.index(aux_pps.slow_ionosphere_removal.polarization_used)
        except ValueError as ex:
            raise StackCalConf.IobConf.IobValueError(
                "Selected calibration reference polarization {} is not available (stack pols: {})".format(
                    aux_pps.slow_ionosphere_removal.polarization_used,
                    [p.value for p in polarizations],
                )
            ) from ex

        iob_conf = StackCalConf.IobConf(
            compensate_l1_iono_phase_screen_flag=aux_pps_iob_conf.compensate_l1_iono_phase_screen_flag,
            range_look_band=aux_pps_iob_conf.range_look_bandwidth,
            range_look_frequency=aux_pps_iob_conf.range_look_frequency,
            ionosphere_latitude_threshold=aux_pps_iob_conf.latitude_threshold,
            polarization_index=calib_polarization_index,
            baseline_method=aux_pps_iob_conf.baseline_method,
            multi_baseline_uniform_weighting=aux_pps_iob_conf.unweighted_multi_baseline_estimation,
            multi_baseline_cb_ratio_threshold=aux_pps_iob_conf.multi_baseline_critical_baseline_threshold,
            sublook_window_sizes=(
                _odd(aux_pps_iob_conf.sublook_window_azimuth_size),
                _odd(aux_pps_iob_conf.sublook_window_range_size),
            ),
            quality_threshold=aux_pps_iob_conf.slow_ionosphere_quality_threshold,
            min_coherence_threshold=aux_pps_iob_conf.min_coherence_threshold,
            max_lh_phase_delta=(
                aux_pps_iob_conf.max_lh_phase_delta
                if aux_pps_iob_conf.max_lh_phase_delta >= 0
                else np.inf  # Disable the unwrapping phase test if the parameter is negative.
            ),
            min_usable_pixel_ratio=aux_pps_iob_conf.min_usable_pixel_ratio,
            use_32bit_precision=aux_pps_iob_conf.use_32bit_flag,
        )

    # Fill parameters of the Phase Plane Removla (PPR) module.
    ppr_conf = None
    if aux_pps.in_sar_calibration.in_sar_calibration_flag:
        try:
            calib_polarization_index = polarizations.index(aux_pps.in_sar_calibration.polarization_used)
        except ValueError as ex:
            raise StackCalConf.PprConf.PprValueError(
                "Selected calibration reference polarization {} is not available (stack pols: {})".format(
                    aux_pps.in_sar_calibration.polarization_used,
                    [p.value for p in polarizations],
                )
            ) from ex

        ppr_conf = StackCalConf.PprConf(
            polarization_index=calib_polarization_index,
            fft2_zero_padding_upsampling_factor=aux_pps.in_sar_calibration.fft2_zero_padding_upsampling_factor,
            fft2_peak_window_size=aux_pps.in_sar_calibration.fft2_peak_window_size,
            use_32bit_precision=aux_pps.in_sar_calibration.use_32bit_flag,
        )

    # Fill parameters of the Sum-of-Kronecker-Products (SKP) module.
    skp_conf = None
    if aux_pps.skp_phase_calibration.skp_phase_estimation_flag:
        aux_pps_skp_conf = aux_pps.skp_phase_calibration
        skp_conf = StackCalConf.SkpConf(
            estimation_window_size=aux_pps_skp_conf.estimation_window_size,
            skp_phase_correction_flag=skp_phase_correction_flag(aux_pps_skp_conf),
            only_flattening_phase_correction_flag=skp_phase_correction_flattening_only_flag(aux_pps_skp_conf),
            skp_calibration_phase_screen_quality_threshold=aux_pps_skp_conf.skp_calibration_phase_screen_quality_threshold,
            output_azimuth_subsampling_step=skp_lut_azimuth_decimation_factor,
            output_range_subsampling_step=skp_lut_range_decimation_factor,
            median_filter_flag=aux_pps_skp_conf.median_filter_flag,
            median_filter_window_size=aux_pps_skp_conf.median_filter_window_size,
            exclude_mpmb_polarization_cross_covariance_flag=aux_pps_skp_conf.exclude_mpmb_polarization_cross_covariance_flag,
            use_32bit_precision=aux_pps_skp_conf.use_32bit_flag,
        )

        if aux_pps_skp_conf.estimation_window_size <= 0.0:
            raise StackCalConf.SkpConf.SkpValueError("Estimation window size must be positive")
        if aux_pps_skp_conf.median_filter_window_size <= 0.0:
            raise StackCalConf.SkpConf.SkpValueError("Median filter window size must be positive")

    return StackCalConf(
        azf_conf=azf_conf,
        iob_conf=iob_conf,
        ppr_conf=ppr_conf,
        skp_conf=skp_conf,
    )


def skp_phase_correction_flag(aux_pps_skp_conf: SkpPhaseCalibrationConf) -> bool:
    """Check if SKP should execute any phase correction."""
    return aux_pps_skp_conf.phase_correction != SkpPhaseCalibrationConf.SkpPhaseCorrectionType.NONE


def skp_phase_correction_flattening_only_flag(aux_pps_skp_conf: SkpPhaseCalibrationConf) -> bool:
    """Check if SKP should execute any phase correction."""
    return aux_pps_skp_conf.phase_correction == SkpPhaseCalibrationConf.SkpPhaseCorrectionType.FLATTENING_PHASE_SCREEN


def parse_user_provided_calib_reference_image_index(
    *,
    job_order: StackJobOrder,
    aux_pps: AuxiliaryStaprocessingParameters,
    coreg_primary_image_index: int,
) -> int | None:
    """
    The user provided calibration reference image.

    This function combines the selected calibration reference from Job Order and
    AUX-PPS. The golden rule is that the Job Order overrides (higher priority)
    the AUX-PPS.

    Parameters
    ----------
    job_order: StackJobOrder
        The stack job-order.

    aux_pps: AuxiliaryStaprocessingParameters
        The configuration from the AUX-PPS.

    coreg_primary_image_index: int
        The coregistration primary image index.

    Raises
    ------
    RuntimeError

    Return
    ------
    int | None
        The calibration reference image index if specified by the
        user, None otherwise.

    """
    # If no calibration reference is specified in the Job Order, we check if the
    # user has selected to use the coreg primary in the AUX-PPS.
    if job_order.processing_parameters.calibration_primary_image is None:
        if aux_pps.slow_ionosphere_removal.primary_image_flag:
            return coreg_primary_image_index
        return None

    # If the user selected a calibration reference image in the Job Order, we
    # use that one.
    calib_primary_image = job_order.processing_parameters.calibration_primary_image
    for image_index, image_path in enumerate(job_order.input_stack):
        if calib_primary_image.name == image_path.name:
            return image_index

    # Oops, probably the selected image is not correct.
    raise RuntimeError(f"could not find calibration reference image ({calib_primary_image})")


def parse_max_num_worker_threads(
    aux_pps: AuxiliaryStaprocessingParameters,
    device_resources: DeviceResources,
) -> int:
    """
    Select the number of worker threads for the calibration stack.

    This value is the smallest of the value provided in the AUX-PPS or the value
    provided in the Job Order. A value 0 in the AUX-PPS indicates to use all cores
    specified in the Job Order.

    Parameters
    ----------
    aux_pps: AuxiliaryStaprocessingParameters
        The configuration from the AUX-PPS.

    device_resources: DeviceResouces
        The device resources configuration as specified in the Job Order.

    Raises
    ------
    ValueError

    Return
    ------
    int
        The maximum number of worker threads.

    """
    if aux_pps.general.outer_parallelization_max_cores < 0:
        raise ValueError(
            "<outerParallelizationMaxCores> (in AUX-PPS) must be a nonnegative integer",
        )
    if device_resources.num_threads <= 0:
        raise ValueError(
            "<Number_of_CPU_Cores> (in Job Order) must be a positive integer",
        )

    if aux_pps.general.outer_parallelization_max_cores == 0:
        return device_resources.num_threads

    return min(
        aux_pps.general.outer_parallelization_max_cores,
        device_resources.num_threads,
    )


def _odd(num: int) -> int:
    """Return the next larger odd number."""
    return num + 1 if num % 2 == 0 else num
