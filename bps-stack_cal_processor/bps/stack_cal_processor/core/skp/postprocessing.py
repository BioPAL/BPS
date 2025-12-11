# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
SKP Postprocessing Module
-------------------------
"""

from concurrent.futures import ThreadPoolExecutor

import numpy.typing as npt
import scipy as sp
from bps.common import bps_logger
from bps.stack_cal_processor.core.floating_precision import (
    EstimationDType,
    assert_list_numeric_types_equal,
)
from bps.stack_cal_processor.core.skp.utils import (
    window_to_azimuth_pixels,
    window_to_range_pixels,
)


def apply_median_filter_skp_phases_multithreaded(
    skp_calibration_phases: tuple[npt.NDArray[float], ...],
    *,
    azimuth_sampling_step: float,
    azimuth_subsampling_step: int,
    range_sampling_step: float,
    range_subsampling_step: int,
    incidence_angle: float,
    satellite_ground_speed: float,
    filter_window_size: float,
    dtypes: EstimationDType,
    num_worker_threads: int = 1,
) -> tuple[npt.NDArray[float], ...]:
    """
    Post-process the SKP calibration phase screen via median filter.

    Parameters
    ----------
    skp_calibration_phases: tuple[npt.NDArray[float], ...] [rad]
        The SKP calibration phases.

    azimuth_sampling_step: float [s]
        Sampling step in azimuth direction.

    azimuth_subsampling_step: int [px]
        Subsampling step in azimuth direction that was used to obtain the
        SKP calibration phase screens.

    range_sampling_step: float [s]
        Sampling step in range direction.

    range_subsampling_step: int [px]
        Subsampling step in range direction that was used to obtain the SKP
        calibration phase screens.

    incidence_angle: float [rad]
        The incidence angle.

    satellite_ground_speed: float [m/s]
        Speed of the satellite at the ground level.

    filter_window_size: float [m]
        Size of the median filter window.

    dtypes: EstimationDType
        Floating point precision used for the estiamtions.

    num_worker_threads: int = 1
        Number of parallel threads assigned to this task.

    Raises
    ------
    ValueError

    Return
    ------
    tuple[npt.NDArray[float], ...] [rad]
        The post-processed SKP calibration phase screens.

    """
    # Minimal check on the input arguments.
    if filter_window_size < 0.0:
        raise ValueError("Median filter window must be positive")
    if num_worker_threads < 1:
        raise ValueError("Number of worker threads must be a positive integer")

    # Just an extra check on the input dtypes.
    assert_list_numeric_types_equal(
        skp_calibration_phases,
        expected_dtype=dtypes.float_dtype,
    )

    # This is always a positive integer as per implementation of the conversion
    # function.
    #
    # NOTE: We need to multiply the sampling step by the subsampling step used
    # to compute the SKP calibration phase screens in order to obtain the
    # correct window size in pixels.
    #
    # NOTE: If it's even, we make it odd. That means that if it's 0, it becomes
    # 1 (i.e. no filtering).
    azimuth_window_size = window_to_azimuth_pixels(
        window_size=filter_window_size,
        azimuth_sampling_step=azimuth_sampling_step * azimuth_subsampling_step,
        satellite_ground_speed=satellite_ground_speed,
    )
    if azimuth_window_size % 2 == 0:
        azimuth_window_size += 1

    # See comment/notes above.
    range_window_size = window_to_range_pixels(
        window_size=filter_window_size,
        range_sampling_step=range_sampling_step * range_subsampling_step,
        incidence_angle=incidence_angle,
    )
    if range_window_size % 2 == 0:
        range_window_size += 1

    # Nothing to do, we simply return the input SKP calibration phases.
    if azimuth_window_size * range_window_size == 1:
        bps_logger.warning("Selected median filter size is too small. Skipping it.")
        return skp_calibration_phases

    bps_logger.debug("Median filter window size: [%d x %d]", azimuth_window_size, range_window_size)

    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # Apply median filter.
        def apply_median_filter_fn(skp_phi, image_index):
            return sp.ndimage.median_filter(
                skp_phi,
                mode="constant",
                cval=0.0,
                size=(azimuth_window_size, range_window_size),
            )

        skp_calibration_phases = tuple(
            executor.map(
                apply_median_filter_fn,
                skp_calibration_phases,
                range(len(skp_calibration_phases)),
            )
        )
        assert_list_numeric_types_equal(skp_calibration_phases, expected_dtype=dtypes.float_dtype)

    return skp_calibration_phases
