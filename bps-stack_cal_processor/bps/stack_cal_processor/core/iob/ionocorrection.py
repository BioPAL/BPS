# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to correct the fast-varying ionosphere
------------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import numpy.typing as npt
from bps.stack_cal_processor.configuration import StackDataSpecs
from bps.stack_cal_processor.core.floating_precision import EstimationDType


def compensate_background_ionosphere_multithreaded(
    *,
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...],
    iono_azimuth_slopes: npt.NDArray[float],
    iono_range_slopes: npt.NDArray[float],
    stack_specs: StackDataSpecs,
    calib_reference_index: int | None,
    estimation_dtypes: EstimationDType,
    num_worker_threads: int,
):
    """
    Compensate (in place) the background ionosphere phase screens.

    Parameters
    ----------
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...]
        The Nimg x Npol input [Nazm x Nrng] stack images.

    iono_azimuth_slopes: npt.NDArray[float] [rad/s]
        The [1 x Nimg] background ionosphere azimuth slopes.

    iono_range_slopes: npt.NDArray[float] [rad/s]
        The [1 x Nimg] background ionosphere range slopes.

    stack_specs: StackDataSpecs
        THe stack specks object.

    calib_reference_index: Optional[int]
        The index of the calibration reference image. If provided, the
        calibration reference image will not be calibrated.

    num_worker_threads: int
        Number of worker threads assigned to the job.

    estimation_dtypes: EstimationDType
        The estimation floating point accuracy.

    Raises
    ------
    IobRuntimeError

    """
    # Compute the background phases.
    iono_phases_screen = tuple(
        compute_background_ionosphere_phase_screen(
            image=stack_image[0],  # Just to read the shape.
            azimuth_sampling_step=stack_specs.azimuth_sampling_step,
            range_sampling_step=stack_specs.range_sampling_step,
            iono_azimuth_slope=iono_azm_slope,
            iono_range_slope=iono_rng_slope,
            estimation_dtypes=estimation_dtypes,
        )
        for stack_image, iono_azm_slope, iono_rng_slope in zip(
            stack_images,
            iono_azimuth_slopes,
            iono_range_slopes,
        )
    )

    # The calibration images.
    image_indices = list(range(len(stack_images)))
    if calib_reference_index is not None:
        image_indices.remove(calib_reference_index)

    # Remove the background phase screen.
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The core removal function.
        def background_ionosphere_compensation_fn(img_pol):
            stack_images[img_pol[0]][img_pol[1]][...] *= np.exp(-1j * iono_phases_screen[img_pol[0]])

        for _ in executor.map(
            background_ionosphere_compensation_fn,
            product(image_indices, range(len(stack_images[0]))),
        ):
            pass


def compute_background_ionosphere_phase_screen(
    *,
    image: npt.NDArray[complex],
    azimuth_sampling_step: float,
    range_sampling_step: float,
    iono_azimuth_slope: float,
    iono_range_slope: float,
    estimation_dtypes: EstimationDType,
) -> npt.NDArray[float]:
    """
    Compute the background ionosphere phase screen. In order to resolve
    the underdetermination, the phase screen is computed so that the it is
    nought in the center of the frame.

    Parameters
    ----------
    image: npt.NDArray[complex]
        A [Nazm x Nrng] stack image.

    azimuth_sampling_step: float [s]
        The azimuth sampling step.

    range_sampling_step: float [s]
        The range sampling step.

    iono_azimuth_slope: float [rad/s]
        The background ionosphere azimuth slope.

    iono_range_slope: float [rad/s]
        The background ionosphere range slope.

    dtypes: EstimationDType
        The estimation floating point accuracy.

    Return
    ------
    npt.NDArray[float] [rad]
        The background ionospheric phase screen.

    """
    # Compute the centered azimuth and range axes.
    azimuth_time_axis = _get_iono_time_axis(image.shape[0], azimuth_sampling_step, estimation_dtypes.float_dtype)
    range_time_axis = _get_iono_time_axis(image.shape[1], range_sampling_step, estimation_dtypes.float_dtype)
    return np.add.outer(iono_azimuth_slope * azimuth_time_axis, iono_range_slope * range_time_axis)


def _get_iono_time_axis(
    num_pixels: int,
    sampling_step: float,
    dtype: np.dtype,
) -> npt.NDArray[float]:
    """The time axis of the ionosphere."""
    time_axis = np.arange(num_pixels, dtype=dtype) * sampling_step
    return time_axis - np.mean(time_axis)
