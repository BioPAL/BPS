# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Phase Plane Correction
----------------------
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import numpy.typing as npt
from bps.stack_cal_processor.configuration import StackDataSpecs
from bps.stack_cal_processor.core.floating_precision import EstimationDType


def compensate_phase_slopes_multithreaded(
    *,
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...],
    azimuth_phase_slopes: npt.NDArray[float],
    range_phase_slopes: npt.NDArray[float],
    stack_specs: StackDataSpecs,
    coreg_primary_image_index: int,
    dtypes: EstimationDType,
    num_worker_threads: int,
):
    """
    Compensate in-place the provided phase planes.

    Parameters
    ----------
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...]
        The input stack images.

    azimuth_phase_slopes: npt.NDArray[float]  [rad/s]
        The phase slopes in along-track direction.

    range_phase_slopes: npt.NDArray[float]  [rad/s]
        The phase slopes in slant-range direction.

    stack_specs: StackDataSpecs
        The stack specifications.

    coreg_primary_image_index: int
        The index of the coregistration primary image.

    num_worker_threads: int
        The number of threads assigned to the job.

    """
    # Compute the phase screens.
    phase_screens = []

    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The shape of the stack images.
        num_azimuths, num_ranges = stack_images[0][0].shape

        # The phase screen calculation.
        def phase_screen_fn(azm_slope_px, rng_slope_px):
            return np.exp(
                -1j
                * np.add.outer(
                    azm_slope_px * np.arange(num_azimuths),
                    rng_slope_px * np.arange(num_ranges),
                    dtype=dtypes.float_dtype,
                ),
                dtype=dtypes.complex_dtype,
            )

        for phase_screen in executor.map(
            phase_screen_fn,
            azimuth_phase_slopes * stack_specs.azimuth_sampling_step,  # [rad/px].
            range_phase_slopes * stack_specs.range_sampling_step,  # [rad/px].
        ):
            phase_screens.append(phase_screen)

    # Calibrate the stack in parallel threads.
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # Just a couple of shortcuts.
        num_images = len(stack_images)
        num_polarizations = len(stack_images[0])

        # The calibration function.
        def phase_removal_fn(img_pol: tuple[int, int]):
            img, pol = img_pol
            if img != coreg_primary_image_index:
                stack_images[img][pol][...] = stack_images[img][pol] * phase_screens[img]

        for _ in executor.map(
            phase_removal_fn,
            product(range(num_images), range(num_polarizations)),
        ):
            pass
