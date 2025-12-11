# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Flattening Phase Screen Compensation
------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
from bps.stack_cal_processor.core.floating_precision import EstimationDType


def compensate_flattening_phase_multithreaded(
    *,
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    reference_polarization_index: int,
    coreg_primary_image_index: int,
    dtypes: EstimationDType,
    num_worker_threads: int,
) -> tuple[npt.NDArray[float], ...]:
    """
    Compensate the flattening phase screens (synthetic interferogram) for a
    selected polarization.

    Parameters
    ----------
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...]
        The input stack image of shape [Nimg x Npol x Nazm x Nrng].

    synth_phases: tuple[npt.NDArray[float], ...]  [rad]
        The [Nimg x Nazm x Nrng] flattening phase screen.

    reference_polarization_index: int
        Only this polarization is returned.

    dtypes: EstimationDType
        The floating-point precision used for the estimation.

    num_worker_threads: int
        The number of threads assigned for the job.

    Return
    ------
    tuple[npt.NDArray[complex], ...]
        The flattened stack, restricted to the selected polarization.

    """
    # The flattened stack images.
    flat_images = []

    # Execute the flattening in separated threads.
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The core flattening function.
        def flatten_phase_screen_fn(image_index):
            return stack_images[image_index][reference_polarization_index] * np.exp(
                1j * synth_phases[image_index], dtype=dtypes.complex_dtype
            )

        for flat_image in executor.map(
            flatten_phase_screen_fn, [i for i in range(len(stack_images)) if i != coreg_primary_image_index]
        ):
            flat_images.append(flat_image)

    # Add the coreg primary image (no flattening needed).
    flat_images.insert(
        coreg_primary_image_index,
        stack_images[coreg_primary_image_index][reference_polarization_index],
    )

    return tuple(flat_images)
