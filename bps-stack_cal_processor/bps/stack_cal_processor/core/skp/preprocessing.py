# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Preprocessing Utilities
-----------------------
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.skp.utils import SkpRuntimeError


def remove_synthetic_phase_multithreaded(
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    num_worker_threads: int,
    dtypes: EstimationDType,
) -> list[list[npt.NDArray[complex], ...], ...]:
    """
    Remove the synthetic inteferogram from data.

    Parameters
    ----------
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...]
        The Nimg x Npol stack images of shape [Nazm x Nrng].

    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic inteferogram from DEM with shape [Nazm x Nrng].

    num_worker_threads: int
        Number of threads assigned to the job.

    dtypes: EstimationDType
        The estimation floating point accuracy.

    Raises
    ------
    SkpRuntimeError, ValueError

    Return
    ------
    list[list[npt.NDArray[complex], ...], ...]
        The pre-processed stack images.

    """
    if len(stack_images) != len(synth_phases):
        raise SkpRuntimeError("The stack is ill-formed")
    if num_worker_threads < 1:
        raise ValueError("Number of threads must be a positive integer")

    # Stack size.
    num_images = len(stack_images)
    num_polarizations = len(stack_images[0])
    stack_size = num_images * num_polarizations
    if stack_size <= 0:
        raise SkpRuntimeError("Empty stack")

    # Prepare the output.
    num_azimuths, num_ranges = stack_images[0][0].shape
    preproc_stack_images = np.empty(
        (num_images, num_polarizations, num_azimuths, num_ranges),
        dtype=dtypes.complex_dtype,
    )

    # Run the process multithreaded.
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The core routine to remove the synthetic interferogram.
        def remove_synthetic_phase_fn(img, pol):
            preproc_stack_images[img][pol] = (
                np.exp(1j * synth_phases[img], dtype=dtypes.complex_dtype) * stack_images[img][pol]
            )

        for _ in executor.map(
            remove_synthetic_phase_fn,
            (k // num_polarizations for k in range(stack_size)),
            (k % num_polarizations for k in range(stack_size)),
        ):
            pass

    return preproc_stack_images
