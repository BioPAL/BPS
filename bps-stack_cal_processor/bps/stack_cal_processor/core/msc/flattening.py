# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to compensate and apply the flattening phases
-------------------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import numpy.typing as npt


def compensate_inplace_dsi_multithreaded(
    *,
    stack_images: tuple[tuple[npt.NDArray[complex], ...] | None, ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    max_num_threads: int = 1,
):
    """
    Compensate in-place the flattening phase screens.

    Parameters
    ----------
    stack_images: tuple[tuple[npt.NDArray[complex], ...] | None, ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images of
        shape [Nazm x Nrng].

    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM of shape [Nazm x Nrng].

    max_num_threads: int = 1,
        Number of threads assigned to the task.

    """
    # Just few shortcuts.
    num_images = len(stack_images)
    num_polarizations = next(len(image) for image in stack_images if image is not None)

    with ThreadPoolExecutor(max_workers=max_num_threads) as executor:

        def compensate_dsi_fn(img_pol):
            img, pol = img_pol
            if stack_images[img] is not None:
                stack_images[img][pol][...] *= np.exp(1j * synth_phases[img])

        for _ in executor.map(compensate_dsi_fn, product(range(num_images), range(num_polarizations))):
            pass


def apply_inplace_dsi_multithreaded(
    *,
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...],
    synth_phases: tuple[npt.NDArray[float], ...],
    max_num_threads: int = 1,
):
    """
    Apply in-place the flattening phase screens.

    Parameters
    ----------
    stack_images: tuple[tuple[npt.NDArray[complex], ...] | None, ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images of
        shape [Nazm x Nrng].

    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM of shape [Nazm x Nrng].

    max_num_threads: int = 1,
        Number of threads assigned to the task.

    """
    # Just few shortcuts.
    num_images = len(stack_images)
    num_polarizations = next(len(image) for image in stack_images if image is not None)

    with ThreadPoolExecutor(max_workers=max_num_threads) as executor:

        def compensate_dsi_fn(img_pol):
            img, pol = img_pol
            if stack_images[img] is not None:
                stack_images[img][pol][...] *= np.exp(-1j * synth_phases[img])

        for _ in executor.map(compensate_dsi_fn, product(range(num_images), range(num_polarizations))):
            pass
