# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Warping Utilities for the Multi-Squint Calibration (MSC)
--------------------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor

import numba as nb
import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.common import bps_logger


@nb.njit(nogil=True, cache=True, parallel=True)
def apply_constant_azimuth_shifts_interp_inplace_multithreaded(
    mpol_image: tuple[npt.NDArray[complex], ...],
    flattening_phase_screen: npt.NDArray[float],
    azimuth_residual_shift: float,
):
    """
    Apply the azimuth shifts using a linear interpolator (with base-banding).

    Parameters
    ----------
    mpol_image: tuple[npt.NDArray[complex], ...]
        The [Npol x Nazm x Nrng] flattened multi-polarimetric image.

    flattening_phases: npt.NDArray[float]
        The [Nazm x Nrng] flattening phase screen for base-banding.

    azimuth_shifts: npt.NDArray[float] [px]
        The [Nazm x Nrng] azimuth shifts applied to the image.

    """
    num_azimuths, num_ranges = mpol_image[0].shape
    azimuth_pixel_axis = np.arange(0.0, num_azimuths)
    azimuth_pixel_shifted_axis = azimuth_pixel_axis - azimuth_residual_shift
    flat_phasors = np.exp(1j * flattening_phase_screen)
    for k in nb.prange(len(mpol_image) * num_ranges):
        pol = k // num_ranges
        rng = k % num_ranges
        mpol_image[pol][:, rng] = (
            _interp1d(
                azimuth_pixel_shifted_axis,
                azimuth_pixel_axis,
                mpol_image[pol][:, rng] * np.conj(flat_phasors[:, rng]),  # Base-banding.
            )
            * flat_phasors[:, rng]
        )


def apply_constant_shifts_fft_inplace_multithreaded(
    *,
    mpol_image: tuple[npt.NDArray[complex], ...],
    azimuth_residual_shift: float,
    range_residual_shift: float,
    flattening_phase_screen: npt.NDArray[float] | None = None,
    max_num_threads: int = 1,
):
    """
    Apply a constant shift via FFT.

    Parameters
    ----------
    mpol_image: tuple[npt.NDArray[complex], ...]
        A tuple of [Naz x Nrg] arrays that stores a multi-polarimetric
        single-look complex.

    azimuth_residual_shift: float  [px]
        A constant shift in along-track direction.

    range_residual_shift: float  [px]
        A constant shift in slant-range direction.

    flattening_phase_screen: npt.NDArray[float] | None = None
        Optionally, re-apply the flattening phase screen to shift in
        base band.

    max_num_threads: int = 1
        Number of threads assigned to the task.

    """
    with ThreadPoolExecutor(max_workers=max_num_threads) as executor:
        # Cache the phasors associated to the flattening phase screens.
        flattening_phasors = 1.0 + 0j
        if flattening_phase_screen is not None:
            flattening_phasors = np.exp(-1j * flattening_phase_screen)

        def warp_fn(image):
            image[...] = _apply_fourier_shifts(
                image=image * flattening_phasors,
                azimuth_residual_shift=azimuth_residual_shift,
                range_residual_shift=range_residual_shift,
            ) * np.conj(flattening_phasors)

        for pol, _ in enumerate(executor.map(warp_fn, mpol_image)):
            bps_logger.debug("Shifted pol=%d", pol)


def _apply_fourier_shifts(
    *,
    image: npt.NDArray[complex],
    azimuth_residual_shift: float,
    range_residual_shift: float,
) -> npt.NDArray[complex]:
    """Apply a constant shift using FFT."""
    # The original image size.
    num_azimuths, num_ranges = image.shape

    # Pad the image.
    az_pad = (image.shape[0] // 2) * (np.abs(azimuth_residual_shift) > 0)
    rg_pad = (image.shape[0] // 2) * (np.abs(range_residual_shift) > 0)
    image_pad = np.pad(
        image,
        ((az_pad, az_pad), (rg_pad, rg_pad)),
        mode="constant",
        constant_values=0.0,
    )

    # The FFT size.
    fft_n_az = sp.fft.next_fast_len(image_pad.shape[0])
    fft_n_rg = sp.fft.next_fast_len(image_pad.shape[1])

    return sp.fft.ifft2(
        sp.ndimage.fourier_shift(
            sp.fft.fft2(image_pad, s=(fft_n_az, fft_n_rg)),
            (-azimuth_residual_shift, -range_residual_shift),
        ),
    )[az_pad : az_pad + num_azimuths, rg_pad : rg_pad + num_ranges]


@nb.njit(nogil=True, cache=True)
def _interp1d(
    x: npt.NDArray[float],
    xp: npt.NDArray[float],
    yp: npt.NDArray[complex],
) -> npt.NDArray[complex]:
    """Interpolation with 0's outsids axis bounds."""
    x_inbound = (xp[0] <= x) & (x <= xp[-1])
    return np.interp(x, xp, yp) * x_inbound
