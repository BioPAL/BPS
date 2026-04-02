# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Phase Plane Estimation and Removal
----------------------------------
"""

from math import ceil

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.signal_processing import compute_coherence
from sklearn.linear_model import LinearRegression


def phase_plane_inplace_compensation(
    *,
    mpol_image: tuple[npt.NDArray[complex], ...],
    phase_slopes: tuple[float, float],
    dtypes: EstimationDType,
):
    """
    Compensate in-place a phase plane.

    Parameters
    ----------
    mpol_image: tuple[npt.NDArray[complex], ...]
        A tuple of [Naz x Nrg] arrays that stores a multi-polarimetric
        single-look complex.

    phase_slopes: tuple[float, float]  [rad/px]
        The phase plane's slopes in along-track and slant-range direction.

    dtypes: EstimationDType
        The floating-point precision used for the operations.

    """
    # Just shortcuts.
    num_azimuths, num_ranges = mpol_image[0].shape

    # Pre-compute the phasors that encodes the phase plane compensation.
    plane_comp_phasor = np.exp(
        -1j
        * np.add.outer(
            phase_slopes[0] * np.arange(num_azimuths, dtype=dtypes.float_dtype),
            phase_slopes[1] * np.arange(num_ranges, dtype=dtypes.float_dtype),
        ),
        dtype=dtypes.complex_dtype,
    )

    # Compensate the images.
    #
    # NOTE: No real need of multi-threading since polarizations are at most
    # 4. There's no significant runtime improvement when assigning a thread per
    # polarization.
    for image in mpol_image:
        image[...] *= plane_comp_phasor


def estimate_phase_slopes_pairwise(
    *,
    image_p: npt.NDArray[complex],
    image_s: npt.NDArray[complex],
    dtypes: EstimationDType,
    coherence_window_size: tuple[int, int] = (5, 5),
    fft2_zero_padding_upsampling_factor: float = 1.2,
    fft2_peak_window_size: int = 3,
) -> tuple[float, float]:
    """
    Estimate the phase slopes for an interferometric pair.

    Parameters
    ----------
    image_p: npt.NDArray[complex]
        The [Nazm x Nrng] primary image (fattened).

    image_s: npt.NDArray[complex]
        The [Nazm x Nrng] secondary image (flattened).

    dtypes: EstimationDType
        The floating-point precision used for the estimation.

    coherence_window_size: tuple[int, int] = (5, 5)
        Size of the multi-looking window used to compute the coherence.

    fft2_zero_padding_upsampling_factor: float = 1.2
        The usampling factor for the FFT2.

    fft2_peak_window_size: int = 3
        The window size used to refine the peak.

    Return
    ------
    float [rad/px]
        The azimuth phase slope in along-track direction.

    float [rad/px]
        The range phase slope in slant-range direction.

    """
    # The azimuth and range 0-padding size.
    azm_pad_size = sp.fft.next_fast_len(ceil(image_p.shape[0] * fft2_zero_padding_upsampling_factor))
    rng_pad_size = sp.fft.next_fast_len(ceil(image_p.shape[1] * fft2_zero_padding_upsampling_factor))

    # Compute the FFT2.
    spectrum = np.abs(
        sp.fft.fftshift(
            sp.fft.fft2(
                compute_coherence(
                    image_p,
                    image_s,
                    filter_window_size=coherence_window_size,
                    dtype=dtypes.complex_dtype,
                ),
                s=(azm_pad_size, rng_pad_size),
            ),
        ),
        dtype=dtypes.float_dtype,
    )

    # compute the peak.
    win = fft2_peak_window_size // 2

    peak_azm, peak_rng = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    delta_peak_azm, delta_peak_rng, _ = refine_peak(
        spectrum[peak_azm - win : peak_azm + win + 1, peak_rng - win : peak_rng + win + 1],
    )
    peak_azm += delta_peak_azm
    peak_rng += delta_peak_rng

    # Compute the phase slopes in rad/px.
    return (
        2 * np.pi * (peak_azm - azm_pad_size // 2) / azm_pad_size,
        2 * np.pi * (peak_rng - rng_pad_size // 2) / rng_pad_size,
    )


def refine_peak(values: npt.NDArray[float]) -> tuple[float, float, npt.NDArray[float]]:
    """Compute the peak of a 2D parabolic interpolator.

    Parmeters
    ---------
    values: npt.NDArray[float]
        A [K x K] real valued matrix representing a window around the
        estimated peak.

    Returns
    -------
    delta_peak_azm: float [px]
        The displacemente from the central pixel (prior peak), in
        azimuth direction (horizontal)

    delta_peak_rng: float [px]
        The displacement from the central pixel (prior peak), in range
        direction (vertical)

    coeffs: npt.NDArray[float]
        The coefficient of the paraboloid, ordered as {a,b,c,d,e,f} with
        ordering ax**2 + by**2 + cxy + dx + ey + f.

    """
    # Shortcut for the window shape.
    n_azm, n_rng = values.shape

    # Build coordinate grid centered around the middle pixel
    c_azm = (n_azm - 1) / 2
    c_rng = (n_rng - 1) / 2
    azm_axes, rng_axes = np.mgrid[0:n_azm, 0:n_rng]
    rng_axes = rng_axes - c_rng
    azm_axes = azm_axes - c_azm

    # Build design matrix for quadratic terms
    phi = np.column_stack(
        [
            np.ravel(azm_axes) ** 2,
            np.ravel(rng_axes) ** 2,
            np.ravel(rng_axes * azm_axes),
            np.ravel(azm_axes),
            np.ravel(rng_axes),
            np.ones(n_azm * n_rng),
        ]
    )
    vals = np.ravel(values)

    # Fit paraboloid coefficients via linear regression.
    reg = LinearRegression(fit_intercept=False)
    reg.fit(phi, vals)

    # Compute vertex (maximum/minimum) analytically.
    A = np.array([[2 * reg.coef_[0], reg.coef_[2]], [reg.coef_[2], 2 * reg.coef_[1]]])
    b_vec = reg.coef_[3:5]
    d_azm, d_rng = -np.linalg.inv(A) @ b_vec

    return d_azm, d_rng, reg.coef_
