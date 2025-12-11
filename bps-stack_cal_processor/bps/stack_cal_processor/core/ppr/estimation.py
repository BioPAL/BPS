# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The PPR Estimation module
-------------------------
"""

from concurrent.futures import ThreadPoolExecutor
from math import ceil

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.stack_cal_processor.configuration import StackDataSpecs
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.signal_processing import compute_coherence
from sklearn.linear_model import LinearRegression


def estimate_phase_slopes_multithreaded(
    images: tuple[npt.NDArray[complex], ...],
    stack_specs: StackDataSpecs,
    coreg_primary_image_index: int,
    dtypes: EstimationDType,
    num_worker_threads: int,
    fft2_zero_padding_upsampling_factor: float = 1.2,
    fft2_peak_window_size: int = 3,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Estimate the phase slopes.

    Parameters
    ----------
    images: tuple[npt.NDArray[complex], ...]
        The [Nimg x Nazm x Nrng] flattened stack associated to the reference polarization.

    stack_specs: StackDataSpecs
        The stack parameters (e.g. range/azimuth sampling steps etc.)

    coreg_primary_image_index: int
        The index of the coregistration primary index.

    dtypes: EstimationDType
        The floating-point precision used for the estimations.

    max_num_threads: int
        The maximum number of threads assigned for the job.

    fft2_zero_padding_upsampling_factor: float = 1.2
        The usampling factor for the FFT2. It defaults to 20%.

    fft2_peak_window_size: int = 3
        The window size used to refine the peak. It defaults to 3.

    Return
    ------
    azimuth_phase_slopes: npt.NDArray[float]  [rad/s]
        The Nimg phase slopes in along-track direction.

    range_phase_slopes: npt.NDArray[float]  [rad/s]
        The Nimg phase slopes in slant-range direction.

    """
    # The output slopes.
    azimuth_phase_slopes = []
    range_phase_slopes = []

    # Execute the estimation in separated threads.
    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The core phase screen estimator.
        def estimate_phase_slopes_fn(image_s):
            return estimate_phase_slopes_pairwise(
                image_p=images[coreg_primary_image_index],
                image_s=image_s,
                azimuth_sampling_step=stack_specs.azimuth_sampling_step,
                range_sampling_step=stack_specs.range_sampling_step,
                dtypes=dtypes,
                fft2_zero_padding_upsampling_factor=fft2_zero_padding_upsampling_factor,
                fft2_peak_window_size=fft2_peak_window_size,
            )

        for azm_slope, rng_slope in executor.map(
            estimate_phase_slopes_fn,
            [img_s for n, img_s in enumerate(images) if n != coreg_primary_image_index],
        ):
            azimuth_phase_slopes.append(azm_slope)
            range_phase_slopes.append(rng_slope)

    # Store the 0's for the the primary.
    azimuth_phase_slopes.insert(coreg_primary_image_index, 0)
    range_phase_slopes.insert(coreg_primary_image_index, 0)

    return (
        np.asarray(azimuth_phase_slopes, dtype=dtypes.float_dtype),
        np.asarray(range_phase_slopes, dtype=dtypes.float_dtype),
    )


def estimate_phase_slopes_pairwise(
    *,
    image_p: npt.NDArray[complex],
    image_s: npt.NDArray[complex],
    azimuth_sampling_step: float,
    range_sampling_step: float,
    fft2_zero_padding_upsampling_factor: float,
    fft2_peak_window_size: int,
    dtypes: EstimationDType,
) -> tuple[float, float]:
    """
    Estimate the phase slopes for an interferometric pair.

    Parameters
    ----------
    image_p: npt.NDArray[complex]
        The [Nazm x Nrng] primary image (fattened).

    image_s: npt.NDArray[complex]
        The [Nazm x Nrng] secondary image (flattened).

    azimuth_sampling_step: float [s]
        The sampling step in along-track direction.

    range_sampling_step: float [s]
        The sampling step in slant-range direction.

    fft2_zero_padding_upsampling_factor: float
        The usampling factor for the FFT2.

    fft2_peak_window_size: int
        The window size used to refine the peak.

    dtypes: EstimationDType
        The floating-point precision used for the estimation.

    Return
    ------
    azimuth_phase_slope: float [rad/s]
        The azimuth phase slope in along-track direction.

    range_phase_slope: float [rad/s]
        The range phase slope in slant-range direction.

    """
    # The azimuth and range 0-padding size.
    azm_pad_size = sp.fft.next_fast_len(ceil(image_p.shape[0] * fft2_zero_padding_upsampling_factor))
    rng_pad_size = sp.fft.next_fast_len(ceil(image_p.shape[1] * fft2_zero_padding_upsampling_factor))

    # Compute the FFT2.
    spectrum = np.abs(
        sp.fft.fftshift(
            sp.fft.fft2(
                compute_coherence(image_p, image_s, dtype=dtypes.complex_dtype),
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
    azm_phase_slope = 2 * np.pi * (peak_azm - azm_pad_size // 2) / azm_pad_size
    rng_phase_slope = 2 * np.pi * (peak_rng - rng_pad_size // 2) / rng_pad_size

    # Return the phase slopes in rad/s.
    return (
        azm_phase_slope / azimuth_sampling_step,
        rng_phase_slope / range_sampling_step,
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
