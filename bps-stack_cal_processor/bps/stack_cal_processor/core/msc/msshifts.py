# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Perform the Multi-Squint Ionospheric Shifts
-------------------------------------------
"""

from dataclasses import dataclass
from math import ceil, floor

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.stack_cal_processor.core.msc.mscoherence import MultiSquintCoherenceEstimationOutput


@dataclass
class MultiSquintShiftsEstimationOutput:
    """Store the output of the Multi-Squint estimation of the along-track shifts."""

    azimuth_shift: float  # [m[.
    """Constant shift in along-track direction caused by the ionosphere [m]."""

    azimuth_frequency_centroid: float  # [1/m].
    """The along-track frequency centroid [1/m]."""


def apply_inplace_centroid_correction(
    *,
    mpol_image: tuple[npt.NDArray[float], ...],
    azimuth_frequency_centroid: float,
    azimuth_space_axis: npt.NDArray[float],
):
    """
    Apply the centroid correction to a multi-polarimetric image.

    Parameters
    ----------
    mpol_image: tuple[npt.NDArray[float], ...]
        The multi-polarimetric image.

    azimuth_frequency_centroid: float  [1/m]
        The azimuth frequency centroid.

    azimuth_space_axis: npt.NDArray[float]  [m]
        The spatial axis in along track direction.

    """
    comp_phasor = np.reshape(
        np.exp(-2j * np.pi * azimuth_frequency_centroid * azimuth_space_axis),
        (-1, 1),
    )
    for image in mpol_image:
        image[...] *= comp_phasor


def compute_azimuth_shifts(
    ms_coherence: MultiSquintCoherenceEstimationOutput,
    azimuth_space_axis: npt.NDArray[float],
    *,
    wavelength: float,
    range_block_size: int = 11,
    spectrum_filter_window_size: tuple[int, int] = (3, 3),
    spectrum_refine_window_size: int = 11,
    spectrum_refine_window_nsamples: int = 71,
) -> MultiSquintShiftsEstimationOutput:
    """
    Estimate the along-track rigid shift caused by ionospheric delay. Compute
    the spectral centroid of the coherence to identify the relationship between
    azimuth frequency and squint-induced phase shifts.

    Parameters
    ----------
    ms_coherence: MultiSquintCoherenceEstimationOutput
        The output of the estimation of the Multi-Squint coherence matrix.

    azimuth_space_axis: npt.NDArray[float] [m]
        The along-track space axis.

    wavelength: float [m]
        The RADAR wavelength.

    range_block_size: int = 11
        The range block size used for the spectral analysis.

    spectrum_filter_window_size: tuple[int, int] = (3, 3)
        Boxcar filter size applied to the spectrum.

    Return
    ------
    MultiSquintShiftsEstimationOutput
        The estimated along-track shift [m] and the azimuth frequency centroid [1/m].

    """
    # Select the range block around the mid range pixel to average the MS coherence.
    ms_coherence_block = ms_coherence.matrix
    if ms_coherence.range_indices.size > range_block_size:
        range_mid = round(ms_coherence.range_indices.size / 2)
        range_begin = max(range_mid - floor(range_block_size / 2), 0)
        range_end = min(range_mid + ceil(range_block_size / 2), ms_coherence.range_indices.size)
        ms_coherence_block = ms_coherence_block[:, range_begin:range_end, :]

    mean_ms_coherence = np.mean(ms_coherence_block, axis=1).T

    # Compute the squint frequency axis.
    squint_axis = (wavelength / 2) * ms_coherence.subband_axis
    squint_step = np.diff(squint_axis[:2])[0]
    num_fp = 2 * 2 ** ceil(np.log2(squint_axis.size))
    squint_freq_axis = sp.fft.fftshift(sp.fft.fftfreq(num_fp, 1 / num_fp)) / (num_fp * squint_step)

    # Compute the azimuth frquency axis [1/m].
    azimuth_space_substep = np.diff(azimuth_space_axis[ms_coherence.azimuth_indices[:2]])[0]
    num_fx = 2 * 2 ** ceil(np.log2(ms_coherence.azimuth_indices.size))
    azimuth_freq_axis = sp.fft.fftshift(sp.fft.fftfreq(num_fx, 1 / num_fx)) / (num_fx * azimuth_space_substep)

    # Run the spectral analysis [Nfx x Nfp]
    #  1- Compute the spectrum
    #  2- Add some filtering/smoothing on it.
    spectrum = sp.fft.fftshift(
        sp.fft.fft2(
            mean_ms_coherence,
            s=(num_fp, num_fx),
        ),
    )

    # Coarse estimation: Find the peak/max of the spectrum.
    peak_fp, peak_fx = np.unravel_index(
        np.argmax(
            sp.ndimage.uniform_filter(
                np.abs(spectrum),
                size=spectrum_filter_window_size,
                mode="nearest",
            ),
        ),
        spectrum.shape,
    )

    # Refine the estimation (local DFT zoom).
    squint_freq_axis = squint_freq_axis[peak_fp] + np.diff(squint_freq_axis[:2])[0] * np.linspace(
        -floor(spectrum_refine_window_size / 2),
        floor(spectrum_refine_window_size / 2),
        spectrum_refine_window_nsamples,
    )
    azimuth_freq_axis = azimuth_freq_axis[peak_fx] + np.diff(azimuth_freq_axis[:2])[0] * np.linspace(
        -floor(spectrum_refine_window_size / 2),
        floor(spectrum_refine_window_size / 2),
        spectrum_refine_window_nsamples,
    )

    # Create the "zoom" matrices.
    W_fp = np.exp(
        -2j
        * np.pi
        * np.outer(
            squint_freq_axis,  # Now higher resolution.
            squint_axis,
        ),
        dtype=ms_coherence.matrix.dtype,
    )
    W_fx = np.exp(
        -2j
        * np.pi
        * np.outer(
            azimuth_space_axis[ms_coherence.azimuth_indices],
            azimuth_freq_axis,  # Now higher resolution.
        ),
        dtype=ms_coherence.matrix.dtype,
    )

    # Find the new peaks.
    peak_fp, peak_fx = np.unravel_index(
        np.argmax(np.abs(W_fp @ mean_ms_coherence @ W_fx)),
        (squint_freq_axis.size, azimuth_freq_axis.size),
    )

    return MultiSquintShiftsEstimationOutput(
        azimuth_shift=(wavelength / 2) * squint_freq_axis[peak_fp],
        azimuth_frequency_centroid=azimuth_freq_axis[peak_fx],
    )
