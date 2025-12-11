# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Azimuth Spectral Filtering Windowing Methods
--------------------------------------------
"""

import numba as nb
import numpy as np
import numpy.typing as npt

# NOTE: Do not set @nb.njit(parallel=True) for these methods as it turn out to
# suck up too much RAM and slowing down things.


@nb.njit(cache=True, nogil=True)
def hamming_window_bank(
    *,
    centers: npt.NDArray[float],
    frequency_bandwidths: npt.NDArray[float],
    sampling_frequency: float,
    window_param: float,
    nsamples: int,
    inverse: bool,
    dtype: np.dtype,
) -> npt.NDArray[float]:
    """
    Compute the Hamming window bank. This function is Numba compiled.

    Parameters
    ----------
    centers: npt.NDArray[float]
        The [1 x Nrng] window centers.

    frequency_bandwidths: npt.NDArray[float]
        The [1 x Nrng] bandwidths.

    sampling_frequency: float
        The (common) sampling frequencies.

    window_param: float
        The Hamming window parameter.

    nsamples: int
        Number of samples.

    inverse: bool
        Return the inverse window, if set to True.

    dtype: np.dtype
        The floating point precision.

    Return
    ------
    npt.NDArray[float] [Hz]
        A [nsamples x Nrng] matrix so that each column is a Hamming window.

    """
    window = np.zeros((nsamples, centers.size), dtype=dtype)

    # pylint: disable=not-an-iterable
    for rng in nb.prange(window.shape[1]):
        param = min(
            np.int32(np.round(0.5 * frequency_bandwidths[rng] * nsamples / sampling_frequency) * 2),
            nsamples,
        )
        window[0:param, rng] = window_param - (1 - window_param) * np.cos(
            (2 * np.pi / (param - 1)) * np.arange(param, dtype=dtype)
        )
        window[:, rng] = np.roll(window[:, rng], int(-param / 2 + centers[rng]))
        if inverse:
            nonzero = window[:, rng] != 0
            window[nonzero, rng] = 1 / window[nonzero, rng]

    return window


@nb.njit(cache=True, nogil=True)
def kaiser_window_bank(
    *,
    centers: npt.NDArray[float],
    frequency_bandwidths: npt.NDArray[float],
    sampling_frequency: float,
    window_param: float,
    nsamples: int,
    inverse: bool,
    dtype: np.dtype,
) -> npt.NDArray[float]:
    """
    Compute the Kaiser window bank. This function is Numba compiled.

    Parameters
    ----------
    centers: npt.NDArray[float]
        The [1 x Nrng] window centers.

    frequency_bandwidths: npt.NDArray[float]
        The [1 x Nrng] bandwidths.

    sampling_frequency: float
        The (common) sampling frequencies.

    window_param: float
        The Kaiser window parameter.

    nsamples: int
        Number of samples.

    inverse: bool
        Return the inverse window, if set to True.

    dtype: np.dtype
        The floating point precision.

    Return
    ------
    npt.NDArray[float] [Hz]
        A [nsamples x Nrng] matrix so that each column is a Kaiser window.

    """
    window = np.zeros((nsamples, centers.size), dtype=dtype)

    # pylint: disable=not-an-iterable
    for rng in nb.prange(window.shape[1]):
        param = min(
            np.int32(np.round(0.5 * frequency_bandwidths[rng] * nsamples / sampling_frequency) * 2),
            nsamples,
        )
        window[0:param, rng] = np.kaiser(param, beta=np.pi * window_param).astype(dtype)
        window[:, rng] = np.roll(window[:, rng], int(-param / 2 + centers[rng]))
        if inverse:
            nonzero = window[:, rng] != 0
            window[nonzero, rng] = 1 / window[nonzero, rng]

    return window


@nb.njit(cache=True, nogil=True)
def none_window_bank(
    *,
    centers: npt.NDArray[float],
    frequency_bandwidths: npt.NDArray[float],
    sampling_frequency: float,
    window_param: float,
    nsamples: int,
    inverse: bool,
    dtype: np.dtype,
) -> npt.NDArray[float]:
    """
    Compute the void window bank (i.e. all ones). This function is used
    to handle the None window type.

    Parameters
    ----------
    centers: npt.NDArray[float]
        The [1 x Nrng] window centers.

    frequency_bandwidths: npt.NDArray[float]
        The [1 x Nrng] bandwidths.

    sampling_frequency: float
        The (common) sampling frequencies.

    window_param: float
        [[UNUSED]]

    nsamples: int
        Number of samples.

    inverse: bool
        Return the inverse window, if set to True.

    dtype: np.dtype
        The floating point precision.

    Return
    ------
    npt.NDArray[float] [Hz]
        A [nsamples x Nrng] matrix so that each column is a window
        containing ones.

    """
    # pylint: disable=unused-argument
    window = np.zeros((nsamples, centers.size), dtype=dtype)

    # pylint: disable=not-an-iterable
    for rng in nb.prange(window.shape[1]):
        param = min(
            np.int32(np.round(0.5 * frequency_bandwidths[rng] * nsamples / sampling_frequency) * 2),
            nsamples,
        )
        window[0:param, rng] = 1
        window[:, rng] = np.roll(window[:, rng], int(-param / 2 + centers[rng]))
        if inverse:
            nonzero = window[:, rng] != 0
            window[nonzero, rng] = 1 / window[nonzero, rng]

    return window
