# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Azimuth Spectral Filtering Utility Module
-----------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numba as nb
import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.common import bps_logger
from bps.stack_cal_processor.configuration import AZF_NAME
from bps.stack_cal_processor.core.azf.windowing import (
    hamming_window_bank,
    kaiser_window_bank,
    none_window_bank,
)
from bps.stack_cal_processor.core.filtering import ConvolutionWindowType
from bps.stack_cal_processor.core.floating_precision import (
    EstimationDType,
    assert_numeric_types_equal,
)
from bps.stack_cal_processor.core.utils import percentage_completed_msg, positive_part

# Map the window types to the window function.
_WINDOW_FUNCTION = {
    ConvolutionWindowType.HAMMING: hamming_window_bank,
    ConvolutionWindowType.KAISER: kaiser_window_bank,
    ConvolutionWindowType.NONE: none_window_bank,
}


class AzfRuntimeError(RuntimeError):
    """Handle an error that occurs while running the AZF."""

    def __init__(self, message: str):
        super().__init__(f"[{AZF_NAME}]: {message}")


def compute_filter_banks_multithreaded(
    *,
    doppler_centroids: tuple[npt.NDArray[float], ...],
    frequencies_xshifts: tuple[npt.NDArray[float], ...],
    frequencies_low: float,
    frequencies_high: float,
    azimuth_bandwidths: tuple[float, ...],
    azimuth_window_types: tuple[ConvolutionWindowType, ...],
    azimuth_window_params: tuple[float, ...],
    common_azimuth_window_type: ConvolutionWindowType,
    common_azimuth_window_param: float,
    azimuth_sampling_step: float,
    num_azimuths: int,
    num_worker_threads: int,
    dtypes: EstimationDType,
) -> tuple[npt.NDArray[float], ...]:
    """
    Compute a bank of filters that will be applied to the input
    stack frames.

    Parameters
    ----------
    doppler_centroids: tuple[npt.NDArray[float], ...] [Hz]
        The Nimg Doppler centroids, each of shape [1 x Nrng].

    frequencies_xshifts: tuple[npt.NDArray[float], ...] [Hz]
        The Nimg cross-shifts, each of shape [1 x Nrng].

    frequencies_high: npt.NDArray[float] [Hz]
        The [1 x Nrng] spectrum upper frequency bounds.

    frequencies_low: npt.NDArray[float] [Hz]
        The [1 x Nrng] spectrum lower frequency bounds.

    azimuth_bandwidths: tuple[float, ...]
        The [1 x Nimg] azimuth bandwidths of the stack.

    azimuth_window_types: tuple[ConvolutionWindowType, ...]
        The [1 x Nimg] azimuth window types of the stack images.

    azimuth_window_params: tuple[float, ...]
        The [1 x Nimg] parameters associated to the azimuth windows.

    common_azimuth_window_type: ConvolutionType
        The output window type applied to all stack image.

    common_azimuth_window_param: float
        The output window parameter associated to the common output window.

    azimuth_sampling_step: float [s]
        The sampling step in azimuth direction.

    num_azimuths: int
        The number of lines of the input stack.

    num_worker_threads: int
        Number of threads associated to the task.

    Raises
    ------
    AzfRuntimeError

    Return
    ------
    tuple[npt.NDArray[float], ...]
        The Nimg filters matices of shape [Nazm x Nrng].

    """
    # Minimal check on the input.
    if not (
        len(doppler_centroids)
        == len(frequencies_xshifts)
        == len(azimuth_bandwidths)
        == len(azimuth_window_types)
        == len(azimuth_window_params)
    ):
        raise AzfRuntimeError("Stack is ill-formed")
    if frequencies_high.shape != frequencies_low.shape:
        raise AzfRuntimeError("Frequencies high/low have mismatching shape")
    if azimuth_sampling_step <= 0:
        raise AzfRuntimeError(f"Invalid azimuth sampling step (step={azimuth_sampling_step})")
    if num_azimuths < 1:
        raise AzfRuntimeError(f"Invalid stack shape (num_azimuths={num_azimuths})")
    if num_worker_threads < 1:
        raise AzfRuntimeError("Number of threads must be at least 1")

    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The core processing routine.
        def compute_filter_bank_fn(dc, fx, bw, win_type, win_par):
            return compute_filter_bank(
                num_azimuths=num_azimuths,
                doppler_centroids=dc,
                frequencies_xshifts=fx,
                frequencies_high=frequencies_high,
                frequencies_low=frequencies_low,
                azimuth_bandwidth=bw,
                forward_azimuth_window_type=win_type,
                forward_azimuth_window_param=win_par,
                backward_azimuth_window_type=common_azimuth_window_type,
                backward_azimuth_window_param=common_azimuth_window_param,
                azimuth_sampling_step=azimuth_sampling_step,
                dtypes=dtypes,
            )

        return list(
            executor.map(
                compute_filter_bank_fn,
                doppler_centroids,
                frequencies_xshifts,
                azimuth_bandwidths,
                azimuth_window_types,
                azimuth_window_params,
            )
        )


def apply_filter_bank_multithreaded(
    *,
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...],
    filter_banks: tuple[npt.NDArray[complex], ...],
    frequencies_high: tuple[float, ...],
    frequencies_low: tuple[float, ...],
    azimuth_sampling_step: float,
    num_worker_threads: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Align the azimuth spectra (and possibly applying a common window)
    by applying the filter banks.

    Parameters
    ----------
    stack_images: tuple[tuple[npt.NDArray[complex], ...], ...]
        The multi-polarimetric image stack, i.e. [Nimg x Npol] images of
        shape [Nazm x Nrng].

    filter_banks: tuple[npt.NDArray[float], ...]
        The filter banks, i.e. Nimg matrices of shape [Nazm x Nrng].

    frequencies_high: npt.NDArray[float] [Hz]
        The [1 x Nrng] spectrum upper frequency bounds.

    frequencies_low: npt.NDArray[float] [Hz]
        The [1 x Nrng] spectrum lower frequency bounds.

    azimuth_sampling_step: float [s]
        The sampling step in azimuth direction.

    num_worker_threads: int
        Number of threads associated to the task.

    Raises
    ------
    AzfRuntimeError

    Return
    ------
    common_bandwidths: tuple[float, ...] [Hz]
        The common bandwidth of each image with respect to the
        primary, packed as [1 x Nimg] tuple.

    central_frequencies: tuple[float, ...] [Hz]
        The central frequency of each image after spectrum alignment,
        packed as a [1 x Nimg] tuple.

    """
    # Minimal check on the input.
    if len(stack_images) != len(filter_banks):
        raise AzfRuntimeError("Stack is ill-formed")
    if frequencies_high.shape != frequencies_low.shape:
        raise AzfRuntimeError("Frequencies high/low have mismatching shape")
    if azimuth_sampling_step <= 0:
        raise AzfRuntimeError(f"Invalid azimuth sampling step (step={azimuth_sampling_step})")
    if num_worker_threads < 1:
        raise AzfRuntimeError("Number of threads must be at least 1")

    # These are computed for every image.
    common_bandwidths = []
    central_frequencies = []

    with ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # The core processing routine.
        def apply_filter_bank_fn(img_pol):
            image_index, polarization_index = img_pol
            return apply_filter_bank(
                image=stack_images[image_index][polarization_index],
                filter_bank=filter_banks[image_index],
                frequencies_high=frequencies_high,
                frequencies_low=frequencies_low,
                azimuth_sampling_step=azimuth_sampling_step,
            )

        # Track progress.
        num_images = len(stack_images)
        num_polarizations = len(stack_images[0])
        num_tasks = num_images * num_polarizations
        completed_tasks = 0

        # Process everything multithreaded.
        valid_solutions = 0
        for common_bandwidth, central_frequency in executor.map(
            apply_filter_bank_fn,
            product(range(num_images), range(num_polarizations)),
        ):
            common_bandwidths.append(common_bandwidth)
            central_frequencies.append(central_frequency)
            # We can trust the solution since bandwidths overlap.
            valid_solutions += common_bandwidth != 0
            # Report progress.
            completed_tasks += 1
            bps_logger.info(percentage_completed_msg(completed_tasks, total=num_tasks))

        bps_logger.debug("Succeeded %d/%d", valid_solutions, num_tasks)

        return common_bandwidths, central_frequencies


def apply_filter_bank(
    *,
    image: npt.NDArray[complex],
    filter_bank: npt.NDArray[float],
    frequencies_low: npt.NDArray[float],
    frequencies_high: npt.NDArray[float],
    azimuth_sampling_step: float,
) -> tuple[float, float]:
    """
    Apply the azimuth spectral filter bank on an image.

    Parameters
    ----------
    image: npt.NDArray[complex]
        The [Nazm x Nrng] stack image.

    filter_bank: npt.NDArray[float]
        The [Nazm x Nrng] filter bank, where Nazm is the dimension of
        the image.

    frequencies_high: npt.NDArray[float] [Hz]
        The [1 x Nrng] spectrum upper frequency bounds.

    frequencies_low: npt.NDArray[float] [Hz]
        The [1 x Nrng] spectrum lower frequency bounds.

    azimuth_sampling_step: float [Hz]
        The (common) azimuth sampling step.

    Return
    ------
    float [Hz]
        The azimuth common bandwidth at the center of the image raster.

    float [Hz]
        The central frequency at the center of the image raster.

    """
    # Extract the image shape.
    num_azimuths, num_ranges = image.shape

    # NOTE: We duplicate the calculation of num_zero_padded_azimuths,
    # common_bandwidths, and central_frequencies since it's very fast and makes
    # code more modular.

    # We pad the image, if needed, to speed up the FFT.
    num_zero_padded_azimuths = sp.fft.next_fast_len(num_azimuths)
    image_pad = np.pad(
        image,
        ((0, num_zero_padded_azimuths - num_azimuths), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Compute the common bandwidths and the central frequencies. Both are
    # [1 x Nrng] real value vector (in [Hz]).
    common_bandwidths = positive_part(frequencies_high - frequencies_low)
    central_frequencies = (frequencies_low + frequencies_high) / 2

    # Shift the spectrum.
    # pylint: disable=invalid-sequence-index
    image[...] = sp.fft.ifft(
        colwise_roll(
            sp.fft.fft(image_pad, axis=0) * filter_bank,
            -np.round(central_frequencies * num_zero_padded_azimuths * azimuth_sampling_step).astype(np.int32),
        ),
        axis=0,
    )[0:num_azimuths, :]

    return (
        common_bandwidths[num_ranges // 2],
        central_frequencies[num_ranges // 2],
    )


def compute_filter_bank(
    *,
    num_azimuths: int,
    doppler_centroids: npt.NDArray[float],
    frequencies_xshifts: npt.NDArray[float],
    frequencies_high: npt.NDArray[float],
    frequencies_low: npt.NDArray[float],
    azimuth_bandwidth: float,
    forward_azimuth_window_type: ConvolutionWindowType,
    forward_azimuth_window_param: float,
    backward_azimuth_window_type: ConvolutionWindowType,
    backward_azimuth_window_param: float,
    azimuth_sampling_step: float,
    dtypes: EstimationDType,
) -> npt.NDArray[float]:
    """
    Compute the filter bank to apply to the images.

    Parameters
    ----------
    num_azimuths: int
        Total number of azimuths (i.e. Nazm).

    doppler_centroids: npt.NDArray[float] [Hz]
        The [1 x Nrng] Doppler centroids.

    frequencies_xshifts: npt.NDArray[float] [Hz]
        The [1 x Nrng] frequency cross-shifts.

    frequencies_high: npt.NDArray[float] [Hz]
        The [1 x Nrng] spectrum upper frequency bounds.

    frequencies_low: npt.NDArray[float] [Hz]
        The [1 x Nrng] spectrum lower frequency bounds.

    forward_azimuth_window_type: ConvolutionWindowType
        The forward window type (e.g. Hamming, Kaiser, etc.)

    forward_azimuth_window_param: float
        The forward window parameter of the current image.

    backward_azimuth_window_type: ConvolutionWindowType
        The common final window type (e.g. Hamming, Kaiser, etc.)

    backward_azimuth_window_param: float
        The parameter of the final common window.

    azimuth_sampling_step: float [Hz]
        The azimuth sampling step.

    dtypes: EstimationDType
        Floating point precision used for the estimation.

    Raises
    ------
    AssertionError, AzfRuntimeError

    Return
    ------
    npt.NDArray[float]
        The [Nazm x Nrng] filter matrix.

    """
    if (
        forward_azimuth_window_type is ConvolutionWindowType.UNIFORM
        or forward_azimuth_window_type is ConvolutionWindowType.UNIFORM
    ):
        raise AzfRuntimeError("Uniform filter window are not supported.")

    # We pad the image, to speed up the FFT.
    num_zero_padded_azimuths = sp.fft.next_fast_len(num_azimuths)

    # The azimuth axes (time and freq) as column vector.
    azimuth_freq_axis = (1 / azimuth_sampling_step) * np.roll(
        np.arange(num_zero_padded_azimuths, dtype=dtypes.float_dtype) / num_zero_padded_azimuths - 0.5,
        shift=int(num_zero_padded_azimuths / 2),
    ).reshape(-1, 1)

    # Compute the common bandwidths and the central frequencies. Both are
    # [1 x Nrng] real value vector (in [Hz]).
    common_bandwidths = positive_part(frequencies_high - frequencies_low)
    central_frequencies = (frequencies_low + frequencies_high) / 2

    # Compute the shifted doppler centroids. These are [1 x Nrng] vectors.
    frequencies_dc_p0 = np.argmin(
        np.abs(azimuth_freq_axis - doppler_centroids),
        axis=0,
    )
    frequencies_dc_p1 = np.argmin(
        np.abs(azimuth_freq_axis - doppler_centroids + central_frequencies + frequencies_xshifts),
        axis=0,
    )

    # Compute the filter windows.
    centers = np.round(central_frequencies * num_zero_padded_azimuths * azimuth_sampling_step).astype(np.int32)

    forward_window = colwise_roll(
        _WINDOW_FUNCTION[forward_azimuth_window_type](
            centers=np.zeros((centers.size,)),
            frequency_bandwidths=np.full((common_bandwidths.size), azimuth_bandwidth),
            sampling_frequency=1 / azimuth_sampling_step,
            window_param=forward_azimuth_window_param,
            nsamples=num_zero_padded_azimuths,
            inverse=True,
            dtype=dtypes.float_dtype,
        ),
        frequencies_dc_p0,
    )
    backward_window = colwise_roll(
        _WINDOW_FUNCTION[backward_azimuth_window_type](
            centers=centers,
            frequency_bandwidths=common_bandwidths,
            sampling_frequency=1 / azimuth_sampling_step,
            window_param=backward_azimuth_window_param,
            nsamples=num_zero_padded_azimuths,
            inverse=False,
            dtype=dtypes.float_dtype,
        ),
        frequencies_dc_p0 - frequencies_dc_p1,
    )
    assert_numeric_types_equal(forward_window, expected_dtype=dtypes.float_dtype)
    assert_numeric_types_equal(backward_window, expected_dtype=dtypes.float_dtype)

    return forward_window * backward_window


def compute_cross_shift_stack(
    *,
    synth_phases: tuple[npt.NDArray[float], ...],
    doppler_centroids: npt.NDArray[float],
    azimuth_sampling_step: float,
    azimuth_bandwidths: tuple[float, ...],
    coreg_primary_image_index: int,
    dtypes: EstimationDType,
) -> tuple[npt.NDArray[float], ...]:
    """
    Compute the cross-shift frequency.

    Parameters
    ----------
    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM of shape [Nazm x Nrng].

    doppler_centroids: npt.NDArray[float] [Hz]
        The Doppler centroids for each image, packed as a numpy array of
        shape [Nimg x Nrng]

    azimuth_sampling_step: float [s]
        The azimuth sampling step.

    azimuth_bandiwdths: tuple[float, ...] [Hz]
        The [1 x Nimg] azimuth bandwiths.

    coreg_primary_image_index: int
        The index of the coregistration primary image.

    dtypes: EstimationDType
        Floating point precision used for the estimation.

    Raises
    ------
    AssertionError, AzfRuntimeError

    Return
    ------
    npt.NDArray[float] [Hz]
        The [1 x Nrng] cross-shifts frequencies.

    npt.NDArray[float] [Hz]
        The [1 x Nrng] upper spectrum bounds.

    npt.NDArray[float] [Hz]
        The [1 x Nrng] lower spectrum bounds.

    """
    # Minimal check on the input.
    if not len(synth_phases) == doppler_centroids.shape[0] == len(azimuth_bandwidths):
        raise AzfRuntimeError("Invalid stack")
    if not 0 <= coreg_primary_image_index < len(synth_phases):
        raise AzfRuntimeError("Invalid coreg primary image index")

    # Compute the cross-shifts as [Nimg x Nrng] vector.
    #
    # Each row of the vector is defined as:
    #
    #     fx[i,:] = 1/(2*\pi) <\grad{t{az}} [\phi{i}[t{az},:]-\phi{p}[t{az},:]>,
    #
    # where <.> denotes the median operator wrt the azimuth dimension and
    # \phi{p} are the synthetic phases associated to the primary image.
    #
    # NOTE: This objects below are packed as [Nimg x Nazm x Nrng] matrices,
    # thus the the azimuth dimension corresponds to '1'.
    #
    freqs_xshifts = np.mean(
        np.gradient(
            np.asarray(synth_phases) - synth_phases[coreg_primary_image_index],
            2 * np.pi * azimuth_sampling_step,
            axis=1,
        ),
        axis=1,
    )
    assert freqs_xshifts.shape == doppler_centroids.shape, (
        "xshfits and DCs have mismatching shapes (xshift={}, DCs={})".format(
            freqs_xshifts.shape, doppler_centroids.shape
        )
    )

    # Compute the upper and lower spectrum bounds. This is packed as a
    # [Nimg x Nrng] vector.
    xshifted_freqs = doppler_centroids + freqs_xshifts

    # Three vectors of shape [1 x Nrng]. We need to exclude the primaries from
    # computing the bandwidth limits.
    azimuth_bandwidths = np.array(azimuth_bandwidths, dtype=dtypes.float_dtype).reshape(-1, 1)
    return (
        freqs_xshifts.astype(dtypes.float_dtype),
        np.min(xshifted_freqs + azimuth_bandwidths / 2, axis=0),
        np.max(xshifted_freqs - azimuth_bandwidths / 2, axis=0),
    )


# NOTE: Do not set parallel=True as it turns out to suck up too much RAM and it
# ends up slowing down things.
@nb.njit(cache=True, nogil=True)
def colwise_roll(data: npt.NDArray, shift: npt.NDArray[int]) -> npt.NDArray:
    """
    Rotate the columns by an column-dependent offset. This function is
    Numba compiled.

    Pseudo-code
    -----------

      rolled_data = empty_like(data)
      for r = 0:num_cols:
          rolled_data.col(r) = roll(data.col(r), shift[r])

    Parameters
    ----------
    data: npt.NDArray
        Some [Nrow x Ncol] data.

    shift: npt.NDArray[int]
            A [1 x Ncol] shift vector.

    Raises
    ------
    AzfRuntimeError

    Return
    ------
    npt.NDArray
        The [Nrow x Ncol] rolled matrix.

    """
    if data.shape[1] != shift.size:
        raise AzfRuntimeError("Data and shifts have incompatible sizes/dimensions")

    rolled_data = np.empty_like(data)

    # pylint: disable=not-an-iterable
    for col in nb.prange(data.shape[1]):
        if shift[col] != 0:
            rolled_data[:, col] = np.roll(data[:, col], shift[col])
        else:
            rolled_data[:, col] = data[:, col]

    return rolled_data
