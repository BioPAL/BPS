# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Basic Signal Processing Utilities
---------------------------------
"""

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.stack_cal_processor.core.filtering import convolve_2d, uniform_filter_2d


def compute_coherence(
    image_p: npt.NDArray[complex],
    image_s: npt.NDArray[complex],
    filter_window_size: tuple[int, int] = (5, 5),
    *,
    decimation_factors: tuple[int, int] = (1, 1),
    dtype: np.dtype = np.complex128,
) -> npt.NDArray[complex]:
    """
    Compute the coherence map (complex).

    The coherence map at a azimuth/range pixel (a,r) is
    defined as:

                                E[S(a, r) * conj(P(a, r))]
       Coh{P, S}(a, r) :=  -----------------------------------
                            sqrt(Var[P(a, r)] * Var[S(a, r)])

    Parameters
    ----------
    image_p: npt.NDArray[complex]
        The [Nazm x Nrng] primary stack image.

    image_s: npt.NDArray[complex]
        The [Nazm x Nrng] secondary stack image.

    filter_window_size: tuple[int, int] = (5, 5)
        The window size of the 2D uniform filter.

    decimation_factors: tuple[int, int] = (1, 1)
        Possibly, subsample the results.

    dtype: np.dtype = np.complex128
        The desired output type. It defaults to np.complex128.

    Return
    ------
    npt.NDArray[complex]
        The [Nazm x Nrng] coherence map.

    """
    if np.all(np.asarray(decimation_factors) > np.asarray(filter_window_size)):
        raise ValueError(f"{decimation_factors=} must be smaller than {filter_window_size=}")

    # NOTE: We ignore the floating point warning errors, we will handle them
    # anyways below.
    np.seterr(all="ignore")

    covariance_sp = uniform_filter_2d(
        image_s * np.conj(image_p),
        filter_window_size,
    )
    variance_p = uniform_filter_2d(
        np.real(image_p * np.conj(image_p)),
        filter_window_size,
    )
    variance_s = uniform_filter_2d(
        np.real(image_s * np.conj(image_s)),
        filter_window_size,
    )

    # We need to make sure that we are not dividing by zeros or doing any nasty
    # operations with NaN, inf etc.
    variance_p_variance_s = variance_p * variance_s
    valid = (variance_p_variance_s > 0.0) & (np.isfinite(variance_p_variance_s))

    coherence = np.empty(covariance_sp.shape, dtype=dtype)
    coherence[valid] = covariance_sp[valid] / np.sqrt(variance_p_variance_s[valid])
    coherence[~valid] = 0

    # NOTE: We throw away the invalid values.
    coherence[np.isnan(coherence)] = 0
    coherence[np.abs(coherence) > 1] = 0

    return coherence[:: decimation_factors[0], :: decimation_factors[1]]


def subsample_data(
    data: npt.NDArray,
    *,
    decimation_factors: tuple[int, int],
    filter_window_sizes: tuple[int, int] | None = None,
) -> npt.NDArray:
    """
    Subsample data applying a box-car filter.

    Parameters
    ----------
    data: npt.NDArray
        The input data.

    decimation_factors: tuple[int, int]
        The subsampling steps.

    filter_window_sizes: tuple[int, int] | None = None
        Optionally the size of the boxcar filter. If None, it uses
        2 * decimation_factors{i} + 1 if None is passed.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[float]

    """
    if decimation_factors[0] <= 0 or decimation_factors[1] <= 0:
        raise ValueError(f"Invalid {decimation_factors=}")
    if filter_window_sizes is None:
        filter_window_sizes = 2 * np.asarray(decimation_factors) + 1

    # Trivial case. Nothing to do.
    if decimation_factors[0] == 1 and decimation_factors[1] == 1:
        return data

    return uniform_filter_2d(data, filter_window_sizes)[:: decimation_factors[0], :: decimation_factors[1]]


def compute_spectrum(
    image: npt.NDArray[complex],
    *,
    axis: int,
    dtype: np.dtype = np.float64,
    center: bool = False,
    return_dB: bool = False,
) -> npt.NDArray[float]:
    """
    Compute the spectrum along a specific direction.

    Parameters
    ----------
    image: npt.NDArray[complex]
        The input image.

    axis: int
        The axis (0: azimuth, 1: range).

    dtype: np.dtype = np.float64
        The floating point precision of the output. It defaults to
        64-bit floating point.

    center: bool = False
        Whether to center the spectrum via FFT-shift. Defaults to
        false.

    return_dB: bool = False
        Return value in dB.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[float]
        The spectrum, possibly in dB, if selected.

    """
    if axis not in (0, 1):
        raise ValueError("Axis can only be 0 (azimuth) or 1 (range).")
    spectrum = np.nanmean(
        np.abs(sp.fft.fft(image, axis=axis)),
        axis=int(not axis),
        dtype=dtype,
    )
    if center:
        spectrum = sp.fft.fftshift(spectrum)
    if return_dB:
        spectrum = 20 * np.log10(spectrum)
    return spectrum


def compute_sublook(
    image: npt.NDArray[complex],
    demodulation_kernel: npt.NDArray[complex],
    filter_kernel: npt.NDArray[complex],
    axis: int,
    apply_remodulation: bool = True,
) -> npt.NDArray[complex]:
    """
    Compute the image sub-look.

    Parameters
    ----------
    image: npt.NDArray[complex]
        The input image.

    demodulation_kernel: npt.NDArray[complex]
        The demodulation  matrix.

    filter_kernel: npt.NDArray[complex]
        A low-pass kernel with support desired sub-look
        bandwidth.

    axis: int
        Sub-look direction.

    apply_remodulation: bool = True
        Whether to remodulate the data or not.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[complex]
        The image's sub-look.

    """
    if axis == 0:
        shape = (-1, 1)
    elif axis == 1:
        shape = (1, -1)
    else:
        raise ValueError("axis must be either 0 or 1")

    sub_look = convolve_2d(
        image * demodulation_kernel.reshape(*shape),
        filter_kernel.reshape(*shape),
    )

    if apply_remodulation:
        return np.conj(demodulation_kernel.reshape(*shape)) * sub_look

    return sub_look


def compute_range_common_bands_and_sublook_freqs(
    *,
    interferometric_pair_indices: tuple[tuple[int, int], ...],
    synth_images: tuple[npt.NDArray[float], ...],
    sampling_frequency: float,
    bandwidth: float,
    look_band: float = 0.0,
    central_frequency: float = 0.0,
    dtype: np.dtype = np.float64,
) -> tuple[dict, dict]:
    """
    Compute the common bands and the auxiliary sub-look frequencies.

    For each interferometric pairs, compute common band's upper and lower
    bound frequencies f_L, f_H and sub-look frequencies f_l, f_h as below

          +---------------- bandwidth -------------+
          |                                        |
          v        >> spectral shift (df) >>       v

              f_L  f_l          f_0          f_h  f_H
          [----{====+============+============+====]----}
          [    {....|............|............|....]    }
          [    {....|............|............|....]    }
          [    {....|............|............|....]    }
          [    {....|............|............|....]    }
          [    {....|............|............|....]    }
      ----[----{----+---------X--+------------+----]----}---->
                             f_c
                    ^            ^            ^
                    |            |            |
                    +-----------] [-----------+
                     (f_H - f_L)   (f_H - f_L)
                       x look_band   x look_band

    Parameters
    ----------
    interferometric_pair_indices: tuple[tuple[int, int], ...]
        Indices of the interferometric stack pairs.

    synth_images: tuple[npt.NDArray[float], ...]  [rad]
        The synthetic phases from the DEM (DSI).

    sampling_frequency: float [Hz]
        The sampling frequency in range.

    bandwidth: float [Hz]
        The range bandwidth.

    look_band: float [%]
        The percentage of common range band that will define
        the distance of the range look frequencies f_l, f_h from the
        mid frequency f_0.

    central_frequency: float [Hz]
        The central (carrier) frequency. If set to 0 Hz (as per default)
        the output frequencies are relative to the central frequency.

    dtype: np.dtype
        The desired floating-point precision.

    Return
    ------
    frequency_LH: dict[tuple[int, int], dict[str, float]] [Hz]
        The range band bounds f_L, f_H indexed over the interferometric
        pair indices and identified via key 'l' and 'h'.

    frequency_lh: dict[tuple[int, int], dict[str, float]] [Hz]
        The range look freqs f_l, f_h indexed over the interferometric
        pair indices and identified via key 'l' and 'h'.

    """
    # Compute the range spectral shifts by averaging the range derivative
    # of the DSI's.
    range_df = np.array(
        [np.mean(np.gradient(synth, 2 * np.pi / sampling_frequency, axis=1), dtype=dtype) for synth in synth_images]
    )

    # Compute the range common bands upper and lower bound f_H, f_L.
    frequency_LH = {
        (p, s): {
            "l": central_frequency + max(range_df[p] - bandwidth / 2, range_df[s] - bandwidth / 2),
            "h": central_frequency + min(range_df[p] + bandwidth / 2, range_df[s] + bandwidth / 2),
        }
        for p, s in interferometric_pair_indices
    }

    # Compute the sub-look frequencies as
    #
    #  f_l/h = (f_L + f_H) / 2 +/- (f_H - f_L) * param,
    #
    frequency_lh = {
        pair: {
            "l": (frequency_LH[pair]["h"] + frequency_LH[pair]["l"]) / 2
            - (frequency_LH[pair]["h"] - frequency_LH[pair]["l"]) * look_band,
            "h": (frequency_LH[pair]["h"] + frequency_LH[pair]["l"]) / 2
            + (frequency_LH[pair]["h"] - frequency_LH[pair]["l"]) * look_band,
        }
        for pair in interferometric_pair_indices
    }

    return frequency_LH, frequency_lh


def compute_demodulation_kernel(
    time_axis: npt.NDArray[float],
    delta_frequency: float,
    *,
    dtype: np.dtype = np.complex128,
) -> npt.NDArray[complex]:
    """
    Compute the demodulation kernel.

    Parameters
    ----------
    time_axis: npt.NDArray[float] [s]
        The time axis values.

    delta_frequency: float [Hz]
        Delta frequency to be shifted.

    dtype: np.dtype = np.complex128
        The desired output type. It defaults to np.complex128.

    Return
    ------
    npt.NDArray[complex]
        The demodulation matrix.

    """
    return np.exp((-2j * delta_frequency * np.pi) * time_axis, dtype=dtype)


def gaussian_kernel(
    time_axis: npt.NDArray[float],
    sub_look_bandwidth: float,
    *,
    kernel_param: int = 7,
    dtype: np.dtype = np.float64,
) -> npt.NDArray[float]:
    """
    Compute a band-pass'ed Gaussian kernel window.

    Parameters
    ----------
    time_axis: npt.NDArray[float] [s]
        The time axis values.

    sub_look_bandwidth: float [Hz]
        The sub-look bandwidth.

    kernel_param: int = 7
        Kernel size parameter.

    dtype: np.dtype = np.float64
        The desired output type. It defaults to np.float64.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[float]
        The Gaussian kernel.

    """
    if sub_look_bandwidth <= 0:
        raise ValueError("Sub-look bandwidth must be positive")
    if kernel_param <= 0:
        raise ValueError("Kernel parameter must be positive")

    sigma = 1 / sub_look_bandwidth
    time_delta = np.mean(np.diff(time_axis))
    time_axis_filter = np.arange(kernel_param * sigma / time_delta) * time_delta
    kernel = np.exp(
        -((time_axis_filter - np.mean(time_axis_filter)) ** 2) / (2 * sigma**2),
        dtype=dtype,
    ) / np.sqrt(2 * np.pi * sigma**2)
    return kernel / np.sum(kernel)


def cosine_spectral_window(
    sampling_frequencies: npt.NDArray[float],
    window_parameter: float = 0.54,
    *,
    dtype: np.dtype = np.float64,
) -> npt.NDArray[float]:
    r"""
    Return the cosine window (use Hamming parameters by default), that is

       a + (1-a) * cos(2 * \pi * n / N), n=-N/2, ... ,N/2

    Parameters
    ----------
    sampling_frequencies: npt.NDArray[float] [Hz]
        The domain of the cosine window.

    window_parameter: float = 0.54
        The window parameter.

    dtype: np.dtype = np.float64
        The desired output type. It defaults to np.float64.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[float]
        The cosine window evaluated in the domain points.

    """
    if not 0 <= window_parameter <= 1:
        raise ValueError("invalid cosine window parameter (not in [0,1])")
    alpha = 2 * np.pi / len(sampling_frequencies)
    window = window_parameter - (1 - window_parameter) * np.cos(
        alpha * sampling_frequencies / np.diff(sampling_frequencies[:2]),
        dtype=dtype,
    )

    return window / np.sqrt(np.mean(np.abs(window) ** 2))


def prepare_spectral_window_removal(
    *,
    compression_window_parameter: float | None,
    compression_window_band: float | None,
    sampling_step: float,
    num_samples: int,
    dtype: np.dtype = np.float64,
) -> tuple[npt.NDArray[float] | None, float | None, npt.NDArray[float] | None]:
    """
    Prepare the spectral window removal.

    Parameters
    ----------
    compression_window_parameter: float | None [adim]
        The a{0} parameter for the cosine window (between 0 and 1).

    compression_window_band: float | None [adim]
        Data banwdith as portion of the spectrum (between 0 and 1).

    sampling_step: float [s]
        The sampling time step.

    num_samples: int [adim]
        The number of samples.

    dtype: np.dtype = np.float64
        The desired output type. It defaults to np.float64.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[float] | None [Hz]
        Optionally, The sampling frequencies.

    float | None [Hz]
        The cut-off frequency (Nyquist)

    npt.NDArray[float] | None [Hz]
        The shifted sampling frequencies of a cosine window.

    """
    # We consider this trivial case only to improve usability within the project.
    if compression_window_parameter is None or compression_window_band is None:
        return (None, None, None)

    if not 0 <= compression_window_parameter <= 1:
        raise ValueError("compression parameter must be in [0,1]")
    if not 0 < compression_window_band <= 1:
        raise ValueError("compression window band must be in (0,1]")
    if num_samples < 1:
        raise ValueError("num samples must be a positive integer")
    if sampling_step <= 0:
        raise ValueError("sampling step [s] must be positive")

    sampling_frequencies = sp.fft.fftfreq(num_samples, sampling_step)
    cutoff_sampling_frequency = compression_window_band / (2 * sampling_step)
    spectral_window = cosine_spectral_window(
        sampling_frequencies[np.abs(sampling_frequencies) < cutoff_sampling_frequency],
        compression_window_parameter,
        dtype=dtype,
    )

    return (
        sampling_frequencies.astype(dtype),
        cutoff_sampling_frequency,
        sp.fft.fftshift(spectral_window),
    )


def spectral_window_removal(
    *,
    image: npt.NDArray[complex],
    sampling_frequencies: npt.NDArray[float],
    spectral_window: npt.NDArray[float],
    cutoff_sampling_frequency: npt.NDArray[float],
    axis: int,
    dtype: np.dtype = np.complex128,
) -> npt.NDArray[complex]:
    """
    Apply the spectral window removal.

    Parameters
    ----------
    image: npt.NDArray[complex]
        The input image.

    sampling_frequencies: npt.NDArray[float] [Hz]
        The frequency axis.

    spectral_window: npt.NDArray[float]
        The spectral window.

    cutoff_sampling_frequency: float [Hz]
        The Nyquist cut-off frequency.

    axis: int
        Indicator for azimuths (0) or ranges (1).

    dtype: np.dtype = np.complex128
        The desired output type. It defaults to np.complex128.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[complex]
        The input data with removed spectral window.

    """
    # pylint seems to be getting confused with sp.fft.fft.
    # pylint: disable=no-member,invalid-sequence-index

    # FFT the original image.
    image_freq = sp.fft.fft(image, axis=axis)

    # Cutoff the values associated to higher frequency than the cut-off.
    update = np.abs(sampling_frequencies) < cutoff_sampling_frequency
    no_update = ~update

    image_no_window = np.empty(image_freq.shape, dtype=dtype)
    if axis == 0:
        image_no_window[no_update, :] = image_freq[no_update, :]
        image_no_window[update, :] = (image_freq[update, :] / spectral_window.reshape(-1, 1),)
    elif axis == 1:
        image_no_window[:, no_update] = image_freq[:, no_update]
        image_no_window[:, update] = (image_freq[:, update] / spectral_window.reshape(1, -1),)
    else:
        raise ValueError("axis must be either 0 (azimuths) or 1 (ranges)")

    return sp.fft.ifft(image_no_window, axis=axis)


def compute_azimuth_common_bandwidth(
    *,
    synth_phases: tuple[npt.NDArray[float], ...],
    doppler_centroids: tuple[npt.NDArray[float], ...],
    coreg_primary_image_index: int,
    azimuth_bandwidths: tuple[float],
    azimuth_sampling_step: float,
) -> tuple[float, tuple[float]]:
    """
    Compute the common azimuth bandwidth and the central frequency.

    Parameters
    ----------
    synth_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg synthetic phases from DEM (i.e. DSI). All must have
        shape [Nazm x Nrng].

    doppler_centroids: tuple[npt.NDArray[float], ...] [Hz]
        The Nimg Doppler centroids, each of shape [1 x Nrng].

    coreg_primary_image_index: int
        The index of the primary image used during coregistration.

    azimuth_bandwidths: tuple[float] [Hz]
        The azimuth bandwiths of the stack images.

    azimuth_sampling_step: float [s]
        The azimuth sampling step (i.e. 1/PRF).

    Raises
    ------
    ValueError

    Return
    ------
    float [Hz]
        The azimuth sampling frequency.

    tuple[float, ...] [Hz]
        The [1 x Nimg] azimuth common bandwidths.

    """
    # Check on the selected primary.
    if not 0 <= coreg_primary_image_index < len(synth_phases):
        raise ValueError(f"{coreg_primary_image_index} is not a valid index")
    if azimuth_sampling_step <= 0:
        raise ValueError("azimuth sampling step must be positive")
    if len(synth_phases) != len(azimuth_bandwidths):
        raise ValueError("wrong number of azimuth bandwidths")

    # The stack shape.
    num_images = len(synth_phases)
    num_azimuths, num_ranges = synth_phases[0].shape

    # The common azimuth bandwidth and central frequency is computed at the
    # central time.
    mid_range = num_ranges // 2

    # Pack primary and secondary objects in a convenient way.
    secondary_image_indices = np.arange(num_images) != coreg_primary_image_index

    doppler_centroids_primary = doppler_centroids[coreg_primary_image_index][..., mid_range]
    doppler_centroids_secondary = doppler_centroids[secondary_image_indices][..., mid_range]

    synth_primary = synth_phases[coreg_primary_image_index][..., mid_range]
    synths_secondary = np.array(
        tuple(synth_phases[i][..., mid_range] for i in range(num_images) if secondary_image_indices[i])
    )

    azimuth_bandwidths = np.array(
        [
            azimuth_bandwidths[coreg_primary_image_index],
            *np.asarray(azimuth_bandwidths)[secondary_image_indices],
        ],
    ).reshape(-1, 1)

    # Compute the cross-shifts as [1 x Nrng] vector.
    frequencies_xshifts = np.median(
        np.gradient(
            synth_primary - synths_secondary,
            2 * np.pi * azimuth_sampling_step,
            axis=1,
        )
    )

    # Compute the upper and lower spectrum bounds.
    xshifted_frequencies = np.array(
        [
            doppler_centroids_primary,
            *(doppler_centroids_secondary + frequencies_xshifts),
        ],
    )
    frequency_high = np.min(xshifted_frequencies + azimuth_bandwidths / 2)
    frequency_low = np.max(xshifted_frequencies - azimuth_bandwidths / 2)

    return (
        (frequency_low + frequency_high) / 2 * sp.fft.next_fast_len(num_azimuths) * azimuth_sampling_step,
        [max(frequency_high - frequency_low, 0)] * num_images,
    )


def range_spectral_filtering(
    *,
    image: npt.NDArray[complex],
    range_time_axis: npt.NDArray[float],
    frequency_high: float,
    frequency_low: float,
    common_band_scaling: float = 1.0,
    dtype: np.dtype = np.complex128,
) -> npt.NDArray[complex]:
    """
    Apply the range spectral filtering to the input images.

    Parameters
    ----------
    image: npt.NDArray[complex]
        The input image.

    range_time_axis: npt.NDArray[float] [s]
        The relative slant range time axis.

    frequency_high: float [Hz]
        The upper range-look frequency.

    frequency_low: float [Hz]
        The lower range-range frequency.

    common_band_scaling: float = 1.0 [adim]
        A scaling factor for the common band. Defaults to 1.0.

    dtype: np.dtype = np.complex128
        The desired output type. It defaults to np.complex128.

    Raises
    ------
    ValueError
        In case the upper and lower frequencies are inconsistent or the
        scaling factor is not positive.

    Return
    ------
    npt.NDArray[complex]
        The filtered image.

    """
    if frequency_high <= frequency_low:
        raise ValueError("Invalid f_h and f_l in range spectral filtering")
    if common_band_scaling <= 0.0:
        raise ValueError("Range spectral filtering scaler must be positive")

    return compute_sublook(
        image,
        compute_demodulation_kernel(
            range_time_axis,
            (frequency_low + frequency_high) / 2,
            dtype=dtype,
        ),
        gaussian_kernel(
            range_time_axis,
            (frequency_high - frequency_low) * common_band_scaling,
            dtype=np.finfo(dtype).dtype,
        ),
        axis=1,  # aka range.
    )


def _fft_pad(
    data: npt.NDArray[float],
    axis: int,
    size: float,
    value: float = 0.0,
) -> npt.NDArray[float]:
    """Pad the FFT for rinterpolation."""
    size_to_pad = data.shape[axis]
    half_size_to_pad = size_to_pad // 2

    if axis == 0:
        return np.concatenate(
            [
                data[:half_size_to_pad, :],
                np.full((size, data.shape[1]), value),
                data[half_size_to_pad:, :],
            ],
            axis=0,
        )
    if axis == 1:
        return np.concatenate(
            [
                data[:, :half_size_to_pad],
                np.full((data.shape[0], size), value),
                data[:, half_size_to_pad:],
            ],
            axis=1,
        )

    raise NotImplementedError("Only axis=0 or axis=1 is supported")


def fft_interpolate(
    data: npt.NDArray[float],
    *,
    oversampling_x: int = 1,
    oversampling_y: int = 1,
):
    """
    FFT-based interpolation.

    Parameters
    ----------
    data: npt.NDArray[float]
        The input data.

    oversampling_x: int = 1
        The oversampling factor in the x-axis. Defaulted to 1.

    oversampling_y: int = 1
        The oversampling factor in the y-axis. Defaulted to 1.

    Raises
    ------
    NotImplementedError

    Returns
    -------
    npt.NDArray[float]
        The interpolated data

    npt.NDArray[float]
        The interpolation x-axis.

    npt.NDArray[float]
        The interpolation y-axis.

    """
    if data.ndim != 2:
        raise NotImplementedError("FFT-interpolation is supported on 2D data only")

    data = sp.fft.fft2(data)

    # Pad the axes.
    size = (oversampling_x - 1) * data.shape[0]
    data = np.concatenate(
        [
            data[: data.shape[0] // 2, :],
            np.zeros((size, data.shape[1])),
            data[data.shape[0] // 2 :, :],
        ],
        axis=0,
    )

    size = (oversampling_y - 1) * data.shape[1]
    data = np.concatenate(
        [
            data[:, : data.shape[1] // 2],
            np.zeros((data.shape[0], size)),
            data[:, data.shape[1] // 2 :],
        ],
        axis=1,
    )

    # Interpolate.
    return (
        sp.fft.ifft2(data),
        np.arange(data.shape[0]) / oversampling_x,
        np.arange(data.shape[1]) / oversampling_y,
    )
