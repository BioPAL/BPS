# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
SKP Postprocessing Module
-------------------------
"""

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.common import bps_logger
from bps.stack_cal_processor.configuration import SkpPostprocessingFilterType
from bps.stack_cal_processor.core.skp.utils import SkpRuntimeError, window_to_azimuth_pixels, window_to_range_pixels


def compute_postprocessing_filter_window_size(
    *,
    filter_type: SkpPostprocessingFilterType,
    azimuth_sampling_step: float,
    range_sampling_step: float,
    satellite_ground_speed: float,
    incidence_angle: float,
    postprocessing_filter_window_size: float,
    goldstein_filter_window_size: int,
) -> tuple[int, int]:
    """
    Compute the size (in pixels) of the filtering window. If "Goldstein" is
    selected, the window size is in frequency domain.

    Parameters
    ----------
    filter_type: SkpPostprocessingFilterType
        The postprocessing filter type.

    azimuth_sampling_step: float [s]
        Sampling step in azimuth direction.

    range_sampling_step: float [s]
        Sampling step in range direction.

    satellite_ground_speed: float [m/s]
        Speed of the satellite at the ground level.

    incidence_angle: float [rad]
        The incidence angle.

    postprocessing_filter_window_size: float  [m]
        The desired postprocessing spatial window size. Used only for "Boxcar"
        and "Median".

    goldstein_filter_window_size: int  [px]
        The filter window size to be used for the Goldstein filter. Used only
        if `filter_type` is "Goldstein".

    Return
    ------
    tuple[int, int]  [px]
        The filtering window in pixels. Both are always odd values.

    """
    if filter_type is SkpPostprocessingFilterType.GOLDSTEIN:
        return (goldstein_filter_window_size, goldstein_filter_window_size)

    azimuth_window_size = window_to_azimuth_pixels(
        window_size=postprocessing_filter_window_size,
        azimuth_sampling_step=azimuth_sampling_step,
        satellite_ground_speed=satellite_ground_speed,
    )
    range_window_size = window_to_range_pixels(
        window_size=postprocessing_filter_window_size,
        range_sampling_step=range_sampling_step,
        incidence_angle=incidence_angle,
    )
    return (
        2 * (azimuth_window_size // 2) + 1,
        2 * (range_window_size // 2) + 1,
    )


def apply_postprocessing_filter(
    skp_calibration_phases: npt.NDArray[float],
    *,
    filter_type: SkpPostprocessingFilterType,
    filter_window: tuple[int, int],
) -> npt.NDArray[float]:
    """
    Apply the selected post-processing filter.

    Parameters
    ----------
    skp_calibration_phases: npt.NDArray[float]  [rad]
        The (wrapped) SKP calibration phase screens.

    filter_type: SkpPostprocessingFilterType
        The selected filter type.

    filter_window: tuple[int, int]  [px]
        Size of the filtering window in pixels. For "Goldstein" this
        is in frequency domain.

    Return
    ------
    npt.NDArray[float]  [rad]
        The (wrapped) filtered SKP phase screens.

    Raises
    ------
    SkpRuntimeError

    """
    # If we get no-filtering, we just skip.
    if filter_type is SkpPostprocessingFilterType.NONE:
        return skp_calibration_phases

    bps_logger.debug("Filter window size [px]: %s", filter_window)

    if filter_type is SkpPostprocessingFilterType.GOLDSTEIN:
        return _apply_goldstein_filter(
            skp_calibration_phases,
            filter_window=filter_window,
        )

    if filter_type is SkpPostprocessingFilterType.BOXCAR:
        return _apply_boxcar_filter(
            skp_calibration_phases,
            filter_window=filter_window,
        )

    if filter_type is SkpPostprocessingFilterType.MEDIAN:
        return np.array(
            [_apply_pivotal_median_filter(phi, filter_window=filter_window) for phi in skp_calibration_phases],
            dtype=skp_calibration_phases[0].dtype,
        )

    raise SkpRuntimeError(f"Unsupported filter type {filter_type}")


def _apply_goldstein_filter(
    skp_calibration_phases: npt.NDArray[float],
    filter_window: tuple[int, int],
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> npt.NDArray[float]:
    """Apply the Goldstein filter to each calibration phase screen."""
    skp_calibration_phases = np.asarray(skp_calibration_phases)
    phi_freq = sp.fft.fftshift(sp.fft.fft2(np.exp(1j * skp_calibration_phases)), axes=(-2, -1))
    power_smooth = sp.ndimage.uniform_filter(np.abs(phi_freq), size=(1, *filter_window))
    weights = (power_smooth / (power_smooth.max(axis=(-2, -1), keepdims=True) + eps)) ** alpha
    return np.angle(sp.fft.ifft2(sp.fft.ifftshift(phi_freq * weights, axes=(-2, -1))))


def _apply_boxcar_filter(
    skp_calibration_phases: npt.NDArray[float],
    filter_window: tuple[int, int],
) -> npt.NDArray[float]:
    """Apply a boxcar filter to each calibration phase screen."""
    return np.arctan2(
        sp.ndimage.uniform_filter(np.sin(skp_calibration_phases), size=(1, *filter_window)),
        sp.ndimage.uniform_filter(np.cos(skp_calibration_phases), size=(1, *filter_window)),
    )


def _apply_pivotal_median_filter(
    skp_calibration_phases: npt.NDArray[float],
    filter_window: tuple[int, int],
) -> npt.NDArray[float]:
    """Apply the pivotal median filter as in Lanari et al. "Generation of
    digital elevation models by using SIR-C/X-SAR multifrequency two-pass
    interferometry: The Etna case study." IEEE Transactions on Geoscience and
    Remote Sensing (1996)

    """

    # Just a shortcut to wrap the phase screen.
    def wrap_phase(ph):
        return (ph + np.pi) % (2 * np.pi) - np.pi

    # Apply some padding to the phase screen.
    win_az, win_rg = filter_window
    pad_az = win_az // 2
    pad_rg = win_rg // 2
    skp_calibration_phases = np.pad(
        skp_calibration_phases,
        ((pad_az, pad_az), (pad_rg, pad_rg)),
        mode="reflect",
    )

    # Create the filter window
    windows = np.lib.stride_tricks.sliding_window_view(skp_calibration_phases, (win_az, win_rg))
    filter_window = windows.reshape(*windows.shape[:2], win_az * win_rg)

    # Compute the pivot as the phase of the medians of real and imaginary parts.
    pivot = np.arctan2(
        np.median(np.sin(filter_window), axis=2),
        np.median(np.cos(filter_window), axis=2),
    )

    # Compute the median of the residuals,
    median_res = np.median(
        wrap_phase(filter_window - pivot[:, :, np.newaxis]),
        axis=2,
    )
    return wrap_phase(pivot + median_res)
