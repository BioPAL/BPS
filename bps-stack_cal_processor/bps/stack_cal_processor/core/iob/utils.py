# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Slow-varying Ionosphere Utilities
---------------------------------
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy as sp
from bps.stack_cal_processor.configuration import IOB_NAME, StackCalConf, StackDataSpecs
from bps.stack_cal_processor.core.filtering import convolve_2d
from bps.stack_cal_processor.core.floating_precision import (
    EstimationDType,
    assert_numeric_types_equal,
)
from bps.stack_cal_processor.core.utils import SubLookFilterType


class IobRuntimeError(RuntimeError):
    """Handle an error that occurs while running the IOB."""

    def __init__(self, message: str):
        super().__init__(f"[{IOB_NAME}]: {message}")


@dataclass
class SubLookConf:
    """The parameters for the sub-look."""

    frequency: float  # [Hz]
    """Where the sub-look is centered."""

    bandwidth: float  # [Hz]
    """The portion of spectrum to be preserved."""

    filter_type: SubLookFilterType = SubLookFilterType.GAUSSIAN
    """The filter type (i.e. 'gaussian' or 'FIR')."""

    filter_param: Union[int, float] = 7.0
    """Parameter of filter (e.g. 7.0 for Gaussian or 17 for FIR). """


def compute_range_sublook(
    *,
    stack_image: npt.NDArray[complex],
    range_sampling_step: float,
    central_frequency: float,
    sublook_conf: SubLookConf,
    dtypes: EstimationDType,
) -> npt.NDArray[complex]:
    """
    Compute the range sub-look of a stack image.

    Parameters
    ----------
    stack_image: npt.NDArray[complex]
        The input image.

    range_sampling_step: float [s]
        The sampling step in range.

    central_frequency: float [Hz]
        The central (carrier) frequency.

    sub_look_conf: SubLookConf
        Parameters to compute the sub-look.

    dtypes: EstimationDType
        The floating precision for the estimation.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[complex]
         The sub-look in range direction.

    """
    if sublook_conf.filter_type == SubLookFilterType.GAUSSIAN:
        return _compute_range_sublook_gaussian(
            stack_image=stack_image,
            range_sampling_step=range_sampling_step,
            central_frequency=central_frequency,
            sublook_frequency=sublook_conf.frequency,
            sublook_bandwidth=sublook_conf.bandwidth,
            gaussian_param=float(sublook_conf.filter_param),
            dtypes=dtypes,
        )
    if sublook_conf.filter_type == SubLookFilterType.FIR:
        return _compute_range_sublook_fir(
            stack_image=stack_image,
            range_sampling_step=range_sampling_step,
            central_frequency=central_frequency,
            sublook_frequency=sublook_conf.frequency,
            sublook_bandwidth=sublook_conf.bandwidth,
            fir_ntaps=int(sublook_conf.filter_param),
            dtypes=dtypes,
        )

    raise ValueError(f"Invalid {sublook_conf.filter_type=}")


def compute_qualities(
    *,
    stack_specs: StackDataSpecs,
    iob_conf: StackCalConf.IobConf,
    coherence: npt.NDArray[float],
    f_l: float,
    f_h: float,
    estimation_dtypes: EstimationDType,
) -> tuple[float, float, float]:
    """
    Compute the multi-baseline weights and quality.

    The multi-baseline weights are computed as

        W := inv(C.T @ G @ C)

    where C is the matrix of the look axes and G is the diagonal matrix
    of the standard deviations of the ionosphere (sigma2_iono).

    The quality is defined as the average of 1 / sqrt(1 + 2 * sigma2_iono).

    Parameters
    ----------
    stack_specs: StackDataSpecs
        The stack specs object.

    iob_conf: StackCalConf.IobConf
        The IOB configurations.

    coherence: npt.NDArray[float]
        The coherence of the inteferometric pair.

    f_l: float [Hz]
        The low sub-look frequency.

    f_h: float [Hz]
        The high sub-look frequency.

    estimation_dtypes: EstimationDType
        The floating point precision for the estimations.

    Raises
    ------
    IobRuntimeError, AssertionError

    Return
    ------
    float [rad^2/s^2]
        The variance of the azimuth iono slope estimation.

    float [rad^2/s^2]
        The variance of the range iono slope estimation.

    float [adim]
        The overall estimation quality.

    """
    # Compute the variance of the sublook.
    num_looks = _num_average_looks(iob_conf.sublook_window_sizes, stack_specs, estimation_dtypes)
    sigma2_sublooks = (1.5 / num_looks) * (1 - coherence**2) / coherence**2
    sigma2_sublooks[~np.isfinite(sigma2_sublooks)] = 0.0
    assert_numeric_types_equal(sigma2_sublooks, expected_dtype=estimation_dtypes.float_dtype)

    # fmt: off
    sigma2_iono = (
        (f_l * f_h / stack_specs.central_frequency / (f_h**2 - f_l**2))**2
        * (f_h**2 + f_l**2) * sigma2_sublooks
    )[
        ::iob_conf.sublook_window_sizes[0] // iob_conf.split_spectrum_decimation_factors[0],
        ::iob_conf.sublook_window_sizes[1] // iob_conf.split_spectrum_decimation_factors[1]
    ]
    assert_numeric_types_equal(sigma2_iono, expected_dtype=estimation_dtypes.float_dtype)
    # fmt: on

    # Compute the the look axis in azimuth direction. We take the average
    # ground velocity of the stack images to convert pixels to meters.
    # fmt: off
    azimuth_look_axis = (
        np.arange(sigma2_iono.shape[0], dtype=estimation_dtypes.float_dtype)
        * stack_specs.azimuth_sampling_step
        * np.mean(stack_specs.satellite_ground_speeds, dtype=estimation_dtypes.float_dtype)
        * iob_conf.sublook_window_sizes[0]
    )  # [m].
    # fmt: on
    azimuth_look_axis -= np.mean(azimuth_look_axis, dtype=estimation_dtypes.float_dtype)
    assert_numeric_types_equal(azimuth_look_axis, expected_dtype=estimation_dtypes.float_dtype)

    # Compute the look axis in range direction.
    # fmt: off
    range_look_axis = (
        np.arange(sigma2_iono.shape[1], dtype=estimation_dtypes.float_dtype)
        * stack_specs.range_sampling_step  # [s].
        * (sp.constants.speed_of_light / 2)  # [m/s].
        * iob_conf.sublook_window_sizes[1]
    )  # [m].
    range_look_axis -= np.mean(range_look_axis, dtype=estimation_dtypes.float_dtype)
    # fmt: on
    assert_numeric_types_equal(range_look_axis, expected_dtype=estimation_dtypes.float_dtype)

    # Build the C matrix. This is a [N x 3] matrix.
    (
        azimuth_look_axis,
        range_look_axis,
    ) = np.meshgrid(azimuth_look_axis, range_look_axis)

    C = np.ones((sigma2_iono.size, 3), dtype=estimation_dtypes.float_dtype)
    C[:, 0] = np.reshape(azimuth_look_axis, -1)
    C[:, 1] = np.reshape(range_look_axis, -1)
    assert_numeric_types_equal(C, expected_dtype=estimation_dtypes.float_dtype)

    # Combute the matrix Gamma. This is a [N x N] matrix.
    sigma2_iono = np.reshape(sigma2_iono, -1)
    valid_sigma2_iono = sigma2_iono != 0
    if valid_sigma2_iono.size == 0:
        raise IobRuntimeError("Matrix [C.T @ inv(G)@ C] is not invertible")

    inv_G = np.zeros_like(sigma2_iono)
    inv_G[valid_sigma2_iono] = 1 / sigma2_iono[valid_sigma2_iono]
    inv_G = sp.sparse.diags(inv_G)

    # Compute the quality of the estimation.
    sigma2 = np.linalg.inv(C.T @ inv_G @ C)
    if np.any(np.isnan(sigma2)):
        raise IobRuntimeError("Inversion of [C.T @ inv(G) @ C] result in NaN's")

    return (
        sigma2[0, 0],
        sigma2[1, 1],
        np.nanmean(1 / np.sqrt(1 + 2 * sigma2_iono)),
    )


def phase_unwrap_validity_mask(
    phi_l: npt.NDArray[float],
    phi_h: npt.NDArray[float],
    max_lh_delta_phase: float,
) -> npt.NDArray[bool]:
    """
    Compute the validity mask by checking the difference between
    the high and low unwrapped phases.

    Parameters
    ----------
    phi_l: npt.NDArray[float]
        The unwrapped interferogram of the low sub-look as an array
        of shape [Nazm x Nrng].

    phi_h: npt.NDArray[float]
        The unwrapped interferogram of the high sub-look as an array
        of shape [Nazm x Nrng].

    max_lh_delta_phase: float [rad]
        The threshold on the phase difference.

    Return
    ------
    npt.NDArray[bool]
        The validty mask.

    """
    delta_phi_lh = phi_l - phi_h
    delta_phi_lh -= np.median(delta_phi_lh)
    return np.abs(delta_phi_lh) <= max_lh_delta_phase


def _compute_range_sublook_gaussian(
    *,
    stack_image: npt.NDArray[complex],
    range_sampling_step: float,
    central_frequency: float,
    sublook_frequency: float,
    sublook_bandwidth: float,
    gaussian_param: float,
    dtypes: EstimationDType,
) -> npt.NDArray[float]:
    """Compute a sub-look using a Gaussian low-pass filter."""
    # The relative range time axis.
    range_time_axis = np.arange(stack_image.shape[1], dtype=dtypes.float_dtype) * range_sampling_step
    # The modulation kernel.
    demodulation_kernel = np.exp(
        2j * np.pi * (sublook_frequency - central_frequency) * range_time_axis,
        dtype=dtypes.complex_dtype,
    )
    # Compute the Gaussian window to extract the sub-look.
    sigma = 1 / (sublook_bandwidth * range_sampling_step)
    gaussian_window = sp.signal.windows.gaussian(round(gaussian_param * sigma), std=sigma).astype(dtypes.float_dtype)
    return (
        convolve_2d(
            stack_image * np.conj(demodulation_kernel),
            np.reshape(gaussian_window / np.sum(gaussian_window), (1, -1)),
        )
        * demodulation_kernel
    )


def _compute_range_sublook_fir(
    *,
    stack_image: npt.NDArray[complex],
    range_sampling_step: float,
    central_frequency: float,
    sublook_frequency: float,
    sublook_bandwidth: float,
    fir_ntaps: int,
    dtypes: EstimationDType,
) -> npt.NDArray[float]:
    """Compute a sub-look using a FIR low-pass filter."""
    # The relative range time axis.
    range_time_axis = np.arange(stack_image.shape[1], dtype=dtypes.float_dtype) * range_sampling_step
    # The modulation kernel.
    demodulation_kernel = np.exp(
        1j * np.pi * (sublook_frequency - central_frequency) * range_time_axis,
        dtype=dtypes.complex_dtype,
    )
    # The filering window to be used.
    fir_window = (
        sp.signal.firwin(fir_ntaps, sublook_bandwidth * range_sampling_step) * demodulation_kernel[:fir_ntaps]
    ).astype(dtypes.complex_dtype)

    return (
        sp.signal.fftconvolve(
            stack_image,
            fir_window.reshape(1, -1),
            mode="same",
        )
        * demodulation_kernel
    )


def _num_average_looks(
    look_window_size: tuple[int, int],
    stack_specs: StackDataSpecs,
    dtypes: EstimationDType,
) -> float:
    """Compute the averange number of looks."""
    baz = np.mean(stack_specs.azimuth_bandwidths, dtype=dtypes.float_dtype)
    brg = stack_specs.range_bandwidth
    faz = 1 / stack_specs.azimuth_sampling_step
    frg = 1 / stack_specs.range_sampling_step
    return look_window_size[0] * baz / faz + look_window_size[1] * brg / frg
