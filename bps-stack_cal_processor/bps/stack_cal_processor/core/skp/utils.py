# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities for the SKP module
----------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
import numpy.typing as npt
from bps.stack_cal_processor.configuration import SKP_NAME
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from scipy import constants
from scipy.stats import circmean, circstd, circvar


class SkpRuntimeError(RuntimeError):
    """Handle an error that occurs while running the SKP."""

    def __init__(self, message: str):
        super().__init__(f"[{SKP_NAME}]: {message}")


@dataclass
class SkpPhaseScreenStats:
    avg_phase: float  # [rad].
    std_phase: float  # [adim].
    var_phase: float  # [adim].
    min_phase: float  # [rad].
    max_phase: float  # [rad].
    mad_phase: float  # [rad].

    @classmethod
    def from_phase_screen(cls, phase_screen: npt.NDArray[float]) -> SkpPhaseScreenStats:
        """Create SkpPhaseScreenStats from a given phase screen."""
        return cls(
            avg_phase=circmean(phase_screen, low=-np.pi, high=np.pi),
            std_phase=circstd(phase_screen, low=-np.pi, high=np.pi),
            var_phase=circvar(phase_screen, low=-np.pi, high=np.pi),
            min_phase=np.min(phase_screen),
            max_phase=np.max(phase_screen),
            mad_phase=circmean(
                np.abs(phase_screen - circmean(phase_screen, low=-np.pi, high=np.pi)),
                low=-np.pi,
                high=np.pi,
            ),
        )


@nb.njit(cache=True, nogil=True)
def joint_diagonalization(
    hermitian_matrix_0: npt.NDArray[complex], hermitian_matrix_1: npt.NDArray[complex]
) -> npt.NDArray[complex]:
    """
    Given two Hermitian matices (i.e. self-adjoint), It finds the matrix U that
    diagonalizes them both, that is, U.H @ hermitian_matrix_0 @ U and
    U.H @ hermitian_matrix_1 @ U are both diagonal matrices.

    Parameters
    ----------
    hermitian_matrix_0: npt.NDArray[complex]
        First Hermitian matrix.

    hermitian_matrix_1: npt.NDArray[complex]
        Second Hermitian matrix.

    Return
    ------
    npt.NDArray[complex]
        The diagonalizing matrix U.

    """
    _, eigenvectors = np.linalg.eig(hermitian_matrix_1 @ np.linalg.inv(hermitian_matrix_0))
    return np.linalg.inv(eigenvectors.conj().T)


# NOTE: Parallelizing this function is discouraged within the scope of the SKP.
# We pioritize outer loops parallelization (what calls this) since it turns out
# to be more effective. We only JIT this one to allow for upstream users to use
# this in GIL-free loops.
@nb.njit(cache=True, nogil=True)
def normalize_coherence(
    covariance: npt.NDArray[complex],
    dtype: np.dtype,
) -> npt.NDArray[complex]:
    """
    Normalize each element of the covariance matrix with respect to the
    corresponding elements on the main diagonal. That is

      Coherence[i,j] = Covariance[i,j] / (Covariance[i,i] * Covariance[j,j])

    Parameters
    ----------
    covariance: npt.NDArray[complex]
        A square covariance matrix.

    dtype: np.dtype
        The floating point precision of the output matrix.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    npt.NDArray[complex]
        Then normalized covariance matrix (i.e. the coherence matrix).

    """
    # pylint: disable=not-an-iterable
    if covariance.shape[0] != covariance.shape[1]:
        raise SkpRuntimeError("matrix must be a square matrix")

    # NOTE: Since this method is meant for small matrices, using a sparse matrix
    # ends up being inefficient. We also use the Lebesgue assumption (0/0 is 0).
    # Note also that if a diagonal term (i,i) is 0, them (i,j) and (j,i) are
    # zeros for any j as well, so we can just use 1 as normalization term.
    diagonal = np.empty((covariance.shape[0],), dtype=dtype)
    for i in nb.prange(diagonal.size):
        diagonal[i] = 1.0 if covariance[i, i] == 0 else np.sqrt(covariance[i, i])

    # NOTE: in principle a covariance matrix should be self-adjoint, we neither
    # check that, nor leverage symmetry to reduce the assemblage of the
    # matrix. This is because we don't need extra performance here so in the
    # case a user passes a non-symmetric matrix, they will notice that from the
    # output.
    coherence = np.empty(covariance.shape, dtype=dtype)
    for i in nb.prange(covariance.shape[0]):
        for j in nb.prange(covariance.shape[1]):
            coherence[i, j] = covariance[i, j] / (diagonal[i] * diagonal[j])
    return coherence


def cache_volume_noise(
    num_images: int,
    num_azimuths: int,
    num_ranges: int,
    *,
    dtypes: EstimationDType,
    seed: int = 1,
) -> npt.NDArray[complex]:
    """
    This function returns the volume noise to apply to the SKP decomposition,
    whenever such a decomposition fails.

    Parameters
    ----------
    num_images: int
        The number of images 3 for INT and up to 7 for TOM.

    num_azimuths: int
        The number of azimuth samples.

    num_ranges: int
        The number of range samples.

    dtypes: EstimationDType
        The estimation floating point accuracy.

    seed: int = 1
        The value to seed the random generator. It defaults to 1 and must
        be a positive integer.

    Raises
    ------
    SkpRuntimeError

    Return
    ------
    npt.NDArray[complex]
        A [Nazm x Nrng x Nimg x Nimg] complex array of uniformly
        distributed noise.

    """
    if num_images <= 0:
        raise SkpRuntimeError("number of images must be a positive integer")
    if num_azimuths <= 0:
        raise SkpRuntimeError("number of azimuths must be a positive integer")
    if num_ranges <= 0:
        raise SkpRuntimeError("number of ranges must be a positive integer")

    # The output size and shape.
    noise_size = num_azimuths * num_ranges * num_images * num_images
    noise_shape = (num_azimuths, num_ranges, num_images, num_images)

    # NOTE: We need to make sure that these are repeatable, so we use a
    # Generator, as per official numpy documentation. Mersenne's Twisters are
    # just fine.
    random_generator = np.random.Generator(np.random.MT19937(seed=seed))

    return -2 * np.reshape(
        random_generator.random(noise_size, dtype=dtypes.float_dtype)
        + 1j * random_generator.random(noise_size, dtype=dtypes.float_dtype),
        noise_shape,
    ) + (1 + 1j)


def estimation_subsampling_steps(
    *,
    satellite_ground_speed: float,
    azimuth_sampling_step: float,
    range_sampling_step: float,
    window_size: float,
    incidence_angle: float,
) -> tuple[int, int]:
    """
    Compute the range and azimuth subsampling size for the estimation grid.
    The steps are chosen so that the estimation grid can be coupled with
    uniform filtering, that is:

               window_size
           +-------------------+
           |                   |
           X---------X---------X---------X--- ....
                     |                   |
                     +-------------------+
                          window_size

    Parameters
    ----------
    satellite_ground_speed: float [m/s]
        The speed of the satellite at the ground.

    azimuth_sampling_step: float [s]
        The sampling time step in azimuth.

    range_sampling_step: float [s]
        The sampling time step in range.

    window_size: float [m]
        Target estimation window.

    incidence_angle: float [rad]
        The incidence angle.

    Return
    ------
    int
        The subsampling step in azimuth.

    int
        The subsampling step in range.

    """
    azimuth_window_samples = window_to_azimuth_pixels(
        window_size=window_size,
        azimuth_sampling_step=azimuth_sampling_step,
        satellite_ground_speed=satellite_ground_speed,
    )
    range_window_samples = window_to_range_pixels(
        window_size=window_size,
        range_sampling_step=range_sampling_step,
        incidence_angle=incidence_angle,
    )
    return (
        max((azimuth_window_samples - 1) // 2, 1),
        max((range_window_samples - 1) // 2, 1),
    )


def window_to_azimuth_pixels(
    *,
    window_size: float,
    azimuth_sampling_step: float,
    satellite_ground_speed: float,
) -> int:
    """
    Convert a window size into pixels in azimuth direction.

    Parameters
    ----------
    window_size: float [m]
        Window size in meters.

    azimuth_sampling_step: float [1/s]
        The sampling step in azimuth direction.

    satellite_ground_speed: float [m/s]
        Speed of the satellite at the ground level.

    Raises
    ------
    ValueError

    Return
    ------
    int
        The corresponding pixels.

    """
    if azimuth_sampling_step <= 0:
        raise ValueError("Azimuth sampling step must be positive")
    if satellite_ground_speed == 0:
        raise ValueError("Satellite ground speed must be nonzero")
    return round(abs(window_size / (azimuth_sampling_step * satellite_ground_speed)))


def window_to_range_pixels(
    window_size: float,
    range_sampling_step: float,
    incidence_angle: float,
) -> int:
    """
    Convert a window size into pixels in range direction.

    Parameters
    ----------
    window_size: float [m]
        Window size in meters.

    range_sampling_step: float [1/s]
        The sampling step in slant-range direction.

    incidence_angle: float [rad]
        Incidence angle.

    Raises
    ------
    ValueError

    Return
    ------
    int
        The corresponding pixels.

    """
    if range_sampling_step <= 0:
        raise ValueError("Range sampling step must be positive")
    if incidence_angle == 0:
        raise ValueError("Incidence angle must be nonzero")
    return round(abs(2 * (np.sin(incidence_angle) * window_size) / (range_sampling_step * constants.speed_of_light)))


def compute_skp_calibration_phases_statistics(
    skp_calibration_phases: tuple[npt.NDArray[float], ...],
) -> tuple[SkpPhaseScreenStats, ...]:
    """
    Compute circular mean, circular variance, circular standard deviation,
    mean absolute deviation, minimum, and maximum for each of the
    skp_calibration_phases.

    Parameters
    ----------
    skp_calibration_phases: tuple[npt.NDArray[float], ...] [rad]
        The Nimg calibration phases from SKP of shape [Nazm' x Nrng'].

    Return
    ------
    tuple[SkpPhaseScreenStats, ...]
        The computed statistics for each calibration phase.
    """

    return tuple(SkpPhaseScreenStats.from_phase_screen(phase_screen) for phase_screen in skp_calibration_phases)
