# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The MSC's Utility Library
-------------------------
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy as sp
from arepytools.io.metadata import EPolarization
from arepytools.io.productfolder2 import ProductFolder2
from bps.common import bps_logger
from bps.common.roi_utils import RegionOfInterest
from bps.stack_cal_processor.configuration import MSC_NAME
from bps.stack_cal_processor.core.utils import read_productfolder_data_by_polarization


class MscRuntimeError(RuntimeError):
    """Handle an error that occurs while running the MSC."""

    def __init__(self, message: str):
        super().__init__(f"[{MSC_NAME}]: {message}")


@dataclass(kw_only=True, frozen=True)
class SecondaryStackManager:
    """
    Manager for the MSC secondary stack (no azimuth residual shifts).

    Attributes
    ----------
    product_cor_pfs : tuple[ProductFolder2 | None, ...]
        Product folders of the the warped stack images. None only for
        the coregistration primary image.
    polarizations: tuple[EPolarization, ...]
        The polarizations of the input stack.
    roi: RegionOfInterest,optional
        The stack region of interest.

    Methods
    -------
    load(image_index: int) -> tuple[npt.NDArray[complex], ...]
        Load all the polarizations of selected secondary frame.

    """

    product_cor_pfs: tuple[ProductFolder2 | None, ...]
    coreg_primary_image_index: int
    polarizations: tuple[EPolarization, ...]
    roi: RegionOfInterest | None

    def __post_init__(self):
        """Minimal validation on the arguments."""
        if self.product_cor_pfs[self.coreg_primary_image_index] is not None:
            raise MscRuntimeError(
                "Secondary stack should not contain the coreg primary",
            )
        if any(pf is None for i, pf in enumerate(self.product_cor_pfs) if i != self.coreg_primary_image_index):
            raise MscRuntimeError(
                "Missing frames in secondary stack",
            )

    def load(self, image_index: int) -> tuple[npt.NDArray[complex], ...]:
        """Load all the polarizations of selected secondary frame."""
        if self.product_cor_pfs[image_index] is None:
            raise MscRuntimeError(
                "Attempted reading coreg primary from secondary stack",
            )

        bps_logger.debug(
            "Loading secondary stack data: frame=%s",
            self.product_cor_pfs[image_index].path,
        )
        return tuple(
            read_productfolder_data_by_polarization(
                self.product_cor_pfs[image_index],
                polarization=polarization,
                roi=self.roi,
            )
            for polarization in self.polarizations
        )


def compute_azimuth_space_resolution(
    azimuth_space_step: float,
    azimuth_sampling_step: float,
    azimuth_bandwidth: float,
) -> float:
    """
    Compute the spatial resolution in along-track direction.

    Parameters
    ----------
    azimuth_space_step: float  [m]
        Pixel spacing in along-track direction.

    azimuth_sampling_step: float  [s]
        Pixel temporal spacing in along-track direction.

    azimuth_bandwidth: float  [Hz]
        RADAR bandwidth in along-track direction.

    Return
    ------
    float  [m]
        The resolution in along-track direction.

    """
    return float(
        azimuth_space_step  # [m],
        / azimuth_sampling_step  # [1/s].
        / azimuth_bandwidth  # [1/Hz].
    )


def compute_azimuth_space_axis(
    *,
    azimuth_sampling_step: float,
    num_azimuths: int,
    satellite_ground_speed: float,
    dtype: np.dtype = np.float64,
) -> npt.NDArray[float]:
    """
    Compute the spatial axis in along-track direction.

    Parameters
    ----------
    azimuth_sampling_step: float  [s]
        The sampling step in along-track direction.

    num_azimuths: int
        Number of pixels/lines in along-track directions.

    satellite_ground_speed: float  [m/s]
        The satellite speed at the ground level.

    dtype: np.dtype = np.float64
        The floating point precision.

    Return
    ------
    npt.NDArray[float]
        The [1 x Naz] array storing the axis.

    """
    dx = float(satellite_ground_speed * azimuth_sampling_step)
    return dx * np.arange(num_azimuths, dtype=dtype)


def compute_slant_range_space_axis(
    *,
    range_sampling_start: float,
    range_sampling_step: float,
    num_ranges: int,
    dtype: np.dtype = np.float64,
) -> npt.NDArray[float]:
    """
    Compute the spatial axis in along-track direction.

    Parameters
    ----------
    range_sampling_start: float  [s]
        The start time in slant-range direction.

    range_sampling_step: float  [s]
        The sampling step in slant-range direction.

    num_azimuths: int
        Number of pixels/samples in range directions.

    dtype: np.dtype = np.float64
        The floating point precision.

    Return
    ------
    npt.NDArray[float]
        The [1 x Nrg] array storing the axis.

    """
    x0 = float(range_sampling_start * sp.constants.speed_of_light / 2)
    dx = float(range_sampling_step * sp.constants.speed_of_light / 2)
    return x0 + dx * np.arange(num_ranges, dtype=dtype)


def coherence_window_size_px(
    size: tuple[float, float],
    *,
    azimuth_space_step: float,
    slant_range_space_step: float,
) -> tuple[int, int]:
    """
    Convert a coherence window size from meters to pixels.

    Parameters
    ----------
    size: tuple[float, float]  [m]
        The window size (along-track/slant-range) in meters

    azimuth_space_step: float  [m]
        The pixel spacing in along-track direction.

    slant_range_space_step: float [m]
        The pixel spacing in slant-range direction.

    Returns
    -------
    tuple[float, float]  [px]
        The window size in pixels.

    """
    return (
        2 * round(0.5 * size[0] / azimuth_space_step) + 1,
        2 * round(0.5 * size[1] / slant_range_space_step) + 1,
    )


def azimuth_meters_to_pixels(
    meters: float,
    *,
    azimuth_sampling_step: float,
    satellite_ground_speed: float,
) -> float:
    """Convert meters to pixels in along-track direction."""
    return float(
        (meters / azimuth_sampling_step)  # [m/s].
        / satellite_ground_speed  # [m/s].
    )


def compute_ionosphere_axes(
    *,
    ionosphere_azimuth_space_axis: npt.NDArray[float],
    ionosphere_slant_range_space_axis: npt.NDArray[float],
    ionosphere_los_range_ground: float,
    satellite_ground_speed: float,
    incidence_angle: float,
    earth_radius: float,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    Compute the ionosphere axes (along-track/slant-range).

    Parameters
    ----------
    ionosphere_azimuth_space_axis: npt.NDArray[float]  [m]
        The space axis in along-track direction.

    ionosphere_slant_range_space_axis: npt.NDArray[float]  [m]
        The space axis in slant-range direction.

    ionosphere_los_range_ground: float  [m]
        The ionosphere LOS range from ground.

    satellite_ground_speed: float  [m/s]
        The satellite speed on ground.

    incidence_angle: float  [rad]
        The incidence angle

    earth_radius: float  [m]
        The Earth's radius to be used for the conversion.

    Return
    ------
    npt.NDArray[float]  [s]
        The ionosphere axis in along-track direction (at iono altitude).

    npt.NDArray[float]  [s]
        The ionosphere axis in slant-range direction (at iono altitude).

    """
    # Convert the ionosphere space axis in slant-range direction
    ionosphere_slant_range_axis = (
        2
        * ionosphere_slant_range_space_axis  # [m].
        / sp.constants.speed_of_light  # [s/m].
    )

    # Convert the ionosphere space axis in along-track direction.
    ionosphere_altitude = los_range_ground_to_altitude(
        ionosphere_los_range_ground,
        incidence_angle=incidence_angle,
    )
    ionosphere_azimuth_axis = (
        ionosphere_azimuth_space_axis  # [m].
        / satellite_ground_speed  # [s/m].
        * ((ionosphere_altitude + earth_radius) / earth_radius)
    )

    return (
        ionosphere_azimuth_axis,
        ionosphere_slant_range_axis,
    )


def los_range_ground_to_altitude(
    los_ground_ranges: npt.NDArray[float],
    *,
    incidence_angle: float,
) -> float:
    """Convert the LOS range from ground to altitude."""
    return los_ground_ranges * np.cos(incidence_angle)


def nanmean(values: npt.ArrayLike, **kwargs) -> npt.ArrayLike:
    """Wrapper around nanmean that returns 0 if all NaN's."""
    if np.all(np.isnan(values)):
        return np.zeros_like(np.mean(values, **kwargs))
    return np.nanmean(values, **kwargs)


def run_multi_squint_iono_correction(
    coherence_improvement: float,
    coherence_improvement_threshold: float,
    force_multi_squint_iono_correction: bool = False,
    disable_multi_squint_iono_correction: bool = False,
) -> bool:
    """
    Check whether the Multi-Squint ionosphere correction must be run or not.

    Parameters
    ----------
    coherence_improvement: float
        The improvement in cohernce with respect to the theoretical
        coherence.

    coherence_improvement_threshold: float
        The threshold on the improvement to activate or not the
        MS ionosphere correction.

    force_multi_squint_iono_correction: bool = False
        If true, always execute the MS coherence correction, unless
        disabled by user. It defaults to False.

    disable_multi_squint_iono_correction: bool = False
        If true, the MS ionosphere correction is never executed. It
        defaults to False.

    Returns
    -------
    bool
        Whether to run or not the MS ionosphere correction.

    """
    if disable_multi_squint_iono_correction:
        bps_logger.info("Skipping the ionosphere correction (disabled by user)")
        return False

    if force_multi_squint_iono_correction:
        bps_logger.info("Starting the ionosphere correction (forced by user)")
        return True

    if coherence_improvement > coherence_improvement_threshold:
        bps_logger.info(
            "Passed coherence improvement test (%.3f > %.3f). Starting the ionosphere correction",
            coherence_improvement,
            coherence_improvement_threshold,
        )
        return True

    bps_logger.info(
        "Failed coherence improvement test (%.3f < %.3f). Skipping the ionosphere correction",
        coherence_improvement,
        coherence_improvement_threshold,
    )
    return False
