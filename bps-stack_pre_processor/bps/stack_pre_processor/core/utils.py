# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities for the stack pre-processor
-------------------------------------
"""

from collections.abc import Callable
from functools import reduce

import numpy as np
import numpy.typing as npt
import scipy as sp
from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.io.metadata import EPolarization, ESideLooking, RasterInfo, StateVectors
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.roi_utils import RegionOfInterest, raise_if_roi_is_invalid
from bps.common.toi_utils import TimeOfInterest, toi_to_axis_slice


class StackPreProcessorRuntimeError(RuntimeError):
    """Handle an invalid set of polarizations in the stack."""


class StackPreProcessorWorkingDirError(RuntimeError):
    """Handle pre-processor's errors when operates in the working dir."""

    @staticmethod
    def on_rmtree_error(*args):
        """Error handler for shutil.rmtree."""
        raise StackPreProcessorWorkingDirError(args[1])


def compute_processing_footprint(
    primary_footprint: list[list[float]],
    primary_raster_info: RasterInfo,
    stack_roi: RegionOfInterest | None,
) -> list[list[float]]:
    """
    Compute the processing footprint.

    Parameters
    ----------
    primary_footprint: list[list[float]]  [deg]
        4D Polygon containing the footpring of the coregistration primary,
        expressed as a list of lat/lon pairs.

    primary_raster_info: RasterInfo
        Raster metadata of the coregistration primary image.

    stack_roi: RegionOfInterest | None
        The stack ROI, if any. If None is provided, the footprint of the
        coregistration primary image is returned.

    Returns
    -------
    list[list[float]]  [deg]
        The stack ROI, encoded as a 4D list of lat/lon pairs.

    """
    if stack_roi is None:
        return primary_footprint

    # Footprints in L1a products are defined as follows:
    #
    #  - footprint[0] <- Lat/Lon(lines_stop, samples_start)
    #  - footprint[1] <- Lat/Lon(lines_stop, samples_stop)
    #  - footprint[2] <- Lat/Lon(lines_start, samples_stop)
    #  - footprint[3] <- Lat/Lon(lines_start, samples_start)
    #
    # See bps.transcoder.sarproduct.dem_footprint_utils.compute_footprint_from_dem and
    # bps.l1_processor.post.interface.
    #
    lines_stop = primary_raster_info.lines_step * (primary_raster_info.lines - 1)
    samples_stop = primary_raster_info.samples_step * (primary_raster_info.samples - 1)

    footprint_lats = np.asarray(primary_footprint)[:, 0]
    footprint_lat_fn = sp.interpolate.RegularGridInterpolator(
        ((lines_stop, 0), (0, samples_stop)),
        ((footprint_lats[0], footprint_lats[1]), (footprint_lats[3], footprint_lats[2])),
        bounds_error=False,
        fill_value=None,
    )

    footprint_lons = np.asarray(primary_footprint)[:, 1]
    footprint_lon_fn = sp.interpolate.RegularGridInterpolator(
        ((lines_stop, 0), (0, samples_stop)),
        ((footprint_lons[0], footprint_lons[1]), (footprint_lons[3], footprint_lons[2])),
        bounds_error=False,
        fill_value=None,
    )

    # Query the ROI corners.
    roi_lines_start = stack_roi[0] * primary_raster_info.lines_step
    roi_lines_stop = (stack_roi[0] + stack_roi[2] - 1) * primary_raster_info.lines_step
    roi_samples_start = stack_roi[1] * primary_raster_info.samples_step
    roi_samples_stop = (stack_roi[1] + stack_roi[3] - 1) * primary_raster_info.samples_step

    footprint_lats = footprint_lat_fn(
        (
            (roi_lines_stop, roi_lines_stop, roi_lines_start, roi_lines_start),
            (roi_samples_start, roi_samples_stop, roi_samples_stop, roi_samples_start),
        )
    )
    footprint_lons = footprint_lon_fn(
        (
            (roi_lines_stop, roi_lines_stop, roi_lines_start, roi_lines_start),
            (roi_samples_start, roi_samples_stop, roi_samples_stop, roi_samples_start),
        )
    )

    return [[float(lat), float(lon)] for lat, lon in zip(footprint_lats, footprint_lons)]


def compute_processing_roi(
    raster_info: RasterInfo,
    toi: TimeOfInterest,
) -> RegionOfInterest | None:
    """
    Compute a ROI associated to a TOI (azimuth). The ROI is encoded as
    per arepytools convention: [azm_0, rng_0, azm_size, rng_size].

    Parameters
    ----------
    raster_info: RasterInfo
        The raster metadata (typically from coregistration primary).

    toi: TimeOfInterest [UTC]
        The selected TOI.

    Raises
    ------
    InvalidTimeOfInterestError, InvalidRegionOfInterestError

    Return
    ------
    Optional[RegionOfInterest]
        The corresponding ROI if smaller than the original image. None
        if the ROI coincides with the full extent of the image.

    """
    azm_begin, azm_end = toi_to_axis_slice(
        toi,
        time_axis=(raster_info.lines_start + np.arange(raster_info.lines) * raster_info.lines_step),
    )

    # If the ROI is the full image, just return None.
    if azm_begin == 0 and azm_end == raster_info.lines - 1:
        return None

    # Create a validate the ROI.
    roi = (
        azm_begin,
        0,
        azm_end - azm_begin + 1,
        raster_info.samples,
    )
    raise_if_roi_is_invalid(raster_info, roi)

    return roi


def common_polarizations(
    polarizations: tuple[tuple[EPolarization, ...], ...],
) -> tuple[EPolarization, ...]:
    """
    Compute polarizations that are common to all frames.

    Parameters
    ----------
    polarizations: tuple[tuple[EPolarization, ...], ...]
        List of lists of polarizations (e.g. list of polarizations
        in a set of frames).

    Return
    ------
    tuple[EPolarization, ...]
        The list of common polarizations.

    """
    return tuple(
        sorted(
            reduce(
                lambda x, y: set(x).intersection(set(y)),
                polarizations,
            ),
            key=lambda p: p.value,
        )
    )


def compute_interferometric_baselines(
    *,
    state_vectors_primary: StateVectors,
    state_vectors_secondary: StateVectors,
    azimuth_time_primary: PreciseDateTime,
    range_time_primary: float,
    look_direction: ESideLooking,
    geodetic_height: float = 0.0,
):
    """
    Compute interferometric baseline.

    Parameters
    ----------
    state_vectors_primary: StateVectors
        The state vector metadata of the primary stack image.

    state_vectors_secondary: StateVectors
        The state vector metadata of the secondary stack image.

    azimuth_time_primary: PreciseDataTime [UTC]
        The reference azimuth time.

    range_time_primary: float [s]
        The reference slant-range time.

    look_direction: ESideLooking
        The looking direction of the satellite (i.e. left or right).

    geodetic_height: float = 0.0 [m]
        The geodetic altitude used as reference altitude for geocoding.
        It defaults to 0 altitude.

    Raises
    ------
    AssertionError

    Return
    ------
    float [m]
        The normal baseline wrt satellite orbit.

    float [m]
        The parallel (LOS) baseline wrt satellite orbit.

    float [m]
        The along-track (AT) baseline wrt the satellite orbit.

    """
    gso_primary = create_general_sar_orbit(state_vectors_primary)
    gso_secondary = create_general_sar_orbit(state_vectors_secondary)

    # Direct geocoding with the primary orbit.
    targets_coords = gso_primary.sat2earth(azimuth_time_primary, range_time_primary, look_direction, geodetic_height)

    # Inverse geocoding with the slave orbit.
    azimuth_time_secondary, _ = gso_secondary.earth2sat(targets_coords[:, 0])
    sensor_secondary_coords = gso_secondary.get_position(azimuth_time_secondary)

    # Computation of the baseline vector.
    # The baseline_vector, los_versor and velocity_versor all has shape [3 x 1].
    sensor_primary_coords = gso_primary.get_position(azimuth_time_primary)
    baseline_vector = sensor_secondary_coords - sensor_primary_coords
    assert baseline_vector.shape == (3, 1), f"{baseline_vector.shape=} is not (3, 1)"

    # Projection of the baseline vector to the SAR directions.
    los_versor = sensor_primary_coords - targets_coords
    los_versor = los_versor / np.linalg.norm(los_versor, axis=0)
    assert los_versor.shape == (3, 1), f"{los_versor.shape=} is not (3, 1)"

    velocity_versor = gso_primary.get_velocity(azimuth_time_primary)
    velocity_versor = velocity_versor / np.linalg.norm(velocity_versor)
    assert velocity_versor.shape == (3, 1), f"{velocity_versor.shape=} is not (3, 1)"

    norm_versor = np.cross(los_versor, velocity_versor, axisa=0, axisb=0).T
    assert norm_versor.shape == (3, 1), f"{norm_versor.shape=} is not (3, 1)"

    return (
        float(baseline_vector.T @ norm_versor),
        float(baseline_vector.T @ los_versor),
        float(baseline_vector.T @ velocity_versor),
    )


def sort_from_pivot(
    values: npt.ArrayLike,
    *,
    pivot_fn: Callable[[npt.ArrayLike], float] = np.median,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Sort values wrt to distance to a pivot value.

    Parameters
    ----------
    values: npt.ArrayLike [any]
        The values that need be sorted.

    pivot_fn: Callable[[npt.ArrayLike], float] = np.median
        Callable that computes the pivot value. Defaulted to median.

    Return
    ------
    tuple[int, ...]
        The permutation that sorts the values in increasing order,
        that is, for all i>0

           |v[i] - pivot| <= |v[i+1] - pivot|,

    """
    return np.argsort(np.abs(np.asarray(values) - pivot_fn(values)))


def compute_faraday_index(fr_phases: npt.NDArray[float]) -> float:
    r"""
    Compute the Faraday Rotation (FR) index. The FR decorrelation
    index is defined as

       \gamma{FR}: = 1 / sqrt(2 * \sigma{FR}^2 + 1),

    with \sigma{FR} the standard deviation of the Faraday Rotation
    phase screen.

    Parameters
    ----------
    fr_phases: npt.NDArray[float] [rad]
        The faraday rotation data.

    Return
    ------
    float
        The FR index (between 0 and 1).

    """
    return 1 / np.sqrt(2 * np.var(fr_phases) + 1)


def compute_rfi_indices(rfi_mask: npt.NDArray[bool]) -> float:
    r"""
    Compute RFI coherence degradation index of the given RFI binary mask.

    The index is defined from an [N x M] binary mask RFI

        \gamma{RFI} := (1 - \sum(RFI) / (N*M))^2

    Parameters
    ----------
    rfi_mask_data: npt.NDArray[float]
        The binary mask.

    Return
    ------
    float
        The RFI index (between 0 and 1).

    """
    return (1 - np.sum(rfi_mask) / rfi_mask.size) ** 2
