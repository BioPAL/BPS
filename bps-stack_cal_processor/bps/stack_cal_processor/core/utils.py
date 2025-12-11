# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utility Library
---------------
"""

import enum
from numbers import Number
from typing import Any

import numba as nb
import numpy as np
import numpy.typing as npt
import scipy as sp
from arepytools.geometry.conversions import llh2xyz, xyz2llh
from arepytools.geometry.generalsarorbit import (
    GeneralSarOrbit,
    compute_incidence_angles_from_orbit,
    compute_look_angles_from_orbit,
)
from arepytools.io import iter_channels, read_metadata, read_raster_with_raster_info
from arepytools.io.metadata import (
    DataSetInfo,
    EPolarization,
    RasterInfo,
    SamplingConstants,
)
from arepytools.io.productfolder2 import ProductFolder2
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.roi_utils import RegionOfInterest, raise_if_roi_is_invalid
from scipy import constants


class SubLookFilterType(enum.Enum):
    """The supported filtering for computing sub-looks."""

    FIR = "FIR"
    GAUSSIAN = "GAUSSIAN"


def critical_vertical_wavenumber(range_bandwidth: float) -> float:
    """
    Compute the critical wavenumber value.

    Parameters
    ----------
    range_bandwidth: float [Hz]
        Bandwidth in slant-range direction.

    Raises
    ------
    ValueError

    Return
    ------
    float [rad/m]
        The critical wavenumber.

    """
    if range_bandwidth <= 0:
        raise ValueError("Range bandwidth must be positive")

    range_resolution = constants.speed_of_light / (2 * range_bandwidth)
    return 2 * np.pi / range_resolution


def positive_part(values: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    Take the positive part of an array.

    Given a real valued array `x`, return an array x+ so that

       x+{i} = x{i} if x{i} > 0, 0 otherwise.

    Parameters
    ----------
    values: npt.NDArray[float]
        A real valued array.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[float]
        The positive part of the input.

    """
    return np.clip(values, 0.0, np.inf)


def get_time_axis(
    raster_info: RasterInfo,
    *,
    axis: int,
    roi: RegionOfInterest | None = None,
    absolute: bool = True,
    dtype: np.dtype = np.float64,
) -> tuple[npt.NDArray, float | PreciseDateTime]:
    """
    Compute azimuth time axis and central time.

    Parameters
    ----------
    raster_info: RasterInfo
        The raster specs.

    axis: int
        The axis (0: azimuth or 1: range)

    roi: Optional[RegionOfInterest] = None
        The region of interests. If None is provided, the
        all raster is used.

    absolute: bool = True
        As to whether the absolute or relative time axis
        must be exported.

    dtype: np.dtype = np.float64
        The precision used to compute the relative time axis.
        It defaults to np.float64.

    Raises
    -----
    InvalidRoiError, ValueError

    Return
    ------
    npt.NDArray [s] or [UTC]
        The time axis.

    Union[float, PreciseDateTime] [s] or [UTC]
        The central time.

    """
    # Select the optional ROI.
    roi = roi or [0, 0, raster_info.lines, raster_info.samples]
    raise_if_roi_is_invalid(raster_info, roi)

    if axis == 0:
        time_step = raster_info.lines_step
        time_start = raster_info.lines_start
    elif axis == 1:
        time_step = raster_info.samples_step
        time_start = raster_info.samples_start
    else:
        raise ValueError("axis must be either 0 (azimuth) or 1 (range)")

    time_axis = np.arange(roi[axis + 2], dtype=dtype) * time_step
    time_mid = (time_axis.size - 1) / 2 * time_step
    if absolute:
        time_offset = roi[axis] * time_step
        time_axis = time_start + time_offset + time_axis
        time_mid = time_start + time_offset + time_mid
    return time_axis, time_mid


def block_slice(
    data: npt.NDArray,
    block: tuple[int, int],
    *,
    axis: int,
) -> npt.NDArray:
    """
    Extract a block slice from data.

    Parameters
    ---------
    data: npt.NDArray
        Any 2D data.

    block: tuple[int, int]
        Begin/end of the slice.

    axis: int
        The axis (0=azimuth, 1=range)

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray
        The sliced data.

    """
    if data.ndim != 2:
        raise ValueError("Slicing requires as input a 2D array")
    if axis == 0:
        return data[block[0] : block[1], :]
    if axis == 1:
        return data[:, block[0] : block[1]]
    raise ValueError("Only azimuth (0) or range (1) axes are supported")


def compute_satellite_ground_speed(
    sar_orbit: GeneralSarOrbit,
    raster_info: RasterInfo,
    dataset_info: DataSetInfo,
    roi: RegionOfInterest | None = None,
) -> float:
    """
    Compute satellite velocity on the ground (down-projected).

    Parameters
    ----------
    sar_orbit: GeneralSarOrbit
        The information on the general SAR orbit.

    raster_info: RasterInfo
        The raster metadata info.

    dataset_info: DataSetInfo
        The dataset metadata info.

    roi: Optional[RegionOfInterest] = None
        Optionally, a region of interest.

    Return
    ------
    float [m/s]
        The satellite velocity.

    """
    if roi is not None:
        raise_if_roi_is_invalid(raster_info, roi)

    _, azimuth_central_time = get_time_axis(raster_info, roi=roi, axis=0, absolute=True)
    _, range_central_time = get_time_axis(raster_info, roi=roi, axis=1, absolute=True)

    sat_unit_vel = sar_orbit.get_velocity(azimuth_central_time)
    sat_unit_vel /= np.linalg.norm(sat_unit_vel)

    sat_proj_ecef = sar_orbit.sat2earth(azimuth_central_time, range_central_time, dataset_info.side_looking.value)
    sat_forward_ecef = sar_orbit.sat2earth(
        azimuth_central_time + 1, range_central_time, dataset_info.side_looking.value
    )

    return float((sat_forward_ecef - sat_proj_ecef).T @ sat_unit_vel)


def compute_satellite_state(
    sar_orbit: GeneralSarOrbit,
    raster_info: RasterInfo,
    dataset_info: DataSetInfo,
    roi: RegionOfInterest | None = None,
) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    """
    Compute satellite state (position and velocity) at the center of the scene.

    Parameters
    ----------
    sar_orbit: GeneralSarOrbit
        The information on the general SAR orbit.

    raster_info: RasterInfo
        The raster metadata info.

    dataset_info: DataSetInfo
        The dataset metadata info.

    roi: Optional[RegionOfInterest] = None
        Optionally, a region of interest.

    Return
    ------
    npt.NDArray[float] [m]
        The satellite ECEF position.

    npt.NDArray[float] [m/s]
        The satellite ECEF velocity.

    npt.NDArray[float] [m]
        The satellite ECEF position on the ground.

    """
    if roi is not None:
        raise_if_roi_is_invalid(raster_info, roi)
    _, azimuth_central_time = get_time_axis(raster_info, roi=roi, axis=0, absolute=True)
    _, range_central_time = get_time_axis(raster_info, roi=roi, axis=1, absolute=True)
    return (
        sar_orbit.get_position(azimuth_central_time),
        sar_orbit.get_velocity(azimuth_central_time),
        sar_orbit.sat2earth(
            azimuth_central_time,
            range_central_time,
            dataset_info.side_looking.value,
        ),
    )


def compute_satellite_altitude(sat_position: npt.NDArray[float]) -> float:
    """Return the altitude of the satellite wrt the WGS84 geoid in [m]."""
    return float(xyz2llh(sat_position)[2])


def compute_target_ground_speed(
    sat_position: npt.NDArray[float],
    sat_velocity: npt.NDArray[float],
    tgt_position: npt.NDArray[float],
) -> float:
    """Compute the velocity of the target point on the ground and satellite [m/s]."""
    earth_radius = compute_earth_radius(sat_position)
    cos_gamma = (tgt_position.T @ sat_position) / (np.linalg.norm(tgt_position) * np.linalg.norm(sat_position))
    return float(
        np.linalg.norm(sat_velocity)
        * earth_radius
        * cos_gamma
        / (earth_radius + compute_satellite_altitude(sat_position))
    )


def compute_earth_radius(xyz: npt.NDArray[float]) -> float:
    """Compute the Earth's radius give a specific direction (in ECEF) [m]."""
    llh = xyz2llh(xyz)
    llh[2] = 0
    return np.linalg.norm(llh2xyz(llh))


def compute_incidence_angle_from_orbit(
    sar_orbit: GeneralSarOrbit,
    raster_info: RasterInfo,
    dataset_info: DataSetInfo,
) -> float:
    """Compute the incidence angle from orbit metadata [rad]."""
    return compute_incidence_angles_from_orbit(
        orbit=sar_orbit,
        azimuth_time=raster_info.lines_start + (raster_info.lines - 1) * raster_info.lines_step / 2,
        range_times=raster_info.samples_start + (raster_info.samples - 1) * raster_info.samples_step / 2,
        look_direction=dataset_info.side_looking.value,
    )


def compute_look_angle_from_orbit(
    raster_info: RasterInfo,
    sar_orbit: GeneralSarOrbit,
    dataset_info: DataSetInfo,
    roi: RegionOfInterest | None = None,
) -> float:
    """Compute the satellite's look angle from orbit metadata [rad]."""
    if roi is not None:
        raise_if_roi_is_invalid(raster_info, roi)
    _, azimuth_central_time = get_time_axis(raster_info, roi=roi, axis=0, absolute=True)
    _, range_central_time = get_time_axis(raster_info, roi=roi, axis=1, absolute=True)
    return compute_look_angles_from_orbit(
        sar_orbit,
        azimuth_central_time,
        range_central_time,
        dataset_info.side_looking.value,
    )


def compute_absolute_distances(
    ecef_tgt_positions_x: npt.NDArray[float],
    ecef_tgt_positions_y: npt.NDArray[float],
    ecef_tgt_positions_z: npt.NDArray[float],
    ecef_ref_position: npt.NDArray[float],
    *,
    dtype: np.dtype = np.float64,
) -> npt.NDArray[float]:
    """
    Compute the (absolute) distances between the a reference position and
    a set of points.

    Parameters
    ----------
    ecef_tgt_positions_x: npt.NDArray[float] [m]
        The ECEF X-coordinates of the target points.

    ecef_tgt_positions_y: npt.NDArray[float] [m]
        The ECEF y-coordiates of the target points.

    ecef_tgt_positions_z: npt.NDArray[float] [m]
        The ECEF z-coordinates of the target points.

    ecef_position_ref: npt.NDArray[float] [m]
        The ECEF reference position (3D).

    dtype: np.dtype = np.float64
        The desired output type. It defaults to np.float64.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[float] [m]
        The ECEF distances between reference and targets.

    """
    # Check on input.
    if ecef_tgt_positions_x.size == 0:
        raise ValueError("No target points to compute distances from")
    if ecef_tgt_positions_x.shape != ecef_tgt_positions_y.shape:
        raise ValueError("X and Y components of target points have mismatching shape")
    if ecef_tgt_positions_x.shape != ecef_tgt_positions_z.shape:
        raise ValueError("X and Z components of target points have mismatching shape")
    if ecef_ref_position.size != 3:
        raise ValueError("Reference position must be a 3D ECEF position")
    return np.sqrt(
        (ecef_ref_position[0] - ecef_tgt_positions_x) ** 2
        + (ecef_ref_position[1] - ecef_tgt_positions_y) ** 2
        + (ecef_ref_position[2] - ecef_tgt_positions_z) ** 2,
        dtype=dtype,
    )


def percentage_completed_msg(partial: float, *, total: float) -> int:
    """
    Return the approximate percentage.

    Parameters
    ----------
    partial: float
        The current value.

    total: float
        The total value (i.e. 100%).

    Raises
    -----
    ValueError

    Return
    ------
    str
        A message.

    """
    if not 0 <= partial <= total:
        raise ValueError(f"{partial=} must be between 0 and {total=}")
    return "Completed {:.1f} %".format(100 * partial / total)


def percentage_valid(mask: npt.NDArray[bool]) -> float:
    """The percentage of 1 in a boolean mask (in 0%-100%)."""
    return np.sum(mask) / mask.size * 100.0


def compute_spatial_azimuth_shifts(
    azimuth_shifts: npt.NDArray[float],
    azimuth_sampling_step: float,
    *,
    ground_speed: float,
    coreg_primary_azimuth_shifts: npt.NDArray[float] | None = None,
) -> npt.NDArray[float]:
    """
    Convert the azimuth shifts (in pixel) to spatial shifts.

    Parameters
    ----------
    azimuth_shifts: npt.NDArray[float] [px]
        Azimuth shifts in pixel.

    azimuth_sampling_step: float [s]
        The azimuth sampling step.

    ground_speed: float [m/s]
        The ground speed at the azimuth pixel position.

    coreg_primary_azimuth_shifts: Optional[npt.ndarray] [samples]
        Optionally, the azimuth shifts of the coregistration primary.

    Return
    ------
    npt.NDArray[float] [m]
        The azimuth shifts in meters.

    """
    return (
        (azimuth_shifts - _value_or(coreg_primary_azimuth_shifts, default_value=0))
        * ground_speed
        * azimuth_sampling_step
    )


def compute_spatial_range_shifts(
    range_shifts: npt.NDArray[float],
    range_sampling_step: float,
    *,
    incidence_angle: float = np.pi / 2.0,
    coreg_primary_range_shifts: npt.NDArray[float] | None = None,
) -> npt.NDArray[float]:
    """
    Convert the shifts (in pixel) to spatial shifts in range. If no incidence
    angles are passed, the shifts are computed in slant-range. If an
    incidence angle is passed, the shifts are computed in ground-range.

    Parameters
    ----------
    range_shifts: npt.NDArray[float] [px]
        Range shifts in pixel.

    range_sampling_step: float [s]
        The range sampling step.

    incidence_angle: float [rad] = pi / 2
        The incidence angle at the center of the scene. It defaults to
        pi/2, i.e., spatial shifts in slant-range.

    coreg_primary_range_shifts: Optional[npt.NDArray[float]] [px]
        Optionally, the range shifts of the coregistration primary.

    Return
    ------
    npt.NDArray[float] [m]
        The range shifts in meters.

    """
    return (
        0.5
        * (range_shifts - _value_or(coreg_primary_range_shifts, default_value=0))
        * range_sampling_step
        * constants.speed_of_light
        / np.sin(incidence_angle)  # Account for the slanted LOS direction wrt geoid.
    )


def interpolate_on_grid(
    grid_values: npt.NDArray,
    *,
    axes_in: tuple[npt.NDArray[float], npt.NDArray[float]],
    axes_out: tuple[npt.NDArray[float], npt.NDArray[float]],
    phase_interpolation: bool = False,
    degree_x: int = 1,
    degree_y: int = 1,
    smoother: float = 0.0,
) -> npt.NDArray:
    """
    Interpolate the grid values over a new grid.

    This method wraps scipy's RectBivariateSpline. It allows for
    standard interpolation (e.g. DSI or other unwrapped phases)
    or to convert the operation to an interpolation of phasors, which
    better copes with wrapped phases.

    Parameters
    ---------
    grid_values: npt.NDArray
        An [N x M] matrix.

    axes_in: tuple[npt.NDArray[float], npt.NDArray[float]]
        The two 1D input axes, respectively of size N and M.

    axes_out: tuple[npt.NDArray[float], npt.NDArray[float]]
        The two 1D output axes.

    phase_interpolation: bool
        As to whether we're interpolating a wrapped phase screen. It
        assumes the input grid to be phase angles [rad].

    degree_x: int = 1
        Degree of the bivariate spline in the first component.

    degree_y: int = 1
        Degree of the bivariate spline in the second component.

    smoother: float = 0.0
        A smoothing factor.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray
        The input values interpolated onto the output axes.

    """
    if axes_in[0].size != grid_values.shape[0]:
        raise ValueError(
            "x-axis and values have mismatching dimensions ({} vs {})".format(
                axes_in[0].size,
                grid_values.shape,
            )
        )
    if axes_in[1].size != grid_values.shape[1]:
        raise ValueError(
            "y-axis and values have mismatching dimensions ({} vs {})".format(
                axes_in[1].size,
                grid_values.shape,
            )
        )
    if phase_interpolation and not np.isrealobj(grid_values):
        raise ValueError("Phase interpolation requires phases (real) in input")

    if phase_interpolation:
        # NOTE: np.arctan2 is faster than np.angle(np.exp(1j * ...)).
        return np.arctan2(
            _interpolate_2d(
                grid_values=np.sin(grid_values),
                axes_in=axes_in,
                axes_out=axes_out,
                degree_x=degree_x,
                degree_y=degree_y,
                smoother=smoother,
            ),
            _interpolate_2d(
                grid_values=np.cos(grid_values),
                axes_in=axes_in,
                axes_out=axes_out,
                degree_x=degree_x,
                degree_y=degree_y,
                smoother=smoother,
            ),
        )

    return _interpolate_2d(
        grid_values=grid_values,
        axes_in=axes_in,
        axes_out=axes_out,
        degree_x=degree_x,
        degree_y=degree_y,
        smoother=smoother,
    )


def interpolate_on_grid_nn(
    grid_values: npt.NDArray,
    *,
    axes_in: tuple[npt.NDArray[float], npt.NDArray[float]],
    axes_out: tuple[npt.NDArray[float], npt.NDArray[float]],
) -> npt.NDArray:
    """
    Fast nearest neighbor interpolation on grids.

    This interpolator decouple the 2D NN-interpolation problem on
    a grid as two independent 1D NN-interpolation on the axes.

    Parameters
    ----------
    grid_values: npt.NDArray
        An [N x M] matrix.

    axes_in: tuple[npt.NDArray[float], npt.NDArray[float]]
        The two 1D input axes, respectively of size N and M.

    axes_out: tuple[npt.NDArray[float], npt.NDArray[float]]
        The two 1D output axes.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray
        The input values interpolated onto the output axes.

    """
    if axes_in[0].size != grid_values.shape[0]:
        raise ValueError(
            "x-axis and values have mismatching dimensions ({} vs {})".format(
                axes_in[0].size,
                grid_values.shape,
            )
        )
    if axes_in[1].size != grid_values.shape[1]:
        raise ValueError(
            "y-axis and values have mismatching dimensions ({} vs {})".format(
                axes_in[1].size,
                grid_values.shape,
            )
        )

    # NOTE: NN interpolation on a grid can be computed by snapping the target
    # grid onto the reference grid, thus using two 1D interpolations. This is
    # way faster then NN saerches (as in sp.interpolate.NearestNDInterpolator).
    indices_out = tuple(
        _interp1d_nn(
            ax_in.astype(np.float64),
            np.arange(ax_in.size, dtype=np.float64),
            ax_out.astype(np.float64),
        ).astype(np.int32)
        for ax_in, ax_out in zip(axes_in, axes_out)
    )
    return grid_values[indices_out[0]][:, indices_out[1]]


def _interp1d_nn(
    xc: npt.NDArray[float],
    yc: npt.NDArray[float],
    xq: npt.NDArray[float],
) -> npt.NDArray[float]:
    """
    The Nearest-Neighbor interpolation. This implementation is equivalent
    to that of scipy.interpolate.interp1d.

    Parameters
    ----------
    xc: npt.NDArray[float]
        The control points' x-values.

    yc: npt.NDArray[float]
        The control points' y-values.

    xq: npt.NDArray[float]
        The query points x-values.

    Returns
    -------
    npt.NDArray[float]
        The y-values of the query points.

    """
    # This is exactly how scipy implements it.
    xc_mid = (xc[1:] + xc[:-1]) / 2
    return yc[np.searchsorted(xc_mid, xq)]


@nb.njit(nogil=True, cache=True)
def query_grid_mask(
    *,
    mask: npt.NDArray,
    x_axis: npt.NDArray[float],
    y_axis: npt.NDArray[float],
    xs: npt.NDArray[float],
    ys: npt.NDArray[float],
    nodata_fill_value: Number,
    nodata_value: Number,
) -> npt.NDArray:
    """
    Query the cell of a discrete grid-mask.

    Parameters
    ----------
    mask: npt.NDArray
        The original [N x M] mask.

    x_axis: npt.NDArray[float]
        The N-dimensional axis along the rows (first index).

    y_axis: npt.NDArray[float]
        The M-dimensional axis along the cols (second index).

    xs: npt.NDArray[float]
        The query x-values (see convention above).

    ys: npt.NDArray[float]
        The query y-values (see convention above).

    nodata_fill_value: Number
        What value to use when encountering a no-data pixel.

    nodata_value: Number = np.nan
        The mask value that represents no-data.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray
        The mask at the query grid values.

    """
    if mask.ndim != 2 or xs.shape != ys.shape:
        raise ValueError(
            "Querying grid mask with inconsistent data points",
        )
    if (
        np.min(xs) < np.min(x_axis)
        or np.max(xs) > np.max(x_axis)
        or np.min(ys) < np.min(y_axis)
        or np.max(ys) > np.max(y_axis)
    ):
        raise ValueError("Query points are outside of the grid domain")

    # Query points packed, to speed up the loop.
    xs_packed = xs.flatten()
    x0 = x_axis[0]
    dx = x_axis[1] - x0

    ys_packed = ys.flatten()
    y0 = y_axis[0]
    dy = y_axis[1] - y0

    output = np.empty(xs.size, mask.dtype)
    # pylint: disable-next=not-an-iterable
    for k in nb.prange(xs.size):
        output[k] = mask[
            np.int32(np.round((xs_packed[k] - x0) / dx)),
            np.int32(np.round((ys_packed[k] - y0) / dy)),
        ]
    output[output == nodata_value] = nodata_fill_value

    return np.reshape(output, xs.shape)


def read_productfolder_data_by_polarization(
    product_folder: ProductFolder2 | None,
    *,
    polarization: EPolarization | None = None,
    roi: RegionOfInterest | None = None,
) -> npt.NDArray:
    """
    Possibly, read data from a product folder given a polarization.

    Parameters
    ----------
    product_folder: Optional[ProductFolder2]
        Optionally, the selected product folder.

    polarization: Optional[EPolarization] = None
        Optionally, a selected polarization. If None, the first available
        polarization is returned.

    roi: Optional[RegionOfInterest] = None
        Optionally, read only a ROI.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray
        The first data matching the provided polarization.

    """
    for ch_index, ch_meta in iter_channels(product_folder, polarization=polarization):
        return read_raster_with_raster_info(
            raster_file=product_folder.get_channel_data(ch_index),
            raster_info=ch_meta.get_raster_info(),
            block_to_read=roi,
        )
    raise ValueError(
        f"{product_folder.path} does not contain polarization {polarization.value}",
    )


def read_productfolder_data_by_channel(
    product_folder: ProductFolder2,
    *,
    channel: int | None = None,
    roi: RegionOfInterest | None = None,
) -> npt.NDArray:
    """
    Possibly, read data from a product folder given channel index.

    Parameters
    ----------
    product_folder: Optional[ProductFolder2]
        Optionally, the selected product folder.

    polarization: Optional[EPolarization] = None
        Optionally, a selected polarization. If None, the first available
        polarization is returned.

    roi: Optional[RegionOfInterest] = None
        Optionally, read only a ROI.

    Return
    ------
    npt.NDArray
        The first data matching the provided polarization.

    """
    channel_metadata = read_metadata(product_folder.get_channel_metadata(channel))
    return read_raster_with_raster_info(
        raster_file=product_folder.get_channel_data(channel),
        raster_info=channel_metadata.get_raster_info(),
        block_to_read=roi,
    )


def read_productfolder_data(
    product_folder: ProductFolder2,
    *,
    roi: RegionOfInterest | None = None,
) -> npt.NDArray:
    """
    Read a single-data product folder.

    Parameters
    ----------
    product_folder: ProductFolder2
        The product folder.

    roi: Optional[RegionOfInterest] = None
        Optionally, read only a ROI.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray
        The data.

    """
    for ch_index, ch_meta in iter_channels(product_folder):
        return read_raster_with_raster_info(
            raster_file=product_folder.get_channel_data(ch_index),
            raster_info=ch_meta.get_raster_info(),
            block_to_read=roi,
        )
    raise ValueError(f"Attempted reading empty product folder {product_folder.path}")


def read_raster_info(product_folder: ProductFolder2) -> RasterInfo:
    """
    Read the raster info of a single-data product folder (or when
    the channel is irrelevant).

    Parameters
    ----------
    product_folder: ProductFolder2
        The product folder.

    Raises
    ------
    ValueError

    Return
    ------
    RasterInfo
        The raster info object.

    """
    for _, ch_meta in iter_channels(product_folder):
        return ch_meta.get_raster_info()
    raise ValueError(f"Attempted reading empty product folder {product_folder.path}")


def read_sampling_constants(product_folder: ProductFolder2) -> SamplingConstants:
    """
    Read the sampling constants of a single-data product folder (or
    when the channel is irrelevant).

    Parameters
    ----------
    product_folder: ProductFolder2
        The product folder.

    Raises
    ------
    ValueError

    Return
    ------
    SamplingConstants
        The sampling constants info object.

    """
    for _, ch_meta in iter_channels(product_folder):
        return ch_meta.get_sampling_constants()
    raise ValueError(f"Attempted reading empty product folder {product_folder.path}")


def _interpolate_2d(
    *,
    grid_values: npt.NDArray,
    axes_in: tuple[npt.NDArray[float], npt.NDArray[float]],
    axes_out: tuple[npt.NDArray[float], npt.NDArray[float]],
    degree_x: int = 1,
    degree_y: int = 1,
    smoother: float = 0.0,
) -> npt.NDArray:
    """Helper method to apply a standard bivariate spline interpolation."""
    return sp.interpolate.RectBivariateSpline(
        axes_in[0],
        axes_in[1],
        grid_values,
        bbox=[
            min(np.min(axes_in[0]), np.min(axes_out[0])),
            max(np.max(axes_in[0]), np.max(axes_out[0])),
            max(np.min(axes_in[1]), np.min(axes_out[1])),
            max(np.max(axes_in[1]), np.max(axes_out[1])),
        ],
        kx=degree_x,
        ky=degree_y,
        s=smoother,
    )(axes_out[0], axes_out[1])


def _value_or(arg: Any | None, *, default_value: Any) -> Any:
    """Return `x` if `x` is not None, `default_value` otherwise."""
    return arg if arg is not None else default_value
