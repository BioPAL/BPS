# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Some Geocoding Utilities
------------------------
"""

import numpy as np
import numpy.typing as npt
import scipy as sp
from arepytools.geometry.conversions import llh2xyz
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.stack_pre_processor.core.utils import StackPreProcessorRuntimeError


def compute_ecef_dem(
    lut_data: dict[str, npt.NDArray[float]],
    lut_axes: tuple[dict[str, npt.NDArray[PreciseDateTime]], dict[str, npt.NDArray[float]]],
    l1a_data_axes: tuple[npt.NDArray[PreciseDateTime], npt.NDArray[float]],
    *,
    degree_x: int = 1,
    degree_y: int = 1,
    smoother: float = 0.0,
) -> tuple[npt.NDArray[float], ...]:
    """
    Compute the DEM by upsampling LLh coordinates.

    Parameters
    ----------
    lut_data: dict[str, npt.NDArray[float]]
        The LUT data. It must contain 'latitude', 'longitude', and
        'height'.

    lut_axes: tuple[dict[str, npt.NDArray[PreciseDateTime]], dict[str, npt.NDArray[float]]]
        THe LUT absolute axes, ordered as azimuth [UTC} and range [s].

    l1a_data_axes: tuple[npt.NDarray[PreciseDateTime], npt.NDArray[float]] [UTC], [s]
        The output axes of the DEM, that is, the axes of the data.

    degree_x: int = 1
        Degree of the bivariate spline in the first component.

    degree_y: int = 1
        Degree of the bivariate spline in the second component.

    smoother: float = 0.0
        A smoothing factor.

    Raises
    ------
    ValueError if necessary LUTs are missing.

    Return
    ------
    npt.NDArray[float] [m]
        The DEM X coordinates on the grid of the L1a data.

    npt.NDArray[float] [m]
        The DEM Y coordinates on the grid of the L1a data.

    npt.NDArray[float] [m]
        The DEM Z coordinates on the grid of the L1a data.

    """
    if any(lut not in lut_data for lut in ("latitude", "longitude", "height")):
        raise StackPreProcessorRuntimeError("Lat/lon/height are missing from LUT data")
    if any(lut not in lut_axes[0] for lut in ("latitude", "longitude", "height")):
        raise StackPreProcessorRuntimeError("Lat/lon/height are missing from LUT azm axes")
    if any(lut not in lut_axes[1] for lut in ("latitude", "longitude", "height")):
        raise StackPreProcessorRuntimeError("Lat/lon/height are missing from LUT rng axes")

    # NOTE: We first convert LLh to ECEF and then upsample each coordinates. No
    # noticeable difference wrt upsample LLh first and then convert to LLh.

    dem = []
    for coord, dem_c in zip(
        ("latitude", "longitude", "height"),
        llh_to_ecef(
            latitudes=np.deg2rad(lut_data["latitude"]),
            longitudes=np.deg2rad(lut_data["longitude"]),
            heights=lut_data["height"],
        ),
    ):
        lut_axis_azm = lut_axes[0][coord]
        lut_axis_rng = lut_axes[1][coord]

        # The LUT axes relative to the grid data axes
        lut_axis_azm = (lut_axis_azm - l1a_data_axes[0][0]).astype(np.float64)
        lut_axis_rng = lut_axis_rng - l1a_data_axes[1][0]

        # The relative grid data axes.
        l1a_data_axis_azm = (l1a_data_axes[0] - l1a_data_axes[0][0]).astype(np.float64)
        l1a_data_axis_rng = l1a_data_axes[1] - l1a_data_axes[1][0]

        # The spline bbox for extrapolation.
        extrapolation_bbox = [
            min(*lut_axis_azm, *l1a_data_axis_azm),
            max(*lut_axis_azm, *l1a_data_axis_azm),
            min(*lut_axis_rng, *l1a_data_axis_rng),
            max(*lut_axis_rng, *l1a_data_axis_rng),
        ]

        dem.append(
            sp.interpolate.RectBivariateSpline(
                lut_axis_azm,
                lut_axis_rng,
                dem_c,
                bbox=extrapolation_bbox,
                kx=degree_x,
                ky=degree_y,
                s=smoother,
            )(l1a_data_axis_azm, l1a_data_axis_rng)
        )

    return tuple(dem)


def llh_to_ecef(
    latitudes: npt.NDArray[float],
    longitudes: npt.NDArray[float],
    heights: npt.NDArray[float],
) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    """
    Convert the LLh (GCS) to ECEF (xyz).

    Parameters
    ----------
    latitudes: npt.NDArray[float] [rad]
        The latitude values.

    longitudes: npt.NDArray[float] [rad]
        The longitude values.

    heights: npt.NDArray[float] [rad]
        The height values.

    Raises
    ------
    StackPreProcessorRuntimeError
        In case dimensios mismatches.

    Return
    ------
    npt.NDArray[float] [m]
        The X values.

    npt.NDArray[float] [m]
        The Y values.

    npt.NDArray[float] [m]
        The Z values.

    """
    if not latitudes.shape == longitudes.shape == heights.shape:
        raise StackPreProcessorRuntimeError("Lat/lon/height data have mismatching dimensions")
    data_shape = latitudes.shape

    packed_xyz = llh2xyz([latitudes.reshape(-1), longitudes.reshape(-1), heights.reshape(-1)])
    return (
        packed_xyz[0].reshape(*data_shape),
        packed_xyz[1].reshape(*data_shape),
        packed_xyz[2].reshape(*data_shape),
    )
