# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Interpolation Utilities
-----------------------
"""

import numpy as np
import numpy.typing as npt
import scipy as sp


def interpolate_points_on_grid(
    grid_values: npt.NDArray,
    axes_in: tuple[npt.NDArray[float], npt.NDArray[float]],
    query_points: tuple[npt.NDArray[float], npt.NDArray[float]],
    *,
    query_points_on_grid: bool = False,
    fill_value: float = None,
) -> npt.NDArray:
    """
    Interpolate the grid values over a new grid.

    The implementation provides an API to specify whether the
    query points are on a regular grid. in such a case, it uses
    scipy.interpolate.RectBivariateSpline (as the calibrator does),
    since RegularGridInterpolator is not optimized for query points
    on a regular grids (hence much slower than the rect-bivariate-spline).

    Parameters
    ---------
    grid_values: npt.NDArray
        An [N x M] matrix.

    axes_in: tuple[npt.NDArray[float], npt.NDArray[float]]
        The two 1D input axes, respectively of size N and M.

    query_points: tuple[npt.NDArray[float], npt.NDArray[float]]
        The query points expressed either as x/y coordinates
        or as axes (see `query_points_on_grid`).

    query_points_on_grid: bool = True
        If true, query_points are treated as axis coordinates, i.e.,
        (x[i], y[j]). If false, as points (x[i], y[i]).

    fill_value: float | None = None
        Value used for extrapolation if `query_points_on_grid` is False.
        It defaults to None (i.e. extrapolation).

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

    # If points are on a grid, RectBivariateSpline seems to be quite efficient.
    if query_points_on_grid:
        interp_fn = sp.interpolate.RectBivariateSpline(
            axes_in[0],
            axes_in[1],
            grid_values,
            bbox=[
                min(*axes_in[0], *query_points[0]),
                max(*axes_in[0], *query_points[0]),
                min(*axes_in[1], *query_points[1]),
                max(*axes_in[1], *query_points[1]),
            ],
            kx=1,
            ky=1,
            s=0,
        )
        return interp_fn(query_points[0], query_points[1])

    if query_points[0].shape != query_points[1].shape:
        raise ValueError("X/Y components of query points have mismatching dimensions")

    # Otherwise, RegularGridInterpolator is faster.
    # NOTE: RegularGridInterpolator requires the query points to be packed
    # into a [N x 2] matrix.
    interp_fn = sp.interpolate.RegularGridInterpolator(
        axes_in,
        grid_values,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )
    return interp_fn(
        np.hstack([query_points[0].reshape(-1, 1), query_points[1].reshape(-1, 1)]),
    ).reshape(query_points[0].shape)
