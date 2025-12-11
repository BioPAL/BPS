# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
LUT Shifting Utilities
----------------------
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from arepytools.io.metadata import RasterInfo
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.roi_utils import RegionOfInterest, raise_if_roi_is_invalid
from bps.stack_coreg_processor.utils import StackCoregProcessorRuntimeError


def coreg_primary_lut_axes(
    raster_info: RasterInfo,
    *,
    target_azimuth_step: float,
    target_range_step: float,
    keep_end_times: bool = False,
    roi: RegionOfInterest | None = None,
) -> tuple[
    npt.NDArray[PreciseDateTime],
    npt.NDArray[float],
    npt.NDArray[int],
    npt.NDArray[int],
]:
    """
    Define the primary LUT grid.

    The output grid will have:
    * Same start time of the raster,
    * 1 out of N elements via subsampling (no interpolation),
    * Optionally, the end times.

    Parameters
    ----------
    raster_info: RasterInfo
        The raster of the data grid of the coreg primary image.

    target_azimuth_step: float [s]
        The target azimuth step/resolution for the output grid.

    target_range_stop: float [s]
        The target range step/resolution for the output grid.

    keep_end_times: bool
        If true, the output axes will also

    roi: Optional[RegionOfInterest] = None
        Optionally, restrict everything to a ROI.

    Raises
    ------
    StackCoregProcessorRuntimeError
        In case the decimation factors are invalid

    Return
    ------
    npt.NDArray[PreciseDateTime] [UTC]
        The azimuth absolute axis of the LUT.

    npt.NDArray[float] [s]
        The range relative axis of the LUT.

    npt.NDArray[int]
        The subsampling azimuth indices wrt the full axis.

    npt.NDArray[int]
        The subsampling range indices wrt the full axis.

    """
    if target_azimuth_step <= 0:
        raise StackCoregProcessorRuntimeError(
            f"LUT azimuth resolution (time) must be positive got [s] {target_azimuth_step}"
        )
    if target_range_step <= 0:
        raise StackCoregProcessorRuntimeError(
            f"LUT range resolution (time) must be positive got [s] {target_range_step}"
        )

    # If ROI is None, use the full extent of the frame.
    if roi is None:
        roi = [0, 0, raster_info.lines, raster_info.samples]
    raise_if_roi_is_invalid(raster_info, roi)

    # The original time axes.
    azm_axis = np.arange(0, raster_info.lines) * raster_info.lines_step
    rng_axis = np.arange(0, raster_info.samples) * raster_info.samples_step
    azm_axis = azm_axis[roi[0] : roi[0] + roi[2]]
    rng_axis = rng_axis[roi[1] : roi[1] + roi[3]]

    # Compute the decimation factor.
    azimuth_decimation_factor = max(round(target_azimuth_step / raster_info.lines_step), 1)
    range_decimation_factor = max(round(target_range_step / raster_info.samples_step), 1)

    # Subsampled indices.
    azm_indices = np.arange(0, azm_axis.size, azimuth_decimation_factor)
    rng_indices = np.arange(0, rng_axis.size, range_decimation_factor)

    if keep_end_times:
        if azm_indices[-1] != azm_axis.size - 1:
            azm_indices = np.append(azm_indices, azm_axis.size - 1)
        if rng_indices[-1] != rng_axis.size - 1:
            rng_indices = np.append(rng_indices, rng_axis.size - 1)

    return (
        azm_axis[azm_indices] + raster_info.lines_start,
        rng_axis[rng_indices] + raster_info.samples_start,
        roi[0] + azm_indices,
        roi[1] + rng_indices,
    )


def shift_lut(
    lut_data: npt.NDArray,
    lut_axes: tuple[npt.NDArray[PreciseDateTime], npt.NDArray[float]],
    azimuth_coreg_shifts: npt.NDArray[float],
    range_coreg_shifts: npt.NDArray[float],
    primary_raster_info: RasterInfo,
    secondary_raster_info: RasterInfo,
    lut_interp_fn: Callable,
) -> npt.NDArray:
    """
    Shift the LUT.

    Parameters
    ----------
    lut_data: npt.NDArray[float]
        The LUT data.

    lut_axes: tuple[npt.NDArray[PreciseDateTime], npt.NDArray[float]] [UTC], [s]
        The LUT axes of the secondary image.

    azimuth_coreg_shifts: npt.NDArray[float] [px]
        The azimuth coregistration shifts in pixels.

    range_coreg_shifts: npt.NDArray[float] [px]
        The range coregistration shifts in pixels.

    primary_raster_info: RasterInfo
        The coregistration primary raster info.

    secondary_raster_info: RasterInfo,
        The secondary raster info.

    lut_interp_fn: Callable
        The interpolation function.

    """
    # Convert the coregistration shifts from pixels to time.
    azimuth_time_shifts = azimuth_coreg_shifts * primary_raster_info.lines_step
    range_time_shifts = range_coreg_shifts * primary_raster_info.samples_step

    # Since the LUT axes and grid axes of the secondary are not aligned, we
    # need to shift the coregistration shifts accordingly, more precisely,
    # Assuming LUT[t{a}, t{r}] is defined on the LUT secondary grid, wetting
    # t'{a} and t'{r} to be times on the grid of the data secondary grid, then
    # it holds:
    #
    #   LUT[t{a}, t{r}] = LUT[t'{a} + dT{Az}, t'{r} + dT{Rg}],
    #
    # for dT{Az}, dT{Rn} azimuth and range time deltas between the two
    # grids. Setting Cor{Az} and Cor{Rg} to be the coregistration time shifts
    # in azimuth and range respectively, Cor{Az} and Cor{Rg} map the data grid
    # of the primary onto the data grid of the secondary. In other words,
    # setting t*{a} and t*{r} to be times on the data grid of the primary, it
    # holds
    #
    #    \bar{t}'{a} = Cor{Az}[t*{a}, t*{r}],
    #    \bar{t}'{r} = Cor{Az}[t*{a}, t*{r}],
    #
    # where \bar{t}'{.} are the time points on the data grid of the secondary
    # that correspond to the time points t*{.} on the data grid of the primary.
    #
    # Owing to the fact that the primary grid of the LUT is by definition
    # aligned to the primary grid of the data, it is easy to see that
    #
    #    LUT[
    #         Cor{Az}[t*{a}, t*{r}] + dT{Az},
    #         Cor{Rg}[t*{a}, t*{r}] + dT{Rg},
    #    ]
    #
    # is the LUT evaluated on the LUT primary grid.

    # The time deltas.
    timedelta_azm = secondary_raster_info.lines_start - lut_axes[0][0]
    timedelta_rng = secondary_raster_info.samples_start - lut_axes[1][0]

    # Convert the coregistration shifts in pixel into time shifts.
    azimuth_time_shifts = azimuth_coreg_shifts * primary_raster_info.lines_step
    range_time_shifts = range_coreg_shifts * primary_raster_info.samples_step

    # Shift the LUTs.
    return lut_interp_fn(
        lut_data,
        (
            (lut_axes[0] - lut_axes[0][0]).astype(np.float64),
            lut_axes[1] - lut_axes[1][0],
        ),
        (
            azimuth_time_shifts.reshape(-1) + timedelta_azm,
            range_time_shifts.reshape(-1) + timedelta_rng,
        ),
    ).reshape(azimuth_time_shifts.shape)
