# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Compute the SKP quality indicator
---------------------------------
"""

# NOTE: As for now, the quality indicator for the SKP is given by a
# Forest/Non-forest (FNF) mask. The mask encoding is the following
#
#  0.0: Non-forested area,
#  0.5: Unknown area,
#  1.0: Forested area,
#
# This relies on the assumption that the SKP works well on the forsted areas
# and fails on the unforested areas, while it has undefined behavior on areas
# that are neither forested nor non-forested.

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from bps.common.fnf_utils import FnFMask
from bps.stack_cal_processor.core.utils import interpolate_on_grid, query_grid_mask


@dataclass
class SkpFnFQualityMask:
    """A class to store the FNF mask information for the SKP quality."""

    fnf_mask: FnFMask
    """The Forest-Nonforest mask of shape [Nrow x Ncol]."""

    latitudes: npt.NDArray[float]
    """The [Nrow x Ncol] latitudes of the FNF mask grid-points. [rad]"""

    longitudes: npt.NDArray[float]
    """The [Nrow x Ncol] longitudes of the FNF mask grid-points. [rad]"""

    azimuth_axis: npt.NDArray[float]
    """The [1 x Nrow] azimuth axis corresponding to the FNF mask grid. [s]"""

    range_axis: npt.NDArray[float]
    """The [1 x Ncol] range axis corresponding to the FNF mask grid. [s]"""


def compute_skp_fnf_quality(
    *,
    skp_fnf_mask: SkpFnFQualityMask,
    skp_azimuth_axis: npt.NDArray[float],
    skp_range_axis: npt.NDArray[float],
) -> npt.NDArray[float]:
    """
    Compute the SKP quality by querying the FNF mask.

    Parameters
    ----------
    skp_fnf_mask: SkpFnFQualityMask
        The FNF mask for the SKP quality.

    skp_azimuth_axis: npt.NDArray[float] [s]
        The SKP relative azimuth time.

    skp_range_axis: npt.NDArray[float] [s]
        The SKP relative slant range time.

    Raises
    ------
    ValueError

    Return
    ------
    npt.NDArray[float] [adim]
        The quality indicator mask.

    """
    # First we interpolate the SKP axis onto the coreg primary grid.
    lat_coreg_primary = interpolate_on_grid(
        skp_fnf_mask.latitudes,
        axes_in=(skp_fnf_mask.azimuth_axis, skp_fnf_mask.range_axis),
        axes_out=(skp_azimuth_axis, skp_range_axis),
    )
    lon_coreg_primary = interpolate_on_grid(
        skp_fnf_mask.longitudes,
        axes_in=(skp_fnf_mask.azimuth_axis, skp_fnf_mask.range_axis),
        axes_out=(skp_azimuth_axis, skp_range_axis),
    )

    # NOTE: no-data values are left as NaN's, thus they will be ignored in the
    # SKP quality mask calculation and substituted by no-data values by the L1c
    # writer.
    return query_grid_mask(
        mask=skp_fnf_mask.fnf_mask.mask,
        x_axis=skp_fnf_mask.fnf_mask.lat_axis,
        y_axis=skp_fnf_mask.fnf_mask.lon_axis,
        xs=lat_coreg_primary,
        ys=lon_coreg_primary,
        nodata_fill_value=np.nan,
        nodata_value=skp_fnf_mask.fnf_mask.nodata_value,
    )
