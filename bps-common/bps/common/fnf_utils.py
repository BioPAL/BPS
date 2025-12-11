# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Forest-Nonforest (FNF) Mask Utilities
-----------------------------------------
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.common.mask_io import read_mask, retrieve_product_content
from osgeo import gdal

gdal.UseExceptions()

# The regex to match the FNF internal resource and the time stamps.
FNF_TIMESTAMP_REGEX = "[0-9]{8}T[0-9]{6}"
FNF_PRODUCT_REGEX = f"^BIO_(AUX_)?((.*_FNF)|FNF)_{FNF_TIMESTAMP_REGEX}_{FNF_TIMESTAMP_REGEX}_"


class InvalidAuxFnFProductError(RuntimeError):
    """Handle invalid FNF internal resource (e.g. wrong naming etc.)."""


@dataclass
class FnFMask:
    """Store the Forest/Non-forest mask data."""

    mask: npt.NDArray[np.uint8]
    """The 0/1 FNF binary mask in Lat/Lon (of type np.uint8)."""

    lat_axis: npt.NDArray[np.float32]
    """The starting latitude [rad] or [deg]."""

    lon_axis: npt.NDArray[np.float32]
    """The longitude axis [rad] or [deg]."""

    geotransform: npt.NDArray[np.float32]
    """The geotransformation [rad] or [deg]."""

    units: Literal["rad", "deg"]
    """The units for the Lat/Lon axes."""

    nodata_value: np.uint8 | None = None
    """The value representing missing data."""


def read_fnf_mask(
    fnf_path: Path,
    latlon_roi: tuple[float, float, float, float],
    *,
    units: Literal["rad", "deg"] = "deg",
    print_info: bool = False,
) -> FnFMask:
    """
    Read a Forest/Non-Forest mask on a region of interest.

    Parameters
    ----------
    fnf_path: Path
        Forest-Non Forest mask (FNF) tiff file path

    latlon_roi: Tuple[float, float, float, float] [rad] or [deg]
        Latitude/Longitude region of interest, in radians or degrees
        and encoded as [lat_min, lat_max, lon_min, lon_max].

    units: Literal["rad", "deg"] = "deg"
        The lat/lon units. This parameter is used both for input
        arguments and for the output mask. It defaults to "deg".

    print_info: bool = False,
        Print FNF information. It defaults to False.

    Raises
    ------
    FileNotFoundError
        In case of a non existing FNF path.

    ValueError
        In case of passing a unit that is not "rad" or "deg".

    Return
    ------
    FnFMask
        The FNF mask.

    """
    if not fnf_path.exists():
        raise FileNotFoundError(f"FNF mask {fnf_path} does not exist")
    if units not in {"deg", "rad"}:
        raise ValueError("Only 'rad' and 'deg' are supported units")

    mask = read_mask(fnf_path, latlon_roi=latlon_roi, units=units)

    if print_info:
        bps_logger.info(
            "FNF: Loaded [%d x %d] mask '%s' @ %.6f %s/px",
            mask.mask.shape[0],
            mask.mask.shape[1],
            fnf_path.name,
            np.abs(mask.lat_axis[1] - mask.lat_axis[0]),
            units,
        )

    return FnFMask(
        mask=mask.mask,
        lat_axis=mask.lat_axis,
        lon_axis=mask.lon_axis,
        geotransform=mask.geotransform,
        units=mask.units,
        nodata_value=mask.nodata_value,
    )


def retrieve_fnf_content(
    fnf_entry_point_path: Path,
    *,
    reference_dates: tuple[PreciseDateTime, ...] | None = None,
) -> None | Path | tuple[Path, ...]:
    """
    Retrieve the FNF object from an entry point path. The entry point path
    can be:

    1) A path to a .tiff file containing the FNF (mostly for debugging purposes).
    2) A path to the FNF internal resource directory

    If 1), the input path is returned, otherwise the most recent file contained
    in the FNF directory is returned. If reference dates are input, the list
    of FNF that are closest to the refernce dates are returned.

    None is returned if no data is available.

    Parameters
    ----------
    fnf_entry_point_path : Path
        The FNF entry point.

    reference_dates: Optional[Tuple[PreciseDateTime, ...]] = None
        Optionally, reference dates. If None is passed, the most recent
        FNF is returned.

    Returns
    -------
    Union[None, Path, Tuple[Path, ...]]
        The path(s) to the FNF(s), or None if no valid FNF are available.

    Raises
    ------
    InvalidAuxFnFProductError

    """
    if not fnf_entry_point_path.exists():
        raise InvalidAuxFnFProductError(f"Specified FNF entry point {fnf_entry_point_path} does not exist")
    if not fnf_entry_point_path.is_dir():
        raise InvalidAuxFnFProductError(f"Specified FNF entry point {fnf_entry_point_path} is not valid")

    return retrieve_product_content(
        fnf_entry_point_path,
        reference_dates=reference_dates,
        product_regex_pattern=FNF_PRODUCT_REGEX,
        timestamp_regex_pattern=FNF_TIMESTAMP_REGEX,
    )
