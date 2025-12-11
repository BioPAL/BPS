# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Land Cover Map (LCM) Utilities
----------------------------------
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

# The regex to match the AUX-LCM product and the time stamps.
AUX_LCM_TIMESTAMP_REGEX = "[0-9]{8}T[0-9]{6}"
AUX_LCM_PRODUCT_REGEX = f"^BIO_AUX_(.)*LCM_{AUX_LCM_TIMESTAMP_REGEX}_{AUX_LCM_TIMESTAMP_REGEX}_"


class InvalidAuxLCMProductError(RuntimeError):
    """Handle invalid AUX-LCM product (e.g. wrong naming etc.)."""


@dataclass
class LCMMask:
    """Store the Land Cover Map data."""

    mask: npt.NDArray[np.uint8]
    """The LCM Map in Lat/Lon (of type np.uint8)."""

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


def read_lcm_mask(
    lcm_path: Path,
    latlon_roi: tuple[float, float, float, float],
    *,
    units: Literal["rad", "deg"] = "deg",
    print_info: bool = False,
) -> LCMMask:
    """
    Read a Land Cover Map on a region of interest.

    Parameters
    ----------
    lcm_path: Path
        Land Cover Map (LCM) tiff file path

    latlon_roi: Tuple[float, float, float, float] [rad] or [deg]
        Latitude/Longitude region of interest, in radians or degrees
        and encoded as [lat_min, lat_max, lon_min, lon_max].

    units: Literal["rad", "deg"] = "deg"
        The lat/lon units. This parameter is used both for input
        arguments and for the output mask. It defaults to "deg".

    print_info: bool = False,
        Print LCM information. It defaults to False.

    Raises
    ------
    FileNotFoundError
        In case of a non existing LCM path.

    ValueError
        In case of passing a unit that is not "rad" or "deg".

    Return
    ------
    LCMMask
        The LCM mask.

    """
    if not lcm_path.exists():
        raise FileNotFoundError(f"LCM mask {lcm_path} does not exist")
    if units not in {"deg", "rad"}:
        raise ValueError("Only 'rad' and 'deg' are supported units")

    mask = read_mask(lcm_path, latlon_roi=latlon_roi, units=units)

    if print_info:
        bps_logger.info(
            "LCM: Loaded [%d x %d] mask '%s' @ %.6f %s/px",
            mask.mask.shape[0],
            mask.mask.shape[1],
            lcm_path.name,
            np.abs(mask.lat_axis[1] - mask.lat_axis[0]),
            units,
        )

    return LCMMask(
        mask=mask.mask,
        lat_axis=mask.lat_axis,
        lon_axis=mask.lon_axis,
        geotransform=mask.geotransform,
        units=mask.units,
        nodata_value=mask.nodata_value,
    )


def retrieve_aux_lcm_content(
    lcm_entry_point_path: Path,
    *,
    reference_dates: tuple[PreciseDateTime, ...] | None = None,
) -> None | Path | tuple[Path, ...]:
    """
    Retrieve the LCM object from an entry point path. The entry point path
    can be:

    1) A path to a .tiff file containing the LCM (mostly for debugging purposes).
    2) A path to the AUX-LCM product directory

    If 1), the input path is returned, otherwise the most recent file contained
    in the AUX-LCM directory is returned. If reference dates are input, the list
    of LCM that are closest to the refernce dates are returned.

    None is returned if no data is available.

    Parameters
    ----------
    lcm_entry_point_path : Path
        The AUX-LCM entry point.

    reference_dates: Optional[Tuple[PreciseDateTime, ...]] = None
        Optionally, reference dates. If None is passed, the most recent
        LCM is returned.

    Returns
    -------
    Union[None, Path, Tuple[Path, ...]]
        The path(s) to the LCM(s), or None if no valid LCM are available.

    Raises
    ------
    InvalidAuxLCMProductError

    """
    if not lcm_entry_point_path.exists():
        raise InvalidAuxLCMProductError(f"Specified AUX-LCM entry point {lcm_entry_point_path} does not exist")
    if not lcm_entry_point_path.is_dir():
        raise InvalidAuxLCMProductError(f"Specified AUX-LCM entry point {lcm_entry_point_path} is not valid")

    return retrieve_product_content(
        lcm_entry_point_path,
        reference_dates=reference_dates,
        product_regex_pattern=AUX_LCM_PRODUCT_REGEX,
        timestamp_regex_pattern=AUX_LCM_TIMESTAMP_REGEX,
    )
