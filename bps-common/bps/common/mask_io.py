# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilties to read LCM/FNF mask products
--------------------------------------
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import numpy.typing as npt
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.common import retrieve_aux_product_data_single_content
from osgeo import gdal


def retrieve_product_content(
    entry_point: Path,
    *,
    reference_dates: Iterable[PreciseDateTime] | None = None,
    product_regex_pattern: str,
    timestamp_regex_pattern: str,
) -> None | Path | tuple[Path, ...]:
    """ """
    if reference_dates is None:
        reference_dates = [PreciseDateTime.now()]

    # If the entry point is already product with return its tiff file.
    if re.match(product_regex_pattern, entry_point.name):
        return retrieve_aux_product_data_single_content(entry_point)

    # Otherwise we look for the product inside the entry point
    products = [
        (file, _get_end_time(re.findall(timestamp_regex_pattern, file.name)))
        for file in entry_point.iterdir()
        if re.match(product_regex_pattern, Path(file).name)
    ]
    if len(products) == 0:
        return None

    output_products = tuple(
        retrieve_aux_product_data_single_content(_find_closest_product(products, reference_date))
        for reference_date in reference_dates
    )

    if len(output_products) == 1:
        return output_products[0]
    return output_products


def _get_end_time(timestamps: list[str]) -> PreciseDateTime:
    """Get the end time of a list of time strings (%Y%m%dT%H%M%S)."""
    if len(timestamps) == 0:
        raise RuntimeError("Invalid product name (no time strings)")

    return sorted(
        map(
            lambda tstr: PreciseDateTime.from_utc_string(f"{tstr}.000000"),
            timestamps,
        )
    )[-1]


def _find_closest_product(
    products: Sequence[tuple[Path, PreciseDateTime]],
    reference_date: PreciseDateTime,
) -> Path:
    """Find the closest product to the reference date."""
    closest_product = int(np.argmin([abs(product_date - reference_date) for _, product_date in products]))
    return products[closest_product][0]


@dataclass
class Mask:
    """Mask data."""

    mask: npt.NDArray[np.uint8]
    """The mask in Lat/Lon (of type np.uint8)."""

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


def read_mask(
    product: Path,
    latlon_roi: tuple[float, float, float, float],
    *,
    units: Literal["rad", "deg"] = "deg",
) -> Mask:
    """"""
    if not product.exists():
        raise FileNotFoundError(f"FNF mask {product} does not exist")
    if units not in {"deg", "rad"}:
        raise ValueError("Only 'rad' and 'deg' are supported units")

    # Since gdal is all in degrees, we convert to degrees. And also convert
    # back (for later usage).
    def to_deg(x):
        return np.rad2deg(x) if units == "rad" else x

    def from_deg(x):
        return np.deg2rad(x) if units == "rad" else x

    # Get min and max lat and lon, with an extra padding, to read only a
    # portion of fnf and making sure we grab the whole portion.
    latlon_roi_deg = list(map(to_deg, latlon_roi))

    pad_deg = 0.5
    lat_min_deg = np.max([latlon_roi_deg[0] - pad_deg, -90])
    lat_max_deg = np.min([latlon_roi_deg[1] + pad_deg, 90])
    lon_min_deg = np.max([latlon_roi_deg[2] - pad_deg, -180])
    lon_max_deg = np.min([latlon_roi_deg[3] + pad_deg, 180])

    # read input fnf metadata
    data = gdal.Open(str(product), 0)
    band = data.GetRasterBand(1)
    geotransform = data.GetGeoTransform()

    # Construct whole input fnf lat lon axis
    # geotransform is [Longitude in, Longitude step, 0, Latitude in, 0 latitude step]
    lon_full_axis_deg = geotransform[0] + geotransform[1] * np.arange(data.RasterXSize)
    lat_full_axis_deg = geotransform[3] + geotransform[5] * np.arange(data.RasterYSize)

    # Get lon/lat indices for reading only a portion of the FNF.
    lon_min_idx = int(np.searchsorted(np.sort(lon_full_axis_deg), lon_min_deg))  # lon offset
    lon_max_idx = int(np.searchsorted(np.sort(lon_full_axis_deg), lon_max_deg))
    lat_max_idx = data.RasterYSize - int(
        np.searchsorted(np.sort(lat_full_axis_deg), lat_min_deg, side="right")
    )  # lat offset
    lat_min_idx = data.RasterYSize - int(np.searchsorted(np.sort(lat_full_axis_deg), lat_max_deg, side="right"))
    lon_size = int(lon_max_idx - lon_min_idx)
    lat_size = int(lat_max_idx - lat_min_idx)

    # Read the FNF data.
    mask = band.ReadAsArray(lon_min_idx, lat_min_idx, lon_size, lat_size)
    assert mask.dtype == np.uint8

    # Assemble FNF lat/lon axes.
    if geotransform[1] > 0:
        lon_axis_deg = lon_min_deg + geotransform[1] * np.arange(lon_size)
    else:
        lon_axis_deg = lon_max_deg + geotransform[1] * np.arange(lon_size)

    if geotransform[5] > 0:
        lat_axis_deg = lat_min_deg + geotransform[5] * np.arange(lat_size)
    else:
        lat_axis_deg = lat_max_deg + geotransform[5] * np.arange(lat_size)

    return Mask(
        mask=mask,
        lat_axis=from_deg(lat_axis_deg),
        lon_axis=from_deg(lon_axis_deg),
        geotransform=from_deg(np.asarray(geotransform)),
        units=units,
        nodata_value=np.uint8(band.GetNoDataValue()),
    )
