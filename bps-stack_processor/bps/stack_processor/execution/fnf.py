# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Utilities to operate with the FNF mask"""

import dataclasses
import pathlib

import numpy as np
from osgeo import gdal


@dataclasses.dataclass(kw_only=True, frozen=True)
class FnfMask:
    """
    Store the FNF mask.

    Attributes
    ----------
    mask : np.ndarray
        A binary mask of dtype np.uint8.
    lat_axis : np.ndarray
        The relative latitude axis [rad].
    lon_axis : np.ndarray
        The relative longitude axis [rad].
    nodata_value : int
        The nodata value.

    """

    mask: np.ndarray
    lat_axis: np.ndarray
    lon_axis: np.ndarray
    nodata_value: int

    def __post_init__(self):
        """
        __post_init__()

        Minimal validation of the input.

        """
        if self.lat_axis.size != self.mask.shape[0]:
            raise RuntimeError("Mask shape does not match the latitude axis")
        if self.lon_axis.size != self.mask.shape[1]:
            raise RuntimeError("Mask shape does not match the longitude axis")


def read_fnf_mask(
    fnf_mask_path: pathlib.Path,
    *,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    pad: float = 0.08,
) -> FnfMask:
    """Read an FNF mask from file.

    Parameters
    ----------
    fnf_mask_path : pathlib.Path
        Read the FNF mask.
    latitudes : np.ndarray
        The latitude map [rad].
    longitudes : np.ndarray
        The longitude map [rad].
    pad : float,default=0.08
        An extra padding [rad].

    Returns
    -------
    FnfMask
        The FNF mask object.

    """
    if pad < 0:
        raise ValueError("FNF padding cannot be negative")

    # Read the FNF mask.
    data = gdal.Open(str(fnf_mask_path), 0)
    band = data.GetRasterBand(1)
    geotransform = np.deg2rad(data.GetGeoTransform())

    if not (
        np.isclose(geotransform[0] * np.sign(geotransform[1]), -np.pi)
        and np.isclose(geotransform[3] * np.sign(geotransform[5]), -np.pi / 2)
    ):
        raise RuntimeError("Provided FNF mask does not covert the entire world")

    if geotransform[5] >= 0:
        raise RuntimeError("FNF mask expects geotransform[5] to be negative")
    if geotransform[1] <= 0:
        raise RuntimeError("FNF mask expects geotransform[1] to be positive")

    # The full axis of the FNF mask.
    fnf_lat_axis = geotransform[3] + geotransform[5] * np.arange(data.RasterYSize)
    fnf_lon_axis = geotransform[0] + geotransform[1] * np.arange(data.RasterXSize)

    # Retrive the latitude indices.
    lat_range = [np.min(latitudes), np.max(latitudes)]
    lat_0_px = _find_pixel(fnf_lat_axis, lat_range[0] - pad)
    lat_1_px = _find_pixel(fnf_lat_axis, lat_range[1] + pad)
    lat_min_px = min(lat_0_px, lat_1_px)
    lat_max_px = max(lat_0_px, lat_1_px)

    # Check if the satellite crosses the antimeridian.
    lon_range = [np.min(longitudes), np.max(longitudes)]
    cross_antimeridian = np.ptp(lon_range) >= np.pi
    if cross_antimeridian:
        lon_range = [np.max(longitudes[longitudes < 0]), np.min(longitudes[longitudes > 0])]
        pad = -pad

    # Retrieve the longitude indices.
    lon_0_px = _find_pixel(fnf_lon_axis, lon_range[0] - pad)
    lon_1_px = _find_pixel(fnf_lon_axis, lon_range[1] + pad)
    lon_min_px = min(lon_0_px, lon_1_px)
    lon_max_px = max(lon_0_px, lon_1_px)

    # We need to distinguish two cases:
    # 1 - No antimeridian crossing: We load 1 connected component.
    # 2 - Antimeridan crossing: We load 2 connected component and we stitch them.
    if np.ptp(lon_range) < np.pi:  # No crossing.
        mask = band.ReadAsArray(
            lon_min_px,
            lat_min_px,
            lon_max_px - lon_min_px + 1,
            lat_max_px - lat_min_px + 1,
        )
        return FnfMask(
            mask=mask,
            lat_axis=fnf_lat_axis[lat_min_px] + geotransform[5] * np.arange(mask.shape[0]),
            lon_axis=fnf_lon_axis[lon_min_px] + geotransform[1] * np.arange(mask.shape[1]),
            nodata_value=np.uint8(band.GetNoDataValue()),
        )

    mask = np.hstack(
        [
            band.ReadAsArray(
                lon_max_px,
                lat_min_px,
                data.RasterXSize - lon_max_px,
                lat_max_px - lat_min_px + 1,
            ),
            band.ReadAsArray(
                0,
                lat_min_px,
                lon_min_px + 1,
                lat_max_px - lat_min_px + 1,
            ),
        ],
    )
    return FnfMask(
        mask=mask,
        lat_axis=fnf_lat_axis[lat_min_px] + geotransform[5] * np.arange(mask.shape[0]),
        lon_axis=fnf_lon_axis[lon_max_px] + geotransform[1] * np.arange(mask.shape[1]),
        nodata_value=np.uint8(band.GetNoDataValue()),
    )


def _find_pixel(axis: np.ndarray, value: float) -> int:
    """Find the pixel in which value is bucketed."""
    return int(np.argmin(np.abs(axis - value)))
