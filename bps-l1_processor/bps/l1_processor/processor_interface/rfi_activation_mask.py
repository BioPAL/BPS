# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
RFI activation mask
-------------------
"""

from pathlib import Path

import numpy as np
from bps.common import bps_logger
from netCDF4 import Dataset
from numpy import typing as npt
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import transform


def _get_utm_converter(
    *,
    lon: float,
    lat: float,
) -> Transformer:
    """Return a Transformer converting WGS84 lat/lon to local UTM coordinates."""

    lon_center_wrapped = ((lon + 180) % 360) - 180  # [-180, 180)
    utm_zone = int((lon_center_wrapped + 180) // 6) + 1
    assert 1 <= utm_zone <= 60

    is_northern = lat >= 0
    proj_crs = f"+proj=utm +zone={utm_zone} +{'north' if is_northern else 'south'} +ellps=WGS84 +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)


def _normalize_longitudes(
    footprint: list[tuple[float, float]], mask_lon_axis: npt.NDArray[np.floating]
) -> tuple[list[tuple[float, float]], npt.NDArray[np.floating]]:
    """Shift longitudes to avoid antimeridian crossing issues."""
    longitude_extent = np.ptp(np.array([lon for lon, _ in footprint]))
    antimeridian_crossed = longitude_extent > 180
    if antimeridian_crossed:
        # Longitude interval is changed from [-180, 180) to [0, 360)
        footprint = [(lon + 360 if lon < 0 else lon, lat) for lon, lat in footprint]  # [0, 360)
        mask_lon_axis = np.where(mask_lon_axis < 0, mask_lon_axis + 360, mask_lon_axis)  # [0, 360)
    return footprint, mask_lon_axis


def _reorder_points_counterclockwise(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Reorder points in counterclockwise order, ensuring the first point is repeated at the end to close the polygon if needed."""

    pts = points[:-1] if points[0] == points[-1] else points
    pts = np.asarray(pts, dtype=float)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    # Sort by angle
    sorted_pts = pts[np.argsort(angles)]

    ordered = [tuple(pt) for pt in sorted_pts]
    ordered.append(ordered[0])

    poly = Polygon(ordered)
    if not poly.exterior.is_ccw:
        ordered = ordered[::-1]

    return ordered


def compute_rfi_activation_mask_overlap_fraction(
    footprint: list[tuple[float, float]],
    activation_mask: npt.NDArray[np.bool_],
    mask_lon_axis: npt.NDArray[np.floating],
    mask_lat_axis: npt.NDArray[np.floating],
) -> float:
    """Compute the fraction [0, 1] of the footprint that overlaps an active mask region."""

    footprint, mask_lon_axis = _normalize_longitudes(footprint, mask_lon_axis)
    footprint = _reorder_points_counterclockwise(footprint)
    footprint_latlon = Polygon(footprint)
    bps_logger.debug(f"RFI mask footprint intersection - footprint after normalization and reordering: {footprint}")

    if not footprint_latlon.is_valid or footprint_latlon.area == 0:
        raise RuntimeError(f"Invalid footprint: {footprint}")

    dlon = np.abs(mask_lon_axis[1] - mask_lon_axis[0])
    dlat = np.abs(mask_lat_axis[1] - mask_lat_axis[0])

    lon_min, lat_min, lon_max, lat_max = footprint_latlon.bounds
    lon_indices = np.where((mask_lon_axis >= lon_min - dlon) & (mask_lon_axis <= lon_max + dlon))[0]
    lat_indices = np.where((mask_lat_axis >= lat_min - dlat) & (mask_lat_axis <= lat_max + dlat))[0]

    lon_center, lat_center = footprint_latlon.centroid.x, footprint_latlon.centroid.y
    latlon_to_utm = _get_utm_converter(lon=lon_center, lat=lat_center).transform
    footprint_utm = transform(latlon_to_utm, footprint_latlon)

    overlap_area = 0.0
    for lat_index in lat_indices:
        lat = mask_lat_axis[lat_index]
        for lon_index in lon_indices:
            lon = mask_lon_axis[lon_index]
            if not activation_mask[lon_index, lat_index]:
                continue

            cell_latlon = box(lon - dlon / 2, lat - dlat / 2, lon + dlon / 2, lat + dlat / 2)
            cell_utm = transform(latlon_to_utm, cell_latlon)

            overlap_area += footprint_utm.intersection(cell_utm).area

    return np.clip(overlap_area / footprint_utm.area, 0, 1)


def read_activation_mask(
    mask_file: Path,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Read activation mask from file"""
    ncfile = Dataset(mask_file, mode="r")

    activation_mask = np.asarray(ncfile.groups["rfiActivationMask"]["mask"][:])

    mask_lon_axis = np.asarray(ncfile.variables["longitude"][:])
    mask_lat_axis = np.asarray(ncfile.variables["latitude"][:])

    return activation_mask, mask_lon_axis, mask_lat_axis


def is_rfi_mitigation_enabled_given_footprint(
    footprint: list[tuple[float, float]], mask_file: Path, overlap_threshold: float
) -> bool:
    """If to activate RFI given a footprint"""
    activation_mask, mask_lon_axis, mask_lat_axis = read_activation_mask(mask_file)
    overlap_fraction = compute_rfi_activation_mask_overlap_fraction(
        footprint, activation_mask, mask_lon_axis, mask_lat_axis
    )
    return overlap_fraction > overlap_threshold
