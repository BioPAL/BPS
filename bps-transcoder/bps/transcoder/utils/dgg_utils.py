# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
DGG utils
---------
"""

import numpy as np
from bps.common.l2_joborder_tags import (
    L2A_OUTPUT_PRODUCT_FD,
    L2A_OUTPUT_PRODUCT_FH,
    L2A_OUTPUT_PRODUCT_GN,
    L2A_OUTPUT_PRODUCT_TFH,
    L2B_OUTPUT_PRODUCT_AGB,
    L2B_OUTPUT_PRODUCT_FD,
    L2B_OUTPUT_PRODUCT_FH,
)


def dgg_search_tiles(latlon_coverage: list[float], create_tiles_dict: bool | None = False):
    """Construct the pixel wise lat lon grid, for all the tiles (list of tiles)"""

    # Tile extent by definition
    tile_extent_lat = 1  # deg
    tile_extent_lon = 1  # deg

    # Copernicus latitude pixel spacing is constant

    pixel_spacing_lat = 3 / 3600  # deg

    # geographic_boundaries
    lat_min = latlon_coverage[0]
    lat_max = latlon_coverage[1]
    lon_min = latlon_coverage[2]
    lon_max = latlon_coverage[3]

    # initialize outputs
    tiles_name_list = []  # list of names for the tiles
    geotransform_list = []  # list of geotransform for each tile
    geotransform_dict = {}  # dict of geotransform for each tile
    tile_extent_lonlat_dict = {}  # list of tile degrees extension in lat / lon for each tile [tile extent lon deg, tile extent lat deg]

    # whole Copernicus latitude vector
    latitude_vector = np.arange(-70, 70, tile_extent_lat) + pixel_spacing_lat / 2

    # Copernicus longitude pixel spacing varies depending on latitude sector
    pixel_spacing_lon_list = np.zeros(latitude_vector.shape).astype(np.float64)
    pixel_spacing_lon_list[np.logical_and(latitude_vector >= -70, latitude_vector < -60)] = 6 / 3600
    pixel_spacing_lon_list[np.logical_and(latitude_vector >= -60, latitude_vector < -50)] = 4.5 / 3600
    pixel_spacing_lon_list[np.logical_and(latitude_vector >= -50, latitude_vector < 50)] = 3 / 3600
    pixel_spacing_lon_list[np.logical_and(latitude_vector >= 50, latitude_vector < 60)] = 4.5 / 3600
    pixel_spacing_lon_list[np.logical_and(latitude_vector >= 60, latitude_vector < 70)] = 6 / 3600

    lat_tiles_name = ["S{:02d}".format(70 - lat) for lat in range(0, 70, tile_extent_lat)] + [
        "N{:02d}".format(lat) for lat in range(0, 89, tile_extent_lat)
    ]

    # extract input requested latitude vectors from the whole ones:
    if lat_min < min(latitude_vector):
        lat_min = min(latitude_vector) + pixel_spacing_lat / 2
        print(f"geographic boundaries lat min changed to {lat_min} [deg]")
    if lat_max > max(latitude_vector):
        lat_max = max(latitude_vector)
        print(f"geographic boundaries lat max changed to {lat_max} [deg]")

    lat_first_tile = np.max(latitude_vector[lat_min > latitude_vector])
    lat_first_tile_idx = np.where(latitude_vector == lat_first_tile)[0][0]
    lat_last_tile = np.min(latitude_vector[lat_max <= latitude_vector])
    lat_last_tile_idx = np.where(latitude_vector == lat_last_tile)[0][0]

    latitude_vector = latitude_vector[lat_first_tile_idx:lat_last_tile_idx]
    pixel_spacing_lon_list = pixel_spacing_lon_list[lat_first_tile_idx:lat_last_tile_idx]
    lat_tiles_name = lat_tiles_name[lat_first_tile_idx:lat_last_tile_idx]

    tiles_dict = {}
    for lat_idx in np.arange(len(latitude_vector)):
        pixel_spacing_lon = pixel_spacing_lon_list[lat_idx]

        # whole Copernicus longitude vector
        longitude_vector = np.arange(-180, 180, tile_extent_lon) - pixel_spacing_lon / 2

        lon_tiles_name = ["W{:03d}".format(180 - lon) for lon in range(0, 180, tile_extent_lon)] + [
            "E{:03d}".format(lon) for lon in range(0, 180, tile_extent_lon)
        ]

        # extract input requested longitude vectors from the whole ones
        if lon_min < min(longitude_vector):
            lon_min = min(longitude_vector) + pixel_spacing_lon / 2
            print(f"geographic boundaries lon min changed to {lon_min} [deg]")
        if lon_max > max(longitude_vector):
            lon_max = max(longitude_vector)
            print(f"geographic boundaries lon max changed to {lon_max} [deg]")

        lon_first_tile = np.max(longitude_vector[lon_min > longitude_vector])
        lon_first_tile_idx = np.where(longitude_vector == lon_first_tile)[0][0]
        lon_last_tile = np.min(longitude_vector[lon_max <= longitude_vector])
        lon_last_tile_idx = np.where(longitude_vector == lon_last_tile)[0][0]

        longitude_vector = longitude_vector[lon_first_tile_idx:lon_last_tile_idx]
        lon_tiles_name = lon_tiles_name[lon_first_tile_idx:lon_last_tile_idx]

        for lon_idx in np.arange(len(longitude_vector)):
            # fill the outputs for current lat / lon tile:

            tile_name_curr = "" + lat_tiles_name[lat_idx] + lon_tiles_name[lon_idx]
            geotransform = [
                longitude_vector[lon_idx],
                pixel_spacing_lon,
                0.0,
                latitude_vector[lat_idx]
                + tile_extent_lat,  # go to end of tile extent, last value of latitude is decrescent in copernicus DEM
                0.0,
                -pixel_spacing_lat,  # minus, because it is decrescent in copernicus DEM
            ]
            tile_extent_lon_lat = [tile_extent_lon, tile_extent_lat]

            tiles_name_list.append(tile_name_curr)
            geotransform_list.append(geotransform)
            geotransform_dict[tile_name_curr] = geotransform
            tile_extent_lonlat_dict[tile_name_curr] = tile_extent_lon_lat

            # fill tiles_dict
            if create_tiles_dict:
                len_lat = int(tile_extent_lat / pixel_spacing_lat)
                len_lon = int(tile_extent_lon / pixel_spacing_lon)

                tile_info = {}
                tile_info["geotransform"] = geotransform
                tile_info["longitude_vector"] = geotransform[0] + geotransform[1] * np.arange(len_lon)
                tile_info["latitude_vector"] = geotransform[3] + geotransform[5] * np.arange(len_lat)
                tiles_dict[tile_name_curr] = tile_info

    return tiles_name_list, geotransform_dict, tile_extent_lonlat_dict, tiles_dict


def create_dgg_sampling_dict(product_type: str) -> dict:
    """
    Create a dictionary containing all the possible DGG grid sampling parameters
    for a 1x1 deg DGG tile.
    DGG varies with the latitude region.

    Parameters
    ----------
    product_type: str
        product type, "FP_FD__L2A", "FP_FH__L2A", "FP_GN__L2A"

    Returns
    -------
    dgg_sampling: dict
        Dictionary containing three string keys,
        one for each DGG sampling latitude region (in degrees):
        "0-50"
        "50-60"
        "60-70"
        Each dict key contains the following DGG parameters for a 1x1 deg DGG tile in the region,
        in sunbdicts:
        latitude_spacing_deg: int
            DGG latitude spacing in degrees
        longitude_spacing_deg:int
            DGG longitude spacing in degrees
        n_lat:int
            DGG number of latitude pixels for a 1x1 degree DGG tile
        n_lon: int
            DGG number of longitude pixels for a 1x1 degree DGG tile
    """

    dgg_sampling_dict = {}

    if product_type in [L2A_OUTPUT_PRODUCT_FD, L2B_OUTPUT_PRODUCT_FD]:
        dgg_sampling_dict["0-50"] = {
            "latitude_spacing_deg": 1.5 / 3600,
            "longitude_spacing_deg": 1.5 / 3600,
            "n_lat": 2401,
            "n_lon": 2401,
        }
        dgg_sampling_dict["50-60"] = {
            "latitude_spacing_deg": 1.5 / 3600,
            "longitude_spacing_deg": 2.25 / 3600,
            "n_lat": 2401,
            "n_lon": 1601,
        }
        dgg_sampling_dict["60-70"] = {
            "latitude_spacing_deg": 1.5 / 3600,
            "longitude_spacing_deg": 3 / 3600,
            "n_lat": 2401,
            "n_lon": 1201,
        }

    if product_type in [
        L2A_OUTPUT_PRODUCT_FH,
        L2A_OUTPUT_PRODUCT_GN,
        L2A_OUTPUT_PRODUCT_TFH,
        L2B_OUTPUT_PRODUCT_FH,
        L2B_OUTPUT_PRODUCT_AGB,
    ]:
        dgg_sampling_dict["0-50"] = {
            "latitude_spacing_deg": 3 / 3600,
            "longitude_spacing_deg": 3 / 3600,
            "n_lat": 1201,
            "n_lon": 1201,
        }
        dgg_sampling_dict["50-60"] = {
            "latitude_spacing_deg": 3 / 3600,
            "longitude_spacing_deg": 4.5 / 3600,
            "n_lat": 1201,
            "n_lon": 801,
        }
        dgg_sampling_dict["60-70"] = {
            "latitude_spacing_deg": 3 / 3600,
            "longitude_spacing_deg": 6 / 3600,
            "n_lat": 1201,
            "n_lon": 601,
        }

    return dgg_sampling_dict
