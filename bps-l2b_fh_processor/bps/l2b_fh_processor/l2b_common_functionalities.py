# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2b common functionalities
--------------------------
"""

from pathlib import Path
from typing import Literal, get_args

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.transcoder.sarproduct.biomass_l2aproduct import BIOMASSL2aProductFD, BIOMASSL2aProductFH, BIOMASSL2aProductGN
from bps.transcoder.sarproduct.biomass_l2bfdproduct import BIOMASSL2bFDProduct
from bps.transcoder.utils.dgg_utils import create_dgg_sampling_dict, dgg_search_tiles
from scipy.interpolate import RegularGridInterpolator

L2B_OUTPUT_PRODUCT_FD_LITERAL = Literal["FP_FD__L2B"]
L2B_OUTPUT_PRODUCT_FH_LITERAL = Literal["FP_FH__L2B"]
L2B_OUTPUT_PRODUCT_AGB_LITERAL = Literal["FP_AGB_L2B"]
L2B_OUTPUT_PRODUCT_FD = get_args(L2B_OUTPUT_PRODUCT_FD_LITERAL)[0]
L2B_OUTPUT_PRODUCT_FH = get_args(L2B_OUTPUT_PRODUCT_FH_LITERAL)[0]
L2B_OUTPUT_PRODUCT_AGB = get_args(L2B_OUTPUT_PRODUCT_AGB_LITERAL)[0]

L2B_OUTPUT_PRODUCTS = Literal[
    L2B_OUTPUT_PRODUCT_FD_LITERAL,
    L2B_OUTPUT_PRODUCT_FH_LITERAL,
    L2B_OUTPUT_PRODUCT_AGB_LITERAL,
]
INT_NODATA_VALUE = int(255)
FLOAT_NODATA_VALUE = -9999.0


def dgg_tiling(
    l2a_products_list: (list[BIOMASSL2aProductFD] | list[BIOMASSL2aProductFH] | list[BIOMASSL2aProductGN]),
    minimum_l2a_coverage: float,
    tile_id: str,
    l2b_product_type: L2B_OUTPUT_PRODUCTS,
    l2b_fd_product_for_fh: BIOMASSL2bFDProduct | None = None,
) -> tuple[bool, dict, np.ndarray, np.ndarray, list[float]]:
    """Tiling:
    placing input L2a products into L2b DGG tile

    Description:
    Input L2a products are already sampled over DGG and contains several DGG tiles coverage
    L2b is generated on a single Tile, specified by the user
    This function also shapes the optput in 3D matrices, already time ordered

    Parameters
    ----------
    l2a_products_list: Union[List[BIOMASSL2aProductFD], List[BIOMASSL2aProductFH], Dict[List[BIOMASSL2aProductGN]]]
        List of L2a products
    minimum_l2a_coverage: float
        Minimum coverage of L2a over the L2b tile_id, to trogger computation [%]
    tile_id: str
        Name of the Tile ID to be processed
    l2b_product_type: 'FP_FD__L2B', 'FP_FH__L2B' or 'FP_AGB_L2B'
        Product Type in l2a_products_list
    l2b_fd_product_for_fh: Optional[BIOMASSL2bFDProduct] = None
        Optional L2B FD product, for FH CFM reading instead of FNF

    Returns
    -------
    skip_computation: bool
        Flag for skipping computation of the tile_id, in case the minimum_l2a_coverage is not satisfied
    data_3d_mat_dict: dict
        The dictionary contains all of the L2a MDS (as fd, cfm, probability, heat map, or fh, fh quality....)
        Each dictionary entry is a 3d matrix of shape [num_lat_dgg_tile, num_lon_dgg_tile, num_l2a_products]
        Note: each 3D matrix third dimension is already temporal ordered (see l2a_temporal_sorting())
    l2b_dgg_tile_latitude_axis_deg: np.ndarray
        L2B DGG tile latitude axis [deg]
    l2b_dgg_tile_longitude_axis_deg: np.ndarray
        L2B DGG tile longitude axis [deg]
    dgg_tile_footprint: List[float]

    """

    bps_logger.info(f"    DGG tiling into '{tile_id}':")

    # First, get all the Tile IDs in the input L2A products
    tile_id_list = []
    for l2a_product in l2a_products_list:
        for tile_id_curr in l2a_product.main_ads_product.tile_id_list:
            if tile_id_curr not in tile_id_list:
                tile_id_list.append(tile_id_curr)

    # Than verify if the Tile ID to be processed is present (L2B processes only one Tile)
    if tile_id not in tile_id_list:
        raise ValueError(f"Tile ID {tile_id} specified in Job Order is not present in any of the L2A input products")

    # bps_logger.info(
    #     f"        L2A input products cover an area of {len(tile_id_list)} tiles: only the coverage of specified input Tile ID {tile_id} will be processed."
    # )

    # Get the geotransform for each DGG tile in the cumulative footprint
    latlon_coverage = _get_l2a_cumulative_footprint(l2a_products_list)
    assert latlon_coverage is not None

    (
        l2b_dgg_tile_latitude_axis_deg,
        l2b_dgg_tile_longitude_axis_deg,
        dgg_tile_footprint,
        dgg_band_key,
    ) = dgg_tiling_create_axis(latlon_coverage, tile_id, l2b_product_type, l2a_products_list)

    # Axis meshgrid, for the 2D interpolation
    lon_meshed, lat_meshed = np.meshgrid(l2b_dgg_tile_longitude_axis_deg, l2b_dgg_tile_latitude_axis_deg)

    # check coverage and discard if not at least of minimum_l2a_coverage percent
    (
        coverage_is_enough,
        _,
        _,
    ) = _check_coverage(
        latlon_coverage,
        lat_meshed,
        lon_meshed,
        minimum_l2a_coverage,
    )

    number_of_data_in_the_tile = 0
    for l2a_product in l2a_products_list:
        if tile_id in l2a_product.main_ads_product.tile_id_list:
            number_of_data_in_the_tile += 1

    data_3d_mat_dict = _init_output_dict(
        l2b_product_type,
        lat_meshed.shape[0],
        lat_meshed.shape[1],
        number_of_data_in_the_tile,
    )
    data_3d_mat_dict["acquisition_ids"] = []
    if coverage_is_enough:
        idx = 0
        for l2a_product in l2a_products_list:
            if tile_id not in l2a_product.main_ads_product.tile_id_list:
                continue
            # check if need to interpolate:
            l2a_band_key = _get_latitude_band_key(
                np.min(l2a_product.measurement.latitude_vec),
                np.max(l2a_product.measurement.latitude_vec),
            )
            if l2a_band_key == dgg_band_key:
                interp_method = "nearest"
            else:
                interp_method = "linear"

            # compute the output L2B product by fusion
            # keep in each tile also the needed additional info: axis, heat map
            if l2b_product_type == L2B_OUTPUT_PRODUCT_FD:
                # preliminary interpolation, if the Lla is between two different

                # FD

                # first convert NODATA to np.nan (casting to float32 first)
                nan_mask = np.logical_or(
                    l2a_product.measurement.data_dict["fd"] == INT_NODATA_VALUE,
                    np.isnan(l2a_product.measurement.data_dict["fd"]),
                )
                fd_input = l2a_product.measurement.data_dict["fd"].astype(np.float64)
                fd_input[nan_mask] = np.nan

                data_3d_mat_dict["fd"][:, :, idx] = _core_interpolator_2d(
                    fd_input,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                )

                # CFM

                # first convert NODATA to np.nan (casting to float32 first)
                nan_mask = np.logical_or(
                    l2a_product.measurement.data_dict["cfm"] == INT_NODATA_VALUE,
                    np.isnan(l2a_product.measurement.data_dict["cfm"]),
                )
                cfm_input = l2a_product.measurement.data_dict["cfm"].astype(np.float64)
                cfm_input[nan_mask] = np.nan

                data_3d_mat_dict["cfm"][:, :, idx] = _core_interpolator_2d(
                    cfm_input,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                )

                # Probability of change

                # first convert NODATA to np.nan
                nan_mask = np.logical_or(
                    l2a_product.measurement.data_dict["probability_ofchange"] == FLOAT_NODATA_VALUE,
                    np.isnan(l2a_product.measurement.data_dict["probability_ofchange"]),
                )
                probability_of_change_input = l2a_product.measurement.data_dict["probability_ofchange"]
                probability_of_change_input[nan_mask] = np.nan

                data_3d_mat_dict["probability_ofchange"][:, :, idx] = _core_interpolator_2d(
                    probability_of_change_input,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                )

            if l2b_product_type == L2B_OUTPUT_PRODUCT_FH:
                if l2b_fd_product_for_fh:
                    temp_fnf = _core_interpolator_2d(
                        l2b_fd_product_for_fh.measurement.data_dict["cfm"],
                        l2b_fd_product_for_fh.measurement.latitude_vec,
                        l2b_fd_product_for_fh.measurement.longitude_vec,
                        lat_meshed,
                        lon_meshed,
                        interp_method,
                        fill_value=INT_NODATA_VALUE,
                    )

                else:
                    temp_fnf = _core_interpolator_2d(
                        l2a_product.lut_ads.lut_fnf,
                        l2a_product.measurement.latitude_vec,
                        l2a_product.measurement.longitude_vec,
                        lat_meshed,
                        lon_meshed,
                        interp_method,
                        fill_value=INT_NODATA_VALUE,
                    )

                nan_mask = np.isnan(temp_fnf)
                temp_fnf[nan_mask] = INT_NODATA_VALUE
                data_3d_mat_dict["bps_fnf"][:, :, idx] = temp_fnf.astype(np.uint8)

                # FH

                # first convert NODATA to np.nan
                nan_mask = np.logical_or(
                    l2a_product.measurement.data_dict["fh"] == FLOAT_NODATA_VALUE,
                    np.isnan(l2a_product.measurement.data_dict["fh"]),
                )
                fh_input = l2a_product.measurement.data_dict["fh"]
                fh_input[nan_mask] = np.nan

                data_3d_mat_dict["fh"][:, :, idx] = _core_interpolator_2d(
                    fh_input,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                )

                # from matplotlib import pyplot as plt
                # footprint = l2a_product.main_ads_input_information.footprint
                # plt.imshow(l2a_product.measurement.data_dict["fh"], origin="lower",  extent=[np.min(footprint[1:8:2]), np.max(footprint[1:8:2]), np.min(footprint[0:8:2]), np.max(footprint[0:8:2])]);plt.axis("equal");plt.show()
                # plt.imshow(data_3d_mat_dict["fh"][:, :, idx], origin="lower",  extent=[np.min(lon_meshed), np.max(lon_meshed), np.min(lat_meshed), np.max(lat_meshed)]);plt.axis("equal");plt.show()

                # FH Quality

                # first convert NODATA to np.nan
                nan_mask = np.logical_or(
                    l2a_product.measurement.data_dict["quality"] == FLOAT_NODATA_VALUE,
                    np.isnan(l2a_product.measurement.data_dict["quality"]),
                )
                fhquality_input = l2a_product.measurement.data_dict["quality"]
                fhquality_input[nan_mask] = np.nan

                data_3d_mat_dict["quality"][:, :, idx] = _core_interpolator_2d(
                    fhquality_input,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                )

            if l2b_product_type == L2B_OUTPUT_PRODUCT_AGB:
                # AGB

                # first convert NODATA to np.nan (casting to float32 first)
                nan_mask_float = np.logical_and(
                    np.logical_and(
                        (l2a_product.measurement.data_dict["HH"] == FLOAT_NODATA_VALUE),
                        (l2a_product.measurement.data_dict["VH"] == FLOAT_NODATA_VALUE),
                    ),
                    (l2a_product.measurement.data_dict["VV"] == FLOAT_NODATA_VALUE),
                )
                nan_mask_nan = np.logical_and(
                    np.logical_and(
                        np.isnan(l2a_product.measurement.data_dict["HH"]),
                        np.isnan(l2a_product.measurement.data_dict["VH"]),
                    ),
                    np.isnan(l2a_product.measurement.data_dict["VV"]),
                )
                nan_mask = np.logical_or(nan_mask_float, nan_mask_nan)

                gn_input_hh = l2a_product.measurement.data_dict["HH"].astype(np.float64)
                gn_input_hh[nan_mask] = np.nan

                gn_input_vh = l2a_product.measurement.data_dict["VH"].astype(np.float64)
                gn_input_vh[nan_mask] = np.nan

                gn_input_vv = l2a_product.measurement.data_dict["VV"].astype(np.float64)
                gn_input_vv[nan_mask] = np.nan

                data_3d_mat_dict["HH"][:, :, idx] = _core_interpolator_2d(
                    gn_input_hh,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                )

                data_3d_mat_dict["VH"][:, :, idx] = _core_interpolator_2d(
                    gn_input_vh,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                )

                data_3d_mat_dict["VV"][:, :, idx] = _core_interpolator_2d(
                    gn_input_vv,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                )

                local_incidence_angle = l2a_product.lut_ads.lut_local_incidence_angle
                local_incidence_angle[nan_mask] = np.nan

                data_3d_mat_dict["local_incidence_angle"][:, :, idx] = _core_interpolator_2d(
                    local_incidence_angle,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                )

                data_3d_mat_dict["temporal_date_sec85"][idx] = l2a_product.main_ads_product.start_time.sec85

                temp_fnf = _core_interpolator_2d(
                    l2a_product.lut_ads.lut_fnf,
                    l2a_product.measurement.latitude_vec,
                    l2a_product.measurement.longitude_vec,
                    lat_meshed,
                    lon_meshed,
                    interp_method,
                    fill_value=INT_NODATA_VALUE,
                )

                nan_mask = np.isnan(temp_fnf)
                temp_fnf[nan_mask] = INT_NODATA_VALUE
                data_3d_mat_dict["bps_fnf"][:, :, idx] = temp_fnf.astype(np.uint8)

            data_3d_mat_dict["acquisition_ids"].append(l2a_product.name)
            idx += 1

    return (
        np.invert(coverage_is_enough),
        data_3d_mat_dict,
        l2b_dgg_tile_latitude_axis_deg,
        l2b_dgg_tile_longitude_axis_deg,
        dgg_tile_footprint,
    )


def dgg_tiling_create_axis(latlon_coverage, tile_id, l2b_product_type, l2a_products_list):
    (
        _,
        geotransform_d,
        _,
        tiles_dict,
    ) = dgg_search_tiles(latlon_coverage, True)
    assert tile_id in geotransform_d
    geotransform = geotransform_d[tile_id]
    tiles_info = tiles_dict[tile_id]
    lat_min = np.min(tiles_info["latitude_vector"])
    lat_max = np.max(tiles_info["latitude_vector"])

    # Tiling
    # Construct the lat lon axis of the output Tile:
    dgg_sampling_dict = create_dgg_sampling_dict(l2b_product_type)
    dgg_band_key = _get_latitude_band_key(lat_min, lat_max)
    dgg_sampling = dgg_sampling_dict[dgg_band_key]

    dgg_lat_sign = 1 if geotransform[5] > 0 else -1
    l2b_dgg_tile_latitude_axis_deg = geotransform[3] + dgg_lat_sign * dgg_sampling["latitude_spacing_deg"] * np.arange(
        dgg_sampling["n_lat"]
    )

    dgg_lon_sign = 1 if geotransform[1] > 0 else -1
    l2b_dgg_tile_longitude_axis_deg = geotransform[0] + dgg_lon_sign * dgg_sampling[
        "longitude_spacing_deg"
    ] * np.arange(dgg_sampling["n_lon"])

    # now check the convention of dgg axis respect to the one in the l2a products
    # and flip dgg if necessary:
    l2a_lat_sign = l2a_products_list[0].measurement.latitude_vec[5] - l2a_products_list[0].measurement.latitude_vec[4]
    l2a_lon_sign = l2a_products_list[0].measurement.longitude_vec[5] - l2a_products_list[0].measurement.longitude_vec[4]
    if (l2a_lat_sign > 0 and dgg_lat_sign < 0) or (l2a_lat_sign < 0 and dgg_lat_sign > 0):
        l2b_dgg_tile_latitude_axis_deg = np.flip(l2b_dgg_tile_latitude_axis_deg)
    if (l2a_lon_sign > 0 and dgg_lon_sign < 0) or (l2a_lon_sign < 0 and dgg_lon_sign > 0):
        l2b_dgg_tile_longitude_axis_deg = np.flip(l2b_dgg_tile_longitude_axis_deg)

    # [ne_lat, ne_lon, se_lat, se_lon, sw_lat, sw_lon, nw_lat, nw_lon]
    # This is easy to write, because Tile is oriented without inclinations
    dgg_tile_footprint = [
        max(l2b_dgg_tile_latitude_axis_deg),
        max(l2b_dgg_tile_longitude_axis_deg),
        min(l2b_dgg_tile_latitude_axis_deg),
        max(l2b_dgg_tile_longitude_axis_deg),
        min(l2b_dgg_tile_latitude_axis_deg),
        min(l2b_dgg_tile_longitude_axis_deg),
        max(l2b_dgg_tile_latitude_axis_deg),
        min(l2b_dgg_tile_longitude_axis_deg),
    ]

    return (
        l2b_dgg_tile_latitude_axis_deg,
        l2b_dgg_tile_longitude_axis_deg,
        dgg_tile_footprint,
        dgg_band_key,
    )


def l2a_temporal_sorting(l2a_paths: tuple[Path]) -> tuple[Path]:
    """Temporal sorting of input l2a product paths

    Parameters
    ----------
    l2a_paths: Tuple[Path]
        Tuple with paths (pathlib) of input L2a products

    Returns
    -------
    l2a_paths_ordered: Tuple[Path]
        Tuple with paths (pathlib) of input L2a products,
        ordered in time from older to newer
    """

    bps_logger.info("L2a input products temporal sorting")

    dates_sec85_list = []

    # fitst get the dates from dict keys, and convert to sec85
    for l2a_path in l2a_paths:
        date = str(l2a_path.name[15:30])
        dates_sec85_list.append(
            PreciseDateTime()
            .from_numeric_datetime(
                int(date[0:4]),
                int(date[4:6]),
                int(date[6:8]),
                int(date[9:11]),
                int(date[11:13]),
                int(date[13:15]),
                0,
            )
            .sec85
        )

    # get the sorting indices of the sec85 dates
    sorting_indexes = np.argsort(dates_sec85_list)
    l2a_paths_ordered = tuple(np.array(l2a_paths)[sorting_indexes])

    return l2a_paths_ordered


def _init_output_dict(
    l2b_product_type: str,
    num_lat,
    num_lon,
    num_imms,
):
    if l2b_product_type == L2B_OUTPUT_PRODUCT_FD:
        # Note: all cast to float to insert np.nan
        return {
            "fd": np.zeros((num_lat, num_lon, num_imms)).astype(np.float64),
            "cfm": np.zeros((num_lat, num_lon, num_imms)).astype(np.float64),
            "probability_ofchange": np.zeros((num_lat, num_lon, num_imms)).astype(np.float32),
        }

    if l2b_product_type == L2B_OUTPUT_PRODUCT_FH:
        # Note also that bps_fnf is a 2D matrix
        return {
            "fh": np.zeros((num_lat, num_lon, num_imms)).astype(np.float64),
            "quality": np.zeros((num_lat, num_lon, num_imms)).astype(np.float64),
            "bps_fnf": np.zeros((num_lat, num_lon, num_imms)).astype(np.uint8),
        }

    if l2b_product_type == L2B_OUTPUT_PRODUCT_AGB:
        # L2B AGB, in input has l2A GN
        return {
            "HH": np.zeros((num_lat, num_lon, num_imms)).astype(np.float64),
            "VH": np.zeros((num_lat, num_lon, num_imms)).astype(np.float64),
            "VV": np.zeros((num_lat, num_lon, num_imms)).astype(np.float64),
            "bps_fnf": np.zeros((num_lat, num_lon, num_imms)).astype(np.uint8),
            "local_incidence_angle": np.zeros((num_lat, num_lon, num_imms)).astype(np.float64),
            "temporal_date_sec85": np.zeros(num_imms, dtype=float),
        }


def _core_interpolator_2d(
    data_in,
    latitude_axis_in,
    longitude_axis_in,
    mesh_latitude_axis_out,
    mesh_longitude_axis_out,
    interp_method,
    fill_value=np.nan,
):
    interp2d_fun = RegularGridInterpolator(
        (
            latitude_axis_in,
            longitude_axis_in,
        ),
        data_in,
        method=interp_method,
        bounds_error=False,
        fill_value=fill_value,
    )

    return interp2d_fun((mesh_latitude_axis_out, mesh_longitude_axis_out))


def _get_l2a_cumulative_footprint(l2a_fh_products_list):
    # This is to be completed: cumulative because each L2a in input may have different coverage

    ne_lat_vec = []
    ne_lon_vec = []
    se_lat_vec = []
    se_lon_vec = []
    sw_lat_vec = []
    sw_lon_vec = []
    nw_lat_vec = []
    nw_lon_vec = []
    latlon_coverage = None
    for l2a_product in l2a_fh_products_list:
        # merge togeter all footprints, take the last one here for testing
        # footprint = [ne_lat, ne_lon, se_lat, se_lon, sw_lat, sw_lon, nw_lat, nw_lon]

        footprint_curr = l2a_product.main_ads_input_information.footprint

        ne_lat_vec.append(footprint_curr[0])
        ne_lon_vec.append(footprint_curr[1])
        se_lat_vec.append(footprint_curr[6])
        se_lon_vec.append(footprint_curr[7])
        sw_lat_vec.append(footprint_curr[4])
        sw_lon_vec.append(footprint_curr[5])
        nw_lat_vec.append(footprint_curr[2])
        nw_lon_vec.append(footprint_curr[3])

    # [lat_min, lat_max, lon_min, lon_max]
    latlon_coverage = [
        np.min([ne_lat_vec, se_lat_vec, sw_lat_vec, nw_lat_vec]),
        np.max([ne_lat_vec, se_lat_vec, sw_lat_vec, nw_lat_vec]),
        np.min([ne_lon_vec, se_lon_vec, sw_lon_vec, nw_lon_vec]),
        np.max([ne_lon_vec, se_lon_vec, sw_lon_vec, nw_lon_vec]),
    ]

    return latlon_coverage


def _check_coverage(
    latlon_coverage: list[float],  # [lat_min, lat_max, lon_min, lon_max]
    mesh_latitude_axis_out: np.ndarray,
    mesh_longitude_axis_out: np.ndarray,
    minimum_l2a_coverage: float | None = None,
) -> tuple[bool | None, float, float]:
    l2a_cumulative_latitude_axis = np.linspace(latlon_coverage[0], latlon_coverage[1], 1000)
    l2a_cumulative_longitude_axis = np.linspace(latlon_coverage[2], latlon_coverage[3], 1000)

    test_data = _core_interpolator_2d(
        np.uint8(np.ones((len(l2a_cumulative_latitude_axis), len(l2a_cumulative_longitude_axis)))),
        l2a_cumulative_latitude_axis,
        l2a_cumulative_longitude_axis,
        mesh_latitude_axis_out,
        mesh_longitude_axis_out,
        interp_method="nearest",
    )
    valid_pixels = np.sum(test_data == np.uint8(1))
    total_pixels = test_data.shape[0] * test_data.shape[1]

    coverage_percentage = 100.0 * valid_pixels / total_pixels

    if minimum_l2a_coverage is None:
        coverage_is_enough = None
    elif coverage_percentage > minimum_l2a_coverage:
        coverage_is_enough = True
        bps_logger.info("        checking coverage of L2a into DGG L2b Tile:")
        bps_logger.info(
            f"            L2b Tile covering of {coverage_percentage:2.1f} % is greater than minimum L2a coverage of {minimum_l2a_coverage:2.1f} %"
        )
    else:
        coverage_is_enough = False
        bps_logger.info("        checking coverage of L2a into DGG L2b Tile:")
        bps_logger.info(
            f"            L2b Tile covering of {coverage_percentage:2.1f} % is not enough (minimum L2a coverage is {minimum_l2a_coverage:2.1f} %)"
        )

    return coverage_is_enough, coverage_percentage, valid_pixels


def _get_latitude_band_key(lat_min, lat_max):
    if lat_min >= 60 or lat_max <= -60:
        return "60-70"
    elif (lat_min < 60 and lat_max > 60) or (lat_max < -60 and lat_min > -60):
        return "60-70"
    elif lat_min >= 50 or lat_max <= -50:
        return "50-60"
    elif (lat_min < 50 and lat_max > 50) or (lat_max < -50 and lat_min > -50):
        return "50-60"
    elif lat_min >= -50 or lat_max <= 50:
        return "0-50"


def compute_l2a_contributing_heat_map(data_3d_mat_dict, key):
    acquisition_id_image = np.ones(
        (data_3d_mat_dict[key].shape),
        dtype=np.uint8,
    )
    for idx in range(data_3d_mat_dict[key].shape[2]):
        acquisition_id_image[:, :, idx][np.isnan(data_3d_mat_dict[key][:, :, idx])] = 0
    return acquisition_id_image


def sort_footprints(footprint_list):
    out_list = []

    for idN, fp in enumerate(footprint_list):
        fp2 = np.copy(fp)

        # find midpoint

        mx_FP = np.mean(fp[1::2])
        my_FP = np.mean(fp[0::2])

        dirs = np.arctan2(fp[0::2] - my_FP, fp[1::2] - mx_FP) * 180.0 / np.pi
        ind = np.argsort(-dirs)
        dirs = np.take_along_axis(dirs, ind, None)
        pivot = np.argmin(np.abs(dirs - 45))
        ind = np.roll(ind, -pivot)

        fp2[0::2] = np.take_along_axis(fp2[0::2], ind, None)
        fp2[1::2] = np.take_along_axis(fp2[1::2], ind, None)

        out_list.append(fp2)

    return out_list
