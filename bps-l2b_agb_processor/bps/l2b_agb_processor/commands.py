# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Commands
--------
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
from bps.common import (
    bps_logger,
    retrieve_aux_product_data_content,
    retrieve_aux_product_data_single_content,
)
from bps.common.lcm_utils import read_lcm_mask, retrieve_aux_lcm_content
from bps.common.translate_job_order import get_bps_logger_level
from bps.l2b_agb_processor import BPS_L2B_AGB_PROCESSOR_NAME
from bps.l2b_agb_processor.agb.agb_commands import AGBL2B, get_5x5_tiles_ids
from bps.l2b_agb_processor.core.parsing import parse_aux_pp2_2b_agb, parse_l2b_job_order
from bps.l2b_agb_processor.core.translate_job_order import (
    L2A_PRODUCT_GN,
    L2B_PRODUCT_AGB,
    L2B_PRODUCT_FD,
)
from bps.l2b_agb_processor.l2b_common_functionalities import l2a_temporal_sorting
from bps.l2b_agb_processor.processor_interface.xsd_validation import (
    validate_aux_pp2_agb,
)
from bps.transcoder.sarproduct.biomass_l2aproduct_reader import BIOMASSL2aProductReader
from bps.transcoder.sarproduct.biomass_l2bagbproduct_reader import (
    BIOMASSL2bAGBProductReader,
)
from bps.transcoder.sarproduct.biomass_l2bagbproduct_writer import FLOAT_NODATA_VALUE
from bps.transcoder.sarproduct.biomass_l2bfdproduct_reader import (
    BIOMASSL2bFDProductReader,
)
from bps.transcoder.sarproduct.mph import MPH_NAMESPACES, read_coverage_and_footprint
from osgeo import gdal

# The regex to match the AUX-FNF product and the time stamps.
AUX_CAL_AB_TIMESTAMP_REGEX = "[0-9]{8}T[0-9]{6}"
AUX_CAL_AB_PRODUCT_REGEX = f"^BIO_AUX_(.)*CAL_AB_{AUX_CAL_AB_TIMESTAMP_REGEX}_{AUX_CAL_AB_TIMESTAMP_REGEX}_"


def run_l2b_agb_processing(
    job_order_file: Path,
    working_dir: Path,
):
    """Performs processing as described in job order.

    Parameters
    ----------
    job_order_file : Path
        job_order xml file
    working_dir : Path
        Working directory: the directory where the orchestrator will write the internal input files.
    """

    assert job_order_file.exists()
    assert working_dir.exists()
    assert job_order_file.is_absolute()
    assert working_dir.is_absolute()

    processing_start_time = datetime.now()

    # Input parsing: joborder
    job_order = parse_l2b_job_order(job_order_file.read_text())

    # Relative paths in the joborder are intended as relative to the directory where the JobOrder is
    job_order_dir = job_order_file.parent
    if not job_order.output_directory.is_absolute():
        job_order.output_directory = job_order_dir.joinpath(job_order.output_directory)

    # Update logging level
    log_level = get_bps_logger_level(
        job_order.processor_configuration.stdout_log_level,
        job_order.processor_configuration.stderr_log_level,
    )
    bps_logger.update_logger(loglevel=log_level)

    # Input parsing: aux_pp2_ab
    bps_logger.info("AUX PP2 AB configuration: %s", job_order.aux_pp2_agb_path)
    aux_pp2_ab_file = retrieve_aux_product_data_single_content(job_order.aux_pp2_agb_path)
    validate_aux_pp2_agb(aux_pp2_ab_file)
    aux_pp2_ab = parse_aux_pp2_2b_agb(aux_pp2_ab_file.read_text())

    # Sorting in time (from older to newer) the paths of input l2a products
    bps_logger.info(f"found #{len(job_order.input_l2a_products)} L2a GN products")
    ordered_l2a_products_paths = l2a_temporal_sorting(job_order.input_l2a_products)

    # Read input L2a products
    l2a_gn_products_list = []
    for l2a_path in ordered_l2a_products_paths:
        l2a_product = BIOMASSL2aProductReader(l2a_path).read()
        assert l2a_product is not None
        if l2a_product.product_type not in L2A_PRODUCT_GN:
            bps_logger.error(
                f"    Biomass L2b AGB Processor: all input L2A products from job order should be \
                    GN L2a products {L2A_PRODUCT_GN}, found not valid product: {l2a_product.product_type}"
            )

        # Convert no data values to nan, for the processing
        l2a_product.lut_ads.lut_local_incidence_angle = np.where(
            l2a_product.lut_ads.lut_local_incidence_angle == FLOAT_NODATA_VALUE,
            np.nan,
            l2a_product.lut_ads.lut_local_incidence_angle,
        )

        l2a_gn_products_list.append(l2a_product)

    # Work In Progress
    # Fast MPH parsing to get input L1c stack cumulative footprint
    latlon_coverage, _ = read_coverage_and_footprint(list(job_order.input_l2a_mph_files))

    # Read input LCM, basing on footprint
    bps_logger.info(f"Reading input LCM from {job_order.lcm_product}")
    # The LCM mask must be read for all the 5x5 tiles coverage, use a gap of 2.5 degrees in each direction should be enough
    gap_degrees = 2.5
    latlon_coverage_enlarged = np.copy(latlon_coverage)
    latlon_coverage_enlarged[0] = latlon_coverage_enlarged[0] - gap_degrees
    latlon_coverage_enlarged[1] = latlon_coverage_enlarged[1] + gap_degrees
    latlon_coverage_enlarged[2] = latlon_coverage_enlarged[2] - gap_degrees
    latlon_coverage_enlarged[3] = latlon_coverage_enlarged[3] + gap_degrees
    lcm_product = read_lcm_mask(
        retrieve_aux_lcm_content(job_order.lcm_product),
        latlon_coverage_enlarged,
        units="deg",
    )

    tile_ids_5x5 = get_5x5_tiles_ids(job_order.processing_parameters.tile_id)
    bps_logger.info(f"Reading the input CAL_AB from {job_order.cal_ab_product}")
    bps_logger.info(
        f"Only the CAL_AB coovering the tiles in the 5x5 neighborhood of central tile {job_order.processing_parameters.tile_id}, will be read"
    )
    cal_agb_product = read_cal_ab(retrieve_aux_cal_ab_content(job_order.cal_ab_product), tile_ids_5x5)

    l2b_agb_product_list = []  # only second iteration
    if job_order.input_l2b_agb_products:
        bps_logger.info(f"found #{len(job_order.input_l2b_agb_products)} optional L2b AGB products")
        for l2b_agb_path in job_order.input_l2b_agb_products:
            l2b_agb_product = BIOMASSL2bAGBProductReader(l2b_agb_path).read()
            assert l2b_agb_product is not None
            if l2b_agb_product.product_type not in L2B_PRODUCT_AGB:
                bps_logger.error(
                    f"    Biomass L2b AGB Processor: all optional input L2B AGB products from job order should be \
                        AGB L2B products {L2B_PRODUCT_AGB}, found not valid product: {l2b_agb_product.product_type}"
                )
            l2b_agb_product_list.append(l2b_agb_product)
    else:
        l2b_agb_product_list = None

    l2b_fd_product_list = []
    if job_order.input_l2b_fd_products:
        bps_logger.info(f"found #{len(job_order.input_l2b_fd_products)} optional L2b FD products")
        for l2b_fd_path in job_order.input_l2b_fd_products:
            l2b_fd_product = BIOMASSL2bFDProductReader(l2b_fd_path).read()
            assert l2b_fd_product is not None
            if l2b_fd_product.product_type not in L2B_PRODUCT_FD:
                bps_logger.error(
                    f"    Biomass L2b AGB Processor: all optional input L2B FD products from job order should be \
                        AGB L2B products {L2B_PRODUCT_FD}, found not valid product: {l2b_fd_product.product_type}"
                )
            l2b_fd_product_list.append(l2b_fd_product)
    else:
        l2b_fd_product_list = None

    # Call Above Ground Biomass L2B processor
    agb_obj = AGBL2B(
        job_order,
        aux_pp2_ab,
        working_dir,
        l2a_gn_products_list,
        lcm_product,
        cal_agb_product,
        l2b_agb_product_list,
        l2b_fd_product_list,
    )

    agb_obj.run_l2b_agb_processing()

    processing_stop_time = datetime.now()
    elapsed_time = processing_stop_time - processing_start_time
    bps_logger.info(
        "%s total processing time: %.3f s",
        BPS_L2B_AGB_PROCESSOR_NAME,
        elapsed_time.total_seconds(),
    )


def retrieve_aux_cal_ab_content(cal_ab_entry_point_path: Path) -> tuple[Path, ...]:
    """
    Retrieve the CAL_AB object from an entry point path. The entry point path
    is a path to the AUX-CAL_AB product directory

    Parameters
    ----------
    cal_ab_entry_point_path : Path
        The AUX-CAL_AB entry point.

    Returns
    -------
    Tuple[Path, ...]:
        The path(s) to all the CAL_AB tifs, for the desired tiles.


    """
    if not cal_ab_entry_point_path.exists():
        raise ValueError(f"Specified AUX-CAL_AB entry point {cal_ab_entry_point_path} does not exist")
    if not cal_ab_entry_point_path.is_dir():
        raise ValueError(f"Specified AUX-CAL_AB entry point {cal_ab_entry_point_path} is not valid")

    # If the input folder is an AUX-FNF product, return the tiff file in it.
    if re.match(AUX_CAL_AB_PRODUCT_REGEX, cal_ab_entry_point_path.name):
        return retrieve_aux_product_data_content(cal_ab_entry_point_path)
    else:
        raise ValueError(f"Specified AUX-CAL_AB entry point {cal_ab_entry_point_path} is not valid")


def tile_ids_from_mph(mph_paths_list: list[Path]) -> list[str]:
    """Fast read the L2a MPH file, without passing through the whole product reading

    Parameters
    ----------

    Returns
    -------

    Raises
    ------

    """
    tile_ids = []

    for mph_path in mph_paths_list:
        root = ET.parse(mph_path).getroot()

        xpath_posList = (
            ".//om:procedure/eop:EarthObservationEquipment/eop:acquisitionParameters/bio:Acquisition/bio:tileID"
        )
        tile_id_node = root.find(xpath_posList, MPH_NAMESPACES)
        assert tile_id_node is not None and tile_id_node.text is not None
        tile_ids.append(tile_id_node.text)
    return list(np.unique(tile_ids))


def read_cal_ab(cal_ab_tif_paths: list[Path], tile_ids: list[str]):
    cal_agb_product_dict = {}
    for cal_ab_tif_path in cal_ab_tif_paths:
        if cal_ab_tif_path.suffix == ".xml":
            continue
        cal_tile_id = cal_ab_tif_path.name[7:-4]
        if cal_tile_id in tile_ids:
            cal_agb_product_dict[cal_tile_id] = {}

            driver = gdal.Open(str(cal_ab_tif_path), 0)
            cal_agb = np.float32(driver.GetRasterBand(1).ReadAsArray())
            cal_agb_se = np.float32(driver.GetRasterBand(2).ReadAsArray())
            cal_agb_geotransform = driver.GetGeoTransform()
            cal_agb_product_dict[cal_tile_id]["cal_ab"] = np.stack([cal_agb, cal_agb_se], axis=2)

            lon_axis_deg = cal_agb_geotransform[0] + cal_agb_geotransform[1] * np.arange(driver.RasterXSize)
            lat_axis_deg = cal_agb_geotransform[3] + cal_agb_geotransform[5] * np.arange(driver.RasterYSize)
            cal_agb_product_dict[cal_tile_id]["dgg_lat_axis_deg"] = lat_axis_deg
            cal_agb_product_dict[cal_tile_id]["dgg_lon_axis_deg"] = lon_axis_deg

    if not cal_agb_product_dict:
        bps_logger.error(
            "Cannot find any CAL_AB inside the 5x5 neighborhood of the central tile specified in job order"
        )

    else:
        bps_logger.info(f"Found Cal_AB products covering following tiles: {list(cal_agb_product_dict.keys())}")
    return cal_agb_product_dict
