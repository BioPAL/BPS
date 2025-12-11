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

from datetime import datetime
from pathlib import Path

import numpy as np
from bps.common import bps_logger, retrieve_aux_product_data_single_content
from bps.common.translate_job_order import get_bps_logger_level
from bps.l2b_fh_processor import BPS_L2B_FH_PROCESSOR_NAME
from bps.l2b_fh_processor.core.parsing import parse_aux_pp2_2b_fh, parse_l2b_job_order
from bps.l2b_fh_processor.core.translate_job_order import L2A_PRODUCT_FH
from bps.l2b_fh_processor.fh.fh_commands import FHL2B
from bps.l2b_fh_processor.l2b_common_functionalities import l2a_temporal_sorting
from bps.l2b_fh_processor.processor_interface.xsd_validation import validate_aux_pp2_fh
from bps.transcoder.sarproduct.biomass_l2aproduct_reader import BIOMASSL2aProductReader
from bps.transcoder.sarproduct.biomass_l2bfdproduct_reader import (
    BIOMASSL2bFDProductReader,
)
from bps.transcoder.sarproduct.biomass_l2bfhproduct_writer import FLOAT_NODATA_VALUE


def run_l2b_fh_processing(
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

    # Input parsing: aux_pp2_fh
    bps_logger.info("AUX PP2 FH configuration: %s", job_order.aux_pp2_fh_path)
    aux_pp2_fh_file = retrieve_aux_product_data_single_content(job_order.aux_pp2_fh_path)
    validate_aux_pp2_fh(aux_pp2_fh_file)
    aux_pp2_fh = parse_aux_pp2_2b_fh(aux_pp2_fh_file.read_text())

    # Sorting in time (from older to newer) the paths of input l2a products
    if "_FH__L2A" in str(job_order.input_l2a_products[0]):
        bps_logger.info(f"found #{len(job_order.input_l2a_products)} L2a FH products")
    elif "_TFH_L2A" in str(job_order.input_l2a_products[0]):
        bps_logger.info(f"found #{len(job_order.input_l2a_products)} L2a TFH products")
    ordered_l2a_products_paths = l2a_temporal_sorting(job_order.input_l2a_products)

    # Read input L2a products
    l2a_products_list = []
    for l2a_path in ordered_l2a_products_paths:
        l2a_product = BIOMASSL2aProductReader(l2a_path).read()
        assert l2a_product is not None
        if l2a_product.product_type not in L2A_PRODUCT_FH:
            bps_logger.error(
                f"    Biomass L2b FH Processor: all input L2A products from job order should be \
                    FH L2a products {L2A_PRODUCT_FH}, found not valid product: {l2a_product.product_type}"
            )

        # Convert no data values to nan, for the processing
        for key, data in l2a_product.measurement.data_dict.items():
            nan_mask = data == FLOAT_NODATA_VALUE
            l2a_product.measurement.data_dict[key][nan_mask] = np.nan

        l2a_products_list.append(l2a_product)

    # Read input L2b FD product (optional)
    if job_order.input_l2b_fd_product:
        bps_logger.info(
            "L2b FD additional product is specified in Job Order, %s :",
            job_order.input_l2b_fd_product,
        )

        l2b_fd_product = BIOMASSL2bFDProductReader(job_order.input_l2b_fd_product).read()
        if job_order.processing_parameters.tile_id not in l2b_fd_product.main_ads_product.tile_id_list:
            raise ValueError(
                f"L2b FD specified in job order covers {l2b_fd_product.main_ads_product.tile_id_list[0]} tile ID, while job order specifies to process '{job_order.processing_parameters.tile_id}' tile ID instead."
            )

    else:
        bps_logger.info(
            "Optional L2b FD additional product not specified in Job Order",
        )
        l2b_fd_product = None

    # Call Forest Height L2B processor
    fh_obj = FHL2B(
        job_order,
        aux_pp2_fh,
        working_dir,
        l2a_products_list,
        l2b_fd_product,
    )
    fh_obj.run_l2b_fh_processing()

    processing_stop_time = datetime.now()
    elapsed_time = processing_stop_time - processing_start_time
    bps_logger.info(
        "%s total processing time: %.3f s",
        BPS_L2B_FH_PROCESSOR_NAME,
        elapsed_time.total_seconds(),
    )
