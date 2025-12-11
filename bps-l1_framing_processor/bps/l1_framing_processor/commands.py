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

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from bps.common import bps_logger
from bps.common.common import retrieve_aux_product_data_single_content
from bps.common.translate_job_order import get_bps_logger_level
from bps.l1_framing_processor import BPS_L1_FRAMING_PROCESSOR_NAME
from bps.l1_framing_processor.check_l1f_device_resources import (
    raise_if_resources_are_not_enough_for_l1f_processing,
)
from bps.l1_framing_processor.core.parsing import parse_l1_job_order
from bps.l1_framing_processor.l1_framer.l1_framer import L1Framer
from bps.l1_framing_processor.l1_vfra_product.l1_vfra_product import L1VFRAProduct
from bps.l1_framing_processor.utils.constants import (
    L0_SENSING_TIME_MARGIN,
    L0_THEORETICAL_TIME_MARGIN,
    L1_PROCESSING_MARGIN,
    MINIMUM_DATATAKE_LENGTH,
    MINIMUM_SLICE_LENGTH,
    SLICE_OVERLAP,
)
from bps.transcoder.sarproduct.biomass_l0product_reader import read_l0_product


def run_l1_framing_processing(
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
    job_order = parse_l1_job_order(job_order_file.read_text())

    raise_if_resources_are_not_enough_for_l1f_processing(provided_resources=job_order.device_resources)

    # Relative paths in the joborder are intended as relative to the directory where the JobOrder is
    job_order_dir = job_order_file.parent
    if not job_order.io_products.output.output_directory.is_absolute():
        job_order.io_products.output.output_directory = job_order_dir.joinpath(
            job_order.io_products.output.output_directory
        )

    # Update logging level
    log_level = get_bps_logger_level(
        job_order.processor_configuration.stdout_log_level,
        job_order.processor_configuration.stderr_log_level,
    )
    bps_logger.update_logger(loglevel=log_level)

    # Run l1 framing processor
    bps_logger.info("Ingest input L0S, L0M and AUX_ORB products")
    l0s_path = job_order.io_products.input.input_standard
    l0m_path = job_order.io_products.input.input_monitoring
    aux_orb_path = job_order.auxiliary_files.orbit
    output_path = job_order.io_products.output.output_directory

    l0s = read_l0_product(l0s_path)
    l0s_start_time = np.datetime64(l0s.start_time.isoformat()[0:29])
    l0s_stop_time = np.datetime64(l0s.stop_time.isoformat()[0:29])
    l0s_sensing_start_time = np.datetime64(l0s.sensing_start_time.isoformat()[0:29])
    l0s_sensing_stop_time = np.datetime64(l0s.sensing_stop_time.isoformat()[0:29])
    l0m = read_l0_product(l0m_path)
    l0m_start_time = np.datetime64(l0m.start_time.isoformat()[0:29])
    l0m_stop_time = np.datetime64(l0m.stop_time.isoformat()[0:29])

    delta = np.timedelta64(L0_SENSING_TIME_MARGIN, "s")
    if (l0m_start_time - l0s_sensing_start_time) > delta or (l0s_sensing_stop_time - l0m_stop_time) > delta:
        bps_logger.warning(
            "Difference between L0M and L0S start/stop times above threshold. "
            "Assuming L0M time validity covers L0S one for computations"
        )
        l0m_start_time = l0s_sensing_start_time
        l0m_stop_time = l0s_sensing_stop_time
    l0m_duration = np.abs(l0m_stop_time - l0m_start_time)
    short_datatake_flag = True if l0m_duration < np.timedelta64(MINIMUM_DATATAKE_LENGTH, "s") else False

    orbit_path = str(retrieve_aux_product_data_single_content(aux_orb_path))

    bps_logger.info("Compute L1 frames contained in input L0 slice")
    l1framer = L1Framer(orbit_path, l0m_start_time, l0m_stop_time)
    l1framer.get_frames_in_datatake()
    frames_in_slice_list = l1framer.get_frames_in_slice(
        l0s_start_time,
        l0s_stop_time,
        l0s_sensing_start_time,
        l0s_sensing_stop_time,
        short_datatake_flag,
        merge_short_frames_flag=True,
        add_l1processing_margins_flag=True,
    )
    if frames_in_slice_list is None:
        raise RuntimeError("Input AUX_ORB product not fully overlapping with input L0S product")

    if l0m_stop_time - l0s_sensing_stop_time > np.timedelta64(L0_THEORETICAL_TIME_MARGIN, "ms"):
        if l0m_stop_time - (l0s_sensing_stop_time - np.timedelta64(SLICE_OVERLAP, "s")) > np.timedelta64(
            MINIMUM_SLICE_LENGTH, "s"
        ):
            bps_logger.info("Remove L1 frames contained in next L0 slice")
            l1framer_next_slice = L1Framer(orbit_path, l0m_start_time, l0m_stop_time)
            l1framer_next_slice.get_frames_in_datatake()
            frames_in_next_slice_list = l1framer_next_slice.get_frames_in_slice(
                l0s_stop_time - np.timedelta64(SLICE_OVERLAP, "s"),
                l0m_stop_time,
                l0s_sensing_stop_time - np.timedelta64(SLICE_OVERLAP, "s"),
                l0m_stop_time,
                short_datatake_flag=False,
                merge_short_frames_flag=True,
                add_l1processing_margins_flag=True,
            )
            if frames_in_next_slice_list is not None:
                if len(frames_in_slice_list) > 1 and len(frames_in_next_slice_list) > 0:
                    if (
                        frames_in_slice_list[-2].stop_time
                        - frames_in_next_slice_list[0].start_time
                        - np.timedelta64(L1_PROCESSING_MARGIN * 2, "s")
                        > 0
                    ):
                        _ = frames_in_slice_list.pop()

    if len(frames_in_slice_list) == 0:
        bps_logger.info("No L1 frames contained in input L0 slice")
    else:
        bps_logger.info("Write output L1 Virtual Frame products")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for frame in frames_in_slice_list:
            bps_logger.debug(str(frame))
            l1vfraproduct = L1VFRAProduct(
                frame,
                file_class=job_order.processor_configuration.file_class,
                product_baseline=job_order.io_products.output.output_baseline,
            )
            l1vfraproduct.write_product(
                output_path,
                os.path.basename(l0s_path),
                os.path.basename(l0m_path),
                os.path.basename(aux_orb_path),
            )

    processing_stop_time = datetime.now()
    elapsed_time = processing_stop_time - processing_start_time
    bps_logger.info(
        "%s total processing time: %.3f s",
        BPS_L1_FRAMING_PROCESSOR_NAME,
        elapsed_time.total_seconds(),
    )
