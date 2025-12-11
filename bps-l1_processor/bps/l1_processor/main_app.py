# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Main app entry point
--------------------
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

from arepyextras.runner import Environment
from bps.common import bps_logger
from bps.common.decorators import log_elapsed_time
from bps.l1_processor import BPS_L1_PROCESSOR_NAME
from bps.l1_processor.check_l1_device_resources import (
    raise_if_resources_are_not_enough_for_l1_processing,
)
from bps.l1_processor.folder_layout import FolderLayout
from bps.l1_processor.processor_interface.joborder_l1 import L1JobOrder
from bps.l1_processor.restart import optional_data_recovery
from bps.l1_processor.run_l1_processor import run_l1_processing_impl
from bps.l1_processor.settings.l1_intermediates import retrieve_additional_core_outputs


@log_elapsed_time(BPS_L1_PROCESSOR_NAME)
def run_l1_processing(
    job_order: L1JobOrder,
    working_dir: Path,
    iers_bulletin_file: Path,
):
    """Performs processing as described in job order.

    Parameters
    ----------
    job_order : L1JobOrder
        job_order object
    working_dir : Path
        Working directory: the directory where the orchestrator will write the internal input files.
    iers_bulletin_file : Path
        path to the iers bulletin file
    """

    # Update running environment: threads from JO and Path from calling env
    env = Environment(working_dir)
    env.setenv("OMP_NUM_THREADS", str(job_order.device_resources.num_threads))
    env.import_env_variable("PATH", is_list=True)
    env.import_env_variable("LD_LIBRARY_PATH", is_list=True)
    env.import_env_variable("PROJ_DATA", is_list=False)
    bps_logger.info(f"Using {env.getenv('OMP_NUM_THREADS')} threads")

    raise_if_resources_are_not_enough_for_l1_processing(provided_resources=job_order.device_resources)

    # Setup working directory layout
    layout = FolderLayout.from_base_dir(working_dir)

    # Update core processor output in layout, based on JO intermediates request
    if job_order.keep_intermediate():
        layout.core_processor_outputs.output_products.update(retrieve_additional_core_outputs(working_dir))

    # Setup recovery dir
    recovery_dir = None
    if job_order.device_resources.ramdisk_mount_point is not None and job_order.keep_intermediate():
        recovery_dir = job_order.intermediate_data_dir

    with optional_data_recovery(recovery_dir, working_dir):
        run_l1_processing_impl(env, job_order, layout, iers_bulletin_file)
