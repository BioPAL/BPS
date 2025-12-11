# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Command line interface - main
-----------------------------
"""

import importlib.resources
from enum import Enum, auto
from pathlib import Path

import bps.l1_processor
import click
from bps.common import bps_logger
from bps.common.processor_init import get_intermediate_data_dir_name, working_directory
from bps.common.translate_job_order import get_bps_logger_level
from bps.l1_processor import BPS_L1_PROCESSOR_NAME, main_app
from bps.l1_processor import __version__ as VERSION
from bps.l1_processor.core.parsing import parse_l1_job_order
from bps.l1_processor.processor_interface.joborder_l1 import L1JobOrder


def get_bulletin_path() -> Path:
    """Get default bulletin position inside the installed package"""
    # pylint: disable-next=too-many-function-args
    return importlib.resources.files(bps.l1_processor).joinpath("resources", "bulletinb-348.txt")


class WorkingDirSource(Enum):
    """Reason a working dir was chosen"""

    CLI_OPTION = auto()
    RAMDISK = auto()
    INTERMEDIATE = auto()
    DEFAULT = auto()


def select_working_dir(
    working_dir: str | None, job_order: L1JobOrder, job_order_path: Path
) -> tuple[Path, WorkingDirSource]:
    """Select working dir"""
    # Working dir precedence
    # 1. command line option
    # 2. ramdisk mount point                                          - default for docker run
    # 3. intermediate_data_dir specified in jo                        - default for debug run
    # 4. a default intermediate_data_dir inside joborder directory    - default behavior for bundle run
    if working_dir is not None:
        return job_order_path.parent.joinpath(working_dir), WorkingDirSource.CLI_OPTION

    if job_order.device_resources.ramdisk_mount_point is not None:
        return job_order.device_resources.ramdisk_mount_point, WorkingDirSource.RAMDISK

    if job_order.intermediate_data_dir is not None and job_order.processor_configuration.keep_intermediate:
        return (
            job_order_path.parent.joinpath(job_order.intermediate_data_dir),
            WorkingDirSource.INTERMEDIATE,
        )

    return (
        get_intermediate_data_dir_name(job_order_path, add_data_dir=True),
        WorkingDirSource.DEFAULT,
    )


def log_working_dir_reason(reason: WorkingDirSource):
    """Log working dir selection decision process"""
    if reason == WorkingDirSource.CLI_OPTION:
        bps_logger.info("Intermediate data directory specified via command line option")
    elif reason == WorkingDirSource.RAMDISK:
        bps_logger.debug("RAMDISK specified via Job Order, using it as intermediate data directory")
    elif reason == WorkingDirSource.INTERMEDIATE:
        bps_logger.info("Using intermediate data directory specified via Job Order")
    else:
        assert reason == WorkingDirSource.DEFAULT


@click.command()
@click.argument(
    "job_order_file",
    nargs=1,
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--working-dir",
    nargs=1,
    type=click.Path(),
    required=False,
    help="Working directory (optional)",
)
@click.version_option(VERSION, help="Show processor version and exit")
@click.help_option()
def run(job_order_file: str, working_dir: str | None):
    """BIOMASS Processing Suite - L1 Processor

    Starts processing from Job Order file
    """
    bps_logger.init_logger(processor="L1_P", task="L1_P", version=VERSION)
    bps_logger.enable_console_logging()

    job_order_path = Path(job_order_file).absolute()
    job_order = parse_l1_job_order(job_order_path.read_text(encoding="utf-8"))

    # Relative paths in the joborder are intended as relative to the directory where the JobOrder is
    job_order_dir = job_order_path.parent
    job_order.io_products.output.output_directory = job_order_dir.joinpath(
        job_order.io_products.output.output_directory
    )
    if job_order.intermediate_data_dir is not None:
        job_order.intermediate_data_dir = job_order_dir.joinpath(job_order.intermediate_data_dir)

    working_dir_path, reason = select_working_dir(working_dir, job_order, job_order_path)

    working_dir_path = working_dir_path.absolute()
    working_dir_path.mkdir(exist_ok=True, parents=True)

    # Logger setup
    bps_logger.enable_file_logging(working_dir_path)

    # Update logger with levels set in JO
    log_level = get_bps_logger_level(
        job_order.processor_configuration.stdout_log_level,
        job_order.processor_configuration.stderr_log_level,
    )
    bps_logger.update_logger(loglevel=log_level)

    bps_logger.info(f"{BPS_L1_PROCESSOR_NAME} started")
    bps_logger.info(f"Job order file: {job_order_path}")

    if job_order.device_resources.ramdisk_mount_point is None:
        bps_logger.warning("RAMDISK not specified via Job Order")

    log_working_dir_reason(reason)

    bps_logger.info(f"Intermediate data directory: {working_dir_path}")

    if job_order.keep_intermediate():
        bps_logger.info("Intermediate data saving enabled")
    else:
        bps_logger.info("Intermediate data saving disabled")

    iers_bulletin = get_bulletin_path()
    assert iers_bulletin.exists()

    with working_directory(working_dir_path):
        main_app.run_l1_processing(job_order, working_dir_path, iers_bulletin)
