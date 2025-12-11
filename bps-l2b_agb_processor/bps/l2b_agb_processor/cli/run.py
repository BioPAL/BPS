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

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import click
from bps.common import bps_logger
from bps.l2b_agb_processor import BPS_L2B_AGB_PROCESSOR_NAME, commands
from bps.l2b_agb_processor import __version__ as VERSION


@contextmanager
def working_directory(path: Path):
    """Run code in path, restore previous dir on exit

    Parameters
    ----------
    path : Path
        working directory
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@click.command()
@click.argument("job_order_file", nargs=1, type=click.Path(exists=True), required=True)
@click.option(
    "--working-dir",
    nargs=1,
    type=click.Path(),
    required=False,
    help="Working directory (optional)",
)
@click.version_option(VERSION, help="Show processor version and exit")
@click.help_option()
def cli(
    job_order_file: str,
    working_dir: str | None,
):
    """BIOMASS Processing Suite - L2b AGB Processor

    Starts processing from Job Order file
    """
    job_order_path = Path(job_order_file).absolute()

    if working_dir is None:
        working_dir = tempfile.mkdtemp(prefix="", dir=job_order_path.parent)

    working_dir_path = Path(working_dir).absolute()
    working_dir_path.mkdir(exist_ok=True, parents=True)

    bps_logger.init_logger(processor="L2B_AGB_P", task="L2B_AGB_P", version=VERSION)
    bps_logger.enable_console_logging()
    bps_logger.enable_file_logging(working_dir_path)

    bps_logger.info("%s started", BPS_L2B_AGB_PROCESSOR_NAME)
    bps_logger.info("Working directory: %s", working_dir_path)
    bps_logger.info("Job order file: %s", job_order_path)

    with working_directory(working_dir_path):
        commands.run_l2b_agb_processing(job_order_path, working_dir_path)
