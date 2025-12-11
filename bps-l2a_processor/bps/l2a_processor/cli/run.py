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

import click
from bps.common.processor_init import processor_setup, working_directory
from bps.l2a_processor import BPS_L2A_PROCESSOR_NAME, commands
from bps.l2a_processor import __version__ as VERSION


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
    """BIOMASS Processing Suite - L2a Processor

    Starts processing from Job Order file
    """
    working_dir_path, job_order_path = processor_setup(
        job_order_file, working_dir, BPS_L2A_PROCESSOR_NAME, "L2A_P", VERSION
    )

    with working_directory(working_dir_path):
        commands.run_l2a_processing(job_order_path, working_dir_path)
