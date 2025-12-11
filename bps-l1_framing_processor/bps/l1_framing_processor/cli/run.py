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
from bps.l1_framing_processor import (
    BPS_L1_FRAMING_PROCESSOR_ID,
    BPS_L1_FRAMING_PROCESSOR_NAME,
)
from bps.l1_framing_processor import __version__ as VERSION

version_option = click.version_option(VERSION, help="Show processor version and exit")


@click.command()
@click.argument("job_order_file", nargs=1, type=click.Path(exists=True), required=True)
@click.option(
    "--working-dir",
    nargs=1,
    type=click.Path(),
    required=False,
    help="Working directory (optional)",
)
@version_option
@click.help_option()
def cli(job_order_file: str, working_dir: str | None):
    """BIOMASS Processing Suite - L1 Framing Processor

    Starts processing from Job Order file
    """
    working_dir_path, job_order_path = processor_setup(
        job_order_file,
        working_dir,
        BPS_L1_FRAMING_PROCESSOR_NAME,
        BPS_L1_FRAMING_PROCESSOR_ID,
        VERSION,
    )

    with working_directory(working_dir_path):
        from bps.l1_framing_processor import commands

        commands.run_l1_framing_processing(job_order_path, working_dir_path)
