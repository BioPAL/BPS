# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Command line interface
----------------------
"""

import click
from bps.stack_processor import __version__ as VERSION
from bps.stack_processor import run_stack_processor


@click.command()
@click.argument(
    "job_order_file",
    nargs=1,
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--working-dir",
    nargs=1,
    type=click.Path(),
    required=False,
    help="Working directory (optional)",
)
@click.version_option(
    VERSION,
    help="Show processor version and exit",
)
@click.help_option()
def cli(
    job_order_file: str,
    working_dir: str | None,
):
    """
    BIOMASS Processing Suite - Stack Processor
    ------------------------------------------

    Start processing a from Job Order file.

    """
    run_stack_processor.stack_processor_main(job_order_file, working_dir)
