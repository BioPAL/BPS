# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""BIOMASS L1 Geocoding Tool CLI"""

from pathlib import Path

import click
from bps.l1_geocoding_tool import __version__ as VERSION
from bps.l1_geocoding_tool.main import run

InputProduct = click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path)
OutputFolder = click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True, path_type=Path)


@click.command()
@click.option("--l1", "l1_product_path", required=True, type=InputProduct, help="Input L1 product")
@click.option("--dem", "dem_path", required=True, type=InputProduct, help="Input DEM database")
@click.option(
    "--latlonstep", "lat_lon_step", required=False, type=str, help="Ouput latitude and longitude sampling step"
)
@click.option("--out", "output_path", required=True, type=OutputFolder, help="Output folder")
@click.version_option(VERSION, help="Show version and exit", prog_name="bps_l1_geocoding_tool")
def run_cli(l1_product_path: Path, dem_path: Path, lat_lon_step: str | None, output_path: Path):
    """BIOMASS L1 Geocoding Tool"""
    run(l1_product_path, dem_path, lat_lon_step, output_path)
