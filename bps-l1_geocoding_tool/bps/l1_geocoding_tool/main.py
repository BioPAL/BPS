# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Main module"""

from pathlib import Path
from time import time

from arepyextras.runner import Environment
from bps.l1_geocoding_tool import OMP_NUM_THREADS
from bps.l1_geocoding_tool.biomass_l1geoproduct_writer import write_l1_product_geocoded
from bps.l1_geocoding_tool.biomass_l1product_reader import read_l1_product
from bps.l1_geocoding_tool.geocoding_utils import geocode_product


def run(l1_product_path: Path, dem_path: Path, lat_lon_step: str | None, output_path: Path):
    """BIOMASS L1 Geocoding Tool"""
    env = Environment(output_path)
    env.setenv("OMP_NUM_THREADS", str(OMP_NUM_THREADS))
    env.import_env_variable("PATH", is_list=True)
    env.import_env_variable("LD_LIBRARY_PATH", is_list=True)

    t_start = time()
    print("\nBIOMASS L1 Geocoding Tool")
    print(f"\nGeocoding started (using {env.getenv('OMP_NUM_THREADS')} threads)")

    print(f"\nInput L1 product: {l1_product_path.name}")
    print(f"Input DEM database: {dem_path.name}")
    print(f"Latitude/Longitude step: {lat_lon_step} [deg]")
    print(f"Output folder: {output_path.name}")

    print("\nReading input L1 product")
    biomass_l1_product = read_l1_product(l1_product_path)

    print("\nGeocoding input L1 product")
    biomass_l1_product_geocoded = geocode_product(biomass_l1_product, dem_path, lat_lon_step, output_path, env)

    print("\nWriting output L1 product (geocoded)")
    write_l1_product_geocoded(biomass_l1_product_geocoded, output_path)

    t_stop = time()
    elapsed_time = t_stop - t_start
    print(f"\nGeocoding completed. Elapsed time: {elapsed_time:.3f}[s]")
