# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""BIOMASS L1 product reader"""

from pathlib import Path

from bps.l1_geocoding_tool import NO_DATA_VALUE
from bps.transcoder.sarproduct.biomass_l1product_reader import BIOMASSL1ProductReader
from bps.transcoder.sarproduct.biomass_stackproduct_reader import BIOMASSStackProductReader
from bps.transcoder.sarproduct.sarproduct import SARProduct


def read_l1_product(l1_product_path: Path) -> SARProduct:
    """Read BIOMASS L1 product"""
    if not l1_product_path.exists():
        raise FileNotFoundError(f"Input L1 product does not exist: {l1_product_path}")
    if "_SCS__1S_" in l1_product_path.name:
        print("L1a product detected")
        product = BIOMASSL1ProductReader(
            product_path=l1_product_path, nodata_fill_value=NO_DATA_VALUE
        ).read_as_sarproduct()
    elif "_DGM__1S_" in l1_product_path.name:
        print("L1b product detected")
        product = BIOMASSL1ProductReader(
            product_path=l1_product_path, nodata_fill_value=NO_DATA_VALUE
        ).read_as_sarproduct()
    elif "_STA__1S_" in l1_product_path.name:
        print("L1c product detected")
        product = BIOMASSStackProductReader(
            product_path=l1_product_path, nodata_fill_value=NO_DATA_VALUE
        ).read_as_sarproduct()
    else:
        raise ValueError(f"Input L1 product is not a valid BIOMASS L1 product: {l1_product_path}")

    return product
