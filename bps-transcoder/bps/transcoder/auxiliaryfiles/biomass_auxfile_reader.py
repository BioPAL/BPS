# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import os

from bps.transcoder.auxiliaryfiles.biomass_auxfile import (
    BIOMASSAuxFile,
    BIOMASSAuxFileStructure,
)


class BIOMASSAuxFileReader:
    def __init__(self, product_path) -> None:
        self.product_path = product_path

        self.product = BIOMASSAuxFile()
        self.product_structure = BIOMASSAuxFileStructure(self.product_path)

    def read(self):
        self.product.name = os.path.basename(self.product_path)

        return self.product
