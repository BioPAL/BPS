# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest

from bps.transcoder.sarproduct.biomass_l1product import BIOMASSL1Product


class BiomassL1ProductTestCase(unittest.TestCase):
    def test_minimal_instance(self):
        l1_product = BIOMASSL1Product(
            product=None,
            is_monitoring=False,
            source=None,
            configuration=None,
        )

        assert l1_product.name is not None
        assert l1_product.name[:-6] == "BIO_S1_SLC__1S_20250101T000000Z_20250101T000025Z_I_G01_M01_C01_T001_F001_01_"


if __name__ == "__main__":
    unittest.main()
