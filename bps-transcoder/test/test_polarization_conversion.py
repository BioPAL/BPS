# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest
from itertools import product

from arepytools.io.metadata import EPolarization
from bps.common.io.common_types import PolarisationType
from bps.transcoder.utils.polarization_conversions import (
    UnsupportedPolarizationError,
    translate_polarization,
    translate_polarization_tag,
)


class TranslatePolarizationTests(unittest.TestCase):
    def test_polarization_tag_translation(self):
        for poltag in map(
            lambda x: "".join(x),
            product(("h", "H"), ("", "/"), ("H", "h")),
        ):
            self.assertEqual(
                translate_polarization_tag(poltag, poltype="bps"),
                PolarisationType.HH,
            )

    def test_polarization_translation_roundtrip(self):
        self.assertEqual(self.__roundtrip("H/H"), PolarisationType.HH)
        self.assertEqual(self.__roundtrip("H/V"), PolarisationType.HV)
        self.assertEqual(self.__roundtrip("X/X"), PolarisationType.XX)
        self.assertEqual(self.__roundtrip("V/H"), PolarisationType.VH)
        self.assertEqual(self.__roundtrip("V/V"), PolarisationType.VV)

    def test_unsupported_polarization(self):
        with self.assertRaises(UnsupportedPolarizationError):
            translate_polarization(EPolarization.crh)
        with self.assertRaises(UnsupportedPolarizationError):
            translate_polarization_tag("foo", poltype="bps")
        with self.assertRaises(UnsupportedPolarizationError):
            translate_polarization_tag("foo", poltype="arepytools")

    def __roundtrip(self, poltag):
        return translate_polarization(translate_polarization_tag(poltag, poltype="arepytools"))


if __name__ == "__main__":
    unittest.main()
