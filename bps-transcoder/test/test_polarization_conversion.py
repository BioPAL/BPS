# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest

from arepytools.io.metadata import EPolarization
from bps.common.io import common
from bps.transcoder.utils.polarization_conversions import (
    UnsupportedPolarizationError,
    translate_arepytools_polarization_to_common,
    translate_common_polarization_to_arepytools,
)


class TestPolarizationConversions(unittest.TestCase):
    def test_arepytools_to_common(self):
        expected = {
            EPolarization.hh: common.PolarisationType.HH,
            EPolarization.hv: common.PolarisationType.HV,
            EPolarization.vh: common.PolarisationType.VH,
            EPolarization.vv: common.PolarisationType.VV,
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(translate_arepytools_polarization_to_common(source), target)

    def test_common_to_arepytools(self):
        expected = {
            common.PolarisationType.HH: EPolarization.hh,
            common.PolarisationType.HV: EPolarization.hv,
            common.PolarisationType.VH: EPolarization.vh,
            common.PolarisationType.VV: EPolarization.vv,
        }
        for source, target in expected.items():
            with self.subTest(source=source):
                self.assertEqual(translate_common_polarization_to_arepytools(source), target)

    def test_unsupported_arepytools_polarization(self):
        with self.assertRaises(UnsupportedPolarizationError):
            translate_arepytools_polarization_to_common(EPolarization.none)

    def test_unsupported_common_polarization(self):
        with self.assertRaises(UnsupportedPolarizationError):
            # pyrefly: ignore[bad-argument-type]
            translate_common_polarization_to_arepytools("")


if __name__ == "__main__":
    unittest.main()
