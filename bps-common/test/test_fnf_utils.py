# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import re
import unittest

from bps.common.fnf_utils import FNF_PRODUCT_REGEX


class TestAuxFnFRegex(unittest.TestCase):
    """Test the AUX-FNF regexes."""

    def test_fnf_match(self):
        match = re.match(
            FNF_PRODUCT_REGEX,
            "BIO_FNF_20200101T000000_20201231T000000_AFRS00",
        )
        self.assertTrue(match is not None)

    def test_fnf_sta_match(self):
        match = re.match(
            FNF_PRODUCT_REGEX,
            "BIO_AUX_STA_FNF_20200101T000000_20201231T000000_AFRS00",
        )
        self.assertTrue(match is not None)

    def test_fnf_l2_match(self):
        match = re.match(
            FNF_PRODUCT_REGEX,
            "BIO_AUX_L2_FNF_20200101T000000_20201231T000000_AFRS00",
        )
        self.assertTrue(match is not None)

    def test_fnf_blank_match(self):
        match = re.match(
            FNF_PRODUCT_REGEX,
            "BIO_AUX____FNF_20200101T000000_20201231T000000_AFRS00",
        )
        self.assertTrue(match is not None)

    def test_fnf_bad_tag_match(self):
        match = re.match(
            FNF_PRODUCT_REGEX,
            "BIO_AUX_badFNF_20200101T000000_20201231T000000_AFRS00",
        )
        self.assertTrue(match is None)

    def test_fnf_bad_time_match(self):
        match = re.match(
            FNF_PRODUCT_REGEX,
            "BIO_AUX_L2_FNF_20200101T0000_20201231T000000_AFRS00",
        )
        self.assertTrue(match is None)


if __name__ == "__main__":
    unittest.main()
