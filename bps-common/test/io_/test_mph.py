# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Tests for bps.common.io.mph
----------------------------
"""

import unittest
from pathlib import Path

from bps.common.io.mph import get_mph_path


class TestGetMphPath(unittest.TestCase):
    """Tests for get_mph_path."""

    def test_lowercase_name(self):
        """MPH filename is the product directory name lowercased with .xml extension."""
        product_dir = Path("/data/FP_GN__L2A_20210601T000000")
        result = get_mph_path(product_dir)
        assert result == Path("/data/FP_GN__L2A_20210601T000000/fp_gn__l2a_20210601t000000.xml")

    def test_already_lowercase_name(self):
        """Product directory name already lowercase: result is unchanged."""
        product_dir = Path("/data/s1_sta__1s_20210601")
        result = get_mph_path(product_dir)
        assert result == Path("/data/s1_sta__1s_20210601/s1_sta__1s_20210601.xml")

    def test_mixed_case_name(self):
        """Mixed-case product name is fully lowercased in the MPH filename."""
        product_dir = Path("/data/MyProduct_V2")
        result = get_mph_path(product_dir)
        assert result == Path("/data/MyProduct_V2/myproduct_v2.xml")

    def test_parent_is_preserved(self):
        """The parent directory of the product is preserved in the result."""
        product_dir = Path("/some/deep/path/FP_AGB_L2B_001")
        result = get_mph_path(product_dir)
        assert result.parent == product_dir

    def test_extension_is_xml(self):
        """Result always ends with .xml."""
        product_dir = Path("/data/PRODUCT")
        result = get_mph_path(product_dir)
        assert result.suffix == ".xml"

    def test_relative_path(self):
        """Works with relative paths."""
        product_dir = Path("relative/FP_FD__L2B_20210101")
        result = get_mph_path(product_dir)
        assert result == Path("relative/FP_FD__L2B_20210101/fp_fd__l2b_20210101.xml")


if __name__ == "__main__":
    unittest.main()
