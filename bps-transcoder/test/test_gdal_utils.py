# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Tests for the GDAL utlities
---------------------------
"""

import unittest

import numpy as np
from arepytools.geometry.conversions import llh2xyz
from bps.transcoder.utils.gdal_utils import _from_gdal_gcp, _to_gdal_gcp


def _make_random_gcp_list(npoints: int) -> list[list[float]]:
    """Make some random GCP."""
    gcps = []
    for _ in range(npoints):
        gcp_xyz = llh2xyz([np.random.rand(), np.random.rand(), np.random.rand()]).flatten()
        gcps.append([*gcp_xyz, np.random.randint(100), np.random.randint(100)])
    return gcps


class TestGDALUtils(unittest.TestCase):
    """Test the GCP conversion utilities."""

    def test_roundtrip_conversion_gcp_list(self):
        np.random.seed(seed=1)
        gt_gcps = _make_random_gcp_list(10)
        for gt_gcp in gt_gcps:
            dut_gcp = _from_gdal_gcp(_to_gdal_gcp(gt_gcp))
            np.testing.assert_almost_equal(dut_gcp[:3], gt_gcp[:3])
            self.assertEqual(dut_gcp[3], gt_gcp[3])
            self.assertEqual(dut_gcp[4], gt_gcp[4])


if __name__ == "__main__":
    unittest.main()
