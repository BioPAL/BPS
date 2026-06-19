# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest
from dataclasses import dataclass

import numpy as np
from arepytools.io.metadata import EPolarization
from bps.stack_pre_processor.core.utils import common_polarizations, compute_processing_footprint


class TestCommonPolarizations(unittest.TestCase):
    """Test computing the stack polarization list."""

    def test_common_polarizations_nominal(self):
        gt = (EPolarization.hh,)
        dut = common_polarizations(
            (
                (EPolarization.hh, EPolarization.vv),
                (EPolarization.hh, EPolarization.hv),
                (EPolarization.vv, EPolarization.hh),
            )
        )
        self.assertTupleEqual(gt, dut)

    def test_common_polarizations_no_intersection(self):
        self.assertEqual(
            common_polarizations(
                (
                    (EPolarization.hv,),
                    (EPolarization.vv,),
                )
            ),
            tuple(),
        )

    def test_common_polarizations_empty(self):
        self.assertTupleEqual(
            common_polarizations((tuple(), (EPolarization.xx,))),
            tuple(),
        )


class TestComputeProcessingFootprint(unittest.TestCase):
    """Test computing the processing footprint."""

    @dataclass
    class RasterInfoProxy:
        """Just a light-weight version of arepytools's RasterInfo."""

        lines: int
        samples: int
        lines_step: float
        samples_step: float

    @classmethod
    def setUpClass(cls):
        cls.footprint = np.array(
            [
                [56.333445, 44.258845],
                [56.195068, 43.613696],
                [55.570093, 44.039912],
                [55.705858, 44.672738],
            ]
        )
        cls.raster_info = TestComputeProcessingFootprint.RasterInfoProxy(
            lines=151,
            samples=91,
            lines_step=1e-2,
            samples_step=1e-4,
        )

    def test_none_roi(self):
        dut = compute_processing_footprint(
            primary_footprint=self.footprint,
            primary_raster_info=self.raster_info,
            stack_roi=None,
        )
        np.testing.assert_equal(dut, self.footprint)

    def test_full_roi(self):
        dut = compute_processing_footprint(
            primary_footprint=self.footprint,
            primary_raster_info=self.raster_info,
            stack_roi=[0, 0, self.raster_info.lines, self.raster_info.samples],
        )
        np.testing.assert_equal(dut, self.footprint)

    def test_nominal_roi(self):
        dut = compute_processing_footprint(
            primary_footprint=self.footprint,
            primary_raster_info=self.raster_info,
            stack_roi=[0, 0, 76, 46],
        )
        gt = [
            self.footprint[3] + (self.footprint[0] - self.footprint[3]) / 2,
            sum(self.footprint[i] for i in range(4)) / 4,
            self.footprint[3] + (self.footprint[2] - self.footprint[3]) / 2,
            self.footprint[3],
        ]
        np.testing.assert_almost_equal(dut, gt)


if __name__ == "__main__":
    unittest.main()
