# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Minimal Testing on LUT shifting
-------------------------------
"""

import unittest
from dataclasses import dataclass

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.stack_coreg_processor.core.shifting import coreg_primary_lut_axes


@dataclass
class StrawManRasterInfo:
    """Just a straw man class."""

    lines: int
    lines_step: float
    lines_start: PreciseDateTime
    samples: int
    samples_step: float
    samples_start: float


def _make_raster_info():
    # Only after 1985 allowed. Nothing interesting have happened in the world
    # in the meantime, So just a date.
    just_a_date = PreciseDateTime.from_utc_string("01-JAN-2000 00:00:00.0")

    return StrawManRasterInfo(
        lines=21395,
        lines_step=1.32183907894041e-07,
        lines_start=just_a_date,
        samples=1650,
        samples_step=0.000981546929703351,
        samples_start=0.005126224129846015,
    )


class CoregPrimaryLutAxesTest(unittest.TestCase):
    """Test the LUT subsampling step"""

    def test_coreg_primary_lut_axes_no_subsampling(self):
        raster_info = _make_raster_info()
        (
            dut_lut_azm_axis,
            dut_lut_rng_axis,
            dut_lut_azm_indices,
            dut_lut_rng_indices,
        ) = coreg_primary_lut_axes(
            raster_info,
            target_azimuth_step=raster_info.lines_step,
            target_range_step=raster_info.samples_step,
            keep_end_times=False,
        )
        self.assertEqual(dut_lut_azm_axis.size, dut_lut_azm_indices.size)
        self.assertEqual(dut_lut_rng_axis.size, dut_lut_rng_indices.size)
        self.assertEqual(np.diff(dut_lut_azm_indices)[0], 1)
        self.assertEqual(np.diff(dut_lut_rng_indices)[0], 1)

    def test_coreg_primary_lut_axes_nominal(self):
        raster_info = _make_raster_info()
        (
            dut_lut_azm_axis,
            dut_lut_rng_axis,
            dut_lut_azm_indices,
            dut_lut_rng_indices,
        ) = coreg_primary_lut_axes(
            raster_info,
            target_azimuth_step=2 * raster_info.lines_step,
            target_range_step=2 * raster_info.samples_step,
            keep_end_times=False,
        )
        self.assertEqual(dut_lut_azm_axis.size, round(raster_info.lines / 2))
        self.assertEqual(dut_lut_rng_axis.size, round(raster_info.samples / 2))
        self.assertEqual(np.diff(dut_lut_azm_indices)[0], 2)
        self.assertEqual(np.diff(dut_lut_rng_indices)[0], 2)

    def test_coreg_primary_lut_axes_upsampling(self):
        raster_info = _make_raster_info()
        (
            dut_lut_azm_axis,
            dut_lut_rng_axis,
            dut_lut_azm_indices,
            dut_lut_rng_indices,
        ) = coreg_primary_lut_axes(
            raster_info,
            target_azimuth_step=raster_info.lines_step / 2,
            target_range_step=raster_info.samples_step / 2,
            keep_end_times=False,
        )
        self.assertEqual(dut_lut_azm_axis.size, dut_lut_azm_indices.size)
        self.assertEqual(dut_lut_rng_axis.size, dut_lut_rng_indices.size)
        self.assertEqual(np.diff(dut_lut_azm_indices)[0], 1)
        self.assertEqual(np.diff(dut_lut_rng_indices)[0], 1)


if __name__ == "__main__":
    unittest.main()
