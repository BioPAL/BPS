# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Unit Tests for the TOI Utility Library
--------------------------------------
"""

import unittest

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.toi_utils import (
    InvalidTimeOfInterestError,
    TimeOfInterest,
    toi_to_axis_slice,
)

# Just a date.
START_TIME = PreciseDateTime().from_numeric_datetime(year=2015, month=9, day=15)


class TestToiToAxisSlice(unittest.TestCase):
    """Test the TOI to axis slice conversion."""

    _time_axis = START_TIME + np.arange(20) * 0.1

    def test_toi_to_axis_slice_exact_match(self):
        dut_begin, dut_end = toi_to_axis_slice(
            TimeOfInterest(
                time_begin=self._time_axis[4],
                time_end=self._time_axis[12],
            ),
            time_axis=self._time_axis,
        )
        self.assertEqual(dut_begin, 4)
        self.assertEqual(dut_end, 12)

    def test_toi_to_axis_slice_approximate_match(self):
        dut_begin, dut_end = toi_to_axis_slice(
            TimeOfInterest(
                time_begin=self._time_axis[4] - 0.08,
                time_end=self._time_axis[12] + 0.002,
            ),
            time_axis=self._time_axis,
        )
        self.assertEqual(dut_begin, 3)
        self.assertEqual(dut_end, 12)

    def test_toi_to_axis_slice_no_time_begin(self):
        dut_begin, dut_end = toi_to_axis_slice(
            TimeOfInterest(time_end=self._time_axis[12]),
            time_axis=self._time_axis,
        )
        self.assertEqual(dut_begin, 0)
        self.assertEqual(dut_end, 12)

    def test_toi_to_axis_slice_no_time_end(self):
        dut_begin, dut_end = toi_to_axis_slice(
            TimeOfInterest(time_begin=self._time_axis[4]),
            time_axis=self._time_axis,
        )
        self.assertEqual(dut_begin, 4)
        self.assertEqual(dut_end, self._time_axis.size - 1)

    def test_toi_to_axis_slice_no_bounds(self):
        dut_begin, dut_end = toi_to_axis_slice(
            TimeOfInterest(time_begin=None, time_end=None),
            time_axis=self._time_axis,
        )
        self.assertEqual(dut_begin, 0)
        self.assertEqual(dut_end, self._time_axis.size - 1)

    def test_toi_to_axis_slice_larger_interval(self):
        dut_begin, dut_end = toi_to_axis_slice(
            TimeOfInterest(
                time_begin=self._time_axis[0] - 1,
                time_end=self._time_axis[-1] + 1,
            ),
            time_axis=self._time_axis,
        )
        self.assertEqual(dut_begin, 0)
        self.assertEqual(dut_end, self._time_axis.size - 1)

    def test_toi_to_axis_slice_intersect_left(self):
        dut_begin, dut_end = toi_to_axis_slice(
            TimeOfInterest(
                time_begin=self._time_axis[0] - 1,
                time_end=self._time_axis[12],
            ),
            time_axis=self._time_axis,
        )
        self.assertEqual(dut_begin, 0)
        self.assertEqual(dut_end, 12)

    def test_toi_to_axis_slice_intersect_right(self):
        dut_begin, dut_end = toi_to_axis_slice(
            TimeOfInterest(
                time_begin=self._time_axis[4],
                time_end=self._time_axis[-1] + 1,
            ),
            time_axis=self._time_axis,
        )
        self.assertEqual(dut_begin, 4)
        self.assertEqual(dut_end, self._time_axis.size - 1)

    def test_toi_to_axis_slice_no_intersection(self):
        with self.assertRaises(InvalidTimeOfInterestError):
            toi_to_axis_slice(
                TimeOfInterest(
                    time_begin=START_TIME - 100,
                    time_end=START_TIME - 99,
                ),
                time_axis=self._time_axis,
            )


if __name__ == "__main__":
    unittest.main()
