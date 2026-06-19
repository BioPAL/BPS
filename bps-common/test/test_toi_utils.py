# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
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
    raise_if_invalid_toi,
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


class TestRaiseIfInvalidToi(unittest.TestCase):
    def test_valid_toi_does_not_raise(self):
        raise_if_invalid_toi(TimeOfInterest(time_begin=START_TIME, time_end=START_TIME + 1.0))

    def test_both_none_does_not_raise(self):
        raise_if_invalid_toi(TimeOfInterest())

    def test_only_begin_does_not_raise(self):
        raise_if_invalid_toi(TimeOfInterest(time_begin=START_TIME))

    def test_only_end_does_not_raise(self):
        raise_if_invalid_toi(TimeOfInterest(time_end=START_TIME))

    def test_begin_equal_end_raises(self):
        with self.assertRaises(InvalidTimeOfInterestError):
            raise_if_invalid_toi(TimeOfInterest(time_begin=START_TIME, time_end=START_TIME))

    def test_begin_after_end_raises(self):
        with self.assertRaises(InvalidTimeOfInterestError):
            raise_if_invalid_toi(TimeOfInterest(time_begin=START_TIME + 1.0, time_end=START_TIME))


if __name__ == "__main__":
    unittest.main()
