# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Unit Tests for roi_utils module
--------------------------------
"""

import unittest
from unittest.mock import MagicMock

from bps.common.roi_utils import InvalidRegionOfInterestError, raise_if_roi_is_invalid


def _make_raster_info(lines: int = 100, samples: int = 200) -> MagicMock:
    raster_info = MagicMock()
    raster_info.lines = lines
    raster_info.samples = samples
    return raster_info


class TestRaiseIfRoiIsInvalid(unittest.TestCase):
    def test_valid_roi_does_not_raise(self):
        raise_if_roi_is_invalid(_make_raster_info(), (0, 0, 10, 20))

    def test_valid_roi_exact_boundary_does_not_raise(self):
        raise_if_roi_is_invalid(_make_raster_info(lines=100, samples=200), (0, 0, 100, 200))

    def test_valid_roi_interior_offset_does_not_raise(self):
        raise_if_roi_is_invalid(_make_raster_info(lines=100, samples=200), (50, 100, 50, 100))

    def test_wrong_length_too_short_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(), (0, 0, 10))  # type: ignore

    def test_wrong_length_too_long_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(), (0, 0, 10, 20, 30))  # pyright: ignore[reportArgumentType]

    def test_negative_azimuth_start_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(), (-1, 0, 10, 20))

    def test_negative_range_start_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(), (0, -1, 10, 20))

    def test_zero_azimuth_span_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(), (0, 0, 0, 20))

    def test_negative_azimuth_span_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(), (0, 0, -5, 20))

    def test_zero_range_span_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(), (0, 0, 10, 0))

    def test_negative_range_span_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(), (0, 0, 10, -5))

    def test_exceeds_azimuth_extent_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(lines=100), (50, 0, 51, 20))

    def test_exceeds_range_extent_raises(self):
        with self.assertRaises(InvalidRegionOfInterestError):
            raise_if_roi_is_invalid(_make_raster_info(samples=200), (0, 150, 10, 51))

    def test_error_is_value_error_subclass(self):
        with self.assertRaises(ValueError):
            raise_if_roi_is_invalid(_make_raster_info(), (-1, 0, 10, 20))

    def test_error_message_contains_roi(self):
        roi = (-1, 0, 10, 20)
        with self.assertRaises(InvalidRegionOfInterestError) as ctx:
            raise_if_roi_is_invalid(_make_raster_info(), roi)
        self.assertIn(str(roi), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
