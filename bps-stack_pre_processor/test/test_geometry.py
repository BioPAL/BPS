# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Unit Tests for the DEM Upsampling
---------------------------------
"""

import unittest

import numpy as np
import numpy.typing as npt
from arepytools.geometry.conversions import llh2xyz
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.stack_pre_processor.core.geometry import compute_ecef_dem


def _make_dem_wgs84(
    num_azimuths: int,
    num_ranges: int,
) -> tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    """Generate a dem on the ellipsoid."""
    # Bounding box around Italy.
    lats_deg = np.linspace(36.6465, 47.09322, num_azimuths).reshape(-1, 1)
    lons_deg = np.linspace(6.1858, 18.5199, num_ranges).reshape(1, -1)
    return (
        lats_deg @ np.ones((1, num_ranges)),
        np.ones((num_azimuths, 1)) @ lons_deg,
        np.zeros((num_azimuths, num_ranges)),
    )


class TestGeometry(unittest.TestCase):
    """Test the DEM usampling."""

    lines: int = 3000
    samples: int = 500
    lines_step: float = 5e-4  # [s].
    samples_step: float = 2e-7  # [s].
    lut_lines_sub: int = 12
    lut_samples_sub: int = 2

    gt_lat_deg: npt.NDArray[float]  # [deg].
    gt_lon_deg: npt.NDArray[float]  # [deg].
    gt_h: npt.NDArray[float]  # [m].

    gt_x: npt.NDArray[float]  # [m].
    gt_y: npt.NDArray[float]  # [m].
    gt_z: npt.NDArray[float]  # [m].

    gt_azm_axis: npt.NDArray[PreciseDateTime]  # [UTC].
    gt_rng_axis: npt.NDArray[float]  # [s].

    @classmethod
    def setUp(cls):
        azm_start = PreciseDateTime.from_numeric_datetime(year=2000, month=1, day=1)
        rng_start = 1e-5  # [s].
        cls.gt_azm_axis = np.arange(cls.lines) * cls.lines_step + azm_start
        cls.gt_rng_axis = np.arange(cls.samples) * cls.samples_step + rng_start

    def test_compute_ecef_dem_wgs84(self):
        # Initialize the DEM.
        gt_lat_deg, gt_lon_deg, gt_h = _make_dem_wgs84(self.lines, self.samples)
        gt_xyz = llh2xyz(
            [
                np.deg2rad(gt_lat_deg.reshape(-1)),
                np.deg2rad(gt_lon_deg.reshape(-1)),
                gt_h.reshape(-1),
            ]
        )

        gt_x = gt_xyz[0].reshape((self.lines, self.samples))
        gt_y = gt_xyz[1].reshape((self.lines, self.samples))
        gt_z = gt_xyz[2].reshape((self.lines, self.samples))

        # Generate a LUT via subsampling.
        gt_lut_lat_deg = gt_lat_deg[:: self.lut_lines_sub, :: self.lut_samples_sub]
        gt_lut_lon_deg = gt_lon_deg[:: self.lut_lines_sub, :: self.lut_samples_sub]
        gt_lut_h = gt_h[:: self.lut_lines_sub, :: self.lut_samples_sub]

        gt_lut_azm_axis = self.gt_azm_axis[:: self.lut_lines_sub]
        gt_lut_rng_axis = self.gt_rng_axis[:: self.lut_samples_sub]

        lut_data = {
            "latitude": gt_lut_lat_deg,
            "longitude": gt_lut_lon_deg,
            "height": gt_lut_h,
        }
        lut_azm_axes = {
            "latitude": gt_lut_azm_axis,
            "longitude": gt_lut_azm_axis,
            "height": gt_lut_azm_axis,
        }
        lut_rng_axes = {
            "latitude": gt_lut_rng_axis,
            "longitude": gt_lut_rng_axis,
            "height": gt_lut_rng_axis,
        }

        # Upsample the DEM.
        dut_x, dut_y, dut_z = compute_ecef_dem(
            lut_data=lut_data,
            lut_axes=(lut_azm_axes, lut_rng_axes),
            l1a_data_axes=(self.gt_azm_axis, self.gt_rng_axis),
        )

        # Evaluate.
        np.testing.assert_almost_equal(
            gt_x[:: self.lut_lines_sub, :: self.lut_samples_sub],
            dut_x[:: self.lut_lines_sub, :: self.lut_samples_sub],
            decimal=8,
        )
        np.testing.assert_almost_equal(
            gt_y[:: self.lut_lines_sub, :: self.lut_samples_sub],
            dut_y[:: self.lut_lines_sub, :: self.lut_samples_sub],
            decimal=8,
        )
        np.testing.assert_almost_equal(
            gt_z[:: self.lut_lines_sub, :: self.lut_samples_sub],
            dut_z[:: self.lut_lines_sub, :: self.lut_samples_sub],
            decimal=8,
        )

    def test_compute_ecef_dem_random_elevation(self):
        # Initialize the DEM.
        gt_lat_deg, gt_lon_deg, _ = _make_dem_wgs84(self.lines, self.samples)
        gt_h = np.full_like(gt_lat_deg, 500) + np.random.randn(self.lines, self.samples)
        gt_xyz = llh2xyz(
            [
                np.deg2rad(gt_lat_deg.reshape(-1)),
                np.deg2rad(gt_lon_deg.reshape(-1)),
                gt_h.reshape(-1),
            ]
        )

        gt_x = gt_xyz[0].reshape((self.lines, self.samples))
        gt_y = gt_xyz[1].reshape((self.lines, self.samples))
        gt_z = gt_xyz[2].reshape((self.lines, self.samples))

        # Generate a LUT via subsampling.
        gt_lut_lat_deg = gt_lat_deg[:: self.lut_lines_sub, :: self.lut_samples_sub]
        gt_lut_lon_deg = gt_lon_deg[:: self.lut_lines_sub, :: self.lut_samples_sub]
        gt_lut_h = gt_h[:: self.lut_lines_sub, :: self.lut_samples_sub]

        gt_lut_azm_axis = self.gt_azm_axis[:: self.lut_lines_sub]
        gt_lut_rng_axis = self.gt_rng_axis[:: self.lut_samples_sub]

        lut_data = {
            "latitude": gt_lut_lat_deg,
            "longitude": gt_lut_lon_deg,
            "height": gt_lut_h,
        }
        lut_azm_axes = {
            "latitude": gt_lut_azm_axis,
            "longitude": gt_lut_azm_axis,
            "height": gt_lut_azm_axis,
        }
        lut_rng_axes = {
            "latitude": gt_lut_rng_axis,
            "longitude": gt_lut_rng_axis,
            "height": gt_lut_rng_axis,
        }

        # Upsample the DEM.
        dut_x, dut_y, dut_z = compute_ecef_dem(
            lut_data=lut_data,
            lut_axes=(lut_azm_axes, lut_rng_axes),
            l1a_data_axes=(self.gt_azm_axis, self.gt_rng_axis),
        )

        # Evaluate.
        np.testing.assert_almost_equal(
            gt_x[:: self.lut_lines_sub, :: self.lut_samples_sub],
            dut_x[:: self.lut_lines_sub, :: self.lut_samples_sub],
            decimal=8,
        )
        np.testing.assert_almost_equal(
            gt_y[:: self.lut_lines_sub, :: self.lut_samples_sub],
            dut_y[:: self.lut_lines_sub, :: self.lut_samples_sub],
            decimal=8,
        )
        np.testing.assert_almost_equal(
            gt_z[:: self.lut_lines_sub, :: self.lut_samples_sub],
            dut_z[:: self.lut_lines_sub, :: self.lut_samples_sub],
            decimal=8,
        )


if __name__ == "__main__":
    unittest.main()
