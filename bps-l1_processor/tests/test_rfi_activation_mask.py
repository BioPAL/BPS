import unittest

import numpy as np
from bps.l1_processor.processor_interface.rfi_activation_mask import compute_rfi_activation_mask_overlap_fraction


def create_mask(n_lon=10, n_lat=10, active_cells=None):
    """Create a simple mask for testing."""
    mask = np.zeros((n_lon, n_lat), dtype=bool)
    if active_cells:
        for i, j in active_cells:
            mask[i, j] = True

    dlon = 360 / n_lon
    dlat = 180 / n_lat
    lon_axis = -180.0 + dlon / 2.0 + dlon * np.arange(n_lon)
    lat_axis = -90 + dlat / 2.0 + dlat * np.arange(n_lat)

    return mask, lon_axis, lat_axis


class TestRFIMask(unittest.TestCase):
    def test_simple_overlap(self):
        """Footprint fully inside one active mask cell."""
        mask, lon_axis, lat_axis = create_mask(n_lon=5, n_lat=5, active_cells=[(2, 2)])
        footprint = [(-10.0, 10), (-10, 20), (0, 20), (0, 10)]
        portion = compute_rfi_activation_mask_overlap_fraction(footprint, mask, lon_axis, lat_axis)
        self.assertGreater(portion, 0)
        self.assertLessEqual(portion, 1)

    def test_no_overlap(self):
        """Footprint outside any active cell."""
        mask, lon_axis, lat_axis = create_mask(n_lon=5, n_lat=5)
        footprint = [(-10.0, 10), (-10, 20), (0, 20), (0, 10)]
        portion = compute_rfi_activation_mask_overlap_fraction(footprint, mask, lon_axis, lat_axis)
        self.assertEqual(portion, 0.0)

    def test_antimeridian_crossing(self):
        """Footprint crosses the antimeridian."""
        mask, lon_axis, lat_axis = create_mask(n_lon=10, n_lat=5, active_cells=[(9, 2), (0, 2)])
        footprint = [(179.0, 10), (179, 20), (-179, 20), (-179, 10)]
        portion = compute_rfi_activation_mask_overlap_fraction(footprint, mask, lon_axis, lat_axis)
        self.assertGreater(portion, 0)
        self.assertLessEqual(portion, 1)

    def test_zero_area_footprint(self):
        """Check handling of degenerate footprint."""
        mask, lon_axis, lat_axis = create_mask(n_lon=5, n_lat=5)
        footprint = [(0.0, 0), (0, 0), (0, 0), (0, 0)]
        with self.assertRaises(RuntimeError):
            compute_rfi_activation_mask_overlap_fraction(footprint, mask, lon_axis, lat_axis)

    def test_partial_overlap(self):
        """Footprint partially overlaps multiple active cells."""
        mask, lon_axis, lat_axis = create_mask(n_lon=5, n_lat=5, active_cells=[(1, 1), (2, 1), (2, 2)])
        footprint = [(-50.0, -20), (-50, 0), (-30, 0), (-30, -20)]
        portion = compute_rfi_activation_mask_overlap_fraction(footprint, mask, lon_axis, lat_axis)
        self.assertGreater(portion, 0)
        self.assertLessEqual(portion, 1)

    def test_mask_edges(self):
        """Footprint on the edge of mask grid."""
        mask, lon_axis, lat_axis = create_mask(n_lon=5, n_lat=5, active_cells=[(0, 0)])
        footprint = [(-180.0, -90), (-180, -80), (-170, -80), (-170, -90)]
        portion = compute_rfi_activation_mask_overlap_fraction(footprint, mask, lon_axis, lat_axis)
        self.assertGreater(portion, 0)
        self.assertLessEqual(portion, 1)

    def test_partial_overlap_large_grid(self):
        """Check partial overlap in a large 100x120 grid using create_mask."""
        n_lat, n_lon = 100, 120

        lat_cells = list(range(40, 45))
        lon_cells = list(range(50, 52))
        active_cells = [(j, i) for i in lat_cells for j in lon_cells]
        mask, lon_axis, lat_axis = create_mask(n_lon=n_lon, n_lat=n_lat, active_cells=active_cells)

        dlon = lon_axis[1] - lon_axis[0]
        dlat = lat_axis[1] - lat_axis[0]

        footprint = [
            (lon_axis[50] - dlon / 2, lat_axis[38] - dlat / 2),
            (lon_axis[53] + dlon / 2, lat_axis[38] - dlat / 2),
            (lon_axis[53] + dlon / 2, lat_axis[41] + dlat / 2),
            (lon_axis[50] - dlon / 2, lat_axis[41] + dlat / 2),
        ]

        portion = compute_rfi_activation_mask_overlap_fraction(footprint, mask, lon_axis, lat_axis)

        self.assertAlmostEqual(portion, 0.25, places=2)


if __name__ == "__main__":
    unittest.main()
