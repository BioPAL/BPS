# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest

import numpy as np
from bps.transcoder.utils.quicklook_utils import compute_quicklook_from_pol_data


def _random_data(num_azimuths: int, num_ranges: int) -> np.ndarray:
    """Just random complex data."""
    return np.random.rand(num_azimuths, num_ranges) + 1j * np.random.rand(num_azimuths, num_ranges)


class QuickLookFromPolDataTest(unittest.TestCase):
    def test_compute_quicklook_from_pol_data_missing_hh(self):
        np.random.seed(seed=1)

        zeros = np.zeros((2000, 100), dtype=np.complex64)
        data_hv = _random_data(2000, 100)
        data_vh = _random_data(2000, 100)
        data_vv = _random_data(2000, 100)

        np.testing.assert_equal(
            compute_quicklook_from_pol_data({"H/H": zeros, "H/V": data_hv, "V/H": data_vh, "V/V": data_vv}),
            compute_quicklook_from_pol_data({"H/V": data_hv, "V/H": data_vh, "V/V": data_vv}),
        )

    def test_compute_quicklook_from_pol_data_missing_hv(self):
        np.random.seed(seed=2)

        zeros = np.zeros((2000, 100), dtype=np.complex64)
        data_hh = _random_data(2000, 100)
        data_vh = _random_data(2000, 100)
        data_vv = _random_data(2000, 100)

        np.testing.assert_equal(
            compute_quicklook_from_pol_data({"H/H": data_hh, "H/V": zeros, "V/H": data_vh, "V/V": data_vv}),
            compute_quicklook_from_pol_data({"H/H": data_hh, "V/H": data_vh, "V/V": data_vv}),
        )

    def test_compute_quicklook_from_pol_data_missing_vh(self):
        np.random.seed(seed=3)

        zeros = np.zeros((2000, 100), dtype=np.complex64)
        data_hh = _random_data(2000, 100)
        data_hv = _random_data(2000, 100)
        data_vv = _random_data(2000, 100)

        np.testing.assert_equal(
            compute_quicklook_from_pol_data({"H/H": data_hh, "H/V": data_hv, "V/H": zeros, "V/V": data_vv}),
            compute_quicklook_from_pol_data({"H/H": data_hh, "H/V": data_hv, "V/V": data_vv}),
        )

    def test_compute_quicklook_from_pol_data_missing_vv(self):
        np.random.seed(seed=4)

        zeros = np.zeros((2000, 100), dtype=np.complex64)
        data_hh = _random_data(2000, 100)
        data_hv = _random_data(2000, 100)
        data_vh = _random_data(2000, 100)

        np.testing.assert_equal(
            compute_quicklook_from_pol_data({"H/H": data_hh, "H/V": data_hv, "V/H": data_vh, "V/V": zeros}),
            compute_quicklook_from_pol_data({"H/H": data_hh, "H/V": data_hv, "V/H": data_vh}),
        )


if __name__ == "__main__":
    unittest.main()
