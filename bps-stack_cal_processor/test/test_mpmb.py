# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import sys
import unittest

import numpy as np
from bps.stack_cal_processor.core.skp.mpmb import get_mpmb_unique_pairs, to_mpmb_indices


# Just a bunch of utilities.
def _random_stack_data(nazm, nrng, npol, nimg):
    """Create a MBMP random data stack."""
    return tuple(
        tuple(np.random.rand(nazm, nrng) + 1j * np.random.rand(nazm, nrng) for p in range(npol)) for i in range(nimg)
    )


def _mbmp_to_mpmb_stack(mbmp_data_stack, transpose=True):
    """Convert a MBMP stack to an MPMB stack. Possilby flip range and azimuth."""
    num_images = len(mbmp_data_stack)
    num_polarizations = len(mbmp_data_stack[0])
    image_shape = mbmp_data_stack[0][0].shape
    if transpose:
        image_shape = np.flip(image_shape)
    mpmb_stack = []
    for p in range(num_polarizations):
        pol_image = np.empty((*image_shape, num_images), dtype=np.complex128)
        for n in range(num_images):
            pol_image[..., n] = mbmp_data_stack[n][p].T if transpose else mbmp_data_stack[n][p]
        mpmb_stack.append(pol_image)
    return mpmb_stack


class MpmbUtilsTest(unittest.TestCase):
    def test_mpmb_unique_pairs_with_diag(self):
        mpmb_unique_pairs = get_mpmb_unique_pairs(num_polarizations=3, num_images=3, include_diagonal=True)
        expected_num_unique_pairs = 9 * (9 + 1) / 2
        self.assertEqual(len(mpmb_unique_pairs), expected_num_unique_pairs)
        self.assertEqual(len(set(mpmb_unique_pairs)), len(mpmb_unique_pairs))

    def test_mpmb_unique_pairs_no_diag(self):
        mpmb_unique_pairs = get_mpmb_unique_pairs(num_polarizations=3, num_images=3, include_diagonal=False)
        expected_num_unique_pairs = 9 * (9 - 1) / 2
        self.assertEqual(len(mpmb_unique_pairs), expected_num_unique_pairs)
        self.assertEqual(len(set(mpmb_unique_pairs)), len(mpmb_unique_pairs))

    def test_to_mpmb_indices(self):
        mpmb_unique_pairs = get_mpmb_unique_pairs(num_polarizations=3, num_images=3, include_diagonal=True)
        triu = np.triu_indices(9)
        for gt_mpmb_i, gt_mpmb_j, (pn0, pn1) in zip(triu[0], triu[1], mpmb_unique_pairs):
            dut_mpmb_i, dut_mpmb_j = to_mpmb_indices(pn0, pn1, num_polarizations=3, num_images=3)
            self.assertEqual(gt_mpmb_i, dut_mpmb_i)
            self.assertEqual(gt_mpmb_j, dut_mpmb_j)


if __name__ == "__main__":
    sys.exit(unittest.main())
