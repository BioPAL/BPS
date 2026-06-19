# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest

import numpy as np
import scipy as sp
from bps.stack_cal_processor.core.filtering import (
    ConvolutionBorderType,
    ConvolutionRoiMode,
    build_sparse_uniform_filter_matrix,
    convolve_2d,
    uniform_filter_2d,
)


class FilteringUtilsTest(unittest.TestCase):
    """Test some utilities used for filtering."""

    def test_build_sparse_uniform_filter_matrix_azimuth(self):
        dut, _ = build_sparse_uniform_filter_matrix(
            input_size=1000,
            subsampling_step=13,
            uniform_filter_window_size=5,
            axis=0,
            border_type=ConvolutionBorderType.ISOLATED,
        )
        np.testing.assert_equal(dut.shape[0], np.arange(0, 1000, 13).size)
        np.testing.assert_equal(dut.shape[1], 1000)
        np.testing.assert_almost_equal(
            np.sum(dut.todense(), axis=1).reshape(-1),
            np.ones((1, dut.shape[0])),
        )

    def test_build_sparse_uniform_filter_matrix_range(self):
        dut, _ = build_sparse_uniform_filter_matrix(
            input_size=1000,
            subsampling_step=9,
            uniform_filter_window_size=17,
            axis=1,
            border_type=ConvolutionBorderType.ISOLATED,
        )
        np.testing.assert_equal(dut.shape[0], 1000)
        np.testing.assert_equal(dut.shape[1], np.arange(0, 1000, 9).size)
        np.testing.assert_almost_equal(
            np.sum(dut.todense(), axis=0).reshape(-1),
            np.ones((1, dut.shape[1])),
        )

    def test_uniform_filtering(self):
        np.random.seed(seed=1)

        data = np.random.rand(1000, 900)

        dut_row, _ = build_sparse_uniform_filter_matrix(
            input_size=1000,
            subsampling_step=15,
            uniform_filter_window_size=5,
            axis=0,
            border_type=ConvolutionBorderType.CONSTANT,
        )
        dut_col, _ = build_sparse_uniform_filter_matrix(
            input_size=900,
            subsampling_step=13,
            uniform_filter_window_size=7,
            axis=1,
            border_type=ConvolutionBorderType.CONSTANT,
        )

        gt = sp.ndimage.uniform_filter(data, (5, 7), mode="constant")[::15, ::13]
        dut = dut_row @ data @ dut_col

        np.testing.assert_almost_equal(dut, gt)

    def test_convolve_2d_axis0(self):
        np.random.seed(seed=2)

        data = np.random.rand(1000, 900)
        kernel = np.hamming(101)[:, np.newaxis]
        kernel /= np.sum(kernel)

        gt = sp.signal.fftconvolve(data, kernel, mode="same", axes=0)
        dut = convolve_2d(data, kernel)

        np.testing.assert_almost_equal(dut, gt)

    def test_convolve_2d_axis1(self):
        np.random.seed(seed=3)

        data = np.random.rand(1000, 900)
        kernel = np.hamming(21)[np.newaxis, :]
        kernel /= np.sum(kernel)

        gt = sp.signal.fftconvolve(data, kernel, mode="same", axes=1)
        dut = convolve_2d(data, kernel)

        np.testing.assert_almost_equal(dut, gt)

    def test_uniform_filter_2d(self):
        np.random.seed(seed=4)

        data = np.random.rand(1000, 900)

        gt = sp.ndimage.uniform_filter(data, (9, 11))
        dut = uniform_filter_2d(data, (9, 11), roi_mode=ConvolutionRoiMode.VALID)

        np.testing.assert_almost_equal(gt[4:-4, 5:-5], dut)


if __name__ == "__main__":
    unittest.main()
