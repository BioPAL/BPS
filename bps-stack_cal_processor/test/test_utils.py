# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest
from itertools import product

import numba as nb
import numpy as np
import scipy as sp
from bps.stack_cal_processor.core.utils import _interp1d_nn, interpolate_on_grid_nn, query_grid_mask


class UtilsTest(unittest.TestCase):
    """Test some utilities."""

    def test_interpolate_on_grid_nn_trivial(self):
        np.random.seed(seed=1)

        axes = (np.linspace(0, 1, 101), np.linspace(0, 1, 111))
        values = np.random.rand(101, 111)

        dut = interpolate_on_grid_nn(values, axes_in=axes, axes_out=axes)
        np.testing.assert_almost_equal(dut, values)

    def test_interpolate_on_grid_nn_noisy_boundary(self):
        np.random.seed(seed=1)

        axes = (np.linspace(0, 1, 101), np.linspace(0, 1, 111))
        values = np.random.rand(101, 111)
        eps0 = 1e-8 * np.array([-1, *np.random.rand(99), 1])
        eps1 = 1e-8 * np.array([-1, *np.random.rand(109), 1])

        dut = interpolate_on_grid_nn(
            values,
            axes_in=axes,
            axes_out=(axes[0] + eps0, axes[1] + eps1),
        )
        np.testing.assert_almost_equal(dut, values)

    def test_interpolate_on_grid_nn_nominal(self):
        np.random.seed(seed=2)

        axes_in = (np.linspace(0, 1, 121), np.linspace(0, 1, 111))
        axes_out = (np.linspace(0, 1, 31), np.linspace(0, 1, 51))
        values = np.random.rand(121, 111)

        gt = sp.interpolate.NearestNDInterpolator(list(product(*axes_in)), values.reshape(-1))(
            *np.meshgrid(*axes_out)
        ).T
        dut = interpolate_on_grid_nn(values, axes_in=axes_in, axes_out=axes_out)
        np.testing.assert_almost_equal(dut, gt)

    def test_query_mask_on_own_grid(self):
        np.random.seed(seed=1)
        nb.set_num_threads(1)

        mask = np.random.rand(20, 40)
        x_axis = np.linspace(0, 10, 20)
        y_axis = np.linspace(-5, 5, 40)

        dut_mask = query_grid_mask(
            mask=mask,
            x_axis=x_axis,
            y_axis=y_axis,
            xs=x_axis,
            ys=y_axis[0:20],
            nodata_fill_value=np.nan,
            nodata_value=np.nan,
        )

        np.testing.assert_equal(np.diag(mask), dut_mask)

    def test_query_mask_increasing_x_increasing_y(self):
        np.random.seed(seed=1)
        nb.set_num_threads(1)

        mask = np.random.rand(21, 41)
        x_axis = np.linspace(0, 10, 21)
        y_axis = np.linspace(-5, 5, 41)

        xs = np.array([[x + 0.01 for x in x_axis[:-1]] for _ in y_axis[:-1]]).T
        ys = np.array([[y + 0.01 for y in y_axis[:-1]] for _ in x_axis[:-1]])

        dut_mask = query_grid_mask(
            mask=mask,
            x_axis=x_axis,
            y_axis=y_axis,
            xs=xs,
            ys=ys,
            nodata_fill_value=np.nan,
            nodata_value=np.nan,
        )

        np.testing.assert_equal(dut_mask, mask[0:-1, 0:-1])

    def test_query_mask_increasing_x_decreasing_y(self):
        np.random.seed(seed=1)
        nb.set_num_threads(1)

        mask = np.random.rand(21, 41)
        x_axis = np.linspace(0, 10, 21)
        y_axis = np.linspace(5, -5, 41)

        xs = np.array([[x + 0.01 for x in x_axis[:-1]] for _ in y_axis[:-1]]).T
        ys = np.array([[y - 0.01 for y in y_axis[:-1]] for _ in x_axis[:-1]])

        dut_mask = query_grid_mask(
            mask=mask,
            x_axis=x_axis,
            y_axis=y_axis,
            xs=xs,
            ys=ys,
            nodata_fill_value=np.nan,
            nodata_value=np.nan,
        )

        np.testing.assert_equal(dut_mask, mask[0:-1, 0:-1])

    def test_query_mask_decreasing_x_increasing_y(self):
        np.random.seed(seed=1)
        nb.set_num_threads(1)

        mask = np.random.rand(21, 41)
        x_axis = np.linspace(10, 0, 21)
        y_axis = np.linspace(-5, 5, 41)

        xs = np.array([[x - 0.01 for x in x_axis[:-1]] for _ in y_axis[:-1]]).T
        ys = np.array([[y + 0.01 for y in y_axis[:-1]] for _ in x_axis[:-1]])

        dut_mask = query_grid_mask(
            mask=mask,
            x_axis=x_axis,
            y_axis=y_axis,
            xs=xs,
            ys=ys,
            nodata_fill_value=np.nan,
            nodata_value=np.nan,
        )

        np.testing.assert_equal(dut_mask, mask[0:-1, 0:-1])

    def test_query_mask_decreasing_x_decreasing_y(self):
        np.random.seed(seed=1)
        nb.set_num_threads(1)

        mask = np.random.rand(21, 41)
        x_axis = np.linspace(10, 0, 21)
        y_axis = np.linspace(5, -5, 41)

        xs = np.array([[x - 0.01 for x in x_axis[:-1]] for _ in y_axis[:-1]]).T
        ys = np.array([[y - 0.01 for y in y_axis[:-1]] for _ in x_axis[:-1]])

        dut_mask = query_grid_mask(
            mask=mask,
            x_axis=x_axis,
            y_axis=y_axis,
            xs=xs,
            ys=ys,
            nodata_fill_value=np.nan,
            nodata_value=np.nan,
        )

        np.testing.assert_equal(dut_mask, mask[0:-1, 0:-1])


class Interp1dTest(unittest.TestCase):
    """Test that _interp1d_nn is equivalent to scipy's interp1d."""

    _sp_interp_options = {
        "kind": "nearest",
        "bounds_error": False,
        "fill_value": "extrapolate",
    }

    def test_nominal(self):
        """Test the standard case."""
        np.random.seed(seed=0)
        xc = np.sort(np.random.rand(1000))
        yc = np.random.randn(1000)
        xq = 1.5 * np.random.rand(250) - 0.75

        gt = sp.interpolate.interp1d(xc, yc, **self._sp_interp_options)(xq)
        dut = _interp1d_nn(xc, yc, xq)
        np.testing.assert_almost_equal(gt, dut)

    def test_nominal_indices(self):
        """Test the indices interpolator."""
        np.random.seed(seed=1)
        xc = np.linspace(0, 1, 101)
        yc = np.arange(101, dtype=np.float64)
        xq = np.linspace(-1.1, 1.1, 701)

        gt = sp.interpolate.interp1d(xc, yc, **self._sp_interp_options)(xq)
        dut = _interp1d_nn(xc, yc, xq)
        np.testing.assert_equal(gt.astype(np.int32), dut.astype(np.int32))


if __name__ == "__main__":
    unittest.main()
