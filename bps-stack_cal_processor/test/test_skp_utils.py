# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest
from dataclasses import fields

import numpy as np
from arepytools.io.metadata import EPolarization
from bps.stack_cal_processor.core.skp.utils import (
    SkpPhaseScreenStats,
    compute_skp_calibration_phases_statistics,
    merge_cross_polarizations,
    normalize_coherence,
)


def _random_hermitian_matrix(shape):
    """Generate a random Hermitian matrix."""
    root = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    return root + root.T.conj()


class SkpUtilsTest(unittest.TestCase):
    def test_normalize_coherence_nominal_case(self):
        cov = np.array([[2, 1], [1, 2]], dtype=np.float32)
        dut = normalize_coherence(cov, np.float32)
        self.assertEqual(dut.dtype, np.float32)
        gt = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        np.testing.assert_almost_equal(dut, gt)

    def test_normalize_coherence_singular_case(self):
        cov = np.array([[2, 0], [0, 0]])
        dut = normalize_coherence(cov, np.float64)
        self.assertEqual(dut.dtype, np.float64)
        gt = np.array([[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_almost_equal(dut, gt)


class TestComputeSkpCalibrationPhasesStatistics(unittest.TestCase):
    def test_basic_statistics_even(self):
        """Test case for compute_skp_calibration_phases_statistics with even data sizes."""
        phases = np.array(
            [
                [[np.pi], [np.pi], [np.pi], [np.pi]],
                [[np.pi], [-np.pi], [np.pi], [-np.pi]],
                [[-np.pi], [0.0], [-np.pi], [0.0]],
                [[0.0], [0.0], [0.0], [0.0]],
            ]
        )
        expected_output = [
            SkpPhaseScreenStats(
                avg_phase=np.pi,
                std_phase=0.0,
                mad_phase=0.0,
                var_phase=0.0,
                min_phase=np.pi,
                max_phase=np.pi,
            ),
            SkpPhaseScreenStats(
                avg_phase=np.pi,
                std_phase=0.0,
                mad_phase=0.0,
                var_phase=0.0,
                min_phase=-np.pi,
                max_phase=np.pi,
            ),
            SkpPhaseScreenStats(
                avg_phase=-np.pi / 2,
                std_phase=8.640816650440966,
                mad_phase=np.pi / 2,
                var_phase=0.9999999999999999,
                min_phase=-np.pi,
                max_phase=0,
            ),
            SkpPhaseScreenStats(
                avg_phase=0.0,
                std_phase=0.0,
                mad_phase=0.0,
                var_phase=0.0,
                min_phase=0.0,
                max_phase=0.0,
            ),
        ]

        output = compute_skp_calibration_phases_statistics(phases)
        for exp_stats, out_stats in zip(expected_output, output):
            for field in fields(SkpPhaseScreenStats):
                np.testing.assert_almost_equal(
                    np.angle(
                        np.exp(1j * (getattr(out_stats, field.name) - getattr(exp_stats, field.name))),
                    ),
                    0,
                    err_msg=f"Field '{field.name}' mismatch: "
                    f"expected {getattr(exp_stats, field.name)}, "
                    f"got {getattr(out_stats, field.name)}",
                )

    def test_basic_statistics_odd(self):
        """Test case for compute_skp_calibration_phases_statistics with odd data sizes."""
        phases = np.array(
            [
                [[np.pi], [np.pi], [np.pi], [np.pi], [np.pi]],
                [[np.pi], [np.pi], [np.pi], [-np.pi], [-np.pi]],
                [[np.pi], [np.pi], [np.pi], [-np.pi], [0.0]],
                [[0.1], [0.2], [0.3], [0.4], [0.5]],
                [[0.0], [0.0], [0.0], [0.0], [0.0]],
            ]
        )
        expected_output = [
            SkpPhaseScreenStats(
                avg_phase=np.pi,
                std_phase=0.0,
                mad_phase=0.0,
                var_phase=0.0,
                min_phase=np.pi,
                max_phase=np.pi,
            ),
            SkpPhaseScreenStats(
                avg_phase=np.pi,
                std_phase=0.0,
                mad_phase=0.0,
                var_phase=0.0,
                min_phase=-np.pi,
                max_phase=np.pi,
            ),
            SkpPhaseScreenStats(
                avg_phase=np.pi,
                std_phase=1.0107676525947897,
                mad_phase=0.0,
                var_phase=0.4,
                min_phase=-np.pi,
                max_phase=np.pi,
            ),
            SkpPhaseScreenStats(
                avg_phase=0.30000000000000027,
                std_phase=0.14157509091289966,
                mad_phase=0.12002403669731443,
                var_phase=0.009971702752292977,
                min_phase=0.1,
                max_phase=0.5,
            ),
            SkpPhaseScreenStats(
                avg_phase=0.0,
                std_phase=0.0,
                mad_phase=0.0,
                var_phase=0.0,
                min_phase=0.0,
                max_phase=0.0,
            ),
        ]

        output = compute_skp_calibration_phases_statistics(phases)
        for exp_stats, out_stats in zip(expected_output, output):
            for field in fields(SkpPhaseScreenStats):
                np.testing.assert_almost_equal(
                    np.angle(
                        np.exp(1j * (getattr(out_stats, field.name) - getattr(exp_stats, field.name))),
                    ),
                    0,
                    err_msg=f"Field '{field.name}' mismatch: "
                    f"expected {getattr(exp_stats, field.name)}, "
                    f"got {getattr(out_stats, field.name)}",
                )


class TestCrossPolMerging(unittest.TestCase):
    def test_cross_pol_merging_off(self):
        """Test cross-pol merging when flag is off."""
        np.random.seed(seed=1)
        gt = np.random.rand(3, 3, 11, 11).astype(np.complex64)
        dut = merge_cross_polarizations(
            gt,
            (EPolarization.hh, EPolarization.vv, EPolarization.hv),
            cross_pol_merging_flag=False,
        )
        np.testing.assert_equal(gt, dut)

    def test_cross_pol_merging_cross_pol_merged_stack(self):
        """Test skipping cross-pol merging since stack is already merged."""
        np.random.seed(seed=2)
        gt = np.random.rand(3, 3, 11, 11).astype(np.complex64)
        dut = merge_cross_polarizations(
            gt,
            (EPolarization.xx, EPolarization.hh, EPolarization.vv),
            cross_pol_merging_flag=True,
        )
        np.testing.assert_equal(gt, dut)

    def test_cross_pol_merging_missing_hv(self):
        """Test skipping cross-pol merging due to missing HV."""
        np.random.seed(seed=3)
        gt = np.random.rand(3, 3, 11, 11).astype(np.complex64)
        dut = merge_cross_polarizations(
            gt,
            (EPolarization.vh, EPolarization.hh, EPolarization.vv),
            cross_pol_merging_flag=True,
        )
        np.testing.assert_equal(gt, dut)

    def test_cross_pol_merging_missing_vh(self):
        """Test skipping cross-pol merging due to missing VH."""
        np.random.seed(seed=4)
        gt = np.random.rand(3, 3, 11, 11).astype(np.complex64)
        dut = merge_cross_polarizations(
            gt,
            (EPolarization.hv, EPolarization.hh, EPolarization.vv),
            cross_pol_merging_flag=True,
        )
        np.testing.assert_equal(gt, dut)

    def test_cross_pol_merging_nominal(self):
        np.random.seed(seed=4)
        data = np.random.rand(3, 4, 11, 11).astype(np.complex64)
        dut = merge_cross_polarizations(
            data,
            (EPolarization.hv, EPolarization.hh, EPolarization.vh, EPolarization.vv),
            cross_pol_merging_flag=True,
        )

        self.assertEqual(len(dut), data.shape[0])
        self.assertTrue(all(len(d) == 3 for d in dut))
        for i in range(3):
            np.testing.assert_equal(data[i, 1, ...], dut[i][0])  # Test HH.
            np.testing.assert_equal(data[i, 3, ...], dut[i][2])  # Test VV.
            np.testing.assert_equal(0.5 * (data[i, 0, ...] + data[i, 2, ...]), dut[i][1])  # Test XX.


if __name__ == "__main__":
    unittest.main()
