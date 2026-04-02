# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Unit Tests for the Memory Checking Module
-----------------------------------------
"""

import unittest

from bps.stack_processor.utils.memorycheck import (
    CRITICAL_SWATH_DURATION,
    azf_memory_usage_forecast_MB,
    coreg_memory_usage_forecast_MB,
    l1c_memory_usage_forecast_MB,
    preproc_memory_usage_forecast_MB,
    skp_memory_usage_forecast_MB,
)

# The nominal swath duration (rounded to half second).
NOMINAL_SWATH_DURATION = 21.5  # [s]

# The default value in the Stack Job Order.
JOB_ORDER_TT_VALUE_MB = 65536  # [MB]


class TestPreprocessorInsufficientMemoryUsage(unittest.TestCase):
    """Minimal testing for the pre-processor's watchdog."""

    def test_preproc_memory_usage_nominal_swath_tom(self):
        self.assertLess(
            preproc_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=NOMINAL_SWATH_DURATION,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )

    def test_preproc_memory_usage_critical_swath_tom(self):
        self.assertLess(
            preproc_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=CRITICAL_SWATH_DURATION,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )


class TestCoregistratorInsufficientMemoryUsage(unittest.TestCase):
    """Minimal testing for the coregistrator's watchdog."""

    def test_coreg_memory_usage_nominal_tom(self):
        self.assertLess(
            coreg_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=NOMINAL_SWATH_DURATION,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )

    def test_coreg_memory_usage_critical_tom(self):
        self.assertLess(
            coreg_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=CRITICAL_SWATH_DURATION,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )


class TestAzimuthSpectralFilterInsufficientMemoryUsage(unittest.TestCase):
    """Minimal testing for the AZF's watchdog."""

    def test_azf_memory_usage_nominal_tom_float32(self):
        self.assertLess(
            azf_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=NOMINAL_SWATH_DURATION,
                use_32bit_flag=True,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )

    def test_azf_memory_usage_nominal_tom_float64(self):
        self.assertLess(
            azf_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=NOMINAL_SWATH_DURATION,
                use_32bit_flag=False,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )

    def test_azf_memory_usage_critical_tom_float32(self):
        self.assertLess(
            azf_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=CRITICAL_SWATH_DURATION,
                use_32bit_flag=True,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )

    def test_azf_memory_usage_critical_tom_float64(self):
        self.assertLess(
            azf_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=CRITICAL_SWATH_DURATION,
                use_32bit_flag=False,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )


class TestSkpPhaseCalibrationInsufficientMemoryUsage(unittest.TestCase):
    """Minimal testing for the SKP's watchdog."""

    def test_skp_memory_usage_nominal_tom_float32(self):
        self.assertLess(
            skp_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=NOMINAL_SWATH_DURATION,
                use_32bit_flag=True,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )

    def test_skp_memory_usage_nominal_tom_float64(self):
        self.assertLess(
            skp_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=NOMINAL_SWATH_DURATION,
                use_32bit_flag=False,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )

    def test_skp_memory_usage_critical_tom_float32(self):
        self.assertLess(
            skp_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=CRITICAL_SWATH_DURATION,
                use_32bit_flag=True,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )

    def test_skp_memory_usage_critical_tom_float64(self):
        self.assertLess(
            skp_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=CRITICAL_SWATH_DURATION,
                use_32bit_flag=False,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )


class TestL1cMemoryUsageInsufficientMemoryUsage(unittest.TestCase):
    """Minimal testing for the L1c writer's watchdog."""

    def test_l1c_memory_usage_nominal_tom(self):
        self.assertLess(
            l1c_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=NOMINAL_SWATH_DURATION,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )

    def test_l1c_memory_usage_critical_tom(self):
        self.assertLess(
            l1c_memory_usage_forecast_MB(
                num_frames=7,
                swath_duration=CRITICAL_SWATH_DURATION,
            ),
            JOB_ORDER_TT_VALUE_MB,
        )


if __name__ == "__main__":
    unittest.main()
