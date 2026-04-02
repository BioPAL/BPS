# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest
from pathlib import Path

import numpy as np
from arepytools.io.metadata import EPolarization
from bps.stack_cal_processor.input_manager import select_calibration_reference_image

# Just some random input stack names.
_INPUT_STACK = list(Path(f"/path/to/l1a_frame{n}") for n in range(1, 8))


class SelectCalibReferenceTests(unittest.TestCase):
    """Test routines for selecting the calibration reference image."""

    def test_select_calibration_reference_image_user_provided(self):
        gt_index = 1
        dut_index = select_calibration_reference_image(
            coreg_primary_image_index=2,
            reference=gt_index,
            input_stack_paths=_INPUT_STACK,
        )
        self.assertEqual(dut_index, gt_index)

    def test_select_calibration_reference_image_index_rfi_indices(self):
        np.random.seed(seed=0)
        rfi_indices = np.random.rand(len(_INPUT_STACK))
        gt_index = np.argmax(rfi_indices)
        dut_index = select_calibration_reference_image(
            coreg_primary_image_index=np.argmin(rfi_indices),
            polarization=EPolarization.hh,
            rfi_indices=[{EPolarization.hh: rfi} for rfi in rfi_indices],
            faraday_decorrelation_indices=None,
            input_stack_paths=_INPUT_STACK,
        )
        self.assertEqual(gt_index, dut_index)

    def test_select_calibration_reference_image_index_fr_indices(self):
        np.random.seed(seed=1)
        fr_indices = np.random.rand(len(_INPUT_STACK))
        gt_index = np.argmax(fr_indices)
        dut_index = select_calibration_reference_image(
            coreg_primary_image_index=np.argmin(fr_indices),
            polarization=EPolarization.hh,
            rfi_indices=None,
            faraday_decorrelation_indices=fr_indices,
            input_stack_paths=_INPUT_STACK,
        )
        self.assertEqual(gt_index, dut_index)

    def test_select_calibration_reference_image_index_rfi_fr_indices(self):
        np.random.seed(seed=2)
        rfi_indices = np.random.rand(len(_INPUT_STACK))
        fr_indices = np.random.rand(len(_INPUT_STACK))
        gt_index = np.argmax(fr_indices * rfi_indices)
        dut_index = select_calibration_reference_image(
            coreg_primary_image_index=np.argmin(fr_indices * rfi_indices),
            polarization=EPolarization.vh,
            rfi_indices=[{EPolarization.vh: rfi} for rfi in rfi_indices],
            faraday_decorrelation_indices=fr_indices,
            input_stack_paths=_INPUT_STACK,
        )
        self.assertEqual(gt_index, dut_index)

    def test_select_calibration_reference_image_fallback_missing_rfi_fr_indices(self):
        gt_index = 2
        dut_index = select_calibration_reference_image(
            coreg_primary_image_index=gt_index,
            rfi_indices=None,
            faraday_decorrelation_indices=None,
            input_stack_paths=_INPUT_STACK,
        )
        self.assertEqual(dut_index, gt_index)

    def test_select_calibration_reference_image_index_fallback_all_NaNs(self):
        gt_index = 2
        dut_index = select_calibration_reference_image(
            coreg_primary_image_index=gt_index,
            polarization=EPolarization.xx,
            rfi_indices=[{EPolarization.xx: np.nan} for _ in _INPUT_STACK],
            faraday_decorrelation_indices=np.full((len(_INPUT_STACK),), np.nan),
            input_stack_paths=_INPUT_STACK,
        )
        self.assertEqual(dut_index, gt_index)


if __name__ == "__main__":
    unittest.main()
