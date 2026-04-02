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
from bps.common.io import common
from bps.stack_pre_processor.configuration import PrimaryImageSelectionConf
from bps.stack_pre_processor.core.preprocessing import (
    baseline_ordering,
    compute_coreg_primary_image_index,
    prepare_stack_data,
)


class TestBaselineOrderingIndices(unittest.TestCase):
    """Some tests on computing the normal baseline indices."""

    def test_standard_ordering(self):
        sp_bsl = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        gt = [0, 1, 2, 3, 4, 5, 6]
        dut = baseline_ordering(sp_bsl)
        np.testing.assert_equal(gt, dut)

    def test_inverse_ordering(self):
        sp_bsl = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]
        gt = [6, 5, 4, 3, 2, 1, 0]
        dut = baseline_ordering(sp_bsl)
        np.testing.assert_equal(gt, dut)

    def test_an_nonnominal_ordering(self):
        cb_bsl = [-15, 0, -45, -30, 45, 15, 30]
        gt = [2, 3, 0, 1, 6, 4, 5]
        dut = baseline_ordering(cb_bsl)
        np.testing.assert_equal(gt, dut)

    def test_skip_flip_ordering(self):
        sp_bsl = [3.0, -2.0, 1.0, 0.0, -1.0, 2.0, -3.0]
        gt = [6, 1, 4, 3, 2, 5, 0]
        dut = baseline_ordering(sp_bsl)
        np.testing.assert_equal(gt, dut)


class TestPreProcessing(unittest.TestCase):
    """Minimal testing of the pre-processing utilities."""

    def test_prepare_stack_data(self):
        common_polarizations = [EPolarization.vh, EPolarization.hh]

        data_list = ["data_hh", "data_hv", "data_vh", "data_vv"]
        polarization_list = ["H/H", "H/V", "V/H", "V/V"]
        raster_info_list = ["raster_hh", "raster_hv", "raster_vh", "raster_vv"]
        swath_info_list = ["swath_hh", "swath_hv", "swath_vh", "swath_vv"]

        prepare_stack_data(
            common_polarizations=common_polarizations,
            data_list=data_list,
            polarization_list=polarization_list,
            metadata_list=[
                raster_info_list,
                swath_info_list,
            ],
        )

        self.assertListEqual(data_list, ["data_vh", "data_hh"])
        self.assertListEqual(polarization_list, ["V/H", "H/H"])
        self.assertListEqual(raster_info_list, ["raster_vh", "raster_hh"])
        self.assertListEqual(swath_info_list, ["swath_vh", "swath_hh"])

    def test_compute_coreg_primary_image_index_joborder(self):
        dut_index, _ = compute_coreg_primary_image_index(
            job_order_input_stack=[
                Path("l1a_int1"),
                Path("l1a_int2"),
                Path("l1a_int3"),
            ],
            job_order_primary_image=Path("l1a_int2"),
            # Unused parameters.
            config=PrimaryImageSelectionConf(
                primary_image_selection_information=common.PrimaryImageSelectionInformationType.GEOMETRY,
                rfi_decorrelation_threshold=0.0,
                faraday_decorrelation_threshold=0.0,
            ),
            stack_spatial_ordering=None,
            stack_temporal_ordering=None,
            reference_polarization=None,
            rfi_coherence_degradation_indices=None,
            faraday_decorrelation_indices=None,
        )

        self.assertEqual(dut_index, 1)

    def test_compute_coreg_primary_image_index_geometry(self):
        dut_index, dut_method = compute_coreg_primary_image_index(
            job_order_input_stack=[
                Path("l1a_int1"),
                Path("l1a_int2"),
                Path("l1a_int3"),
            ],
            job_order_primary_image=None,
            config=PrimaryImageSelectionConf(
                primary_image_selection_information=common.PrimaryImageSelectionInformationType.GEOMETRY,
                rfi_decorrelation_threshold=0.0,
                faraday_decorrelation_threshold=0.0,
            ),
            stack_spatial_ordering=np.array([2, 0, 1]),
            # Unused.
            stack_temporal_ordering=None,
            reference_polarization=None,
            rfi_coherence_degradation_indices=None,
            faraday_decorrelation_indices=None,
        )

        self.assertEqual(dut_index, 2)
        self.assertEqual(dut_method, common.PrimaryImageSelectionInformationType.GEOMETRY)

    def test_compute_coreg_primary_image_index_temporal_baseline(self):
        dut_index, dut_method = compute_coreg_primary_image_index(
            job_order_input_stack=[
                Path("l1a_int1"),
                Path("l1a_int2"),
                Path("l1a_int3"),
            ],
            job_order_primary_image=None,
            config=PrimaryImageSelectionConf(
                primary_image_selection_information=common.PrimaryImageSelectionInformationType.TEMPORAL_BASELINE,
                rfi_decorrelation_threshold=0.0,
                faraday_decorrelation_threshold=0.0,
            ),
            stack_temporal_ordering=np.array([1, 0, 2]),
            # Unused.
            stack_spatial_ordering=None,
            reference_polarization=None,
            rfi_coherence_degradation_indices=None,
            faraday_decorrelation_indices=None,
        )

        self.assertEqual(dut_index, 1)
        self.assertEqual(
            dut_method,
            common.PrimaryImageSelectionInformationType.TEMPORAL_BASELINE,
        )

    def test_compute_coreg_primary_image_index_geometry_and_rfi(self):
        dut_index, dut_method = compute_coreg_primary_image_index(
            job_order_input_stack=[
                Path("l1a_int1"),
                Path("l1a_int2"),
                Path("l1a_int3"),
            ],
            job_order_primary_image=None,
            config=PrimaryImageSelectionConf(
                primary_image_selection_information=common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_CORRECTION,
                rfi_decorrelation_threshold=0.5,
                faraday_decorrelation_threshold=0.0,
            ),
            reference_polarization=EPolarization.hh,
            stack_spatial_ordering=np.array([2, 0, 1]),
            rfi_coherence_degradation_indices=(
                {EPolarization.hh: 0.12, EPolarization.vv: 0.99},
                {EPolarization.hh: 0.22, EPolarization.vh: 0.81},
                {EPolarization.hh: 0.95, EPolarization.hv: 0.21},
            ),
            # Unused.
            faraday_decorrelation_indices=(0.0, 0.1, 0.2),
            stack_temporal_ordering=None,
        )

        self.assertEqual(dut_index, 2)
        self.assertEqual(
            dut_method,
            common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_CORRECTION,
        )

    def test_compute_coreg_primary_image_index_geometry_and_fr(self):
        dut_index, dut_method = compute_coreg_primary_image_index(
            job_order_input_stack=[
                Path("l1a_int1"),
                Path("l1a_int2"),
                Path("l1a_int3"),
            ],
            job_order_primary_image=None,
            config=PrimaryImageSelectionConf(
                primary_image_selection_information=common.PrimaryImageSelectionInformationType.GEOMETRY_AND_FR_CORRECTION,
                rfi_decorrelation_threshold=0.0,
                faraday_decorrelation_threshold=0.5,
            ),
            stack_spatial_ordering=np.array([2, 0, 1]),
            faraday_decorrelation_indices=(0.1, 0.9, 0.2),
            rfi_coherence_degradation_indices=({}, {}, {}),
            # Unused.
            reference_polarization=None,
            stack_temporal_ordering=None,
        )

        self.assertEqual(dut_index, 1)
        self.assertEqual(
            dut_method,
            common.PrimaryImageSelectionInformationType.GEOMETRY_AND_FR_CORRECTION,
        )

    def test_compute_coreg_primary_image_index_geometry_and_rfi_fr(self):
        dut_index, dut_method = compute_coreg_primary_image_index(
            job_order_input_stack=[
                Path("l1a_int1"),
                Path("l1a_int2"),
                Path("l1a_int3"),
            ],
            job_order_primary_image=None,
            config=PrimaryImageSelectionConf(
                primary_image_selection_information=common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS,
                rfi_decorrelation_threshold=0.5,
                faraday_decorrelation_threshold=0.5,
            ),
            stack_spatial_ordering=np.array([1, 2, 0]),
            reference_polarization=EPolarization.hh,
            rfi_coherence_degradation_indices=(
                {EPolarization.hh: 0.12, EPolarization.vv: 0.99},
                {EPolarization.hh: 0.91, EPolarization.vh: 0.81},
                {EPolarization.hh: 0.9, EPolarization.hv: 0.21},
            ),
            faraday_decorrelation_indices=(0.1, 0.99, 0.5),
            # Unused.
            stack_temporal_ordering=None,
        )

        self.assertEqual(dut_index, 1)
        self.assertEqual(
            dut_method,
            common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS,
        )

    def test_compute_coreg_primary_image_index_fallback_geometry(self):
        dut_index, dut_method = compute_coreg_primary_image_index(
            job_order_input_stack=[
                Path("l1a_int1"),
                Path("l1a_int2"),
                Path("l1a_int3"),
            ],
            job_order_primary_image=None,
            config=PrimaryImageSelectionConf(
                primary_image_selection_information=common.PrimaryImageSelectionInformationType.GEOMETRY_AND_RFI_FR_CORRECTIONS,
                rfi_decorrelation_threshold=0.99,
                faraday_decorrelation_threshold=0.99,
            ),
            stack_spatial_ordering=np.array([1, 2, 0]),
            reference_polarization=EPolarization.hh,
            rfi_coherence_degradation_indices=(
                {EPolarization.hh: 0.12, EPolarization.vv: 0.98},
                {EPolarization.hh: 0.91, EPolarization.vh: 0.81},
                {EPolarization.hh: 0.9, EPolarization.hv: 0.21},
            ),
            faraday_decorrelation_indices=(0.1, 0.99, 0.5),
            # Unused.
            stack_temporal_ordering=None,
        )

        self.assertEqual(dut_index, 1)
        self.assertEqual(dut_method, common.PrimaryImageSelectionInformationType.GEOMETRY)


if __name__ == "__main__":
    unittest.main()
