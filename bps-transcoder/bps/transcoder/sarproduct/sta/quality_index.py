# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1c Quality Index
-----------------
"""

from __future__ import annotations

from dataclasses import dataclass

from bps.transcoder.utils.quality_index import QualityIndex


@dataclass
class StackQualityIndex(QualityIndex):
    """Quality index of STA_P."""

    l1a_source_product_is_invalid: bool = False

    pre_processor_expected_lut_is_missing: bool = False
    pre_processor_input_with_non_nominal_set_of_polarizations: bool = False
    pre_processor_input_with_non_nominal_swath_duration: bool = False
    pre_processor_input_with_inconsistent_phase_screen_and_range_shifts_luts: bool = False

    coregistration_fitting_quality_below_threshold: bool = False
    coregistration_shift_accuracy_below_threshold: bool = False

    slow_ionosphere_removal_misses_l1_iono_phase_screen: bool = False
    slow_ionosphere_removal_estimation_quality_below_threshold: bool = False
    slow_ionosphere_removal_invalid_interferometric_pair_in_mb_estimation: bool = False
    slow_ionosphere_removal_above_latitude_threshold: bool = False

    sum_kronecker_products_estimation_window_too_small_wrt_dem_resolution: bool = False
    sum_kronecker_products_phase_screen_estimation_failure: bool = False
    sum_kronecker_products_phase_screen_correction_failure: bool = False
    sum_kronecker_products_estimation_quality_below_threshold: bool = False
    sum_kronecker_products_unsupported_number_of_polarizations: bool = False

    l1a_coregistration_primary_source_product_is_invalid: bool = False
    l1a_calibration_reference_source_product_is_invalid: bool = False
    l1a_stack_contains_other_invalid_products: bool = False

    _bit_map = {
        # Input control's bit (1).
        "l1a_source_product_is_invalid": 0,
        # Preprocessor's bits (2-5).
        "pre_processor_expected_lut_is_missing": 1,
        "pre_processor_input_with_non_nominal_set_of_polarizations": 2,
        "pre_processor_input_with_non_nominal_swath_duration": 3,
        "pre_processor_input_with_inconsistent_phase_screen_and_range_shifts_luts": 4,
        # Coregistration's bits (6-9). Bits #8 and #9 unassigned.
        "coregistration_fitting_quality_below_threshold": 5,
        "coregistration_shift_accuracy_below_threshold": 6,
        # AZF bits (10-11). Bits #10 and #11 unassigned.
        # IOB bits (12-15).
        "slow_ionosphere_removal_misses_l1_iono_phase_screen": 11,
        "slow_ionosphere_removal_estimation_quality_below_threshold": 12,
        "slow_ionosphere_removal_invalid_interferometric_pair_in_mb_estimation": 13,
        "slow_ionosphere_removal_above_latitude_threshold": 14,
        # InSAR calibration bits (16-22). All unassigned.
        # SKP bits (24-29). Bit #29 unassigned.
        "sum_kronecker_products_estimation_window_too_small_wrt_dem_resolution": 23,
        "sum_kronecker_products_phase_screen_estimation_failure": 24,
        "sum_kronecker_products_phase_screen_correction_failure": 25,
        "sum_kronecker_products_estimation_quality_below_threshold": 26,
        "sum_kronecker_products_unsupported_number_of_polarizations": 27,
        # Input L1a stack bits (30-32).
        "l1a_coregistration_primary_source_product_is_invalid": 29,
        "l1a_calibration_reference_source_product_is_invalid": 30,
        "l1a_stack_contains_other_invalid_products": 31,
    }
