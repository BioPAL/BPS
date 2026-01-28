# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""BPS L1ab quality index"""

from __future__ import annotations

from dataclasses import dataclass

from bps.transcoder.utils.quality_index import QualityIndex


@dataclass
class L1QualityIndex(QualityIndex):
    """L1 Quality Index"""

    raw_data_stats_out_of_boundaries: bool = False

    int_cal_sequence_above_threshold: bool = False

    dc_fallback_activated: bool = False

    dc_rmse_above_threshold: bool = False

    ionosphere_height_estimation_failure: bool = False

    iri_model_used_as_fallback: bool = False

    gaussian_filter_size_out_of_boundaries: bool = False

    inconsistent_phasescreen_and_rangeshifts_luts: bool = False

    geomagnetic_equator_fallback_activated: bool = False

    number_of_failing_estimations_above_threshold: bool = False

    _bit_map = {
        "raw_data_stats_out_of_boundaries": 4,
        "int_cal_sequence_above_threshold": 8,
        "dc_fallback_activated": 16,
        "dc_rmse_above_threshold": 17,
        "ionosphere_height_estimation_failure": 24,
        "iri_model_used_as_fallback": 25,
        "gaussian_filter_size_out_of_boundaries": 26,
        "inconsistent_phasescreen_and_rangeshifts_luts": 27,
        "geomagnetic_equator_fallback_activated": 28,
        "number_of_failing_estimations_above_threshold": 30,
    }
