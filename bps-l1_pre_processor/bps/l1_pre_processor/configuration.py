# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 PreProcessor configuration structure
---------------------------------------
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class L1PreProcessorConfiguration:
    """Swath based configuration of the L1 pre processor"""

    class Source(Enum):
        """Internal calibration source"""

        EXTRACTED = "EXTRACTED"
        MODEL = "MODEL"

    max_isp_gap: int

    raw_mean_expected: float
    raw_mean_threshold: float

    raw_std_expected: float
    raw_std_threshold: float

    correct_bias: bool
    correct_gain_imbalance: bool
    correct_non_orthogonality: bool

    internal_calibration_source: Source

    max_drift_amplitude_error: float
    max_drift_amplitude_std_fraction: float

    max_drift_phase_error: float
    max_drift_phase_std_fraction: float

    max_invalid_drift_fraction: float

    enable_channel_delays_annotation: bool | None
    enable_internal_calibration: bool | None

    swath: str | None = None
    """Swath name"""


@dataclass
class L1PreProcessorConfigurationFile:
    """Configuration of the L1 pre processor"""

    configurations: list[L1PreProcessorConfiguration]
