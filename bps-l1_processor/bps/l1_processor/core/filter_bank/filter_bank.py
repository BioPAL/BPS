# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Filter bank utils
-----------------
"""

from enum import Enum, auto
from pathlib import Path
from typing import Literal

import numpy as np
from bps.common import bps_logger
from bps.l1_processor.core.filter_bank.filter_creation import (
    FirFilterBuilder,
    SincFilterBuilder,
    create_filter_bank,
)
from bps.l1_processor.core.filter_bank.filter_write import write_filter_bank_product


class FilterType(Enum):
    """Type of filters"""

    FIR = auto()
    SINC = auto()


def create_filter_builder(
    filter_type: FilterType,
    input_sampling_frequency: float,
    filter_bandwidth: float,
    ovs_factor: int,
):
    bandwidth = min(filter_bandwidth / input_sampling_frequency, 0.99)
    bps_logger.debug(
        f"Resampling filter: relative bandwidth: {bandwidth}, input sampling frequency: {input_sampling_frequency}"
    )

    if filter_type == FilterType.FIR:
        bps_logger.debug("Resampling filter: FIR filter selected")
        fir_bands = np.array([0, bandwidth / ovs_factor, bandwidth, 1])
        fir_desired_gains = np.array([1, 1, 0, 0])
        return FirFilterBuilder(fir_bands, fir_desired_gains)
    elif filter_type == FilterType.SINC:
        bps_logger.debug("Resampling filter: SINC filter selected")
        if ovs_factor != 1:
            raise RuntimeError("Oversampling not supported")
        return SincFilterBuilder(bandwidth=bandwidth)
    raise RuntimeError("Unknown filter")


def write_resampling_filter(
    resampling_product: Path,
    filter_type: Literal["SINC", "FIR"],
    filter_bandwidth,
    sampling_frequency,
    filter_length,
    bank_size,
):
    """Write resampling filter product"""
    oversampling_factor = 1
    sampling_step = 1.0 / oversampling_factor

    filter_builder = create_filter_builder(
        FilterType[filter_type],
        sampling_frequency,
        filter_bandwidth,
        oversampling_factor,
    )

    filter_bank = create_filter_bank(
        filter_builder,
        filter_length,
        bank_size,
    )

    write_filter_bank_product(filter_bank, resampling_product, sampling_step)
