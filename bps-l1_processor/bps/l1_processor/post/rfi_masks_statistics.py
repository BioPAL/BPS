# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
RFI Masks statistics computation module
---------------------------------------
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
from arepytools.io import (
    iter_channels,
    metadata,
    open_product_folder,
    read_raster_with_raster_info,
)
from bps.common import Polarization
from bps.transcoder.sarproduct.l1_annotations import (
    FrequencyMaskStats,
    RFIMasksStatistics,
    TimeMaskStats,
)


def _translate_epolarization_to_biomass(pol: metadata.EPolarization) -> Polarization:
    return Polarization(pol.name.upper())


def compute_rfi_frequency_domain_stats(mask: Path) -> list[FrequencyMaskStats]:
    """RFI frequency domain stats"""

    class FreqMaskValues(Enum):
        """Possible non-zero mask value"""

        ISOLATED = 1
        PERSISTENT = 2
        BOTH = 3

    statistics = []

    pf = open_product_folder(mask)
    for channel_id, channel_metadata in iter_channels(pf):
        raster_info = channel_metadata.get_raster_info()

        data = read_raster_with_raster_info(pf.get_channel_data(channel_id), raster_info)

        both = data == FreqMaskValues.BOTH.value
        isolated = np.logical_or(data == FreqMaskValues.ISOLATED.value, both)
        persistent = np.logical_or(data == FreqMaskValues.PERSISTENT.value, both)

        isolated_affected_samples_per_lines = np.sum(isolated, axis=1)
        isolated_affected_lines_percentage = (
            np.count_nonzero(isolated_affected_samples_per_lines) / raster_info.lines
        ) * 100
        isolated_max_affected_bandwidth_percentage = (
            np.max(isolated_affected_samples_per_lines) / raster_info.samples
        ) * 100
        isolated_avg_affected_bandwidth_percentage = (
            np.mean(isolated_affected_samples_per_lines) / raster_info.samples
        ) * 100

        persistent_affected_samples_per_lines = np.sum(persistent, axis=1)
        persistent_affected_lines_percentage = (
            np.count_nonzero(persistent_affected_samples_per_lines) / raster_info.lines
        ) * 100
        persistent_max_affected_bandwidth_percentage = (
            np.max(persistent_affected_samples_per_lines) / raster_info.samples
        ) * 100
        persistent_avg_affected_bandwidth_percentage = (
            np.mean(persistent_affected_samples_per_lines) / raster_info.samples
        ) * 100

        statistics.append(
            FrequencyMaskStats(
                polarization=_translate_epolarization_to_biomass(channel_metadata.get_swath_info().polarization),
                isolated_affected_lines_percentage=isolated_affected_lines_percentage,
                isolated_max_affected_bandwidth_percentage=isolated_max_affected_bandwidth_percentage,
                isolated_avg_affected_bandwidth_percentage=isolated_avg_affected_bandwidth_percentage,
                persistent_max_affected_bandwidth_percentage=persistent_max_affected_bandwidth_percentage,
                persistent_avg_affected_bandwidth_percentage=persistent_avg_affected_bandwidth_percentage,
                persistent_affected_lines_percentage=persistent_affected_lines_percentage,
            )
        )
    return statistics


def compute_rfi_time_domain_stats(mask: Path) -> list[TimeMaskStats]:
    """RFI time domain statistics computation"""

    statistics = []
    pf = open_product_folder(mask)
    for channel_id, channel_metadata in iter_channels(pf):
        raster_info = channel_metadata.get_raster_info()

        data = read_raster_with_raster_info(pf.get_channel_data(channel_id), raster_info)

        affected_samples_per_lines = np.sum(data, axis=1)
        affected_lines_percentage = (np.count_nonzero(affected_samples_per_lines) / raster_info.lines) * 100
        avg_affected_samples_percentage = (np.mean(affected_samples_per_lines) / raster_info.samples) * 100
        max_affected_samples_percentage = (np.max(affected_samples_per_lines) / raster_info.samples) * 100

        statistics.append(
            TimeMaskStats(
                polarization=_translate_epolarization_to_biomass(channel_metadata.get_swath_info().polarization),
                affected_lines_percentage=affected_lines_percentage,
                average_affected_samples_percentage=avg_affected_samples_percentage,
                max_affected_samples_percentage=max_affected_samples_percentage,
            )
        )

    return statistics


def compute_rfi_masks_statistics(rfi_frequency_mask: Path | None, rfi_time_mask: Path | None) -> RFIMasksStatistics:
    rfi_frequency_stats = []
    if rfi_frequency_mask is not None and rfi_frequency_mask.exists():
        rfi_frequency_stats = compute_rfi_frequency_domain_stats(rfi_frequency_mask)

    rfi_time_stats = []
    if rfi_time_mask is not None and rfi_time_mask.exists():
        rfi_time_stats = compute_rfi_time_domain_stats(rfi_time_mask)

    return RFIMasksStatistics(time_stats=rfi_time_stats, freq_stats=rfi_frequency_stats)
