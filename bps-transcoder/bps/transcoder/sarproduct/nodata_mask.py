# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class _SamplingFactors:
    azimuth_up_sampling: int
    azimuth_down_sampling: int
    range_up_sampling: int
    range_down_sampling: int

    @classmethod
    def from_processing_parameters(cls, processing_parameters, swath: str):
        """Retrieve factors from configuration of given swath"""
        return cls(
            azimuth_up_sampling=processing_parameters.azimuth_upsampling_factor[swath],
            azimuth_down_sampling=processing_parameters.azimuth_downsampling_factor[swath],
            range_up_sampling=processing_parameters.range_upsampling_factor[swath],
            range_down_sampling=processing_parameters.range_downsampling_factor[swath],
        )


def _create_no_data_mask_from_margins(data_shape: tuple[int, int], margins: tuple[int, int]):
    nodata_mask = np.full(tuple(np.subtract(data_shape, margins)), False)
    nodata_mask = np.pad(
        nodata_mask,
        (
            (int(margins[0] / 2),) * 2,
            (int(margins[1] / 2),) * 2,
        ),
        mode="constant",
        constant_values=True,
    )
    return nodata_mask


class _MarginsComputer(Protocol):
    def compute(self, overlaps: tuple[int, int]) -> tuple[int, int]: ...


def _create_no_data_mask_on_focusing_margins_kept(
    data_shape: tuple[int, int],
    margins_computer: _MarginsComputer,
    overlaps: tuple[int, int],
):
    borders = margins_computer.compute(overlaps)
    return _create_no_data_mask_from_margins(data_shape, borders)


def _compute_margins_for_dgm(overlaps, sampling_factors: _SamplingFactors):
    margins_multipliers = (
        sampling_factors.azimuth_up_sampling / sampling_factors.azimuth_down_sampling,
        sampling_factors.range_up_sampling / sampling_factors.range_down_sampling / 2,
    )
    margins = tuple(np.multiply(overlaps, margins_multipliers))
    margins = tuple(round(v / 2) * 2 for v in margins)
    return margins


@dataclass
class _DGMMarginsComputer:
    sampling_factors: _SamplingFactors

    def compute(self, overlaps: tuple[int, int]) -> tuple[int, int]:
        """Apply DGM resampling"""
        return _compute_margins_for_dgm(overlaps, self.sampling_factors)


class _SCSMarginsComputer:
    def compute(self, overlaps: tuple[int, int]) -> tuple[int, int]:
        """Nothing to do"""
        return overlaps


def create_no_data_mask(data_shape, product_type: str, processing_parameters, swath: str):
    """Compute nodata mask based on processing configuration and data level"""

    azimuth_focusing_margins_removed = processing_parameters.azimuth_focusing_margins_removal_flag
    if azimuth_focusing_margins_removed:
        return np.full(data_shape, False)

    overlaps = (
        processing_parameters.azimuth_compression_block_overlap_lines,
        processing_parameters.azimuth_compression_block_overlap_samples,
    )
    if product_type == "DGM":
        sampling_factors = _SamplingFactors.from_processing_parameters(processing_parameters, swath)
        margins_computer = _DGMMarginsComputer(sampling_factors)
    else:
        margins_computer = _SCSMarginsComputer()

    return _create_no_data_mask_on_focusing_margins_kept(data_shape, margins_computer, overlaps)
