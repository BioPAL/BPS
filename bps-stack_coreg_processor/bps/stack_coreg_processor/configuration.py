# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Coregistration Executable Internal Configurations
-------------------------------------------------
"""

from dataclasses import dataclass, field
from enum import Enum

from arepytools.io.metadata import EPolarization
from bps.common.utils import LogLevel


@dataclass
class GeneralCoregStackProcessorInternalConfiguration:
    """Internal general configuration of the coregistratior's binary"""

    @dataclass
    class LoggerConfType:
        """Stack core logging configuration"""

        enable_log_file: bool | None = False
        enable_std_out: bool | None = False
        report_level: LogLevel | None = LogLevel.LOW

    logger_conf: LoggerConfType = field(default_factory=LoggerConfType)


@dataclass
class FullAccuracyPreProcessingConfType:
    """Configuration of the pre-processing step for Full-Accuracy coregistration."""

    coreg_reference_polarization: EPolarization  # Set from AUX-PPS.
    enable_common_band_range_filter: int = 0
    range_max_shift: int = 5
    azimuth_max_shift: int = 5
    range_block_size: int = 51
    azimuth_block_size: int = 101
    coarse_input: int = 0
    range_min_overlap: int = 0
    azimuth_min_overlap: int = 0
    fitting_quality_threshold: float = 0.0
    memory: int = 256
    verbose: int = 0
    report_level: int = 0


@dataclass
class NonStationaryAreasConfType:
    """Configuration of the non-stationary areas."""

    low_pass_filter_type: str
    low_pass_filter_order: int
    low_pass_filter_std_dev: float


@dataclass
class FullAccuracyPostProcessingConfType:
    """Configuration of the post-processing step for Full-Accuracy coregistration."""

    quality_threshold_for_automatic_mode: float  # Set from AUX-PPS.
    min_valid_blocks: int  # Set from AUX-PPS.
    weight_threshold_refine_rg: float = 0.01
    weight_threshold_refine_az: float = 0.01
    non_stationary_coreg_conf: NonStationaryAreasConfType | None = None


@dataclass
class ReinterpolationConfType:
    """Configuration of the warping step."""

    filter_length: int = 11
    bank_size: int = 101
    bandwidth: float = 0.80000001
    range_overlap: int = 1
    demodulation_type: int = 0
    unsigned_flag: int = 0
    memory: int = 256
    verbose: int = 0
    report_level: int = 0


@dataclass
class CoregistrationOutputProductsConfType:
    """What products the coregistrator writes on disk."""

    remove_ancillary_coregistration_data: int = 1
    provide_coregistration_shifts: int = 1
    provide_geometry_shifts: int = 1
    provide_coregistration_accuracy_stats: int = 1
    xcorr_azimuth_min_overlap: int | None = None
    provide_products_for_each_polarization: int = 0
    provide_wavenumbers: int = 1
    shifts_only_estimation: int = 0
    provide_absolute_primary_distance: int = 1


@dataclass
class CoregStackProcessorInternalConfiguration:
    """Internal coregistration parameters of the coregistrator binary."""

    class CoregMode(Enum):
        """Coregistration modalities."""

        FULL_ACCURACY = "FULL_ACCURACY"  # i.e. Geometry And Data.
        GEOMETRY = "GEOMETRY"
        AUTOMATIC = "AUTOMATIC"

    coreg_mode: CoregMode  # This is set from AUX-PPS.
    full_accuracy_preproc_conf: FullAccuracyPreProcessingConfType = field(
        default_factory=FullAccuracyPreProcessingConfType
    )
    full_accuracy_postproc_conf: FullAccuracyPostProcessingConfType = field(
        default_factory=FullAccuracyPostProcessingConfType
    )
    reinterp_conf: ReinterpolationConfType = field(default_factory=ReinterpolationConfType)
    coreg_output_products_conf: CoregistrationOutputProductsConfType = field(
        default_factory=CoregistrationOutputProductsConfType
    )
    temp_remove_flag: int = 1
    memory_sar_geometry: int = 256
