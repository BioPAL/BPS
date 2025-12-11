# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack processor job order
-------------------------
"""

from dataclasses import dataclass
from pathlib import Path

from bps.common.joborder import DeviceResources, ProcessorConfiguration


@dataclass
class StackProcessingParameters:
    """High level processing parameters."""

    primary_image: Path | None = None
    """Path of L1 product to be selected as primary image in the coregistration."""

    calibration_primary_image: Path | None = None
    """Path of L1 product to be selected as primary image in the multi-baseline calibration."""

    range_interval: tuple[float, float] | None = None
    """Interval in range times - start, stop [s]."""


@dataclass
class StackExternalsProducts:
    """External products."""

    dem_database_entry_point: Path | None = None
    """DEM database entry point location."""

    fnf_database_entry_point: Path | None = None
    """FNF database entry point location."""

    lcm_database_entry_point: Path | None = None
    """LCM database entry point location."""


@dataclass
class StackOutputProducts:
    """Relevant Output products when processing stripmap products."""

    output_directory: Path
    """Output products directory."""

    sta_standard_required: bool
    """Sx_STA__1S, Standard stack product."""

    sta_monitoring_required: bool
    """Sx_STA__1M, Monitoring stack product."""

    product_baseline: int
    """Baseline specified in the JobOrder"""


@dataclass
class StackJobOrder:
    """Job order data for Stack processing."""

    input_stack: tuple[Path, ...]
    """Stack of SCS products."""

    output_path: StackOutputProducts
    """Outputs directory of Stack processor."""

    auxiliary_files: Path
    """The auxiliary files - pps."""

    external_products: StackExternalsProducts
    """External products."""

    device_resources: DeviceResources
    """Available device resources."""

    processor_configuration: ProcessorConfiguration
    """Processor configuration."""

    config_file: Path | None
    """Additional XML configuration file."""

    processing_parameters: StackProcessingParameters | None
    """High level processing parameters."""

    intermediate_files: dict[str, str]
    """Intermediate files map: id to filename."""
