# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2B AGB processor job order
---------------------------
"""

from dataclasses import dataclass
from pathlib import Path

from bps.common.joborder import DeviceResources, ProcessorConfiguration


@dataclass
class L2BAGBProcessingParameters:
    """High level processing parameters"""

    tile_id: str = ""
    """Tile ID to be processed"""


@dataclass
class L2bAGBJobOrder:
    """Job order data for L2B AGB processing."""

    input_l2a_products: tuple[Path, ...]
    """Paths of L2a GN input products"""

    input_l2a_mph_files: tuple[Path, ...]
    """Paths of each l2a MPH main annotation file"""

    output_directory: Path
    """Output products common directory"""

    output_product: str
    """Enabled output product name"""

    aux_pp2_agb_path: Path
    """The auxiliary file: aux_pp2_ab"""

    lcm_product: Path
    """The auxiliary file: LCM, Land Cover Map"""

    cal_ab_product: Path
    """The auxiliary file: Calibration AGB"""

    device_resources: DeviceResources
    """Available device resources"""

    processor_configuration: ProcessorConfiguration
    """Processor configuration"""

    processing_parameters: L2BAGBProcessingParameters
    """High level processing parameters"""

    input_l2b_fd_products: tuple[Path, ...] | None = None
    """L2b FD Products optional input, only INT phase (over green + blue area)"""

    input_l2b_agb_products: tuple[Path, ...] | None = None
    """L2b AGB Producst optional input (over green area)"""

    l2b_agb_p_conf: Path | None = None
    """Additional XML configuration file"""

    output_baseline: int | None = None
    """Output baseline string"""
