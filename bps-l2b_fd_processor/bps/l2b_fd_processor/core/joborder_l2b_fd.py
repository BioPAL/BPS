# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2B FD processor job order
--------------------------
"""

from dataclasses import dataclass
from pathlib import Path

from bps.common.joborder import DeviceResources, ProcessorConfiguration, TileProcessingParameters


@dataclass
class L2bFDJobOrder:
    """Job order data for L2B FH processing."""

    input_l2a_products: list[Path]
    """Paths of L2a input products"""

    output_directory: Path
    """Output products common directory"""

    output_product: str
    """Enabled output product name"""

    aux_pp2_fd_path: Path
    """The auxiliary file: aux_pp2_fd"""

    device_resources: DeviceResources
    """Available device resources"""

    processor_configuration: ProcessorConfiguration
    """Processor configuration"""

    processing_parameters: TileProcessingParameters
    """High level processing parameters"""

    l2b_fd_p_conf: Path | None = None
    """Additional XML configuration file"""

    output_baseline: int | None = None
    """Output baseline string"""
