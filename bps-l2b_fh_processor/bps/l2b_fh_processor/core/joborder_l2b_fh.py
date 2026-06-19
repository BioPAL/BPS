# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2B FH processor job order
--------------------------
"""

from dataclasses import dataclass
from pathlib import Path

from bps.common.joborder import DeviceResources, ProcessorConfiguration, TileProcessingParameters


@dataclass
class L2bFHJobOrder:
    """Job order data for L2B FH processing."""

    input_l2a_products: list[Path]
    """Paths of L2a input products"""

    output_directory: Path
    """Output products common directory"""

    output_product: str
    """Enabled output product name"""

    aux_pp2_fh_path: Path
    """The auxiliary file: aux_pp2_fh"""

    device_resources: DeviceResources
    """Available device resources"""

    processor_configuration: ProcessorConfiguration
    """Processor configuration"""

    processing_parameters: TileProcessingParameters
    """High level processing parameters"""

    input_l2b_fd_product: Path | None = None
    """L2b FD Product optional input"""

    l2b_fh_p_conf: Path | None = None
    """Additional XML configuration file"""

    output_baseline: int | None = None
    """Output baseline string"""
