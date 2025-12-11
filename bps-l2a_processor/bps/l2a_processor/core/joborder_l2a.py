# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2A processor job order
-----------------------
"""

from dataclasses import dataclass
from pathlib import Path

from bps.common.joborder import DeviceResources, ProcessorConfiguration


@dataclass
class L2aJobOrder:
    """Job order data for L2A processing."""

    input_stack_acquisitions: tuple[Path, ...]
    """Stack of L1c input products"""

    input_stack_mph_files: tuple[Path, ...]
    """Paths of each stack acquisition MPH main annotation file"""

    output_directory: Path
    """Output products common directory"""

    output_products: list[str]
    """Enabled output products names"""

    aux_pp2_2a_path: Path
    """The auxiliary file: aux_pp2_2a"""

    fnf_directory: Path
    """The FNF directory"""

    device_resources: DeviceResources
    """Available device resources"""

    processor_configuration: ProcessorConfiguration
    """Processor configuration"""

    input_l2a_fd_product: Path | None = None
    """L2a FD Product optional input"""

    l2a_p_conf: Path | None = None
    """Additional XML configuration file"""

    output_baselines: list[int] | None = None
    """Output baselines strings"""
