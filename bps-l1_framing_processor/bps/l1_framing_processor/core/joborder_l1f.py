# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 framing processor job order
------------------------------
"""

from dataclasses import dataclass
from pathlib import Path

from bps.common.joborder import DeviceResources, ProcessorConfiguration


@dataclass
class L1StripmapInputProducts:
    """Relevant Input products when processing stripmap products"""

    input_standard: Path
    """Sx_RAW__0S, Stripmap Level-0 - Standard product"""

    input_monitoring: Path
    """Sx_RAW__0M, Stripmap Level-0 - Monitoring product"""


@dataclass
class L1VirtualFrameOutputProducts:
    """Relevant Output products when processing stripmap products"""

    output_directory: Path
    """Output products directory"""

    output_baseline: int
    """Output products baseline"""

    vfra_standard_required: bool
    """CPF_L1VFRA, L1 Virtual Frame product"""


@dataclass
class L1StripmapProducts:
    """Relevant I/O products when processing stripmap products"""

    input: L1StripmapInputProducts
    """input products"""

    output: L1VirtualFrameOutputProducts
    """output products"""


@dataclass
class L1AuxiliaryProducts:
    """Job order auxiliary files"""

    orbit: Path
    """AUX_ORB, Auxiliary Orbit"""


@dataclass
class L1FJobOrder:
    """Job order data for L1 processing."""

    io_products: L1StripmapProducts
    """Input/Output products"""

    auxiliary_files: L1AuxiliaryProducts
    """The auxiliary files"""

    device_resources: DeviceResources
    """Available device resources"""

    processor_configuration: ProcessorConfiguration
    """Processor configuration"""

    intermediate_files: dict[str, str]
    """Intermediate files map: id to filename"""
