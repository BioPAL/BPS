# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Coregistration input file structure
-----------------------------------------
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CoregProcessorInputFile:
    """Input file of the Stack Coregistration Processor app"""

    primary_product: Path
    """Input primary product"""

    secondary_product: Path
    """Input secondary product"""

    ecef_grid_product: Path
    """ECEF grid product"""

    output_path: Path
    """Output path"""

    coreg_conf_file: Path | None = None
    """Coregistration configuration XML file"""

    rg_shifts_product: Path | None = None
    """Optionally, externally provided range coreg shift product"""

    az_shifts_product: Path | None = None
    """Optionally, externally provided azimuth coreg shift product"""


@dataclass
class BPSCoregProcessorInputFile:
    """Input file of the BPS Stack Coregistration Processor app"""

    coregistration_input: CoregProcessorInputFile
    """Input file of the Stack Geometry Processor app"""

    bps_configuration_file: Path
    """BPS configuration file"""

    bps_log_file: Path
    """BPS log file"""
