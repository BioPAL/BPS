# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Preprocessor Processor interface module
---------------------------------------------
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class EarthGeometryType(Enum):
    """Supported Earth's models."""

    ELLIPSOID = "ELLIPSOID"
    DEM = "DEM"


@dataclass
class GeometryProcessorInputFile:
    """Input file of the Stack Geometry Processor app."""

    primary_product: Path
    """Input primary product."""

    secondary_product: Path
    """Input secondary product."""

    output_path: Path
    """Output path."""

    earth_geometry: EarthGeometryType | None = EarthGeometryType("DEM")
    """Earth geometry type."""

    geometry_conf_file: Path | None = None
    """Coregistration configuration XML file."""

    general_conf_file: Path | None = None
    """General configuration XML file."""


@dataclass
class BPSGeometryProcessorInputFile:
    """Input file of the BPS Stack Geometry Processor app."""

    geometry_input: GeometryProcessorInputFile
    """Input file of the Stack Geometry Processor app."""

    bps_configuration_file: Path
    """BPS configuration file."""

    bps_log_file: Path
    """BPS log file."""
