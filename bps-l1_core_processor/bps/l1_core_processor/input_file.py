# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Input file structures
---------------------
"""

from dataclasses import dataclass, field
from pathlib import Path

from bps.l1_core_processor.pf_selector_input_file import (
    PFSelectorAreaOptions,
    PFSelectorPolarization,
)


@dataclass
class AntennaProducts:
    """Biomass antenna pattern products"""

    d1h_pattern_product: Path
    """First doublet, h polarization"""

    d2h_pattern_product: Path
    """Second doublet, h polarization"""

    d1v_pattern_product: Path
    """First doublet, v polarization"""

    d2v_pattern_product: Path
    """Second doublet, v polarization"""

    tx_power_tracking_product: Path
    """Transmission power tracking"""


@dataclass
class CoreProcessorInputs:
    """Core processor input file data"""

    input_level0_product: Path
    """Input L0 extracted product"""

    processing_options_file: Path
    """Processing option XML file"""
    processing_parameters_file: Path
    """Processing parameters XML file"""

    output_directory: Path
    """Output directory"""

    input_chirp_replica_product: Path | None = None
    """Input chirp replica product"""
    input_per_line_dechirping_reference_times_product: Path | None = None
    """Input per line dechirping reference time"""
    input_per_line_correction_factors_product: Path | None = None
    """Input per line correction factors product"""
    input_noise_product: Path | None = None
    """Input estimated noise product"""
    input_processing_dc_poly_file_name: Path | None = None
    """Input doppler centroid polynomials file"""

    polarization_to_process: list[PFSelectorPolarization] = field(default_factory=list)
    """Which polarizations to process"""
    area_to_process: PFSelectorAreaOptions | None = None
    """Which area to process"""


@dataclass
class BPSL1CoreProcessorInputFile:
    """Input file of the BPS L1 Core processor"""

    core_processor_input: CoreProcessorInputs
    """Core input parameters"""

    input_antenna_products: AntennaProducts | None
    """Biomass antenna pattern products"""

    input_geomagnetic_field_model_product: Path | None
    """Input geomagnetic field folder containing the model"""

    input_tec_map_product: Path | None
    """Input tec map field product"""

    input_climatological_model_file: Path | None
    """Input xml file with ionospheric height"""

    input_faraday_rotation_product: Path | None
    """Input Faraday rotation product"""

    input_phase_screen_product: Path | None
    """Input phase screen product"""

    bps_configuration_file: Path
    """BPS configuration file"""

    bps_log_file: Path
    """BPS log file"""
