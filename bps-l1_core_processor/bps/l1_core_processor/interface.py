# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 Core Processor interface module
----------------------------------
"""

from dataclasses import dataclass
from pathlib import Path

from bps.l1_core_processor.input_file import BPSL1CoreProcessorInputFile
from bps.l1_core_processor.processing_options import BPSL1CoreProcessingOptions
from bps.l1_core_processor.processing_parameters import SarfocProcessingParameters
from bps.l1_core_processor.serialization import (
    serialize_bps_l1_core_processor_input_file,
    serialize_bps_l1_core_processor_processing_options,
    serialize_sarfoc_processing_parameters,
)


def write_l1_coreproc_input_file(input_file: BPSL1CoreProcessorInputFile, file: Path):
    """Write L1 Core Processor input file"""
    file.write_text(serialize_bps_l1_core_processor_input_file(input_file), encoding="utf-8")


def write_l1_coreproc_options_file(options: BPSL1CoreProcessingOptions, file: Path):
    """Write L1 Core Processor processing options file"""
    file.write_text(serialize_bps_l1_core_processor_processing_options(options), encoding="utf-8")


def write_l1_coreproc_parameters_file(params: SarfocProcessingParameters, file: Path):
    """Write L1 Core Processor processing parameters file"""
    file.write_text(serialize_sarfoc_processing_parameters(params), encoding="utf-8")


@dataclass
class L1CoreProcessorInterface:
    """L1 Core Processor interface objects"""

    input_file: BPSL1CoreProcessorInputFile
    """Input file"""
    options: BPSL1CoreProcessingOptions
    """Configuration"""
    params: SarfocProcessingParameters
    """Processing parameters"""
