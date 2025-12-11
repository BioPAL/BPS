# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 PreProcessor interface module
--------------------------------
"""

from dataclasses import dataclass
from pathlib import Path

from bps.l1_pre_processor.configuration import L1PreProcessorConfigurationFile
from bps.l1_pre_processor.input_file import L1PreProcessorInputFile
from bps.l1_pre_processor.serialization import (
    serialize_l1_preprocessor_configuration,
    serialize_l1_preprocessor_input_file,
)


def write_l1_preproc_input_file(input_file: L1PreProcessorInputFile, file: Path):
    """Write L1 Pre Processor input file"""
    file.write_text(serialize_l1_preprocessor_input_file(input_file), encoding="utf-8")


def write_l1_preproc_configuration_file(conf: L1PreProcessorConfigurationFile, file: Path):
    """Write L1 Pre Processor configuration file"""
    file.write_text(serialize_l1_preprocessor_configuration(conf), encoding="utf-8")


@dataclass
class L1PreProcessorInterface:
    """L1 PreProcessor interface objects"""

    input_file: L1PreProcessorInputFile
    conf: L1PreProcessorConfigurationFile
