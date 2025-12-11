# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Serialization
-------------
"""

from bps.common.io.parsing import serialize
from bps.stack_coreg_processor.configuration import (
    CoregStackProcessorInternalConfiguration,
)
from bps.stack_coreg_processor.input_file import BPSCoregProcessorInputFile
from bps.stack_coreg_processor.translate import (
    translate_bps_coreg_input_file_to_model,
    translate_coreg_configuration_to_model,
)


def serialize_coreg_input_file(
    input_file: BPSCoregProcessorInputFile,
) -> str:
    """
    Serialize a Stack Coregistration Processor input file to string.

    Parameters
    ----------
    input_file: CoregProcessorInputFile
        Input file object.

    Return
    ------
    str
        Serialized input file content.
    """
    input_file_model = translate_bps_coreg_input_file_to_model(input_file)
    return serialize(input_file_model)


def serialize_coreg_config_file(
    config: CoregStackProcessorInternalConfiguration,
) -> str:
    """
    Serialize a Coreg Stack Configuration object to string.

    Parameters
    ----------
    config: CoregStackProcessorInternalConfiguration
        Coregistrator configuration object.

    Return
    ------
    str
        Serialized config file content.

    """
    config_model = translate_coreg_configuration_to_model(config)
    return serialize(config_model)
