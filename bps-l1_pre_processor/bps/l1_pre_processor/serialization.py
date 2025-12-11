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
from bps.l1_pre_processor.configuration import L1PreProcessorConfigurationFile
from bps.l1_pre_processor.input_file import L1PreProcessorInputFile
from bps.l1_pre_processor.translate import (
    translate_l1preprocessor_configuration_file_to_model,
    translate_l1preprocessor_input_file_to_model,
)


def serialize_l1_preprocessor_input_file(
    input_file: L1PreProcessorInputFile,
) -> str:
    """Serialize a L1 PreProcessor input file to string

    Parameters
    ----------
    input_file : L1PreProcessorInputFile
        L1 preprocessor input file object

    Returns
    -------
    str
        serialized input file content
    """
    model = translate_l1preprocessor_input_file_to_model(input_file)
    return serialize(model)


def serialize_l1_preprocessor_configuration(
    configuration: L1PreProcessorConfigurationFile,
) -> str:
    """Serialize a L1 PreProcessor configuration file to string

    Parameters
    ----------
    configuration : L1PreProcessorConfigurationFile
        L1 preprocessor input file object

    Returns
    -------
    str
        serialized configuration file content
    """
    model = translate_l1preprocessor_configuration_file_to_model(configuration)
    return serialize(model)
