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
from bps.l1_core_processor.input_file import BPSL1CoreProcessorInputFile
from bps.l1_core_processor.processing_options import BPSL1CoreProcessingOptions
from bps.l1_core_processor.processing_parameters import SarfocProcessingParameters
from bps.l1_core_processor.translate import (
    translate_bps_l1_core_processor_input_file_to_model,
    translate_bps_l1_core_processor_processing_options_to_model,
    translate_sarfoc_processing_parameters_to_model,
)


def serialize_bps_l1_core_processor_input_file(
    input_file: BPSL1CoreProcessorInputFile,
) -> str:
    """Serialize a BPS L1 core processor input file to string

    Parameters
    ----------
    input_file : BPSL1CoreProcessorInputFile
        Input file object

    Returns
    -------
    str
        serialized input file content
    """
    input_file_model = translate_bps_l1_core_processor_input_file_to_model(input_file)
    return serialize(input_file_model)


def serialize_bps_l1_core_processor_processing_options(
    processing_options: BPSL1CoreProcessingOptions,
) -> str:
    """Serialize a BPSL1CoreProcessor processing options object to string

    Parameters
    ----------
    processing_options : BPSL1CoreProcessingOptions
        Processing options object

    Returns
    -------
    str
        serialized processing option file content
    """
    model = translate_bps_l1_core_processor_processing_options_to_model(processing_options)
    return serialize(model)


def serialize_sarfoc_processing_parameters(
    processing_parameters: SarfocProcessingParameters,
) -> str:
    """Serialize a Sarfoc processing parameters object to string

    Parameters
    ----------
    processing_parameters : SarfocProcessingParameters
        Processing parameters object

    Returns
    -------
    str
        serialized processing parameters file content
    """
    model = translate_sarfoc_processing_parameters_to_model(processing_parameters)
    return serialize(model)
