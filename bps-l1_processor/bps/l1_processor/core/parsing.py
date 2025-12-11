# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Parsing
-------
"""

from bps.common.io import joborder_models
from bps.common.io.parsing import ParsingError, parse
from bps.l1_processor.io import aux_pp1_models
from bps.l1_processor.processor_interface.aux_pp1 import AuxProcessingParametersL1
from bps.l1_processor.processor_interface.joborder_l1 import L1JobOrder
from bps.l1_processor.processor_interface.translate_aux_pp1 import (
    translate_model_to_aux_processing_parameters_l1,
)
from bps.l1_processor.processor_interface.translate_job_order import (
    translate_model_to_l1_job_order,
)


class InvalidJobOrder(RuntimeError):
    """Raised when the job order content is not the correct format"""


class InvalidAuxPP1(RuntimeError):
    """Raised when the aux pp1 content is not the correct format"""


def parse_l1_job_order(job_order_content: str) -> L1JobOrder:
    """Job order XML parsing

    Parameters
    ----------
    job_order_content : str
        content of the job order XML file

    Returns
    -------
    L1JobOrder
        Object containing the job order for the L1 processing

    Raises
    ------
    InvalidJobOrder
    """
    try:
        job_order_model = parse(job_order_content, joborder_models.JobOrder)
    except ParsingError as exc:
        raise InvalidJobOrder from exc

    return translate_model_to_l1_job_order(job_order_model)


def parse_aux_pp1(aux_pp1_content: str) -> AuxProcessingParametersL1:
    """Aux PP1 XML parsing

    Parameters
    ----------
    aux_pp1_content : str
        content of the aux pp1 XML file

    Returns
    -------
    AuxProcessingParametersL1
        Object containing the L1 Processing Parameters

    Raises
    ------
    InvalidAuxPP1
    """
    try:
        model = parse(aux_pp1_content, aux_pp1_models.AuxiliaryL1ProcessingParameters)
    except ParsingError as exc:
        raise InvalidAuxPP1 from exc

    return translate_model_to_aux_processing_parameters_l1(model)
