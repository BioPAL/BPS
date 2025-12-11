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
from bps.l2a_processor.core.aux_pp2_2a import AuxProcessingParametersL2A
from bps.l2a_processor.core.joborder_l2a import L2aJobOrder
from bps.l2a_processor.core.translate_aux_pp2_2a import (
    translate_model_to_aux_processing_parameters_l2a,
)
from bps.l2a_processor.core.translate_job_order import translate_model_to_l2a_job_order
from bps.l2a_processor.io import aux_pp2_2a_models


class InvalidJobOrder(RuntimeError):
    """Raised when the job order content is not the correct format"""


class InvalidAuxPP2_2A(RuntimeError):
    """Raised when the aux pp2 2a content is not the correct format"""


def parse_l2a_job_order(job_order_content: str) -> L2aJobOrder:
    """Job order XML parsing

    Parameters
    ----------
    job_order_content : str
        content of the job order XML file

    Returns
    -------
    L2AJobOrder
        Object containing the job order for the L2a processing

    Raises
    ------
    InvalidJobOrder
    """
    try:
        job_order_model = parse(job_order_content, joborder_models.JobOrder)
    except ParsingError as exc:
        raise InvalidJobOrder from exc

    return translate_model_to_l2a_job_order(job_order_model)


def parse_aux_pp2_2a(aux_pp2_2a_content: str) -> AuxProcessingParametersL2A:
    """Aux PP1 XML parsing

    Parameters
    ----------
    aux_pp2_2a_content : str
        content of the aux pp2 2a XML file

    Returns
    -------
    AuxProcessingParametersL2A
        Object containing the L2a Processing Parameters

    Raises
    ------
    InvalidAuxPP2_2A
    """

    try:
        model = parse(aux_pp2_2a_content, aux_pp2_2a_models.AuxiliaryL2AProcessingParameters)
    except ParsingError as exc:
        raise InvalidAuxPP2_2A from exc

    return translate_model_to_aux_processing_parameters_l2a(model)
