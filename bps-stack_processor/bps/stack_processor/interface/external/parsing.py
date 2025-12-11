# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Parsing Utilities for AUX-PPS and JobOrders
-------------------------------------------
"""

from bps.common.io import joborder_models
from bps.common.io.parsing import ParsingError, parse
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
)
from bps.stack_processor.interface.external.joborder_stack import StackJobOrder
from bps.stack_processor.interface.external.translate_aux_pps import (
    AuxPPSParsingError,
    translate_model_to_sta_product_conf,
)
from bps.stack_processor.interface.external.translate_job_order import (
    StackJobOrderParsingError,
    translate_model_to_stack_job_order,
)
from bps.stack_processor.io import aux_pps_models


def parse_stack_job_order(job_order_content: str) -> StackJobOrder:
    """
    Job order XML parsing.

    Parameters
    ----------
    job_order_content : str
        content of the Stack Job Order XML file

    Return
    ------
    StackJobOrder
        Object containing the job order for the Stack processing

    Raises
    ------
    StackJobOrderParsingError

    """
    try:
        job_order_model = parse(job_order_content, joborder_models.JobOrder)
    except ParsingError as exc:
        raise StackJobOrderParsingError(exc) from exc

    return translate_model_to_stack_job_order(job_order_model)


def parse_aux_pps(aux_pps_content: str) -> AuxiliaryStaprocessingParameters:
    """
    AUX-PPS XML parsing.

    Parameters
    ----------
    aux_pps_content : str
        content of the AUX-PPS XML file

    Return
    ------
    AuxStaprocessingParameters
        Object containing the Stack Processing Parameters

    Raises
    ------
    AuxPPSParsingError

    """
    try:
        model = parse(aux_pps_content, aux_pps_models.AuxiliaryStaprocessingParameters)
    except ParsingError as exc:
        raise AuxPPSParsingError(exc) from exc

    return translate_model_to_sta_product_conf(model)
