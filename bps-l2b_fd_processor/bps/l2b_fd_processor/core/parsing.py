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
from bps.l2b_fd_processor.core.aux_pp2_2b_fd import AuxProcessingParametersL2BFD
from bps.l2b_fd_processor.core.joborder_l2b_fd import L2bFDJobOrder
from bps.l2b_fd_processor.core.translate_aux_pp2_2b_fd import (
    translate_model_to_aux_processing_parameters_l2b_fd,
)
from bps.l2b_fd_processor.core.translate_job_order import (
    translate_model_to_l2b_fd_job_order,
)
from bps.l2b_fd_processor.io import aux_pp2_2b_fd_models


class InvalidJobOrder(RuntimeError):
    """Raised when the job order content is not the correct format"""


class InvalidAuxPP2_2B_FD(RuntimeError):
    """Raised when the aux pp2 2b FD content is not the correct format"""


def parse_l2b_job_order(job_order_content: str) -> L2bFDJobOrder:
    """Job order XML parsing

    Parameters
    ----------
    job_order_content : str
        content of the job order XML file

    Returns
    -------
    L2BJobOrder
        Object containing the job order for the L2b processing

    Raises
    ------
    InvalidJobOrder
    """
    try:
        job_order_model = parse(job_order_content, joborder_models.JobOrder)
    except ParsingError as exc:
        raise InvalidJobOrder from exc

    return translate_model_to_l2b_fd_job_order(job_order_model)


def parse_aux_pp2_2b_fd(aux_pp2_2b_fd_content: str) -> AuxProcessingParametersL2BFD:
    """Aux PP2 FD XML parsing

    Parameters
    ----------
    aux_pp2_2b_fd_content : str
        content of the aux pp2 2b FD XML file

    Returns
    -------
    AuxProcessingParametersL2BFF
        Object containing the L2bFFProcessing Parameters


    Raises
    ------
    InvalidAuxPP2_FD
    """

    try:
        model = parse(
            aux_pp2_2b_fd_content,
            aux_pp2_2b_fd_models.AuxiliaryL2BFdprocessingParameters,
        )
    except ParsingError as exc:
        raise InvalidAuxPP2_2B_FD from exc

    return translate_model_to_aux_processing_parameters_l2b_fd(model)
