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
from bps.l1_framing_processor.core.joborder_l1f import L1FJobOrder
from bps.l1_framing_processor.core.translate_job_order import (
    translate_model_to_l1f_job_order,
)


class InvalidJobOrder(RuntimeError):
    """Raised when the job order content is not the correct format"""


class InvalidAuxPP1(RuntimeError):
    """Raised when the aux pp1 content is not the correct format"""


def parse_l1_job_order(job_order_content: str) -> L1FJobOrder:
    """Job order XML parsing

    Parameters
    ----------
    job_order_content : str
        content of the job order XML file

    Returns
    -------
    L1FJobOrder
        Object containing the job order for the L1 framing processing

    Raises
    ------
    InvalidJobOrder
    """
    try:
        job_order_model = parse(job_order_content, joborder_models.JobOrder)
    except ParsingError as exc:
        raise InvalidJobOrder from exc

    return translate_model_to_l1f_job_order(job_order_model)
