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

from bps.common.io.parsing import ParsingError, parse
from bps.l1_pre_processor.aux_ins.aux_ins import AuxInsParameters
from bps.l1_pre_processor.aux_ins.translate_aux_ins import (
    translate_model_to_aux_ins_parameters,
)
from bps.l1_pre_processor.io import aux_ins_models


class InvalidAuxINS(RuntimeError):
    """Raised when the aux ins content is not the correct format"""


def parse_aux_ins(aux_ins_content: str) -> AuxInsParameters:
    """Aux INS XML parsing

    Parameters
    ----------
    aux_ins_content : str
        content of the aux ins XML file

    Returns
    -------
    AuxIns
        Object containing the Auxiliary Instrument Parameters

    Raises
    ------
    InvalidAuxINS
    """
    try:
        model = parse(aux_ins_content, aux_ins_models.AuxiliaryInstrumentParameters)
    except ParsingError as exc:
        raise InvalidAuxINS from exc

    return translate_model_to_aux_ins_parameters(model)
