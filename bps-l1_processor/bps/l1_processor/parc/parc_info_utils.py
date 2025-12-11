# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to read a ParcInfo product
------------------------------------
"""

from bps.common.io.parsing import ParsingError, parse
from bps.l1_processor.io import parc_info_models
from bps.l1_processor.parc.parc_info import ParcInfoList
from bps.l1_processor.parc.translate_parc_info import translate_model_to_parc_info_list


class InvalidParcInfo(RuntimeError):
    """Raised when the parc info content is not the correct format"""


def parse_parc_info(
    parc_info_content: str,
) -> ParcInfoList:
    """PARC info XML parsing

    Parameters
    ----------
    parc_info_content : str
        content of the PARC info XML file

    Returns
    -------
    ParcInfoList
        list of PARC info

    Raises
    ------
    InvalidParcInfo
    """
    try:
        model = parse(parc_info_content, parc_info_models.AuxiliaryCalSiteInformation)
    except ParsingError as exc:
        raise InvalidParcInfo from exc

    return translate_model_to_parc_info_list(model)
