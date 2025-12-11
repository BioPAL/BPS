# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to parse a Channel Imbalance file
-------------------------------------------
"""

from dataclasses import dataclass

from bps.common.io.parsing import ParsingError, parse
from bps.l1_processor.io.biomass_channel_imbalance import models


class InvalidChannelImbalance(RuntimeError):
    """Raised when the channel imbalance content is not the correct format"""


@dataclass
class ChannelImbalanceProcessingParametersL1:
    """Channel imbalance parameters"""

    tx: complex
    rx: complex


def translate_f_complex_type_to_complex(number: models.FcomplexNumberType) -> complex:
    """Translate FcomplexNumberType to complex data type"""

    assert number.real is not None
    assert number.imag is not None

    return complex(real=number.real, imag=number.real)


def translate_model_to_channel_imbalance_processing_parameters(
    model: models.ChannelImbalance,
) -> ChannelImbalanceProcessingParametersL1:
    """Translate channel imbalance to the corresponding structure"""

    assert model.tx is not None
    assert model.rx is not None

    tx = translate_f_complex_type_to_complex(model.tx)
    rx = translate_f_complex_type_to_complex(model.rx)

    return ChannelImbalanceProcessingParametersL1(tx=tx, rx=rx)


def parse_channel_imbalance(
    channel_imbalance_content: str,
) -> ChannelImbalanceProcessingParametersL1:
    """Channel Imbalance XML parsing

    Parameters
    ----------
    channel_imbalance_content : str
        content of the channel Imbalance XML file

    Returns
    -------
    ChannelImbalanceProcessingParametersL1
        Object containing the ChannelImbalanceProcessingParametersL1

    Raises
    ------
    InvalidChannelImbalance
    """
    try:
        model = parse(channel_imbalance_content, models.ChannelImbalance)
    except ParsingError as exc:
        raise InvalidChannelImbalance from exc

    return translate_model_to_channel_imbalance_processing_parameters(model)
