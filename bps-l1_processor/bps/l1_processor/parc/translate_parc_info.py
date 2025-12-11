# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
PARC info Translation
---------------------
"""

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.l1_processor.io.parc_info_models import (
    AuxiliaryCalSiteInformation,
    DelayListType,
    ParcType,
    RcsListType,
)
from bps.l1_processor.parc.parc_info import ParcInfo, ParcInfoList, ScatteringResponse


class InvalidParcInfo(RuntimeError):
    """Raised when information inside parc info are invalid"""


def translate_model_to_delays(
    delays_model: DelayListType,
) -> dict[ScatteringResponse, float]:
    """Translate delays"""
    assert delays_model.delay_gt1 is not None
    assert delays_model.delay_gt1.value is not None
    assert delays_model.delay_gt2 is not None
    assert delays_model.delay_gt2.value is not None
    assert delays_model.delay_x is not None
    assert delays_model.delay_x.value is not None
    assert delays_model.delay_y is not None
    assert delays_model.delay_y.value is not None

    return {
        ScatteringResponse.GT1: delays_model.delay_gt1.value,
        ScatteringResponse.GT2: delays_model.delay_gt2.value,
        ScatteringResponse.X: delays_model.delay_x.value,
        ScatteringResponse.Y: delays_model.delay_y.value,
    }


def translate_model_to_rcs(rcs_model: RcsListType) -> dict[ScatteringResponse, float]:
    """Translate rcs"""
    assert rcs_model.rcs_gt1 is not None
    assert rcs_model.rcs_gt1.value is not None
    assert rcs_model.rcs_gt2 is not None
    assert rcs_model.rcs_gt2.value is not None
    assert rcs_model.rcs_x is not None
    assert rcs_model.rcs_x.value is not None
    assert rcs_model.rcs_y is not None
    assert rcs_model.rcs_y.value is not None

    return {
        ScatteringResponse.GT1: rcs_model.rcs_gt1.value,
        ScatteringResponse.GT2: rcs_model.rcs_gt2.value,
        ScatteringResponse.X: rcs_model.rcs_x.value,
        ScatteringResponse.Y: rcs_model.rcs_y.value,
    }


def translate_model_to_parc_info(parc_info_model: ParcType) -> ParcInfo:
    """Translate parc info"""
    assert parc_info_model.parc_id is not None
    assert parc_info_model.validity_start is not None
    assert parc_info_model.validity_stop is not None
    assert parc_info_model.position_x is not None
    assert parc_info_model.position_y is not None
    assert parc_info_model.position_z is not None
    assert parc_info_model.delay_list is not None
    assert parc_info_model.rcs_list is not None
    return ParcInfo(
        parc_id=parc_info_model.parc_id,
        validity_interval=(
            PreciseDateTime.fromisoformat(str(parc_info_model.validity_start)),
            PreciseDateTime.fromisoformat(str(parc_info_model.validity_stop)),
        ),
        position=np.array(
            [
                parc_info_model.position_x,
                parc_info_model.position_y,
                parc_info_model.position_z,
            ]
        ),
        delays=translate_model_to_delays(parc_info_model.delay_list),
        rcs=translate_model_to_rcs(parc_info_model.rcs_list),
    )


def translate_model_to_parc_info_list(
    parc_info_model: AuxiliaryCalSiteInformation,
) -> ParcInfoList:
    """Translate parc info model into corresponding object"""
    assert parc_info_model.parc_list is not None
    assert parc_info_model.parc_list.count is not None

    if parc_info_model.parc_list.count != len(parc_info_model.parc_list.parc):
        raise InvalidParcInfo(
            "Count does not correspond to parc info list size: "
            + f"{parc_info_model.parc_list.count} != {len(parc_info_model.parc_list.parc)}"
        )

    return [translate_model_to_parc_info(info_model) for info_model in parc_info_model.parc_list.parc]
