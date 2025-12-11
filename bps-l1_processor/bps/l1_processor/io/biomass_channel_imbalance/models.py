# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD BPS Channel imbalance file models
-------------------------------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class FcomplexNumberType:
    class Meta:
        name = "FComplexNumberType"
        target_namespace = "biomass_common"

    real: Optional[float] = field(
        default=None,
        metadata={
            "name": "Real",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    imag: Optional[float] = field(
        default=None,
        metadata={
            "name": "Imag",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


class PolarizationType(Enum):
    H_H = "H/H"
    H_V = "H/V"
    V_H = "V/H"
    V_V = "V/V"


class SwathType(Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


class UnitType(Enum):
    S = "s"
    UTC = "Utc"


@dataclass
class ChannelDelayType:
    class Meta:
        target_namespace = "biomass_common"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    swath: Optional[SwathType] = field(
        default=None,
        metadata={
            "name": "Swath",
            "type": "Attribute",
            "required": True,
        },
    )
    polarization: Optional[PolarizationType] = field(
        default=None,
        metadata={
            "name": "Polarization",
            "type": "Attribute",
            "required": True,
        },
    )
    uo_m: UnitType = field(
        init=False,
        default=UnitType.S,
        metadata={
            "name": "UoM",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ChannelImbalanceType:
    class Meta:
        target_namespace = "biomass_common"

    tx: Optional[FcomplexNumberType] = field(
        default=None,
        metadata={
            "name": "TX",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rx: Optional[FcomplexNumberType] = field(
        default=None,
        metadata={
            "name": "RX",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class ChannelImbalance(ChannelImbalanceType):
    pass


@dataclass
class ChannelDelaysType:
    class Meta:
        target_namespace = "biomass_common"

    channel_delay: list[ChannelDelayType] = field(
        default_factory=list,
        metadata={
            "name": "ChannelDelay",
            "type": "Element",
            "namespace": "",
        },
    )
