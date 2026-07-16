# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD BPS Channel imbalance file models
-------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(kw_only=True)
class FcomplexNumberType:
    class Meta:
        name = "FComplexNumberType"
        target_namespace = "biomass_common"

    real: float = field(
        metadata={
            "name": "Real",
            "type": "Element",
            "namespace": "",
        }
    )
    imag: float = field(
        metadata={
            "name": "Imag",
            "type": "Element",
            "namespace": "",
        }
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


@dataclass(kw_only=True)
class ChannelDelayType:
    class Meta:
        target_namespace = "biomass_common"

    value: float = field()
    swath: SwathType = field(
        metadata={
            "name": "Swath",
            "type": "Attribute",
        }
    )
    polarization: PolarizationType = field(
        metadata={
            "name": "Polarization",
            "type": "Attribute",
        }
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


@dataclass(kw_only=True)
class ChannelImbalanceType:
    class Meta:
        target_namespace = "biomass_common"

    tx: FcomplexNumberType = field(
        metadata={
            "name": "TX",
            "type": "Element",
            "namespace": "",
        }
    )
    rx: FcomplexNumberType = field(
        metadata={
            "name": "RX",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
class ChannelImbalance(ChannelImbalanceType):
    pass


@dataclass(kw_only=True)
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
