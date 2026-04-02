# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD PARC info models
--------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from bps.common.io.common_types import (
    AzimuthPolynomialType,
    ChannelImbalanceList,
    ChannelType,
    Complex,
    ComplexArray,
    CrossTalkList,
    DatumType,
    DoubleArray,
    DoubleArrayWithUnits,
    DoubleWithUnit,
    FloatArray,
    FloatArrayWithUnits,
    FloatWithChannel,
    FloatWithPolarisation,
    FloatWithUnit,
    GeodeticReferenceFrameType,
    GroupType,
    HeightModelBaseType,
    HeightModelType,
    IntArray,
    InterferometricPairListType,
    InterferometricPairType,
    LayerListType,
    LayerType,
    MinMaxType,
    MinMaxTypeWithUnit,
    PolarisationType,
    SlantRangePolynomialType,
    StateType,
    TimeTypeWithPolarisation,
    UnsignedIntWithGroup,
    UomType,
)


@dataclass(kw_only=True)
class DelayListType:
    """
    Parameters
    ----------
    delay_gt1
        Delay for PARC Gt1 scattering response [s].
    delay_gt2
        Delay for PARC Gt2 scattering response [s].
    delay_x
        Delay for PARC X scattering response [s].
    delay_y
        Delay for PARC Y scattering response [s].
    """

    class Meta:
        name = "delayListType"

    delay_gt1: FloatWithUnit = field(
        metadata={
            "name": "delayGt1",
            "type": "Element",
            "namespace": "",
        },
    )
    delay_gt2: FloatWithUnit = field(
        metadata={
            "name": "delayGt2",
            "type": "Element",
            "namespace": "",
        },
    )
    delay_x: FloatWithUnit = field(
        metadata={
            "name": "delayX",
            "type": "Element",
            "namespace": "",
        },
    )
    delay_y: FloatWithUnit = field(
        metadata={
            "name": "delayY",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class RcsListType:
    """
    Parameters
    ----------
    rcs_gt1
        RCS for PARC Gt1 scattering response [dBm2].
    rcs_gt2
        RCS for PARC Gt2 scattering response [dBm2].
    rcs_x
        RCS for PARC X scattering response [dBm2].
    rcs_y
        RCS for PARC Y scattering response [dBm2].
    """

    class Meta:
        name = "rcsListType"

    rcs_gt1: FloatWithUnit = field(
        metadata={
            "name": "rcsGt1",
            "type": "Element",
            "namespace": "",
        },
    )
    rcs_gt2: FloatWithUnit = field(
        metadata={
            "name": "rcsGt2",
            "type": "Element",
            "namespace": "",
        },
    )
    rcs_x: FloatWithUnit = field(
        metadata={
            "name": "rcsX",
            "type": "Element",
            "namespace": "",
        },
    )
    rcs_y: FloatWithUnit = field(
        metadata={
            "name": "rcsY",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class ParcType:
    """
    Parameters
    ----------
    parc_id
        PARC identifier.
    validity_start
        PARC validity start time. Indicates when the calibrator starts being active (and then can be seen in the
        acquired images).
    validity_stop
        PARC validity stop time. Indicates when the calibrator stops being active (and then cannot be seen anymore
        in the acquired images).
    position_x
        PARC ECEF X coordinate.
    position_y
        PARC ECEF Y coordinate.
    position_z
        PARC ECEF Z coordinate.
    delay_list
        List of PARC delays for each scattering response.
    rcs_list
        List of PARC RCS values for each scattering response.
    """

    class Meta:
        name = "parcType"

    parc_id: str = field(
        metadata={
            "name": "parcID",
            "type": "Element",
            "namespace": "",
        },
    )
    validity_start: str = field(
        metadata={
            "name": "validityStart",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    validity_stop: str = field(
        metadata={
            "name": "validityStop",
            "type": "Element",
            "namespace": "",
            "pattern": "\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{6}",
        }
    )
    position_x: float = field(
        metadata={
            "name": "positionX",
            "type": "Element",
            "namespace": "",
        },
    )
    position_y: float = field(
        metadata={
            "name": "positionY",
            "type": "Element",
            "namespace": "",
        },
    )
    position_z: float = field(
        metadata={
            "name": "positionZ",
            "type": "Element",
            "namespace": "",
        },
    )
    delay_list: DelayListType = field(
        metadata={
            "name": "delayList",
            "type": "Element",
            "namespace": "",
        },
    )
    rcs_list: RcsListType = field(
        metadata={
            "name": "rcsList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class ParcListType:
    """
    Parameters
    ----------
    parc
        Parameters for a given calibration target (PARC).
    count
    """

    class Meta:
        name = "parcListType"

    parc: list[ParcType] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryCalSiteInformationType:
    """
    Parameters
    ----------
    parc_list
        List of calibration targets (PARC) parameters.
    """

    class Meta:
        name = "auxiliaryCalSiteInformationType"

    parc_list: ParcListType = field(
        metadata={
            "name": "parcList",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class AuxiliaryCalSiteInformation(AuxiliaryCalSiteInformationType):
    """
    BIOMASS auxiliary calibration site information element.
    """

    class Meta:
        name = "auxiliaryCalSiteInformation"
