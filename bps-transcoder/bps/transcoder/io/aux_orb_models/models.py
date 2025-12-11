# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Orbit models
----------------
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class OrbitFileVariableHeaderRefFrame(Enum):
    BAR_MEAN_2000 = "BAR_MEAN_2000"
    HEL_MEAN_2000 = "HEL_MEAN_2000"
    GEO_MEAN_2000 = "GEO_MEAN_2000"
    MEAN_DATE = "MEAN_DATE"
    TRUE_DATE = "TRUE_DATE"
    EARTH_FIXED = "EARTH_FIXED"
    BAR_MEAN_1950 = "BAR_MEAN_1950"
    QUASI_MEAN_DATE = "QUASI_MEAN_DATE"
    PSE_TRUE_DATE = "PSE_TRUE_DATE"
    PSEUDO_EARTH_FIXED = "PSEUDO_EARTH_FIXED"


class OrbitFileVariableHeaderTimeReference(Enum):
    TAI = "TAI"
    UTC = "UTC"
    UT1 = "UT1"


@dataclass
class PositionComponentType:
    class Meta:
        name = "Position_Component_Type"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: str = field(
        init=False,
        default="m",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class SourceType:
    class Meta:
        name = "Source_Type"

    system: Optional[str] = field(
        default=None,
        metadata={
            "name": "System",
            "type": "Element",
            "required": True,
        },
    )
    creator: Optional[str] = field(
        default=None,
        metadata={
            "name": "Creator",
            "type": "Element",
            "required": True,
        },
    )
    creator_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "Creator_Version",
            "type": "Element",
            "required": True,
        },
    )
    creation_date: Optional[str] = field(
        default=None,
        metadata={"name": "Creation_Date", "type": "Element", "required": True, "length": 23, "pattern": r"UTC=.*"},
    )


@dataclass
class ValidityPeriodType:
    class Meta:
        name = "Validity_Period_Type"

    validity_start: Optional[str] = field(
        default=None,
        metadata={"name": "Validity_Start", "type": "Element", "required": True, "length": 23, "pattern": r"UTC=.*"},
    )
    validity_stop: Optional[str] = field(
        default=None,
        metadata={"name": "Validity_Stop", "type": "Element", "required": True, "length": 23, "pattern": r"UTC=.*"},
    )


@dataclass
class VelocityComponentType:
    class Meta:
        name = "Velocity_Component_Type"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: str = field(
        init=False,
        default="m/s",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class FixedHeaderType:
    class Meta:
        name = "Fixed_Header_Type"

    file_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Name",
            "type": "Element",
            "required": True,
            "pattern": r"(bio_aux_orb____.+)|(bio_.+_orb)",
        },
    )
    file_description: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Description",
            "type": "Element",
            "required": True,
        },
    )
    notes: Optional[str] = field(
        default=None,
        metadata={
            "name": "Notes",
            "type": "Element",
            "required": True,
        },
    )
    mission: Optional[str] = field(
        default=None, metadata={"name": "Mission", "type": "Element", "required": True, "pattern": r"BIOMASS"}
    )
    file_class: Optional[str] = field(
        default=None, metadata={"name": "File_Class", "type": "Element", "required": True, "pattern": r"OPER"}
    )
    file_type: Optional[str] = field(
        default=None, metadata={"name": "File_Type", "type": "Element", "required": True, "pattern": r"AUX_ORB___"}
    )
    validity_period: Optional[ValidityPeriodType] = field(
        default=None,
        metadata={
            "name": "Validity_Period",
            "type": "Element",
            "required": True,
        },
    )
    file_version: Optional[str] = field(
        default=None, metadata={"name": "File_Version", "type": "Element", "required": True, "pattern": r"[0-9]{4}"}
    )
    eoffs_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "EOFFS_Version",
            "type": "Element",
            "required": True,
        },
    )
    source: Optional[SourceType] = field(
        default=None,
        metadata={
            "name": "Source",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class OsvType:
    class Meta:
        name = "OSV_Type"

    tai: Optional[str] = field(
        default=None, metadata={"name": "TAI", "type": "Element", "required": True, "pattern": r"TAI=.*"}
    )
    utc: Optional[str] = field(
        default=None, metadata={"name": "UTC", "type": "Element", "required": True, "pattern": r"UTC=.*"}
    )
    ut1: Optional[str] = field(
        default=None, metadata={"name": "UT1", "type": "Element", "required": True, "pattern": r"UT1=.*"}
    )
    absolute_orbit: Optional[int] = field(
        default=None,
        metadata={
            "name": "Absolute_Orbit",
            "type": "Element",
            "required": True,
        },
    )
    x: Optional[PositionComponentType] = field(
        default=None,
        metadata={
            "name": "X",
            "type": "Element",
            "required": True,
        },
    )
    y: Optional[PositionComponentType] = field(
        default=None,
        metadata={
            "name": "Y",
            "type": "Element",
            "required": True,
        },
    )
    z: Optional[PositionComponentType] = field(
        default=None,
        metadata={
            "name": "Z",
            "type": "Element",
            "required": True,
        },
    )
    vx: Optional[VelocityComponentType] = field(
        default=None,
        metadata={
            "name": "VX",
            "type": "Element",
            "required": True,
        },
    )
    vy: Optional[VelocityComponentType] = field(
        default=None,
        metadata={
            "name": "VY",
            "type": "Element",
            "required": True,
        },
    )
    vz: Optional[VelocityComponentType] = field(
        default=None,
        metadata={
            "name": "VZ",
            "type": "Element",
            "required": True,
        },
    )
    quality: Optional[str] = field(
        default=None,
        metadata={
            "name": "Quality",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class OrbitFileVariableHeader:
    class Meta:
        name = "Orbit_File_Variable_Header"

    ref_frame: Optional[OrbitFileVariableHeaderRefFrame] = field(
        default=None,
        metadata={
            "name": "Ref_Frame",
            "type": "Element",
            "required": True,
        },
    )
    time_reference: Optional[OrbitFileVariableHeaderTimeReference] = field(
        default=None,
        metadata={
            "name": "Time_Reference",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class ListOfOsvsType:
    class Meta:
        name = "List_of_OSVs_Type"

    osv: list[OsvType] = field(
        default_factory=list,
        metadata={
            "name": "OSV",
            "type": "Element",
            "min_occurs": 1,
        },
    )
    count: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class RestitutedOrbitHeaderType:
    class Meta:
        name = "Restituted_Orbit_Header_Type"

    fixed_header: Optional[FixedHeaderType] = field(
        default=None,
        metadata={
            "name": "Fixed_Header",
            "type": "Element",
            "required": True,
        },
    )
    variable_header: Optional[OrbitFileVariableHeader] = field(
        default=None,
        metadata={
            "name": "Variable_Header",
            "type": "Element",
            "required": True,
        },
    )
    schema_version: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
        },
    )


@dataclass
class RestitutedOrbitDataBlockType:
    class Meta:
        name = "Restituted_Orbit_Data_Block_Type"

    list_of_osvs: Optional[ListOfOsvsType] = field(
        default=None,
        metadata={
            "name": "List_of_OSVs",
            "type": "Element",
            "required": True,
        },
    )
    type_value: str = field(
        init=False,
        default="xml",
        metadata={
            "name": "type",
            "type": "Attribute",
            "required": True,
        },
    )
    schema_version: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
        },
    )


@dataclass
class RestitutedOrbitFileType:
    class Meta:
        name = "Restituted_Orbit_File_Type"

    earth_observation_header: Optional[RestitutedOrbitHeaderType] = field(
        default=None,
        metadata={
            "name": "Earth_Observation_Header",
            "type": "Element",
            "required": True,
        },
    )
    data_block: Optional[RestitutedOrbitDataBlockType] = field(
        default=None,
        metadata={
            "name": "Data_Block",
            "type": "Element",
            "required": True,
        },
    )
    schema_version: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
        },
    )


@dataclass
class EarthObservationFile(RestitutedOrbitFileType):
    class Meta:
        name = "Earth_Observation_File"
