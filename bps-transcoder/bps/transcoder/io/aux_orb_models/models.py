# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Orbit models
----------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum


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


@dataclass(kw_only=True)
class PositionComponentType:
    class Meta:
        name = "Position_Component_Type"

    value: Decimal = field()
    unit: str = field(
        init=False,
        default="m",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass(kw_only=True)
class SourceType:
    class Meta:
        name = "Source_Type"

    system: str = field(
        metadata={
            "name": "System",
            "type": "Element",
        },
    )
    creator: str = field(
        metadata={
            "name": "Creator",
            "type": "Element",
        },
    )
    creator_version: str = field(
        metadata={
            "name": "Creator_Version",
            "type": "Element",
        },
    )
    creation_date: str = field(
        metadata={"name": "Creation_Date", "type": "Element", "length": 23, "pattern": r"UTC=.*"}
    )


@dataclass(kw_only=True)
class ValidityPeriodType:
    class Meta:
        name = "Validity_Period_Type"

    validity_start: str = field(
        metadata={"name": "Validity_Start", "type": "Element", "length": 23, "pattern": r"UTC=.*"}
    )
    validity_stop: str = field(
        metadata={"name": "Validity_Stop", "type": "Element", "length": 23, "pattern": r"UTC=.*"}
    )


@dataclass(kw_only=True)
class VelocityComponentType:
    class Meta:
        name = "Velocity_Component_Type"

    value: Decimal = field()
    unit: str = field(
        init=False,
        default="m/s",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass(kw_only=True)
class FixedHeaderType:
    class Meta:
        name = "Fixed_Header_Type"

    file_name: str = field(
        metadata={"name": "File_Name", "type": "Element", "pattern": r"(bio_aux_orb____.+)|(bio_.+_orb)"}
    )
    file_description: str = field(
        metadata={
            "name": "File_Description",
            "type": "Element",
        },
    )
    notes: str = field(
        metadata={
            "name": "Notes",
            "type": "Element",
        },
    )
    mission: str = field(metadata={"name": "Mission", "type": "Element", "pattern": r"BIOMASS"})
    file_class: str = field(metadata={"name": "File_Class", "type": "Element", "pattern": r"OPER"})
    file_type: str = field(metadata={"name": "File_Type", "type": "Element", "pattern": r"AUX_ORB___"})
    validity_period: ValidityPeriodType = field(
        metadata={
            "name": "Validity_Period",
            "type": "Element",
        },
    )
    file_version: str = field(metadata={"name": "File_Version", "type": "Element", "pattern": r"[0-9]{4}"})
    eoffs_version: str = field(
        metadata={
            "name": "EOFFS_Version",
            "type": "Element",
        },
    )
    source: SourceType = field(
        metadata={
            "name": "Source",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class OsvType:
    class Meta:
        name = "OSV_Type"

    tai: str = field(metadata={"name": "TAI", "type": "Element", "pattern": r"TAI=.*"})
    utc: str = field(metadata={"name": "UTC", "type": "Element", "pattern": r"UTC=.*"})
    ut1: str = field(metadata={"name": "UT1", "type": "Element", "pattern": r"UT1=.*"})
    absolute_orbit: int = field(
        metadata={
            "name": "Absolute_Orbit",
            "type": "Element",
        },
    )
    x: PositionComponentType = field(
        metadata={
            "name": "X",
            "type": "Element",
        },
    )
    y: PositionComponentType = field(
        metadata={
            "name": "Y",
            "type": "Element",
        },
    )
    z: PositionComponentType = field(
        metadata={
            "name": "Z",
            "type": "Element",
        },
    )
    vx: VelocityComponentType = field(
        metadata={
            "name": "VX",
            "type": "Element",
        },
    )
    vy: VelocityComponentType = field(
        metadata={
            "name": "VY",
            "type": "Element",
        },
    )
    vz: VelocityComponentType = field(
        metadata={
            "name": "VZ",
            "type": "Element",
        },
    )
    quality: str = field(
        metadata={
            "name": "Quality",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class OrbitFileVariableHeader:
    class Meta:
        name = "Orbit_File_Variable_Header"

    ref_frame: OrbitFileVariableHeaderRefFrame = field(
        metadata={
            "name": "Ref_Frame",
            "type": "Element",
        },
    )
    time_reference: OrbitFileVariableHeaderTimeReference = field(
        metadata={
            "name": "Time_Reference",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class RestitutedOrbitHeaderType:
    class Meta:
        name = "Restituted_Orbit_Header_Type"

    fixed_header: FixedHeaderType = field(
        metadata={
            "name": "Fixed_Header",
            "type": "Element",
        },
    )
    variable_header: OrbitFileVariableHeader = field(
        metadata={
            "name": "Variable_Header",
            "type": "Element",
        },
    )
    schema_version: None | Decimal = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class RestitutedOrbitDataBlockType:
    class Meta:
        name = "Restituted_Orbit_Data_Block_Type"

    list_of_osvs: ListOfOsvsType = field(
        metadata={
            "name": "List_of_OSVs",
            "type": "Element",
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
    schema_version: None | Decimal = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class RestitutedOrbitFileType:
    class Meta:
        name = "Restituted_Orbit_File_Type"

    earth_observation_header: RestitutedOrbitHeaderType = field(
        metadata={
            "name": "Earth_Observation_Header",
            "type": "Element",
        },
    )
    data_block: RestitutedOrbitDataBlockType = field(
        metadata={
            "name": "Data_Block",
            "type": "Element",
        },
    )
    schema_version: None | Decimal = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class EarthObservationFile(RestitutedOrbitFileType):
    class Meta:
        name = "Earth_Observation_File"
