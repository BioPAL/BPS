# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Attitude models
-------------------
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


@dataclass
class AngleType:
    class Meta:
        name = "Angle_Type"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: str = field(
        init=False,
        default="deg",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class MaxGapType:
    class Meta:
        name = "Max_Gap_Type"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: str = field(
        init=False,
        default="s",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class QuaternionComponentType:
    class Meta:
        name = "Quaternion_Component_Type"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
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


class TimeReferenceType(Enum):
    UTC = "UTC"
    UT1 = "UT1"
    TAI = "TAI"
    GPS = "GPS"


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
class AngleTimeType:
    class Meta:
        name = "Angle_Time_Type"

    value: str = field(
        default="",
        metadata={
            "required": True,
            "pattern": r"[A-Z0-9]{3}=(\d{4}-(((01|03|05|07|08|10|12)-(0[1-9]|[1,2][0-9]|3[0,1]))|((04|06|09|11)-(0[1-9]|[1,2][0-9]|30))|(02-(0[1-9]|[1,2][0-9])))T([0,1][0-9]|2[0-3])(:[0-5][0-9]){2}|0000-00-00T00:00:00|9999-99-99T99:99:99)(.\d*)?",
        },
    )
    ref: Optional[TimeReferenceType] = field(
        default=None,
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
            "pattern": r"(bio_aux_att____.+)|(bio_.+_att)",
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
        default=None, metadata={"name": "File_Type", "type": "Element", "required": True, "pattern": r"AUX_ATT___"}
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
class AnglesDataType:
    class Meta:
        name = "Angles_Data_Type"

    time: Optional[AngleTimeType] = field(
        default=None,
        metadata={
            "name": "Time",
            "type": "Element",
            "required": True,
        },
    )
    pitch: Optional[AngleType] = field(
        default=None,
        metadata={
            "name": "Pitch",
            "type": "Element",
            "required": True,
        },
    )
    roll: Optional[AngleType] = field(
        default=None,
        metadata={
            "name": "Roll",
            "type": "Element",
            "required": True,
        },
    )
    yaw: Optional[AngleType] = field(
        default=None,
        metadata={
            "name": "Yaw",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class AttitudeHeaderType:
    class Meta:
        name = "Attitude_Header_Type"

    fixed_header: Optional[FixedHeaderType] = field(
        default=None,
        metadata={
            "name": "Fixed_Header",
            "type": "Element",
            "required": True,
        },
    )
    variable_header: Optional[object] = field(
        default=None,
        metadata={
            "name": "Variable_Header",
            "type": "Element",
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
class QuaternionType:
    class Meta:
        name = "Quaternion_Type"

    time: Optional[AngleTimeType] = field(
        default=None,
        metadata={
            "name": "Time",
            "type": "Element",
            "required": True,
        },
    )
    q1: Optional[QuaternionComponentType] = field(
        default=None,
        metadata={
            "name": "Q1",
            "type": "Element",
            "required": True,
        },
    )
    q2: Optional[QuaternionComponentType] = field(
        default=None,
        metadata={
            "name": "Q2",
            "type": "Element",
            "required": True,
        },
    )
    q3: Optional[QuaternionComponentType] = field(
        default=None,
        metadata={
            "name": "Q3",
            "type": "Element",
            "required": True,
        },
    )
    q4: Optional[QuaternionComponentType] = field(
        default=None,
        metadata={
            "name": "Q4",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class ListOfAttitudeAnglesType:
    class Meta:
        name = "List_of_Attitude_Angles_Type"

    attitude_angles: list[AnglesDataType] = field(
        default_factory=list,
        metadata={
            "name": "Attitude_Angles",
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
class ListOfQuaternionsType:
    class Meta:
        name = "List_of_Quaternions_Type"

    quaternions: list[QuaternionType] = field(
        default_factory=list,
        metadata={
            "name": "Quaternions",
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
class AttitudeAnglesDataType:
    class Meta:
        name = "Attitude_Angles_Data_Type"

    reference_frame: Optional[str] = field(
        default=None,
        metadata={
            "name": "Reference_Frame",
            "type": "Element",
            "required": True,
        },
    )
    list_of_attitude_angles: Optional[ListOfAttitudeAnglesType] = field(
        default=None,
        metadata={
            "name": "List_of_Attitude_Angles",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class QuaternionDataType:
    class Meta:
        name = "Quaternion_Data_Type"

    reference_frame: Optional[str] = field(
        default=None,
        metadata={
            "name": "Reference_Frame",
            "type": "Element",
            "required": True,
        },
    )
    list_of_quaternions: Optional[ListOfQuaternionsType] = field(
        default=None,
        metadata={
            "name": "List_of_Quaternions",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class AttitudeDataBlockType:
    class Meta:
        name = "Attitude_Data_Block_Type"

    attitude_file_type: Optional[str] = field(
        default=None,
        metadata={
            "name": "Attitude_File_Type",
            "type": "Element",
            "required": True,
        },
    )
    attitude_data_type: Optional[str] = field(
        default=None,
        metadata={
            "name": "Attitude_Data_Type",
            "type": "Element",
            "required": True,
        },
    )
    max_gap: Optional[MaxGapType] = field(
        default=None,
        metadata={
            "name": "Max_Gap",
            "type": "Element",
            "required": True,
        },
    )
    attitude_angles_data: Optional[AttitudeAnglesDataType] = field(
        default=None,
        metadata={
            "name": "Attitude_Angles_Data",
            "type": "Element",
        },
    )
    quaternion_data: Optional[QuaternionDataType] = field(
        default=None,
        metadata={
            "name": "Quaternion_Data",
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
    schema_version: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
        },
    )


@dataclass
class AttitudeFileType:
    class Meta:
        name = "Attitude_File_Type"

    earth_observation_header: Optional[AttitudeHeaderType] = field(
        default=None,
        metadata={
            "name": "Earth_Observation_Header",
            "type": "Element",
            "required": True,
        },
    )
    data_block: Optional[AttitudeDataBlockType] = field(
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
class EarthObservationFile(AttitudeFileType):
    class Meta:
        name = "Earth_Observation_File"
