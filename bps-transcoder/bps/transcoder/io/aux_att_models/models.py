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

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum


@dataclass(kw_only=True)
class AngleType:
    class Meta:
        name = "Angle_Type"

    value: Decimal = field()
    unit: str = field(
        init=False,
        default="deg",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass(kw_only=True)
class MaxGapType:
    class Meta:
        name = "Max_Gap_Type"

    value: Decimal = field()
    unit: str = field(
        init=False,
        default="s",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass(kw_only=True)
class QuaternionComponentType:
    class Meta:
        name = "Quaternion_Component_Type"

    value: Decimal = field()


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


class TimeReferenceType(Enum):
    UTC = "UTC"
    UT1 = "UT1"
    TAI = "TAI"
    GPS = "GPS"


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
class AngleTimeType:
    class Meta:
        name = "Angle_Time_Type"

    value: str = field(
        default="",
        metadata={
            "pattern": r"[A-Z0-9]{3}=(\d{4}-(((01|03|05|07|08|10|12)-(0[1-9]|[1,2][0-9]|3[0,1]))|((04|06|09|11)-(0[1-9]|[1,2][0-9]|30))|(02-(0[1-9]|[1,2][0-9])))T([0,1][0-9]|2[0-3])(:[0-5][0-9]){2}|0000-00-00T00:00:00|9999-99-99T99:99:99)(.\d*)?"
        },
    )
    ref: TimeReferenceType = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class FixedHeaderType:
    class Meta:
        name = "Fixed_Header_Type"

    file_name: str = field(
        metadata={"name": "File_Name", "type": "Element", "pattern": r"(bio_aux_att____.+)|(bio_.+_att)"}
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
    file_type: str = field(metadata={"name": "File_Type", "type": "Element", "pattern": r"AUX_ATT___"})
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
class AnglesDataType:
    class Meta:
        name = "Angles_Data_Type"

    time: AngleTimeType = field(
        metadata={
            "name": "Time",
            "type": "Element",
        },
    )
    pitch: AngleType = field(
        metadata={
            "name": "Pitch",
            "type": "Element",
        },
    )
    roll: AngleType = field(
        metadata={
            "name": "Roll",
            "type": "Element",
        },
    )
    yaw: AngleType = field(
        metadata={
            "name": "Yaw",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class AttitudeHeaderType:
    class Meta:
        name = "Attitude_Header_Type"

    fixed_header: FixedHeaderType = field(
        metadata={
            "name": "Fixed_Header",
            "type": "Element",
        },
    )
    variable_header: None | object = field(
        default=None,
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
class QuaternionType:
    class Meta:
        name = "Quaternion_Type"

    time: AngleTimeType = field(
        metadata={
            "name": "Time",
            "type": "Element",
        },
    )
    q1: QuaternionComponentType = field(
        metadata={
            "name": "Q1",
            "type": "Element",
        },
    )
    q2: QuaternionComponentType = field(
        metadata={
            "name": "Q2",
            "type": "Element",
        },
    )
    q3: QuaternionComponentType = field(
        metadata={
            "name": "Q3",
            "type": "Element",
        },
    )
    q4: QuaternionComponentType = field(
        metadata={
            "name": "Q4",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
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
    count: int = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class AttitudeAnglesDataType:
    class Meta:
        name = "Attitude_Angles_Data_Type"

    reference_frame: str = field(
        metadata={
            "name": "Reference_Frame",
            "type": "Element",
        },
    )
    list_of_attitude_angles: ListOfAttitudeAnglesType = field(
        metadata={
            "name": "List_of_Attitude_Angles",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class QuaternionDataType:
    class Meta:
        name = "Quaternion_Data_Type"

    reference_frame: str = field(
        metadata={
            "name": "Reference_Frame",
            "type": "Element",
        },
    )
    list_of_quaternions: ListOfQuaternionsType = field(
        metadata={
            "name": "List_of_Quaternions",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class AttitudeDataBlockType:
    class Meta:
        name = "Attitude_Data_Block_Type"

    attitude_file_type: str = field(
        metadata={
            "name": "Attitude_File_Type",
            "type": "Element",
        },
    )
    attitude_data_type: str = field(
        metadata={
            "name": "Attitude_Data_Type",
            "type": "Element",
        },
    )
    max_gap: MaxGapType = field(
        metadata={
            "name": "Max_Gap",
            "type": "Element",
        },
    )
    attitude_angles_data: None | AttitudeAnglesDataType = field(
        default=None,
        metadata={
            "name": "Attitude_Angles_Data",
            "type": "Element",
        },
    )
    quaternion_data: None | QuaternionDataType = field(
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
    schema_version: None | Decimal = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class AttitudeFileType:
    class Meta:
        name = "Attitude_File_Type"

    earth_observation_header: AttitudeHeaderType = field(
        metadata={
            "name": "Earth_Observation_Header",
            "type": "Element",
        },
    )
    data_block: AttitudeDataBlockType = field(
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
class EarthObservationFile(AttitudeFileType):
    class Meta:
        name = "Earth_Observation_File"
