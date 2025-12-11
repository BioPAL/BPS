# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional, Union

__NAMESPACE__ = "http://eop-cfi.esa.int/CFI"


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
class AnyTypeType:
    class Meta:
        name = "AnyType_Type"

    content: list[object] = field(
        default_factory=list,
        metadata={
            "type": "Wildcard",
            "namespace": "##any",
            "mixed": True,
        },
    )


@dataclass
class DeltaUt1Type:
    class Meta:
        name = "Delta_UT1_Type"

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
class DistanceType:
    class Meta:
        name = "Distance_Type"

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
class EquatorCrossLongType:
    class Meta:
        name = "Equator_Cross_Long_Type"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: str = field(
        init=False,
        default="10-6deg",
        metadata={
            "type": "Attribute",
        },
    )


class FrameNumberTypeValue(Enum):
    VALUE = "___"


class FrameStatusType(Enum):
    NOMINAL = "NOMINAL"
    MERGED = "MERGED"
    PARTIAL = "PARTIAL"
    INCOMPLETE = "INCOMPLETE"
    NOT_FRAMED = "NOT_FRAMED"


@dataclass
class FreqType:
    class Meta:
        name = "Freq_Type"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: str = field(
        init=False,
        default="MHz",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class LatType:
    class Meta:
        name = "Lat_Type"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: str = field(
        init=False,
        default="10-6deg",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class LongType:
    class Meta:
        name = "Long_Type"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: str = field(
        init=False,
        default="10-6deg",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


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


class OrbitScenarioVariableHeaderTimeReference(Enum):
    UT1 = "UT1"


@dataclass
class PositionType:
    class Meta:
        name = "Position_Type"

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
class ProductConfidenceDataType:
    class Meta:
        name = "Product_Confidence_Data_Type"

    num_isps: Optional[int] = field(
        default=None,
        metadata={
            "name": "Num_ISPs",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    num_missing_isps: Optional[int] = field(
        default=None,
        metadata={
            "name": "Num_Missing_ISPs",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    num_error_isps: Optional[int] = field(
        default=None,
        metadata={
            "name": "Num_Error_ISPs",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    num_discarded_isps: Optional[int] = field(
        default=None,
        metadata={
            "name": "Num_Discarded_ISPs",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    num_rs_isps: Optional[int] = field(
        default=None,
        metadata={
            "name": "Num_RS_ISPs",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    num_rs_corrections: Optional[int] = field(
        default=None,
        metadata={
            "name": "Num_RS_Corrections",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )


class RefractionModelType(Enum):
    NO_REF = "NO_REF"
    STD_REF = "STD_REF"
    USER_REF = "USER_REF"
    PRED_REF = "PRED_REF"


@dataclass
class RelTimeAscNodeType:
    class Meta:
        name = "Rel_Time_ASC_Node_Type"

    value: str = field(
        default="",
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
class SecondsTimeType:
    class Meta:
        name = "Seconds_Time_Type"

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
class SourceType:
    class Meta:
        name = "Source_Type"

    system: Optional[str] = field(
        default=None,
        metadata={
            "name": "System",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    creator: Optional[str] = field(
        default=None,
        metadata={
            "name": "Creator",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    creator_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "Creator_Version",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    creation_date: Optional[str] = field(
        default=None,
        metadata={
            "name": "Creation_Date",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "length": 23,
            "pattern": r"UTC=.*",
        },
    )


@dataclass
class TotSizeType:
    class Meta:
        name = "Tot_Size_Type"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    unit: str = field(
        init=False,
        default="bytes",
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ValidityPeriodBomEomType:
    class Meta:
        name = "Validity_Period_BOM_EOM_Type"

    validity_start: Optional[str] = field(
        default=None,
        metadata={
            "name": "Validity_Start",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"UTC=0000-00-00T00:00:00",
        },
    )
    validity_stop: Optional[str] = field(
        default=None,
        metadata={
            "name": "Validity_Stop",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"UTC=9999-99-99T99:99:99",
        },
    )


@dataclass
class ValidityPeriodEomType:
    class Meta:
        name = "Validity_Period_EOM_Type"

    validity_start: Optional[str] = field(
        default=None,
        metadata={
            "name": "Validity_Start",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "length": 23,
            "pattern": r"UTC=.*",
        },
    )
    validity_stop: Optional[str] = field(
        default=None,
        metadata={
            "name": "Validity_Stop",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"UTC=9999-99-99T99:99:99",
        },
    )


@dataclass
class ValidityPeriodType:
    class Meta:
        name = "Validity_Period_Type"

    validity_start: Optional[str] = field(
        default=None,
        metadata={
            "name": "Validity_Start",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "length": 23,
            "pattern": r"UTC=.*",
        },
    )
    validity_stop: Optional[str] = field(
        default=None,
        metadata={
            "name": "Validity_Stop",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "length": 23,
            "pattern": r"UTC=.*",
        },
    )


@dataclass
class VelocityType:
    class Meta:
        name = "Velocity_Type"

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
class Declination(AngleType):
    pass


@dataclass
class FixedHeaderBomEomType:
    class Meta:
        name = "Fixed_Header_BOM_EOM_Type"

    file_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Name",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    file_description: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Description",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    notes: Optional[str] = field(
        default=None,
        metadata={
            "name": "Notes",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    mission: Optional[str] = field(
        default=None,
        metadata={
            "name": "Mission",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    file_class: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Class",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"OPER|OFFL|NRT_|RPRO|STV[0-3]|GSOV|TEST|TD[0-9][0-9]|Routine Operations|Off-Line Processing|near-real-Time Processing|Re-Processing|Satellite Validation Test [0-3]|Ground Segment Overall Validation test|Generated test files|Test Data Set [0-9][0-9]",
        },
    )
    file_type: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Type",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"[A-Z0-9_]{10}",
        },
    )
    validity_period: Optional[ValidityPeriodBomEomType] = field(
        default=None,
        metadata={
            "name": "Validity_Period",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    file_version: Optional[int] = field(
        default=None,
        metadata={
            "name": "File_Version",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    source: Optional[SourceType] = field(
        default=None,
        metadata={
            "name": "Source",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )


@dataclass
class FixedHeaderEomType:
    class Meta:
        name = "Fixed_Header_EOM_Type"

    file_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Name",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    file_description: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Description",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    notes: Optional[str] = field(
        default=None,
        metadata={
            "name": "Notes",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    mission: Optional[str] = field(
        default=None,
        metadata={
            "name": "Mission",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    file_class: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Class",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"OPER|OFFL|NRT_|RPRO|STV[0-3]|GSOV|TEST|TD[0-9][0-9]|Routine Operations|Off-Line Processing|near-real-Time Processing|Re-Processing|Satellite Validation Test [0-3]|Ground Segment Overall Validation test|Generated test files|Test Data Set [0-9][0-9]",
        },
    )
    file_type: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Type",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"[A-Z0-9_]{10}",
        },
    )
    validity_period: Optional[ValidityPeriodEomType] = field(
        default=None,
        metadata={
            "name": "Validity_Period",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    file_version: Optional[int] = field(
        default=None,
        metadata={
            "name": "File_Version",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    source: Optional[SourceType] = field(
        default=None,
        metadata={
            "name": "Source",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
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
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"([A-Z0-9_]){3}_([A-Z0-9_]){4}_([A-Z0-9_]){10}_([A-Z0-9_]){1,41}",
        },
    )
    file_description: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Description",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    notes: Optional[str] = field(
        default=None,
        metadata={
            "name": "Notes",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    mission: Optional[str] = field(
        default=None,
        metadata={
            "name": "Mission",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    file_class: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Class",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"OPER|OFFL|NRT_|RPRO|STV[0-3]|GSOV|TEST|TD[0-9][0-9]|Routine Operations|Off-Line Processing|near-real-Time Processing|Re-Processing|Satellite Validation Test [0-3]|Ground Segment Overall Validation test|Generated test files|Test Data Set [0-9][0-9]",
        },
    )
    file_type: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Type",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"[A-Z0-9_]{10}",
        },
    )
    validity_period: Optional[ValidityPeriodType] = field(
        default=None,
        metadata={
            "name": "Validity_Period",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    file_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "File_Version",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"[0-9]{4}",
        },
    )
    source: Optional[SourceType] = field(
        default=None,
        metadata={
            "name": "Source",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )


@dataclass
class HeightType(DistanceType):
    class Meta:
        name = "Height_Type"


@dataclass
class L1VirtualFrameDataBlockType:
    class Meta:
        name = "L1_Virtual_Frame_Data_Block_Type"

    source_l0_s: Optional[str] = field(
        default=None,
        metadata={
            "name": "source_L0S",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    source_l0_m: Optional[str] = field(
        default=None,
        metadata={
            "name": "source_L0M",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    source_aux_orb: Optional[str] = field(
        default=None,
        metadata={
            "name": "source_AUX_ORB",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    frame_id: Optional[Union[int, FrameNumberTypeValue]] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"[0-9]{3}",
        },
    )
    frame_start_time: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"UTC=.*",
        },
    )
    frame_stop_time: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"UTC=.*",
        },
    )
    frame_status: Optional[FrameStatusType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    ops_angle_start: Optional[AngleType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    ops_angle_stop: Optional[AngleType] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
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
class MphType:
    class Meta:
        name = "MPH_Type"

    product: Optional[str] = field(
        default=None,
        metadata={
            "name": "Product",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    proc_stage_code: Optional[str] = field(
        default=None,
        metadata={
            "name": "Proc_Stage_Code",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    ref_doc: Optional[str] = field(
        default=None,
        metadata={
            "name": "Ref_Doc",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    proc_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "Proc_Time",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    software_version: Optional[str] = field(
        default=None,
        metadata={
            "name": "Software_Version",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    phase: Optional[str] = field(
        default=None,
        metadata={
            "name": "Phase",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    cycle: Optional[int] = field(
        default=None,
        metadata={
            "name": "Cycle",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    rel_orbit: Optional[int] = field(
        default=None,
        metadata={
            "name": "Rel_Orbit",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    abs_orbit: Optional[int] = field(
        default=None,
        metadata={
            "name": "Abs_Orbit",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    state_vector_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "State_Vector_Time",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"UTC=.*",
        },
    )
    delta_ut1: Optional[DeltaUt1Type] = field(
        default=None,
        metadata={
            "name": "Delta_UT1",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    x_position: Optional[PositionType] = field(
        default=None,
        metadata={
            "name": "X_Position",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    y_position: Optional[PositionType] = field(
        default=None,
        metadata={
            "name": "Y_Position",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    z_position: Optional[PositionType] = field(
        default=None,
        metadata={
            "name": "Z_Position",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    x_velocity: Optional[VelocityType] = field(
        default=None,
        metadata={
            "name": "X_Velocity",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    y_velocity: Optional[VelocityType] = field(
        default=None,
        metadata={
            "name": "Y_Velocity",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    z_velocity: Optional[VelocityType] = field(
        default=None,
        metadata={
            "name": "Z_Velocity",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    state_vector_source: Optional[str] = field(
        default=None,
        metadata={
            "name": "State_Vector_Source",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"..",
        },
    )
    product_err: Optional[int] = field(
        default=None,
        metadata={
            "name": "Product_Err",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    tot_size: Optional[TotSizeType] = field(
        default=None,
        metadata={
            "name": "Tot_Size",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
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
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    time_reference: Optional[OrbitFileVariableHeaderTimeReference] = field(
        default=None,
        metadata={
            "name": "Time_Reference",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )


@dataclass
class OrbitInformationType:
    class Meta:
        name = "Orbit_Information_Type"

    sensing_start: Optional[str] = field(
        default=None,
        metadata={
            "name": "Sensing_Start",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"TAI=.*",
        },
    )
    abs_orbit_start: Optional[int] = field(
        default=None,
        metadata={
            "name": "Abs_Orbit_Start",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    rel_time_asc_node_start: Optional[RelTimeAscNodeType] = field(
        default=None,
        metadata={
            "name": "Rel_Time_ASC_Node_Start",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    sensing_stop: Optional[str] = field(
        default=None,
        metadata={
            "name": "Sensing_Stop",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "pattern": r"TAI=.*",
        },
    )
    abs_orbit_stop: Optional[int] = field(
        default=None,
        metadata={
            "name": "Abs_Orbit_Stop",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    rel_time_asc_node_stop: Optional[RelTimeAscNodeType] = field(
        default=None,
        metadata={
            "name": "Rel_Time_ASC_Node_Stop",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    equator_cross_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "Equator_Cross_Time",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    equator_cross_long: Optional[EquatorCrossLongType] = field(
        default=None,
        metadata={
            "name": "Equator_Cross_Long",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    ascending_flag: Optional[str] = field(
        default=None,
        metadata={
            "name": "Ascending_Flag",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
            "length": 1,
        },
    )


@dataclass
class OrbitScenarioVariableHeader:
    class Meta:
        name = "Orbit_Scenario_Variable_Header"

    time_reference: Optional[OrbitScenarioVariableHeaderTimeReference] = field(
        default=None,
        metadata={
            "name": "Time_Reference",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )


@dataclass
class ProductLocationType:
    class Meta:
        name = "Product_Location_Type"

    start_lat: Optional[LatType] = field(
        default=None,
        metadata={
            "name": "Start_Lat",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    start_long: Optional[LongType] = field(
        default=None,
        metadata={
            "name": "Start_Long",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    stop_lat: Optional[LatType] = field(
        default=None,
        metadata={
            "name": "Stop_Lat",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    stop_long: Optional[LongType] = field(
        default=None,
        metadata={
            "name": "Stop_Long",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )


@dataclass
class RefractionType:
    class Meta:
        name = "Refraction_Type"

    model: Optional[RefractionModelType] = field(
        default=None,
        metadata={
            "name": "Model",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    freq: Optional[FreqType] = field(
        default=None,
        metadata={
            "name": "Freq",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )


@dataclass
class RightAsc(AngleType):
    class Meta:
        name = "Right_Asc"


@dataclass
class SecondsDurationType(SecondsTimeType):
    class Meta:
        name = "Seconds_Duration_Type"


@dataclass
class EarthExplorerHeaderType:
    class Meta:
        name = "Earth_Explorer_Header_Type"

    fixed_header: Optional[FixedHeaderType] = field(
        default=None,
        metadata={
            "name": "Fixed_Header",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    variable_header: Optional[AnyTypeType] = field(
        default=None,
        metadata={
            "name": "Variable_Header",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    schema_version: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "schemaVersion",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class L1VirtualFrameHeaderType:
    class Meta:
        name = "L1_Virtual_Frame_Header_Type"

    fixed_header: Optional[FixedHeaderType] = field(
        default=None,
        metadata={
            "name": "Fixed_Header",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    variable_header: Optional[AnyTypeType] = field(
        default=None,
        metadata={
            "name": "Variable_Header",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
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
class SphType:
    class Meta:
        name = "SPH_Type"

    sph_descriptor: Optional[str] = field(
        default=None,
        metadata={
            "name": "SPH_Descriptor",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    orbit_information: Optional[OrbitInformationType] = field(
        default=None,
        metadata={
            "name": "Orbit_Information",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    product_location: Optional[ProductLocationType] = field(
        default=None,
        metadata={
            "name": "Product_Location",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    product_confidence_data: Optional[ProductConfidenceDataType] = field(
        default=None,
        metadata={
            "name": "Product_Confidence_Data",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )


@dataclass
class L1VirtualFrameFileType:
    class Meta:
        name = "L1_Virtual_Frame_File_Type"

    earth_explorer_header: Optional[L1VirtualFrameHeaderType] = field(
        default=None,
        metadata={
            "name": "Earth_Explorer_Header",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
            "required": True,
        },
    )
    data_block: Optional[L1VirtualFrameDataBlockType] = field(
        default=None,
        metadata={
            "name": "Data_Block",
            "type": "Element",
            "namespace": "http://eop-cfi.esa.int/CFI",
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
class EarthExplorerFile(L1VirtualFrameFileType):
    class Meta:
        name = "Earth_Explorer_File"
        namespace = "http://eop-cfi.esa.int/CFI"
