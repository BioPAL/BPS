# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Attitude file utilities
-----------------------
"""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io.parsing import parse, serialize
from bps.transcoder.io import aux_att_models


@dataclass
class Attitude:
    """Attitude minimal info"""

    max_time_gap: float

    time_axis_origin: PreciseDateTime
    relative_time_axis: np.ndarray
    quaternions: np.ndarray


def translate_precise_date_time_from_angle_time_model(
    time: aux_att_models.AngleTimeType,
) -> PreciseDateTime:
    """Convert AngleTimeType to PreciseDateTime"""
    return PreciseDateTime.from_utc_string(time.value.split("=")[1])


def translate_precise_date_time_to_angle_time_model(
    time: PreciseDateTime,
) -> aux_att_models.AngleTimeType:
    """PreciseDateTime is translated to AngleTimeType"""
    return aux_att_models.AngleTimeType(
        value="UTC=" + time.isoformat(timespec="microseconds").strip("Z"),
        ref=aux_att_models.TimeReferenceType.UTC,
    )


@dataclass
class AttitudeRecord:
    """Single attitude record with quaternion"""

    time: PreciseDateTime
    quaternion: np.ndarray


def translate_attitude_record_from_quaternion_model(
    record: aux_att_models.QuaternionType,
) -> AttitudeRecord:
    """Convert quaternion model to attitude record"""
    assert record.time is not None
    assert record.q1 is not None and record.q1.value is not None
    assert record.q2 is not None and record.q2.value is not None
    assert record.q3 is not None and record.q3.value is not None
    assert record.q4 is not None and record.q4.value is not None

    return AttitudeRecord(
        time=translate_precise_date_time_from_angle_time_model(record.time),
        quaternion=np.array(
            [
                float(record.q1.value),
                float(record.q2.value),
                float(record.q3.value),
                float(record.q4.value),
            ]
        ),
    )


def float_to_quaternion_component_type(
    value: float,
) -> aux_att_models.QuaternionComponentType:
    """Convert float to proper quaternion type"""
    return aux_att_models.QuaternionComponentType(Decimal(value=f"{value:.9f}"))


def translate_attitude_record_to_quaternion_model(
    attitude_record: AttitudeRecord,
) -> aux_att_models.QuaternionType:
    """Fill a quaternions"""

    return aux_att_models.QuaternionType(
        time=translate_precise_date_time_to_angle_time_model(attitude_record.time),
        q1=float_to_quaternion_component_type(attitude_record.quaternion[0]),
        q2=float_to_quaternion_component_type(attitude_record.quaternion[1]),
        q3=float_to_quaternion_component_type(attitude_record.quaternion[2]),
        q4=float_to_quaternion_component_type(attitude_record.quaternion[3]),
    )


def fill_attitude_from_model(
    attitude_model: aux_att_models.EarthObservationFile,
) -> Attitude:
    """Retrieve some information from attitude file"""
    assert attitude_model.data_block is not None
    assert attitude_model.data_block.max_gap is not None
    assert attitude_model.data_block.max_gap.value is not None
    assert attitude_model.data_block.quaternion_data is not None
    assert attitude_model.data_block.quaternion_data.list_of_quaternions is not None

    max_gap = float(attitude_model.data_block.max_gap.value)

    times = []
    quaternions = []
    for attitude_record in attitude_model.data_block.quaternion_data.list_of_quaternions.quaternions:
        record = translate_attitude_record_from_quaternion_model(attitude_record)
        times.append(record.time)
        quaternions.append(record.quaternion)

    first_time = times[0]
    times = np.array([time - first_time for time in times])
    quaternions = np.asarray(quaternions)

    return Attitude(
        max_time_gap=max_gap,
        time_axis_origin=first_time,
        relative_time_axis=times,
        quaternions=quaternions,
    )


def update_model_with_additional_attitude_records(
    attitude_model: aux_att_models.EarthObservationFile,
    reference_time: PreciseDateTime,
    additional_times: np.ndarray,
    additional_quaternions: np.ndarray,
):
    """Insert new attitude records in the model"""
    missing_quaternions_models = [
        translate_attitude_record_to_quaternion_model(AttitudeRecord(time=reference_time + time, quaternion=quaternion))
        for time, quaternion in zip(additional_times, additional_quaternions)
    ]

    assert attitude_model.data_block is not None
    assert attitude_model.data_block.quaternion_data is not None
    assert attitude_model.data_block.quaternion_data.list_of_quaternions is not None
    assert attitude_model.data_block.quaternion_data.list_of_quaternions.count is not None
    attitude_model.data_block.quaternion_data.list_of_quaternions.count += additional_times.size
    attitude_model.data_block.quaternion_data.list_of_quaternions.quaternions.extend(missing_quaternions_models)
    attitude_model.data_block.quaternion_data.list_of_quaternions.quaternions.sort(
        key=lambda x: translate_precise_date_time_from_angle_time_model(x.time)  # type: ignore
    )


def read_attitude_file(attitude_file: Path) -> aux_att_models.EarthObservationFile:
    """Read attitude file"""
    return parse(
        attitude_file.read_text(encoding="utf-8"),
        aux_att_models.EarthObservationFile,
    )


def write_attitude_file(
    file: Path,
    attitude_model: aux_att_models.EarthObservationFile,
    update_file_name: bool = True,
):
    """Write attitude file"""
    assert attitude_model.earth_observation_header is not None
    assert attitude_model.earth_observation_header.fixed_header is not None

    # update file name
    if update_file_name:
        attitude_model.earth_observation_header.fixed_header.file_name = file.stem

    attitude_content = serialize(attitude_model)
    file.write_text(attitude_content)
