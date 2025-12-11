# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Navigation XML file utilities"""

from decimal import Decimal

import numpy as np
import numpy.typing as npt
from arepytools.geometry.attitude_utils import (
    compute_euler_angles_from_antenna_reference_frame,
)
from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.geometry.reference_frames import compute_sensor_local_axis
from arepytools.io import metadata
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io.parsing import serialize
from bps.transcoder.auxiliaryfiles.aux_attitude import (
    AttitudeRecord,
    fill_attitude_from_model,
    translate_attitude_record_to_quaternion_model,
)
from bps.transcoder.io import aux_att_models, aux_orb_models
from scipy.spatial import transform

CREATION_DATE_TEMPLATE_STRING = "CREATION_DATE_TEMPLATE"
FILE_NAME_TEMPLATE_STRING = "FILE_NAME_TEMPLATE"


def replace_template_string(template_string: str, creation_date: str, file_name: str) -> str:
    """Replace creation_date string and file_name string with actual content"""
    return template_string.replace(CREATION_DATE_TEMPLATE_STRING, creation_date).replace(
        FILE_NAME_TEMPLATE_STRING, file_name
    )


def fill_attitude_file_template_str(arf: np.ndarray, time_array: np.ndarray) -> str:
    """Fill an attitude XML string leaving creation date and file name as a template string

    Parameters
    ----------
    arf : np.ndarray
        antenna reference frame
    time_array: np.ndarray
        time arrayb (PreciseDateTime)
    Returns
    -------
    str
        template string
    """
    creation_date = CREATION_DATE_TEMPLATE_STRING
    file_name = FILE_NAME_TEMPLATE_STRING

    quaternions = transform.Rotation.from_matrix(arf).as_quat(canonical=True)
    reference_frame = "EARTH_FIXED"

    start: PreciseDateTime = time_array[0]
    stop: PreciseDateTime = time_array[-1]
    validity_start = "UTC=" + start.isoformat(timespec="seconds")[:-1]
    validity_stop = "UTC=" + stop.isoformat(timespec="seconds")[:-1]

    file_description = "Attitude File"
    notes = ""
    mission = "BIOMASS"
    file_class = "OPER"
    file_type = "AUX_ATT___"
    validity_period = aux_att_models.ValidityPeriodType(validity_start=validity_start, validity_stop=validity_stop)
    file_version = "0001"
    eoffs_version = "3.0"
    source = aux_att_models.SourceType(
        system="",
        creator="BPS",
        creator_version="1.0.0",
        creation_date="UTC=" + creation_date,
    )
    fixed_header = aux_att_models.FixedHeaderType(
        file_name,
        file_description,
        notes,
        mission,
        file_class,
        file_type,
        validity_period,
        file_version,
        eoffs_version,
        source,
    )

    variable_header = ""

    earth_observation_header = aux_att_models.AttitudeHeaderType(fixed_header, variable_header)

    attitude_file_type = "Sat_Nominal_Attitude"
    attitude_data_type = "Quaternions"
    max_gap = aux_att_models.MaxGapType(value=Decimal(str(2.0 * np.mean(np.diff(time_array)))))

    quaternions_list = [
        translate_attitude_record_to_quaternion_model(AttitudeRecord(time=time, quaternion=quaternion))
        for time, quaternion in zip(time_array, quaternions)
    ]

    list_of_quaternions = aux_att_models.ListOfQuaternionsType(quaternions_list, count=len(quaternions_list))
    quaternion_data = aux_att_models.QuaternionDataType(reference_frame, list_of_quaternions)

    data_block = aux_att_models.AttitudeDataBlockType(
        attitude_file_type, attitude_data_type, max_gap, None, quaternion_data
    )

    schema_version = "3.0"

    earth_observation_file_model = aux_att_models.EarthObservationFile(
        earth_observation_header, data_block, Decimal(value=schema_version)
    )

    # Write attitude file
    return serialize(earth_observation_file_model)


def fill_orbit_file_template_str(
    position: np.ndarray,
    velocity: np.ndarray,
    start_time: PreciseDateTime,
    time_step: float,
    tai_utc_difference: int,
) -> str:
    """Fill an orbit XML string leaving creation date and file name as a template string

    Parameters
    ----------
    position : np.ndarray
        sensor position
    velocity : np.ndarray
        sensor velocity
    start_time : PreciseDateTime
        orbit start time
    time_step : float
        orbit time step
    tai_utc_difference : int
        number of seconds between TAI and UTC time (for conversions)

    Returns
    -------
    str
        orbit template XML string
    """
    nsv = position.shape[0]
    validity_start = start_time.isoformat(timespec="seconds")[:-1]
    validity_stop = (start_time + (nsv - 1) * time_step).isoformat(timespec="seconds")[:-1]

    file_description = "Orbit File"
    notes = ""
    mission = "BIOMASS"
    file_class = "OPER"
    file_type = "AUX_ORB___"
    validity_period = aux_orb_models.ValidityPeriodType(
        validity_start="UTC=" + validity_start, validity_stop="UTC=" + validity_stop
    )
    file_version = "0001"
    eoffs_version = "3.0"
    source = aux_orb_models.SourceType(
        system="",
        creator="BPS",
        creator_version="1.0.0",
        creation_date="UTC=" + CREATION_DATE_TEMPLATE_STRING,
    )
    fixed_header = aux_orb_models.FixedHeaderType(
        FILE_NAME_TEMPLATE_STRING,
        file_description,
        notes,
        mission,
        file_class,
        file_type,
        validity_period,
        file_version,
        eoffs_version,
        source,
    )

    ref_frame = aux_orb_models.OrbitFileVariableHeaderRefFrame.EARTH_FIXED
    time_reference = aux_orb_models.OrbitFileVariableHeaderTimeReference.UTC
    variable_header = aux_orb_models.OrbitFileVariableHeader(ref_frame, time_reference)

    earth_observation_header = aux_orb_models.RestitutedOrbitHeaderType(fixed_header, variable_header)

    osv_list = list()
    for sv in range(nsv):
        tai = "TAI=" + (start_time + sv * time_step + tai_utc_difference).isoformat(timespec="microseconds")[:-1]
        utc = "UTC=" + (start_time + sv * time_step).isoformat(timespec="microseconds")[:-1]
        ut1 = "UT1=" + (start_time + sv * time_step).isoformat(timespec="microseconds")[:-1]
        absolute_orbit = 0
        x = aux_orb_models.PositionComponentType(value=float(position[sv, 0]))
        y = aux_orb_models.PositionComponentType(value=float(position[sv, 1]))
        z = aux_orb_models.PositionComponentType(value=float(position[sv, 2]))
        vx = aux_orb_models.VelocityComponentType(value=float(velocity[sv, 0]))
        vy = aux_orb_models.VelocityComponentType(value=float(velocity[sv, 1]))
        vz = aux_orb_models.VelocityComponentType(value=float(velocity[sv, 2]))
        quality = "0000000000000"
        osv = aux_orb_models.OsvType(tai, utc, ut1, absolute_orbit, x, y, z, vx, vy, vz, quality)
        osv_list.append(osv)
    list_of_osvs = aux_orb_models.ListOfOsvsType(osv_list, count=nsv)

    data_block = aux_orb_models.RestitutedOrbitDataBlockType(list_of_osvs)

    schema_version = "3.0"

    earth_observation_file_model = aux_orb_models.EarthObservationFile(
        earth_observation_header, data_block, schema_version
    )

    return serialize(earth_observation_file_model)


def compute_yaw_pitch_roll_from_antenna_reference_frame(
    reference_frame: str,
    antenna_reference_frame: np.ndarray,
    positions: npt.ArrayLike,
    velocities: npt.ArrayLike,
    order: str,
) -> np.ndarray:
    """Compute yaw pitch roll from antenna reference frame"""
    initial_frame = compute_sensor_local_axis(positions, velocities, reference_frame)

    return compute_euler_angles_from_antenna_reference_frame(initial_frame, antenna_reference_frame, order)


def translate_attitude_file_to_attitude_info(
    attitude_model: aux_att_models.EarthObservationFile,
    state_vectors: metadata.StateVectors,
) -> metadata.AttitudeInfo:
    """Read attitude xml file"""
    attitude = fill_attitude_from_model(attitude_model)

    t0 = attitude.time_axis_origin
    delta_t = float(np.mean(np.diff(attitude.relative_time_axis)))

    reference_frame = "ZERODOPPLER"
    rotation_order = "YPR"

    if np.sum(np.abs(attitude.quaternions[:, 3])) == 0.0:
        yaw = attitude.quaternions[:, 0]
        pitch = attitude.quaternions[:, 1]
        roll = attitude.quaternions[:, 2]
    else:
        orbit = create_general_sar_orbit(state_vectors, ignore_anx_after_orbit_start=True)
        time_axis = t0 + attitude.relative_time_axis
        positions = orbit.get_position(time_axis).T
        velocities = orbit.get_velocity(time_axis).T
        antenna_reference_frame = transform.Rotation.from_quat(attitude.quaternions).as_matrix()

        ypr = np.rad2deg(
            compute_yaw_pitch_roll_from_antenna_reference_frame(
                reference_frame,
                antenna_reference_frame,
                positions,
                velocities,
                rotation_order,
            )
        )
        yaw = ypr[..., 0]
        pitch = ypr[..., 1]
        roll = ypr[..., 2]

    return metadata.AttitudeInfo(yaw, pitch, roll, t0, delta_t, reference_frame, rotation_order)
