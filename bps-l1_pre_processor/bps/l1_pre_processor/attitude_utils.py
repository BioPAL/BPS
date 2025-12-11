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

from functools import partial
from pathlib import Path

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.transcoder.auxiliaryfiles.aux_attitude import (
    Attitude,
    fill_attitude_from_model,
    read_attitude_file,
    update_model_with_additional_attitude_records,
)
from bps.transcoder.io import aux_att_models
from scipy.spatial import transform


def find_gap_positions(attitude: Attitude) -> list[int]:
    """Retrieve gap positions"""
    margin = 0.2
    max_gap = attitude.max_time_gap + margin
    deltas = np.diff(attitude.relative_time_axis)
    gap_positions = np.where(deltas >= max_gap)[0]
    return gap_positions.tolist()


def are_overlapping(
    interval_a,
    interval_b,
) -> bool:
    """Check that the two intervals overlap"""

    (
        a_start,
        a_stop,
    ) = interval_a

    (
        b_start,
        b_stop,
    ) = interval_b

    assert a_start <= a_stop and b_start <= b_stop

    return (
        b_start <= a_start <= b_stop
        or b_start <= a_stop <= b_stop
        or a_start <= b_start <= a_stop
        or a_start <= b_stop <= a_stop
    )


def compute_missing_records(
    attitude: Attitude, gap_relative_time_intervals: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the gaps in attitude"""
    # Average time steps excluding gaps
    steps = np.diff(attitude.relative_time_axis)
    average_steps = np.mean(steps[steps < attitude.max_time_gap])

    # Time axis with resonable steps to cover the gaps
    missing_times = []
    for start_gap, stop_gap in gap_relative_time_intervals:
        missing_times.append(np.arange(start_gap + average_steps, stop_gap, step=average_steps))
    missing_times = np.sort(np.concatenate(missing_times))

    # Quaternions are converted to euler angles for interpolation
    default_euler_sequence = "xyz"
    euler_angles = transform.Rotation.from_quat(attitude.quaternions).as_euler(seq=default_euler_sequence)

    # Euler angles interpolation
    _interpolate_on_missing_times = partial(np.interp, x=missing_times, xp=attitude.relative_time_axis)
    missing_euler_angles = np.vstack(
        tuple(_interpolate_on_missing_times(fp=euler_angles[:, coord]) for coord in range(3))
    ).T

    # Euler angles are converted back to quaternions
    missing_quaternions = transform.Rotation.from_euler(
        seq=default_euler_sequence,
        angles=missing_euler_angles,
    ).as_quat(canonical=True)

    return missing_times, missing_quaternions


def repair_attitude_if_needed(
    attitude_xml_file: Path,
    processing_interval: tuple[PreciseDateTime, PreciseDateTime],
) -> aux_att_models.EarthObservationFile | None:
    """Analyse attitude file, looking for gaps in data, possibly filling them"""
    attitude_model = read_attitude_file(attitude_xml_file)
    attitude = fill_attitude_from_model(attitude_model)
    gap_positions = find_gap_positions(attitude)
    if len(gap_positions) == 0:
        return None

    bps_logger.warning(f"Found {len(gap_positions)} time gap in attitude input file: {attitude_xml_file}.")

    gap_relative_time_intervals: list[tuple[float, float]] = [
        (
            attitude.relative_time_axis[gap_position],
            attitude.relative_time_axis[gap_position + 1],
        )
        for gap_position in gap_positions
    ]

    gap_time_intervals = [
        (t0 + attitude.time_axis_origin, t1 + attitude.time_axis_origin) for t0, t1 in gap_relative_time_intervals
    ]

    for gap in gap_time_intervals:
        if are_overlapping(gap, processing_interval):
            raise RuntimeError(
                f"Attitude file: {attitude_xml_file} has a time gap: {gap}"
                + f" overlapping the processing time interval {processing_interval}"
            )

        bps_logger.warning(f"Attitude time gap {gap} outside of processing time interval {processing_interval}")

    missing_times, missing_quaternions = compute_missing_records(attitude, gap_relative_time_intervals)

    update_model_with_additional_attitude_records(
        attitude_model, attitude.time_axis_origin, missing_times, missing_quaternions
    )

    return attitude_model
