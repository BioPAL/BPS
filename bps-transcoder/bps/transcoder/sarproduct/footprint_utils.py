# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Footprint utilities"""

import numpy as np
from arepytools.geometry.conversions import xyz2llh
from arepytools.geometry.generalsarorbit import GeneralSarOrbit
from arepytools.timing.precisedatetime import PreciseDateTime


def compute_footprint(
    time_corners: tuple[float, float, PreciseDateTime, PreciseDateTime],
    gso: GeneralSarOrbit,
    look_direction: str,
) -> list[list[float]]:
    """Compute footprint"""
    samples_start, samples_stop, lines_start, lines_stop = time_corners

    footprint_nn = gso.sat2earth(lines_start, samples_start, look_direction)
    footprint_fn = gso.sat2earth(lines_start, samples_stop, look_direction)
    footprint_nf = gso.sat2earth(lines_stop, samples_start, look_direction)
    footprint_ff = gso.sat2earth(lines_stop, samples_stop, look_direction)

    footprint: list[list[float]] = []
    for f in [
        footprint_nf,
        footprint_ff,
        footprint_fn,
        footprint_nn,
    ]:
        footprint.append(list(np.degrees(xyz2llh(f)[[0, 1]].squeeze())))

    return footprint


def _parse_footprint_string_with_commas(footprint_str: str) -> list[list[float]]:
    return [[float(v) for v in f.split(",")] for f in footprint_str.split(" ")]


def _parse_footprint_string_with_spaces(footprint_str) -> list[list[float]]:
    coords = [float(value) for value in footprint_str.split()]
    if len(coords) != 8:
        raise RuntimeError(f"Cannot read footprint: {footprint_str}")

    return [[coords[2 * corner_index], coords[2 * corner_index + 1]] for corner_index in range(4)]


def parse_footprint_string(footprint_str: str, closed: bool = False) -> list[list[float]]:
    """Parse footprint string

    "1 2 3 4 5 6 7 8" and "1,2 3,4 5,6 7,8" are both supported
    and converted to
    [[1,2], [3,4], [5,6], [7,8]]

    If `closed` is set to True, the footprint is returned as a closed
    polygon, e.g. [[1,2], [3,4], [5,6], [7,8], [1,2]].

    """
    if "," in footprint_str:
        footprint = _parse_footprint_string_with_commas(footprint_str)
    else:
        footprint = _parse_footprint_string_with_spaces(footprint_str)

    if len(footprint) != 4:
        raise RuntimeError(f"Cannot read footprint: {footprint_str}")

    if closed:
        footprint.append(footprint[0])

    return footprint


def serialize_footprint(footprint: list[list[float]]) -> str:
    """Serialize footprint

    [[1,2], [3,4], [5,6], [7,8]] is converted to
    "1 2 3 4 5 6 7 8"
    """
    return " ".join([" ".join(str(value) for value in corner) for corner in footprint])


DEFAULT_GCP_RANGE_SAMPLING = 500
DEFAULT_GCP_AZIMUTH_SAMPLING = 2000


def gcp_axis_sampling(
    range_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    samples_sub_sampling: int | None,
    lines_sub_sampling: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub sample axis for gcp computation"""
    samples_sub_sampling = samples_sub_sampling or DEFAULT_GCP_RANGE_SAMPLING
    lines_sub_sampling = lines_sub_sampling or DEFAULT_GCP_AZIMUTH_SAMPLING

    samples_indexes = np.append(
        np.arange(0, range_axis.size, samples_sub_sampling),
        range_axis.size - 1,
    )
    lines_indexes = np.append(
        np.arange(0, azimuth_axis.size, lines_sub_sampling),
        azimuth_axis.size - 1,
    )
    return samples_indexes, lines_indexes


def compute_ground_corner_points_on_wgs84(
    range_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    gso: GeneralSarOrbit,
    look_direction: str,
    samples_sub_sampling: int | None = None,
    lines_sub_sampling: int | None = None,
) -> list[list]:
    """Compute ground corner points on ellipsoid"""
    samples_indexes, lines_indexes = gcp_axis_sampling(
        range_axis, azimuth_axis, samples_sub_sampling, lines_sub_sampling
    )

    samples_axis = range_axis[samples_indexes]
    lines_axis = azimuth_axis[lines_indexes]

    gcp_list: list[list] = []
    for sample, sample_index in zip(samples_axis, samples_indexes):
        for line, line_index in zip(lines_axis, lines_indexes):
            gcp = gso.sat2earth(line, sample, look_direction).squeeze().tolist()
            gcp_list.append(gcp + [int(sample_index), int(line_index)])

    return gcp_list


def is_null_footprint(footprint: list[list[float]]) -> bool:
    """Check if the footprint is a void one."""
    return footprint == [[0.0, 0.0]] * 4 or footprint == [[0.0, 0.0]] * 5
