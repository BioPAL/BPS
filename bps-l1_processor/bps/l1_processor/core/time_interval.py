# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Time interval utilities
-----------------------
"""

from arepytools.timing.precisedatetime import PreciseDateTime

TimeInterval = tuple[PreciseDateTime, PreciseDateTime]


def contains(
    interval_a: TimeInterval,
    interval_b: TimeInterval,
):
    """Wether a contains b"""
    (
        a_start,
        a_stop,
    ) = interval_a

    (
        b_start,
        b_stop,
    ) = interval_b

    assert a_start <= a_stop and b_start <= b_stop

    return a_start <= b_start <= b_stop <= a_stop


def are_overlapping(
    interval_a: TimeInterval,
    interval_b: TimeInterval,
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
