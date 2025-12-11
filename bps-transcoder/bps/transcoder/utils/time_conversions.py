# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT


"""
Time conversions module
-----------------------
"""

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.transcoder.utils.constants import REFERENCE_EPOCH

REFERENCE_PDT = PreciseDateTime.fromisoformat(REFERENCE_EPOCH)


def pdt_to_compact_string(pdt: PreciseDateTime) -> str:
    """Time PreciseDateTime-to-string (compact) conversion

    Parameters
    ----------
    pdt : PreciseDateTime
        Time (PreciseDateTime)

    Returns
    -------
    str
        Time (str, compact)
    """
    return pdt.isoformat().replace("-", "").replace(":", "").split(".")[0]


def pdt_to_compact_date(pdt: PreciseDateTime, ref: PreciseDateTime | None = None) -> str:
    """Time PreciseDateTime-to-compact date conversion

    Parameters
    ----------
    pdt : PreciseDateTime
        Time (PreciseDateTime)
    ref : PreciseDateTime, optional
        Reference epoch, by default REFERENCE_PDT

    Returns
    -------
    str
        Relative time base-36 encoded
    """
    ref = REFERENCE_PDT if ref is None else ref

    pdt_rel = int(pdt - ref)
    return np.base_repr(pdt_rel, 36)


def no_zulu_isoformat(
    time: PreciseDateTime,
    *,
    timespec: str,
) -> str:
    """Remove Zulu 'Z' suffix from ISO format string."""
    return time.isoformat(timespec=timespec)[:-1]


def round_precise_datetime(
    time: PreciseDateTime,
    *,
    timespec: str,
) -> PreciseDateTime:
    """
    Round the time stamp to a target specified resolution.

    Parameters
    ----------
    time: PreciseDateTime [UTC]
        The timestamp.

    timespec: str
        The target resolution. This can be 'milliseconds', 'microseconds',
        or 'nanoseconds'.

    Raises
    ------
    ValueError: When an invalid time-spec is provided.

    Return
    ------
    PreciseDateTime [UTC]
        The rounded timestamp.

    """
    picoseconds = time.picosecond_of_second
    if timespec == "milliseconds":
        picoseconds = np.round(picoseconds * 1e-12, 3) * 1e12
    elif timespec == "microseconds":
        picoseconds = np.round(picoseconds * 1e-12, 6) * 1e12
    elif timespec == "nanoseconds":
        picoseconds = np.round(picoseconds * 1e-12, 9) * 1e12
    else:
        raise ValueError(f"Unsupported timespec '{timespec}'")

    return PreciseDateTime.from_numeric_datetime(
        year=time.year,
        month=time.month,
        day=time.day_of_the_month,
        hours=time.hour_of_day,
        minutes=time.minute_of_hour,
        seconds=time.second_of_minute,
        picoseconds=picoseconds,
    )
