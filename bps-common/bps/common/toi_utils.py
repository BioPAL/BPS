# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Some common time utilities.
---------------------------
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from arepytools.timing.precisedatetime import PreciseDateTime


class InvalidTimeOfInterestError(ValueError):
    """Handle errors of invalid time-of-interests."""


@dataclass
class TimeOfInterest:
    """An absolute time slice in azimuth, aka a TOI."""

    time_begin: PreciseDateTime | None = None  # [UTC].
    """The start time in UTC. It defaults to no start time."""

    time_end: PreciseDateTime | None = None  # [UTC].
    """The end time in UTC. It defaults to no end time."""


def toi_to_axis_slice(
    toi: TimeOfInterest,
    time_axis: npt.NDArray[PreciseDateTime],
) -> tuple[int, int]:
    """
    Find the slice on a reference time axis that is associated to
    a TOI.

    Parameters
    ----------
    toi: TimeOfInterest [UTC]
        The query TOI.

    time_axis: npt.NDArray[PreciseDateTime] [UTC]
        The reference time axis.

    Raises
    ------
    InvalidTimeOfInterestError

    Return
    ------
    tuple[int, int]
        Begin and end indices corresponding to the start and end
        time of the TOI.

    """
    # Check the input TOI.
    raise_if_invalid_toi(toi)

    # Verify that the reference time axis and the TOI interesect.
    time_begin = toi.time_begin or time_axis[0]
    time_end = toi.time_end or time_axis[-1]

    if time_begin > time_axis[-1] or time_end < time_axis[0]:
        raise InvalidTimeOfInterestError("TOI and time axis do not intersect")

    return (
        int(np.argmin([abs(t - time_begin) for t in time_axis])),
        int(np.argmin([abs(t - time_end) for t in time_axis])),
    )


def raise_if_invalid_toi(toi: TimeOfInterest):
    """
    Check that the TOI is valid.

    Parameters
    ----------
    toi: TimeOfInterest [UTC]
        The time of interest.

    Raises
    ------
    InvalidTimeOfInterestError

    """
    if toi.time_begin is not None and toi.time_end is not None and toi.time_begin >= toi.time_end:
        raise InvalidTimeOfInterestError("End time must be later than start time")
