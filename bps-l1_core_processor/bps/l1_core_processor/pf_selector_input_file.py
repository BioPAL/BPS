# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
PF selector input structures
----------------------------
"""

from dataclasses import dataclass, field
from enum import Enum, auto

from arepytools.timing.precisedatetime import PreciseDateTime


class PFSelectorPolarization(Enum):
    """PF selector supported polarizations"""

    HH = auto()
    HV = auto()
    VH = auto()
    VV = auto()


@dataclass
class IndexInterval:
    """Index interval: for bursts, lines, or samples"""

    start_index: int
    length: int


@dataclass
class PFSelectorAreaSwathsBursts:
    """Selection by swath name and, optionally, burst interval"""

    @dataclass
    class Swath:
        """Burst interval, swath selection"""

        name: str
        burst_interval: IndexInterval | None = None

    swaths: list[str]


@dataclass
class PFSelectorAreaRasterCoordinates:
    """Selection by swath name and raster coordinates: lines and samples"""

    @dataclass
    class Swath:
        """Raster coordinates interval, swath selection"""

        name: str

        lines_interval: IndexInterval | None = None
        samples_interval: IndexInterval | None = None

    swaths: list[Swath]


@dataclass
class PFSelectorAreaGeographicCoordinates:
    """Selection by tie points and swath names"""

    @dataclass
    class TiePoint:
        """Tie point in lat lon [deg]"""

        lat: float
        lon: float

    tie_points: list[TiePoint]
    swaths: list[str] = field(default_factory=list)


@dataclass
class PFSelectorAreaTimeCoordinates:
    """Selection by swath name and time coordinates"""

    @dataclass
    class AzimuthTimeInterval:
        """Azimuth time interval"""

        start_time: PreciseDateTime
        duration: float

    @dataclass
    class RangeTimeInterval:
        """Range time interval"""

        start_time: float
        duration: float

    azimuth_time_interval: AzimuthTimeInterval | None
    range_time_interval: RangeTimeInterval | None
    swaths: list[str] = field(default_factory=list)


PFSelectorAreaOptions = (
    PFSelectorAreaSwathsBursts
    | PFSelectorAreaRasterCoordinates
    | PFSelectorAreaGeographicCoordinates
    | PFSelectorAreaTimeCoordinates
)
"""All PFSelector area options"""
