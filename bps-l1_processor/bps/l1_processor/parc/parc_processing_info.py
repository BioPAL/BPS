# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
PARC processing info
--------------------
"""

from dataclasses import dataclass, field

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.l1_processor.parc.parc_info import ScatteringResponse


@dataclass
class Window:
    """Time-based description of product window"""

    azimuth_start: PreciseDateTime
    azimuth_stop: PreciseDateTime
    range_start: float
    range_stop: float


@dataclass
class Delays:
    """Azimuth and range time delays"""

    azimuth_delay: float
    range_delay: float


@dataclass
class ParcProcessingInfo:
    """Information for processing around Parcs"""

    parc_id: str
    processing_data: dict[ScatteringResponse, tuple[Delays, Window]] = field(default_factory=dict)
