# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
PARC info
---------
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime


class ScatteringResponse(Enum):
    """Scattering responses"""

    GT1 = 1
    GT2 = 2
    X = 3
    Y = 4


@dataclass
class ParcInfo:
    """ParcInfo"""

    parc_id: str
    validity_interval: tuple[PreciseDateTime, PreciseDateTime]
    position: np.ndarray
    delays: dict[ScatteringResponse, float]
    rcs: dict[ScatteringResponse, float]


ParcInfoList = list[ParcInfo]
