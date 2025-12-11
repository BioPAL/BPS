# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Orbital computations utils
"""

import numpy as np
from arepytools.geometry.conversions import xyz2llh
from arepytools.geometry.generalsarorbit import GeneralSarOrbit
from arepytools.timing.precisedatetime import PreciseDateTime


def compute_platform_heading(
    time_corners: tuple[float, float, PreciseDateTime, PreciseDateTime],
    gso: GeneralSarOrbit,
    look_direction: str,
) -> float:
    """Compute platform heading (in degrees)"""
    samples_start, _, lines_start, lines_stop = time_corners

    footprint_nn = xyz2llh(gso.sat2earth(lines_start, samples_start, look_direction)).squeeze()
    footprint_nf = xyz2llh(gso.sat2earth(lines_stop, samples_start, look_direction)).squeeze()

    delta_latitude = footprint_nf[0] - footprint_nn[0]
    delta_longitude = footprint_nf[1] - footprint_nn[1]

    return np.degrees(np.arctan2(delta_longitude, delta_latitude))
