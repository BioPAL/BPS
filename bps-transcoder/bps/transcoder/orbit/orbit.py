# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Orbit module
"""

import enum

from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.io import metadata


class EOrbitType(enum.Enum):
    """EOrbitType class

    Parameters
    ----------
    enum : enum.Enum
        Generic enumeration class
    """

    DOWNLINK = "DOWNLINK"
    PREDICTED = "PREDICTED"
    RESTITUTED = "RESTITUTED"
    PRECISE = "PRECISE"
    UNKNOWN = "UNKNOWN"


class Orbit:
    """Orbit class"""

    def __init__(self, orbit_file):
        """Initialise Orbit object

        Parameters
        ----------
        orbit_file : str
            Path to orbit file
        """
        self.orbit_file = orbit_file

        self.name = None
        self.mission = None
        self.type = None
        self.start_time = None
        self.stop_time = None

        self.position_sv = []
        self.velocity_sv = []
        self.reference_time = None
        self.delta_time = None

    def get_orbit_direction(self, current_time):
        """Get orbit direction

        Parameters
        ----------
        current_time : PreciseDateTime
            Current time [UTC]

        Returns
        -------
        str
            Orbit direction

        Raises
        ------
        RuntimeError
            Orbit not initialized
        """
        if self.name is None:
            raise RuntimeError("Orbit not initialized")

        current_ind = int((current_time - self.reference_time) / self.delta_time)
        if self.velocity_sv[current_ind][2] > 0:
            return metadata.EOrbitDirection.ascending.value
        else:
            return metadata.EOrbitDirection.descending.value

    def get_gso(self):
        """Get General SAR Orbit representation

        Returns
        -------
        GeneralSarOrbit
            General SAR Orbit object
        """
        state_vectors = metadata.StateVectors(
            self.position_sv,
            self.velocity_sv,
            self.reference_time,
            self.delta_time,
        )

        gso = create_general_sar_orbit(state_vectors)

        return gso
