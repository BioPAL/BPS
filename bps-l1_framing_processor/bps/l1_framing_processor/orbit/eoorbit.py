# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Earth Observation orbit module
"""

import os
from pathlib import Path

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.l1_framing_processor.io import aux_orb_models
from bps.l1_framing_processor.orbit.orbit import EOrbitType, Orbit
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser

_CONTEXT = XmlContext()
_PARSER = XmlParser(context=_CONTEXT)


class EOOrbit(Orbit):
    """EOOrbit class

    Parameters
    ----------
    Orbit : Orbit
        Generic Orbit class
    """

    orbit_type_dict = {
        "AUX_PREORB": EOrbitType.PREDICTED.value,
        "AUX_RESORB": EOrbitType.RESTITUTED.value,
        "AUX_POEORB": EOrbitType.PRECISE.value,
    }

    def __init__(self, orbit_file):
        """Initialise EOOrbit object

        Parameters
        ----------
        orbit_file : str
            Path to orbit file
        """
        Orbit.__init__(self, orbit_file)

        self.__read_orbit()

    def __read_orbit(self):
        """Read orbit file

        Returns
        -------
        bool
            Status (True for success, False for unsuccess)
        """
        # Set orbit file name
        self.name = os.path.basename(self.orbit_file)

        # Read orbit file
        # - Initialize model
        orbit_model = _PARSER.from_string(Path(self.orbit_file).read_text(), aux_orb_models.EarthObservationFile)

        # - Set orbit metadata
        fixed_header = orbit_model.earth_observation_header.fixed_header
        self.mission = fixed_header.mission
        self.type = self.orbit_type_dict.get(fixed_header.file_type, EOrbitType.UNKNOWN.value)
        self.start_time = PreciseDateTime().set_from_utc_string(
            fixed_header.validity_period.validity_start[4:] + ".000000"
        )
        self.stop_time = PreciseDateTime().set_from_utc_string(
            fixed_header.validity_period.validity_stop[4:] + ".000000"
        )

        # - Set orbit data
        list_of_osvs = orbit_model.data_block.list_of_osvs
        sv_count = len(list_of_osvs.osv)
        self.position_sv = np.zeros((sv_count, 3))
        self.velocity_sv = np.zeros((sv_count, 3))
        for sv in range(sv_count):
            self.position_sv[sv][0] = float(list_of_osvs.osv[sv].x.value)
            self.position_sv[sv][1] = float(list_of_osvs.osv[sv].y.value)
            self.position_sv[sv][2] = float(list_of_osvs.osv[sv].z.value)
            self.velocity_sv[sv][0] = float(list_of_osvs.osv[sv].vx.value)
            self.velocity_sv[sv][1] = float(list_of_osvs.osv[sv].vy.value)
            self.velocity_sv[sv][2] = float(list_of_osvs.osv[sv].vz.value)
        self.reference_time = PreciseDateTime().set_from_utc_string(list_of_osvs.osv[0].utc[4:])
        self.delta_time = PreciseDateTime().set_from_utc_string(list_of_osvs.osv[1].utc[4:]) - self.reference_time

        return True
