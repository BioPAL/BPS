# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Earth Observation orbit module
------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io.parsing import parse
from bps.transcoder.io import aux_orb_models
from bps.transcoder.orbit.orbit import EOrbitType, Orbit


@dataclass
class StateVectors:
    """State vectors"""

    positions: np.ndarray
    velocities: np.ndarray
    time_start: PreciseDateTime
    time_step: float

    @classmethod
    def translate_state_vectors(
        cls,
        state_vectors: list[aux_orb_models.OsvType],
    ) -> StateVectors:
        """Translate state vectors"""
        sv_count = len(state_vectors)
        positions = np.zeros((sv_count, 3))
        velocities = np.zeros((sv_count, 3))
        for state_vector, position, velocity in zip(state_vectors, positions, velocities):
            position[:], velocity[:] = _translate_state_vector(state_vector)

        time_start = _get_time_from_state_vector(state_vectors[0])
        time_step = _get_time_from_state_vector(state_vectors[1]) - time_start

        return cls(positions, velocities, time_start, time_step)


def _get_time_from_state_vector(
    state_vector: aux_orb_models.OsvType,
) -> PreciseDateTime:
    assert state_vector.utc is not None
    return PreciseDateTime().set_from_utc_string(state_vector.utc[4:])


def _translate_state_vector(
    state_vector: aux_orb_models.OsvType,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate state vector"""

    assert state_vector.x is not None
    assert state_vector.y is not None
    assert state_vector.z is not None
    assert state_vector.vx is not None
    assert state_vector.vy is not None
    assert state_vector.vz is not None
    assert state_vector.x.value is not None
    assert state_vector.y.value is not None
    assert state_vector.z.value is not None
    assert state_vector.vx.value is not None
    assert state_vector.vy.value is not None
    assert state_vector.vz.value is not None

    position = np.array(
        [
            float(state_vector.x.value),
            float(state_vector.y.value),
            float(state_vector.z.value),
        ]
    )

    velocity = np.array(
        [
            float(state_vector.vx.value),
            float(state_vector.vy.value),
            float(state_vector.vz.value),
        ]
    )

    return position, velocity


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

        self._read_orbit()

    def _read_orbit(self):
        """Read orbit file"""

        # Set orbit file name
        self.name = os.path.basename(self.orbit_file)

        # Read orbit file
        # - Initialize model

        orbit_model: aux_orb_models.EarthObservationFile = parse(
            Path(self.orbit_file).read_text(encoding="utf-8"),
            aux_orb_models.EarthObservationFile,
        )

        # - Set orbit metadata
        assert orbit_model.earth_observation_header is not None
        assert orbit_model.earth_observation_header.fixed_header is not None
        fixed_header = orbit_model.earth_observation_header.fixed_header

        self.mission = fixed_header.mission

        assert fixed_header.file_type is not None
        assert fixed_header.validity_period is not None
        assert fixed_header.validity_period.validity_start is not None
        assert fixed_header.validity_period.validity_stop is not None

        self.type = self.orbit_type_dict.get(fixed_header.file_type, EOrbitType.UNKNOWN.value)

        self.start_time = PreciseDateTime().set_from_utc_string(
            fixed_header.validity_period.validity_start[4:] + ".000000"
        )
        self.stop_time = PreciseDateTime().set_from_utc_string(
            fixed_header.validity_period.validity_stop[4:] + ".000000"
        )

        # - Set orbit data
        assert orbit_model.data_block is not None
        assert orbit_model.data_block.list_of_osvs is not None
        state_vectors = StateVectors.translate_state_vectors(orbit_model.data_block.list_of_osvs.osv)

        self.position_sv = state_vectors.positions
        self.velocity_sv = state_vectors.velocities
        self.reference_time = state_vectors.time_start
        self.delta_time = state_vectors.time_step
