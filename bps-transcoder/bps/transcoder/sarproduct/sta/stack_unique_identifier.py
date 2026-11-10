# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Stack Unique ID
-------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from bps.transcoder.utils.production_model_utils import encode_product_name_id_value
from bps.transcoder.utils.time_conversions import pdt_to_compact_date, pdt_to_compact_string
from perseo_core.timing import PreciseDateTime


@dataclass
class StackUniqueID:
    """
    The stack Unique Identifier.

       BIO_S1_STA__ID_20250523T160904_20250526T161054_C_G___M___N0136_E0322_A_01_D9A8AY

    where:
    * 20250523T160904 is the earliest start time of the input stack products
    * 20250526T161054 is the latest stop time of the input stack products
    * C is the phase (i.e. 'C', 'T', 'I')
    * G__ is the global coverage ID
    * M__ is the major cycle ID
    * N0136 is the latittude of the center of the scene
    * E0322 is the longitude of the center of the scene
    * A is the orbit direction (A=Ascending, D=Descending)
    * 01 is the version (baseline ID)
    * D9A8AY is the compact creation date.

    """

    swath: str
    stack_start_time: PreciseDateTime
    stack_stop_time: PreciseDateTime
    mission_phase: str
    global_coverage_id: int
    major_cycle_id: int
    latitude_deg: float
    longitude_deg: float
    orbit_direction: str
    version: int
    creation_timestamp: PreciseDateTime

    def to_id(self) -> str:
        """Create the Stack's UID."""
        return f"BIO_{self.swath:s}_STA__ID_{pdt_to_compact_string(self.stack_start_time):s}_{pdt_to_compact_string(self.stack_stop_time):s}_{self.mission_phase[0]:s}_G{encode_product_name_id_value(self.global_coverage_id, npad=2):s}_M{encode_product_name_id_value(self.major_cycle_id, npad=2):s}_{_format_latitude_str(self.latitude_deg):s}_{_format_longitude_str(self.longitude_deg):s}_{self.orbit_direction[0]:s}_{self.version:02d}_{pdt_to_compact_date(self.creation_timestamp):s}"


def _format_latitude_str(latitude_deg: float) -> str:
    """Convert a latitude (in degrees) to a latitude string."""
    if latitude_deg >= 0:
        return f"N{round(latitude_deg * 10):04d}"
    return f"S{round(-latitude_deg * 10):04d}"


def _format_longitude_str(longitude_deg: float) -> str:
    """Convert a longitude (in degrees) to a longitude string."""
    if longitude_deg >= 0:
        return f"E{round(longitude_deg * 10):04d}"
    return f"W{round(-longitude_deg * 10):04d}"
