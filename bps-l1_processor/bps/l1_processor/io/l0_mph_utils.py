# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Manifest file utilities
-----------------------
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import AcquisitionMode, MissionPhaseID, Swath
from bps.transcoder.sarproduct.mph import get_footprint, get_phenomenon_time, get_sensor


@dataclass
class L0MainProductHeader:
    """Information from MPH file"""

    swath: Swath
    mission_phase_id: MissionPhaseID
    phenomenon_time_begin: PreciseDateTime
    phenomenon_time_end: PreciseDateTime

    footprint: list[tuple[float, float]] | None
    """L0 footprint corners: (lon, lat) tuples, deg"""

    @property
    def acquisition_mode(self) -> AcquisitionMode:
        """The acquisition mode: swath and mission phase"""
        return self.swath, self.mission_phase_id

    @property
    def phenomenon_time(self) -> tuple[PreciseDateTime, PreciseDateTime]:
        """Phenomenon time"""
        return (
            self.phenomenon_time_begin,
            self.phenomenon_time_end,
        )

    @classmethod
    def from_file(cls, mph_file: Path) -> Self:
        """Retrieve information from mph file"""
        root = ET.parse(mph_file).getroot()

        sensor_info = get_sensor(root)

        phenomenon_time = get_phenomenon_time(root)
        if sensor_info.swath is None or sensor_info.mission_phase is None or phenomenon_time is None:
            raise RuntimeError(f"Error during parsing of {mph_file}")

        footprint = get_footprint(root)  # not always present, may be None
        if footprint is not None:
            footprint = [(b, a) for a, b in zip(footprint[::2], footprint[1::2])]

        return cls(
            swath=Swath(sensor_info.swath),
            mission_phase_id=MissionPhaseID[sensor_info.mission_phase],
            phenomenon_time_begin=phenomenon_time[0],
            phenomenon_time_end=phenomenon_time[1],
            footprint=footprint,
        )

    @classmethod
    def from_product(cls, product: Path) -> Self:
        """Retrieve basic swath information from product"""
        xml_content = list(filter(lambda x: x.suffix == ".xml", product.iterdir()))
        if len(xml_content) != 1:
            raise RuntimeError(f"Error during retrieval of manifest file in product: {product}")

        try:
            manifest_swath_info = cls.from_file(xml_content[0])
        except Exception as exc:
            raise RuntimeError(f"Error during parsing of manifest file: {xml_content[0]}") from exc

        return manifest_swath_info
