# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Main product header management"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger
from bps.transcoder.utils.production_model_utils import decode_mph_id_value

MPH_NAMESPACES = {
    "bio": "http://earth.esa.int/biomass/1.0",
    "eop": "http://www.opengis.net/eop/2.1",
    "gml": "http://www.opengis.net/gml/3.2",
    "om": "http://www.opengis.net/om/2.0",
    "ows": "http://www.opengis.net/ows/2.0",
    "sar": "http://www.opengis.net/sar/2.1",
    "xlink": "http://www.w3.org/1999/xlink",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def get_phenomenon_time(root: ET.Element) -> tuple[PreciseDateTime, PreciseDateTime] | None:
    phenomenon_time = root.find("om:phenomenonTime", MPH_NAMESPACES)
    if phenomenon_time is None:
        return None

    return _get_time_period(phenomenon_time)


def get_valid_time(root: ET.Element) -> tuple[PreciseDateTime, PreciseDateTime] | None:
    valid_time = root.find("om:validTime", MPH_NAMESPACES)
    if valid_time is None:
        return None

    return _get_time_period(valid_time)


@dataclass
class SensorInfo:
    swath: str
    mission_phase: str


def get_sensor(root: ET.Element):
    swath_elem = root.find(".//eop:swathIdentifier", MPH_NAMESPACES)
    assert swath_elem is not None
    assert swath_elem.text is not None

    mission_phase_elem = root.find(".//bio:missionPhase", MPH_NAMESPACES)
    assert mission_phase_elem is not None
    assert mission_phase_elem.text is not None

    return SensorInfo(swath=swath_elem.text, mission_phase=mission_phase_elem.text)


def _get_time_period(elem: ET.Element) -> tuple[PreciseDateTime, PreciseDateTime]:
    sensing_start_time_elem = elem.find("gml:TimePeriod/gml:beginPosition", MPH_NAMESPACES)
    assert sensing_start_time_elem is not None and sensing_start_time_elem.text is not None
    sensing_start_time = PreciseDateTime.fromisoformat(sensing_start_time_elem.text)

    sensing_stop_time_elem = elem.find("gml:TimePeriod/gml:endPosition", MPH_NAMESPACES)
    assert sensing_stop_time_elem is not None and sensing_stop_time_elem.text is not None
    sensing_stop_time = PreciseDateTime.fromisoformat(sensing_stop_time_elem.text)
    return sensing_start_time, sensing_stop_time


@dataclass
class MPHAcquisition:
    orbit_number: int
    orbit_direction: Literal["ASCENDING", "DESCENDING"]
    track_number: int
    slice_number: int
    anx_time: PreciseDateTime
    mission_phase_id: str
    instrument_configuration_id: int
    datatake_id: int
    orbit_drift_flag: bool
    global_coverage_id: int
    major_cycle_id: int
    repeat_cycle_id: int


def get_acquisition(root: ET.Element) -> MPHAcquisition:
    acquisition = root.find(
        "om:procedure/eop:EarthObservationEquipment/eop:acquisitionParameters/bio:Acquisition", MPH_NAMESPACES
    )
    assert acquisition is not None

    orbit_number_elem = acquisition.find("eop:orbitNumber", MPH_NAMESPACES)
    orbit_direction_elem = acquisition.find("eop:orbitDirection", MPH_NAMESPACES)
    track_number_elem = acquisition.find("eop:wrsLongitudeGrid", MPH_NAMESPACES)
    slice_number_elem = acquisition.find("eop:wrsLatitudeGrid", MPH_NAMESPACES)
    anx_time_elem = acquisition.find("eop:ascendingNodeDate", MPH_NAMESPACES)
    mission_phase_id_elem = acquisition.find("bio:missionPhase", MPH_NAMESPACES)
    instrument_configuration_id_elem = acquisition.find("bio:instrumentConfID", MPH_NAMESPACES)
    datatake_id_elem = acquisition.find("bio:dataTakeID", MPH_NAMESPACES)
    orbit_drift_flag_elem = acquisition.find("bio:orbitDriftFlag", MPH_NAMESPACES)
    global_coverage_id_elem = acquisition.find("bio:globalCoverageID", MPH_NAMESPACES)
    major_cycle_id_elem = acquisition.find("bio:majorCycleID", MPH_NAMESPACES)
    repeat_cycle_id_elem = acquisition.find("bio:repeatCycleID", MPH_NAMESPACES)
    assert orbit_number_elem is not None
    assert orbit_direction_elem is not None
    assert track_number_elem is not None
    assert slice_number_elem is not None
    assert anx_time_elem is not None
    assert mission_phase_id_elem is not None
    assert instrument_configuration_id_elem is not None
    assert datatake_id_elem is not None
    assert orbit_drift_flag_elem is not None
    assert global_coverage_id_elem is not None
    assert major_cycle_id_elem is not None
    assert repeat_cycle_id_elem is not None

    orbit_number_text = orbit_number_elem.text
    orbit_direction_text = orbit_direction_elem.text
    track_number_text = track_number_elem.text
    slice_number_text = slice_number_elem.text
    anx_time_text = anx_time_elem.text
    mission_phase_id_text = mission_phase_id_elem.text
    instrument_configuration_id_text = instrument_configuration_id_elem.text
    datatake_id_text = datatake_id_elem.text
    orbit_drift_flag_text = orbit_drift_flag_elem.text
    global_coverage_id_text = global_coverage_id_elem.text
    major_cycle_id_text = major_cycle_id_elem.text
    repeat_cycle_id_text = repeat_cycle_id_elem.text
    assert orbit_number_text is not None
    assert orbit_direction_text is not None
    assert track_number_text is not None
    assert slice_number_text is not None
    assert anx_time_text is not None
    assert mission_phase_id_text is not None
    assert instrument_configuration_id_text is not None
    assert datatake_id_text is not None
    assert orbit_drift_flag_text is not None
    assert global_coverage_id_text is not None
    assert major_cycle_id_text is not None
    assert repeat_cycle_id_text is not None

    if orbit_direction_text not in ("ASCENDING", "DESCENDING"):
        raise RuntimeError(f"Unexpected orbit direction: {orbit_direction_text}")

    return MPHAcquisition(
        orbit_number=int(orbit_number_text),
        orbit_direction=orbit_direction_text,
        track_number=int(track_number_text),
        slice_number=0 if slice_number_text == "___" else int(slice_number_text),
        anx_time=PreciseDateTime.fromisoformat(anx_time_text),
        mission_phase_id=mission_phase_id_text,
        instrument_configuration_id=int(instrument_configuration_id_text),
        datatake_id=int(datatake_id_text),
        orbit_drift_flag=orbit_direction_text.lower() == "true",
        global_coverage_id=decode_mph_id_value(global_coverage_id_text),
        major_cycle_id=decode_mph_id_value(major_cycle_id_text),
        repeat_cycle_id=decode_mph_id_value(repeat_cycle_id_text),
    )


@dataclass
class MPHProductInformation:
    baseline_id: int
    file_sizes: dict[str, int] = field(default_factory=dict)


def get_product_information(root: ET.Element) -> MPHProductInformation:
    prod_list = root.findall("om:result/eop:EarthObservationResult/eop:product", MPH_NAMESPACES)

    baseline_id = 1
    file_sizes: dict[str, int] = {}

    for prod in prod_list:
        prod_information = prod.find("bio:ProductInformation", MPH_NAMESPACES)
        if prod_information is None:
            prod_information = prod.find(
                "eop:ProductInformation", MPH_NAMESPACES
            )  # kept for retrocompatibility with previous MPH versions
        assert prod_information is not None

        prod_version = prod_information.find("eop:version", MPH_NAMESPACES)
        if prod_version is not None:
            assert prod_version.text is not None
            baseline_id = int(prod_version.text)

        prod_size = None
        size_element = prod_information.find("eop:size", MPH_NAMESPACES)
        if size_element is not None:
            assert size_element.text is not None
            prod_size = int(size_element.text)

        file_name = prod_information.find("eop:fileName", MPH_NAMESPACES)
        if file_name is not None:
            reference = file_name.find("ows:ServiceReference", MPH_NAMESPACES)
            if reference is not None:
                name = reference.attrib["{http://www.w3.org/1999/xlink}href"]
                if name is not None and prod_size is not None:
                    file_sizes[name] = prod_size

    return MPHProductInformation(baseline_id=baseline_id, file_sizes=file_sizes)


@dataclass
class MPHMetadata:
    is_partial: bool | None
    is_merged: bool | None
    doi: str


def get_metadata(root: ET.Element) -> MPHMetadata:
    metadata_property = root.find("eop:metaDataProperty", MPH_NAMESPACES)
    assert metadata_property is not None

    eo_metadata = metadata_property.find("bio:EarthObservationMetaData", MPH_NAMESPACES)
    if eo_metadata is None:
        eo_metadata = metadata_property.find(
            "eop:EarthObservationMetaData", MPH_NAMESPACES
        )  # kept for retrocompatibility with previous MPH versions
    assert eo_metadata is not None

    doi_elem = eo_metadata.find("eop:doi", MPH_NAMESPACES)
    assert doi_elem is not None and doi_elem.text is not None
    doi = doi_elem.text

    metadata = MPHMetadata(is_partial=None, is_merged=None, doi=doi)

    is_partial_flag = eo_metadata.find("bio:isPartial", MPH_NAMESPACES)
    if is_partial_flag is not None:
        assert is_partial_flag.text is not None
        metadata.is_partial = is_partial_flag.text.lower() == "true"

    is_merged_flag = eo_metadata.find("bio:isMerged", MPH_NAMESPACES)
    if is_merged_flag is not None:
        assert is_merged_flag.text is not None
        metadata.is_merged = is_merged_flag.text.lower() == "true"

    return metadata


def read_product_doi(mph_path: Path) -> str:
    """read product DOI from MPH

    The function is very fast by reading directly the xml field needed;
    used in product reader, to retrieve product DOI

    Parameters
    ----------
    mph_path: Path,
        Path of MPH xml file

    Returns
    -------
    product_doi: str
        String identifying the product DOI, the one originally coming from AUX_PP file
    """

    root = ET.parse(mph_path).getroot()

    metadata = get_metadata(root)

    return metadata.doi


@dataclass
class MPHFootprint:
    ne_lat: float
    ne_lon: float
    se_lat: float
    se_lon: float
    sw_lat: float
    sw_lon: float
    nw_lat: float
    nw_lon: float

    def to_list(self) -> list[float]:
        """To list in proper ordering"""
        return [self.ne_lat, self.ne_lon, self.se_lat, self.se_lon, self.sw_lat, self.sw_lon, self.nw_lat, self.nw_lon]

    @classmethod
    def from_list(cls, footprint_list) -> Self:
        return cls(
            ne_lat=footprint_list[0],
            ne_lon=footprint_list[1],
            se_lat=footprint_list[2],
            se_lon=footprint_list[3],
            sw_lat=footprint_list[4],
            sw_lon=footprint_list[5],
            nw_lat=footprint_list[6],
            nw_lon=footprint_list[7],
        )

    def coverage(self) -> tuple[float, float, float, float]:
        """return (lat_min, lat_max, lon_min, lon_max)"""
        footprint = self.to_list()
        lat_min = min(footprint[0::2])
        lat_max = max(footprint[0::2])
        lon_min = min(footprint[1::2])
        lon_max = max(footprint[1::2])

        return lat_min, lat_max, lon_min, lon_max


def compute_coverage(footprints: list[MPHFootprint]) -> tuple[float, float, float, float]:
    """return (lat_min, lat_max, lon_min, lon_max)"""
    values = [coord for footprint in footprints for coord in footprint.to_list()]
    latitudes = values[0::2]
    longitudes = values[1::2]
    return (
        np.min(latitudes),
        np.max(latitudes),
        np.min(longitudes),
        np.max(longitudes),
    )


def get_footprint(root: ET.Element) -> list[float] | None:
    xpath_posList = ".//om:featureOfInterest/eop:Footprint/eop:multiExtentOf/gml:MultiSurface/gml:surfaceMember/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList"
    footprint_node = root.find(xpath_posList, MPH_NAMESPACES)
    if footprint_node is not None and footprint_node.text is not None:
        return [float(elem) for elem in footprint_node.text.split()]
    else:
        return None


def read_coverage_and_footprint(
    mph_files: list[Path],
) -> tuple[list[float], list[MPHFootprint]]:
    """Fast read the stack MPH file, without passing through the whole product reading"""

    footprints = [MPHFootprint.from_list(get_footprint(ET.parse(mph_path).getroot())) for mph_path in mph_files]

    # [lat_min, lat_max, lon_min, lon_max]
    latlon_coverage = list(compute_coverage(footprints))

    bps_logger.info(f"Computing latitude longitude coverage from #{len(mph_files)} product footprints:")
    bps_logger.info(f"    latitude [min, max] = [{latlon_coverage[0]:2.2f}, {latlon_coverage[1]:2.2f}] deg")
    bps_logger.info(f"    longitude [min, max] = [{latlon_coverage[2]:2.2f}, {latlon_coverage[3]:2.2f}] deg")
    return latlon_coverage, footprints
