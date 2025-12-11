# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Overlay utilities"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bps.common.io.parsing import serialize
from bps.transcoder.io import overlay
from bps.transcoder.utils.product_name import parse_l1product_name, parse_l2aproduct_name, parse_l2bproduct_name
from bps.transcoder.utils.production_model_utils import encode_product_name_id_value


@dataclass
class Overlay:
    """Minimal information to write an overlay file"""

    overlay_file_name: str
    quicklook_file_name: str
    product_name: str
    footprint: list[tuple[float, float]]
    document_description: str
    product_type: str
    product_start_time: datetime
    product_stop_time: datetime
    mission_phase_id: str
    global_coverage_id: str
    major_cycle_id: str
    repeat_cycle_id: str
    track_number: str
    frame_number: str
    baseline_id: str


def crosses_antimeridian(footprint: list[tuple[float, float]]) -> bool:
    """Check if footprint crosses antimeridian"""
    longitudes = [lon for _, lon in footprint]
    return any(abs(lon1 - lon2) > 180 for lon1, lon2 in zip(longitudes, longitudes[1:]))


def normalize_longitudes(footprint: list[tuple[float, float]]) -> list[tuple[float, float]]:
    "Normalize footprint longitude coordinates if it crosses antimeridian"
    return [(lat, lon if lon >= 0 else lon + 360) for lat, lon in footprint]


def translate_overlay_file(overlay_info: Overlay) -> overlay.Kml:
    """Translate overlay to model"""

    def datetime_to_str(dt: datetime) -> str:
        """Convert datetime to string in ISO format"""
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def footprint_to_str(footprint: list[tuple[float, float]], add_zero_height: bool = False) -> str:
        """Convert footprint to string"""
        height_string = ",0" if add_zero_height else ""
        return " ".join(f"{lon:.6f},{lat:.6f}" + height_string for lat, lon in footprint)

    product_duration = overlay_info.product_stop_time - overlay_info.product_start_time
    product_mid_time = overlay_info.product_start_time + product_duration / 2

    time_stamp = overlay.TimeStampType(datetime_to_str(product_mid_time))

    ground_overlay = overlay.GroundOverlayType(
        name="Ground overlay",
        visibility=1,
        time_stamp=time_stamp,
        icon=overlay.IconType(href=overlay_info.quicklook_file_name),
        lat_lon_quad=overlay.LatLonQuad(footprint_to_str(overlay_info.footprint)),
    )

    placemark = overlay.PlacemarkType(
        name="Placemark",
        visibility=1,
        time_stamp=time_stamp,
        style=overlay.StyleType(
            line_style=overlay.LineStyleType(color="40ffffff", width=1),
            poly_style=overlay.PolyStyleType(color="00000000"),
        ),
        extended_data=overlay.ExtendedDataType(
            data=[
                overlay.DataType(overlay_info.product_name, "Product Name"),
                overlay.DataType(overlay_info.product_type, "Product Type"),
                overlay.DataType(datetime_to_str(overlay_info.product_start_time), "Start Date and Time"),
                overlay.DataType(datetime_to_str(overlay_info.product_stop_time), "Stop Date and Time"),
                overlay.DataType(str(int(product_duration.total_seconds())), "Duration [s]"),
                overlay.DataType(overlay_info.mission_phase_id, "Mission Phase ID"),
                overlay.DataType(overlay_info.global_coverage_id, "Global Coverage ID"),
                overlay.DataType(overlay_info.major_cycle_id, "Major Cycle ID"),
                overlay.DataType(overlay_info.repeat_cycle_id, "Repeat Cycle ID"),
                overlay.DataType(overlay_info.track_number, "Track Number"),
                overlay.DataType(overlay_info.frame_number, "Frame Number"),
                overlay.DataType(overlay_info.baseline_id, "Baseline ID"),
            ]
        ),
        polygon=overlay.PolygonType(
            tessellate=1,
            altitude_mode="clampToGround",
            outer_boundary_is=overlay.OuterBoundaryIsType(
                linear_ring=overlay.LinearRingType(
                    footprint_to_str([*overlay_info.footprint, overlay_info.footprint[0]], add_zero_height=True)
                )
            ),
        ),
    )

    return overlay.Kml(
        overlay.DocumentType(
            overlay_info.overlay_file_name,
            overlay_info.document_description,
            ground_overlay,
            placemark,
        )
    )


def write_overlay_file(
    overlay_file: Path,
    quicklook_file: Path,
    product_name: str,
    footprint: list[tuple[float, float]],
    description: str,
    utc_start_time: str | None = None,
    utc_stop_time: str | None = None,
):
    """Write overlay file"""

    if "L2A" in product_name:
        product_info = parse_l2aproduct_name(product_name, time_format="str")
        assert isinstance(product_info.utc_start_time, str)
        assert isinstance(product_info.utc_stop_time, str)
        utc_start_time = product_info.utc_start_time
        utc_stop_time = product_info.utc_stop_time

    elif "L2B" in product_name:
        product_info = parse_l2bproduct_name(product_name)
        assert isinstance(utc_start_time, str)
        assert isinstance(utc_stop_time, str)
    else:
        product_info = parse_l1product_name(product_name, time_format="str")
        assert isinstance(product_info.utc_start_time, str)
        assert isinstance(product_info.utc_stop_time, str)
        utc_start_time = product_info.utc_start_time
        utc_stop_time = product_info.utc_stop_time

    overlay_info = Overlay(
        overlay_file_name=overlay_file.name,
        quicklook_file_name=quicklook_file.name,
        product_name=product_name,
        footprint=footprint if not crosses_antimeridian(footprint) else normalize_longitudes(footprint),
        document_description=description,
        product_type=product_info.product_type,
        product_start_time=datetime.strptime(utc_start_time, "%Y%m%dT%H%M%S"),
        product_stop_time=datetime.strptime(utc_stop_time, "%Y%m%dT%H%M%S"),
        mission_phase_id=product_info.mission_id,
        global_coverage_id=encode_product_name_id_value(product_info.coverage, npad=2),
        major_cycle_id="-1"
        if "L2B" in product_name
        else encode_product_name_id_value(product_info.major_cycle, npad=2),
        repeat_cycle_id="-1"
        if "L2B" in product_name
        else encode_product_name_id_value(product_info.repeat_cycle, npad=2),
        track_number="-1" if "L2B" in product_name else encode_product_name_id_value(product_info.track_number, npad=3),
        frame_number="-1" if "L2B" in product_name else encode_product_name_id_value(product_info.frame_number, npad=3),
        baseline_id=f"{product_info.baseline}".rjust(2, "0"),
    )

    kml_model = translate_overlay_file(overlay_info)

    kml_text = serialize(
        kml_model,
        ns_map={
            "gx": "http://www.google.com/kml/ext/2.2",
            "kml": "http://www.opengis.net/kml/2.2",
        },
    )
    overlay_file.write_text(kml_text, encoding="utf-8")
