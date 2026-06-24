# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Main module"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self
import numpy as np
from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.io import metadata
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.transcoder.orbit.eoorbit import EOOrbit
from bps.transcoder.sarproduct.footprint_utils import compute_footprint
from bps.transcoder.sarproduct.l0.product_content import L0ProductContent
from isp_stream import BiomassIspStream
import sys
import folium
import json
import argparse
# Resolve BiomassProduct from the sibling scripts/ folder (../scripts/) so this
# module works regardless of the absolute install location.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from BiomassProduct import BiomassProductL1VFRA



F_USO = 37.826087e6  # [Hz]
K_CONST = 4 / F_USO  # [s]
LEAP_SECONDS = 37  # [s]


@dataclass(frozen=True)
class AUXORBProductContent:
    """AUX_ORB product content"""

    mph_file: Path
    parameters_file_folder: Path
    parameters_file: Path
    schema_file_folder: Path
    schema_file: Path

    @classmethod
    def from_name(cls, name: str) -> Self:
        """Create AUX_ORB product content from name"""
        lower_name = name.lower()
        name_root = lower_name[:-10]

        mph_file = Path(lower_name + ".xml")

        parameters_file_folder = Path("data")
        parameters_file = Path(name_root + "_orbit.xml")

        schema_file_folder = Path("support")
        schema_file = Path("bio_aux_orbit.xsd")

        return cls(
            mph_file=mph_file,
            parameters_file_folder=parameters_file_folder,
            parameters_file=parameters_file,
            schema_file_folder=schema_file_folder,
            schema_file=schema_file,
        )


def read_ispstream(file_in):
    """Read ISP stream"""
    isp_stream = BiomassIspStream(file_in)
    isp_stream.read()
    isp_stream.fh.seek(0, os.SEEK_SET)
    return isp_stream


def get_azimuth_time_given_index(ispstream, index) -> PreciseDateTime:
    """Get azimuth time given packet index"""
    coarse_time = ispstream.df["coarse_time"][index]
    fine_time = ispstream.df["fine_time"][index] * K_CONST
    return PreciseDateTime.from_numeric_datetime(year=2000, month=1, day=1) + (coarse_time + fine_time) - LEAP_SECONDS


def get_range_time_given_index(ispstream, index) -> tuple[float, float]:
    """Get range time given packet index"""
    swp = ispstream.df["swp"][index] / F_USO
    swl = ispstream.df["swl"][index] / F_USO
    rank = ispstream.df["rank"][index]
    pri = ispstream.df["pri"][160] / F_USO
    swp_tot = swp + rank * pri
    return swp_tot, swp_tot + swl


def compute_footprint_from_l0s_custom(
    l0_product_path: Path,
    aux_orb_product_path: Path,
    start_time: PreciseDateTime,
    stop_time: PreciseDateTime,
) -> list[list[float]]:

    """
    Compute the L0 footprint for a custom time interval within the L0 product.
    Only packets with STYP==0 and azimuth time within [start_time, stop_time] are used.
    Returns a list of [lat, lon] corner coordinates.
    """
    l0_product_content = L0ProductContent.from_name(name=l0_product_path.name)
    ispstream = read_ispstream(file_in=l0_product_path.joinpath(l0_product_content.rxh))

    # Select only science packets (STYP==0)
    indexes = np.where(ispstream.df["sigtyp"] == 0)[0]

    # Compute azimuth time for all science packets
    all_times = np.array([
        float(get_azimuth_time_given_index(ispstream, i) - PreciseDateTime.from_numeric_datetime(2000, 1, 1))
        for i in indexes
    ])

    t_start_ref = float(start_time - PreciseDateTime.from_numeric_datetime(2000, 1, 1))
    t_stop_ref  = float(stop_time  - PreciseDateTime.from_numeric_datetime(2000, 1, 1))

    # Keep only packets within the requested time interval
    mask = (all_times >= t_start_ref) & (all_times <= t_stop_ref)
    filtered_indexes = indexes[mask]

    if len(filtered_indexes) == 0:
        raise ValueError("No packets found in the specified time interval!")

    print(f"Packets found in interval: {len(filtered_indexes)}")

    samples_start, samples_stop = get_range_time_given_index(ispstream, filtered_indexes[0])
    lines_start = get_azimuth_time_given_index(ispstream, filtered_indexes[0])
    lines_stop  = get_azimuth_time_given_index(ispstream, filtered_indexes[-1])
    # Load orbit data from AUX_ORB product
    aux_orb_product_content = AUXORBProductContent.from_name(name=aux_orb_product_path.name)
    eo_orbit = EOOrbit(
        orbit_file=aux_orb_product_path
            .joinpath(aux_orb_product_content.parameters_file_folder)
            .joinpath(aux_orb_product_content.parameters_file)
    )
    metadata_sv = metadata.StateVectors(
        eo_orbit.position_sv,
        eo_orbit.velocity_sv,
        eo_orbit.reference_time,
        eo_orbit.delta_time,
    )
    gso = create_general_sar_orbit(metadata_sv, ignore_anx_after_orbit_start=True)

    return compute_footprint(
        time_corners=(samples_start, samples_stop, lines_start, lines_stop),
        gso=gso,
        look_direction="LEFT"
    )


def compute_footprint_from_l0s(l0_product_path: Path, aux_orb_product_path: Path) -> list[list[float]]:
    """Compute L0 footprint given L0 product and AUX_ORB product paths"""
    l0_product_content = L0ProductContent.from_name(name=l0_product_path.name)

    # Read ISP stream (one polarization only)
    ispstream = read_ispstream(file_in=l0_product_path.joinpath(l0_product_content.rxh))

    # Derive useful parameters
    indexes = np.where(ispstream.df["sigtyp"] == 0)[0]  # find STYP==0
    samples_start, samples_stop = get_range_time_given_index(
        ispstream, indexes[0]
    )  # FIXME: compute min and max, not just first
    lines_start = get_azimuth_time_given_index(ispstream, indexes[0])
    lines_stop = get_azimuth_time_given_index(ispstream, indexes[-1])

    aux_orb_product_content = AUXORBProductContent.from_name(name=aux_orb_product_path.name)

    eo_orbit = EOOrbit(
        orbit_file=aux_orb_product_path.joinpath(aux_orb_product_content.parameters_file_folder).joinpath(
            aux_orb_product_content.parameters_file
        )
    )
    metadata_sv = metadata.StateVectors(
        eo_orbit.position_sv,
        eo_orbit.velocity_sv,
        eo_orbit.reference_time,
        eo_orbit.delta_time,
    )
    gso = create_general_sar_orbit(metadata_sv, ignore_anx_after_orbit_start=True)

    return compute_footprint(
        time_corners=(samples_start, samples_stop, lines_start, lines_stop), gso=gso, look_direction="LEFT"
    )

import click

def parse_precisedatetime(dt_str: str) -> PreciseDateTime:
    """Converte stringa ISO in PreciseDateTime"""
    date, time_part = dt_str.split("T")
    y, mo, d = date.split("-")
    h, mi, s = time_part.split(":")
    sec, microsec = s.split(".")
    subseconds = float(f"0.{microsec}")
    return PreciseDateTime.from_numeric_datetime(int(y), int(mo), int(d), int(h), int(mi), int(sec), subseconds)



# ============================================================
# HIGH-LEVEL API — used by the Jupyter notebook
# ============================================================

def get_frames_footprints(input_folder: str, frames_dir: Path) -> list[dict]:
    """
    Parse all EOF files in frames_dir and compute the footprint for each frame.
    RAW_0S and AUX_ORB paths are resolved automatically from input_folder/Inputs/
    using the product names stored inside each EOF file.
    Returns a list of dicts with keys: id, status, start, stop, corners.
    """
    inputs_dir = Path(input_folder) / "Inputs"

    # Load all EOF files
    eof_files = sorted(frames_dir.glob("*.EOF"))
    print(f"📄 EOF files found: {len(eof_files)}")

    frames_data = []
    for eof in eof_files:
        # Parse frame metadata from EOF file
        try:
            f = BiomassProductL1VFRA(eof)
        except Exception as e:
            print(f"  ⚠️  {eof.name} could not be parsed: {e}")
            continue

        # Resolve RAW_0S and AUX_ORB paths from Inputs folder
        raw_0s  = inputs_dir / f.source_L0S
        aux_orb = inputs_dir / f.source_AUX_ORB

        if not raw_0s.exists():
            print(f"  ⚠️  RAW_0S not found: {raw_0s}")
            continue
        if not aux_orb.exists():
            print(f"  ⚠️  AUX_ORB not found: {aux_orb}")
            continue

        print(f"  🔄 Frame {f.frame_id} [{f.frame_status}]  {f.frame_start_time} -> {f.frame_stop_time}")

        # Compute footprint for this frame's time interval
        try:
            start   = parse_precisedatetime(f.frame_start_time)
            stop    = parse_precisedatetime(f.frame_stop_time)
            corners = compute_footprint_from_l0s_custom(raw_0s, aux_orb, start, stop)
            print(f"     ✅ Corners computed")
        except Exception as e:
            print(f"     ⚠️  Footprint failed: {e}")
            corners = None

        frames_data.append({
            "id":      f.frame_id,
            "status":  f.frame_status,
            "start":   f.frame_start_time,
            "stop":    f.frame_stop_time,
            "corners": corners,
        })

    # Sort by frame ID
    frames_data.sort(key=lambda x: int(x["id"]))
    return frames_data


def build_folium_map(frames_data: list[dict]) -> folium.Map:
    """
    Build a folium interactive map with one polygon per frame footprint.
    Each polygon is color-coded and shows frame ID, status and time range on click.
    """
    colors = ["red", "blue", "green", "orange", "purple", "darkred", "cadetblue", "darkgreen"]
    m = None

    for i, frame in enumerate(frames_data):
        if frame["corners"] is None:
            continue

        corners = frame["corners"]
        color = colors[i % len(colors)]
        center_lat = np.mean([c[0] for c in corners])
        center_lon = np.mean([c[1] for c in corners])

        # Initialize map centered on the first valid frame
        if m is None:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            #, width=1000, height=400)

        # Draw frame footprint polygon
        folium.Polygon(
            locations=list(corners) + [corners[0]],  # close the polygon
            color=color,
            fill=True,
            fill_opacity=0.2,
            tooltip=f"Frame {frame['id']} | {frame['status']}",
            popup=folium.Popup(
                f"<b>Frame ID:</b> {frame['id']}<br>"
                f"<b>Status:</b> {frame['status']}<br>"
                f"<b>Start:</b> {frame['start']}<br>"
                f"<b>Stop:</b> {frame['stop']}",
                max_width=300
            )
        ).add_to(m)

        # Add frame ID label at polygon center
        folium.Marker(
            location=[center_lat, center_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;color:{color};font-weight:bold;">F{frame["id"]}</div>'
            )
        ).add_to(m)

    return m


# ============================================================
# CLI entry point
# ============================================================
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=False)
    parser.add_argument("--frames_dir",   required=False)
    parser.add_argument("--output_json",  required=False)
    parser.add_argument("--raw_0s",       required=False)
    parser.add_argument("--aux_orb",      required=False)
    parser.add_argument("--start_time",   required=False)
    parser.add_argument("--stop_time",    required=False)
    args = parser.parse_args()

    if args.input_folder:
        # Called from notebook bash cell: compute all frames and save to JSON
        frames_data = get_frames_footprints(args.input_folder, Path(args.frames_dir))
        with open(args.output_json, "w") as f:
            json.dump(frames_data, f)
        print(f"✅ Saved: {args.output_json}")
    else:
        # Called from CLI: compute single frame
        footprint = compute_footprint_from_l0s_custom(
            l0_product_path=Path(args.raw_0s),
            aux_orb_product_path=Path(args.aux_orb),
            start_time=parse_precisedatetime(args.start_time),
            stop_time=parse_precisedatetime(args.stop_time),
        )
        print("\nComputed corners:")
        for corner in footprint:
            print(f"lat {corner[0]:.6f}, lon {corner[1]:.6f}")
