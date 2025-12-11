# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX INS Pattern management
--------------------------
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    metadata,
    write_metadata,
    write_raster_with_raster_info,
)
from bps.common import Polarization
from bps.l1_pre_processor.aux_ins import netcdf_utils


class DoubletID(Enum):
    """Biomass doublet indexes"""

    D1 = "D1"
    D2 = "D2"


class AntennaProductPolarization(Enum):
    """Antenna polarizations"""

    H = "H"
    V = "V"


AntennaProductID = tuple[DoubletID, AntennaProductPolarization]
"""ID of the antenna product. E.g. (d1, h) or (d2, v). Each product than contains the copol and crosspol pattern"""


PatternID = tuple[DoubletID, Polarization]
"""Patterns ID"""


@dataclass
class BiomassAntennaPattern:
    """Antenna pattern data"""

    samples_axis: npt.NDArray[np.float64]
    lines_axis: npt.NDArray[np.float64]
    pattern: npt.NDArray[np.complex128]


class InvalidAntennaPatternsNETCDFFile(RuntimeError):
    """Raised when an antenna patterns netcdf file is invalid"""


NETCDF_ANTENNA_PATTERNS_GROUP = "antennaPatterns"
NETCDF_ANTENNA_PATTERNS_VARIABLES: dict[PatternID, str] = {
    (DoubletID.D1, Polarization.HH): "patternD1HH",
    (DoubletID.D1, Polarization.HV): "patternD1HV",
    (DoubletID.D2, Polarization.HH): "patternD2HH",
    (DoubletID.D2, Polarization.HV): "patternD2HV",
    (DoubletID.D1, Polarization.VV): "patternD1VV",
    (DoubletID.D1, Polarization.VH): "patternD1VH",
    (DoubletID.D2, Polarization.VV): "patternD2VV",
    (DoubletID.D2, Polarization.VH): "patternD2VH",
}
NETCDF_ELEVATION_AXIS_NAME = "elevationAngle"
NETCDF_AZIMUTH_AXIS_NAME = "azimuthAngle"


def transcode_input_antenna_patterns_to_product_folder(
    netcdf_file: Path,
    *,
    ant_d1_h_product: Path,
    ant_d1_v_product: Path,
    ant_d2_h_product: Path,
    ant_d2_v_product: Path,
):
    """Translate antenna patterns from netcdf file to product folders"""
    patterns = read_antenna_patterns(netcdf_file)
    write_antenna_patterns_to_product_folder(
        output_d1_h_product=ant_d1_h_product,
        output_d1_v_product=ant_d1_v_product,
        output_d2_h_product=ant_d2_h_product,
        output_d2_v_product=ant_d2_v_product,
        biomass_patterns=patterns,
    )


def fill_bps_antenna_info(doublet: DoubletID, polarization: Polarization):
    """Fill basic antenna info with bps convention"""

    sensor_name = "NOT SET"
    polarization_str = polarization.value[0] + "/" + polarization.value[1]
    mode = "STRIPMAP"
    beam = doublet.value
    return metadata.AntennaInfo(sensor_name, polarization_str, mode, beam)


def get_copol_polarization(
    antenna_polarization: AntennaProductPolarization,
) -> Polarization:
    """Get co-polarization pair from antenna polarization"""
    if antenna_polarization == AntennaProductPolarization.H:
        return Polarization.HH
    if antenna_polarization == AntennaProductPolarization.V:
        return Polarization.VV
    raise RuntimeError(f"Unknown antenna polarization: {antenna_polarization}")


def get_crosspol_polarization(
    antenna_polarization: AntennaProductPolarization,
) -> Polarization:
    """Get cross-polarization pair from antenna polarization"""
    if antenna_polarization == AntennaProductPolarization.H:
        return Polarization.HV
    if antenna_polarization == AntennaProductPolarization.V:
        return Polarization.VH
    raise RuntimeError(f"Unknown antenna polarization: {antenna_polarization}")


def write_antenna_patterns_to_product_folder(
    output_d1_h_product: Path,
    output_d1_v_product: Path,
    output_d2_h_product: Path,
    output_d2_v_product: Path,
    biomass_patterns: dict[PatternID, BiomassAntennaPattern],
):
    """Write biomass antenna patterns to antenna patterns product folders

    Parameters
    ----------
    output_d1_h_product : Path
        antenna product containing d1 h copol and crosspol patterns
    output_d1_v_product : Path
        antenna product containing d2 h copol and crosspol patterns
    output_d2_h_product : Path
        antenna product containing d1 v copol and crosspol patterns
    output_d2_v_product : Path
        antenna product containing d2 v copol and crosspol patterns
    biomass_patterns : Dict[PatternID, BiomassAntennaPattern]
        all the 8 necessary patterns

    Raises
    ------
    FileExistsError
        when at least one of the output products is already present
    """

    output_products_info = {
        (DoubletID.D1, AntennaProductPolarization.H, output_d1_h_product),
        (DoubletID.D1, AntennaProductPolarization.V, output_d1_v_product),
        (DoubletID.D2, AntennaProductPolarization.H, output_d2_h_product),
        (DoubletID.D2, AntennaProductPolarization.V, output_d2_v_product),
    }

    for doublet_id, antenna_polarization, output_product_path in output_products_info:
        co_polarization = get_copol_polarization(antenna_polarization)
        cross_polarization = get_crosspol_polarization(antenna_polarization)

        output_pattern_list = [
            (
                doublet_id,
                co_polarization,
                biomass_patterns[doublet_id, co_polarization],
            ),
            (
                doublet_id,
                cross_polarization,
                biomass_patterns[doublet_id, cross_polarization],
            ),
        ]

        antenna_product_folder = create_product_folder(output_product_path, overwrite_ok=True)

        for channel_index, (doublet, polarization, pattern) in enumerate(output_pattern_list):
            ch_idx = channel_index + 1
            raster_name = antenna_product_folder.get_channel_data(ch_idx)
            meta = create_new_metadata()
            raster_info = metadata.RasterInfo(
                lines=pattern.lines_axis.size,
                samples=pattern.samples_axis.size,
                celltype=metadata.ECellType.fcomplex,
                filename=raster_name.name,
            )

            raster_info.set_lines_axis(
                lines_start=pattern.lines_axis[0],
                lines_start_unit="rad",
                lines_step=float(pattern.lines_axis[1] - pattern.lines_axis[0] if pattern.lines_axis.size > 1 else 0),
                lines_step_unit="rad",
            )

            raster_info.set_samples_axis(
                samples_start=pattern.samples_axis[0],
                samples_start_unit="rad",
                samples_step=float(
                    pattern.samples_axis[1] - pattern.samples_axis[0] if pattern.samples_axis.size > 1 else 0
                ),
                samples_step_unit="rad",
            )
            meta.insert_element(raster_info)

            antenna_info = fill_bps_antenna_info(doublet, polarization)
            meta.insert_element(antenna_info)

            write_metadata(meta, metadata_file=antenna_product_folder.get_channel_metadata(ch_idx))
            write_raster_with_raster_info(
                raster_file=raster_name,
                data=np.reshape(
                    np.conj(pattern.pattern),  # apply complex conjugate operator to input antenna patterns
                    (pattern.lines_axis.size, pattern.samples_axis.size),
                ),
                raster_info=raster_info,
            )


def read_antenna_patterns(
    antenna_pattern_netcdf_file: Path,
) -> dict[PatternID, BiomassAntennaPattern]:
    """Read all antenna patterns from the input netcdf file

    Parameters
    ----------
    antenna_pattern_netcdf_file : Path
        input netcdf file describing all the eight patterns

    Returns
    -------
    Dict[PatternID, BiomassAntennaPattern]
        all the patterns stored in BiomassAntennaPattern as numpy arrays

    Raises
    ------
    InvalidAntennaPatternsNETCDFFile
        In case of expected information is not found in the netcdf file
    """

    dataset = netcdf_utils.get_dataset(antenna_pattern_netcdf_file)

    samples_axis = np.deg2rad(netcdf_utils.read_dimension(dataset, NETCDF_ELEVATION_AXIS_NAME))
    lines_axis = np.deg2rad(netcdf_utils.read_dimension(dataset, NETCDF_AZIMUTH_AXIS_NAME))
    if samples_axis is None or lines_axis is None:
        raise InvalidAntennaPatternsNETCDFFile(antenna_pattern_netcdf_file)

    output_patterns: dict[PatternID, BiomassAntennaPattern] = {}
    for pattern_id, variable_name in NETCDF_ANTENNA_PATTERNS_VARIABLES.items():
        pattern = netcdf_utils.read_group_variable(dataset, NETCDF_ANTENNA_PATTERNS_GROUP, variable_name)
        if pattern is None:
            raise InvalidAntennaPatternsNETCDFFile(antenna_pattern_netcdf_file)

        output_patterns[pattern_id] = BiomassAntennaPattern(
            samples_axis=samples_axis, lines_axis=lines_axis, pattern=pattern
        )

    return output_patterns
