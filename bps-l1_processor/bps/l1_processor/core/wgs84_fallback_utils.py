# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to handle WGS84 fallback
----------------------------------
"""

from pathlib import Path

import numpy as np
from arepytools.geometry.direct_geocoding import direct_geocoding_monostatic
from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.geometry.geometric_functions import (
    compute_incidence_angles,
    compute_look_angles,
)
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    metadata,
    write_metadata,
    write_raster_with_raster_info,
)


def write_slant_dem_product_on_ellipsoid(
    product_metadata: metadata.MetaData,
    slant_dem_product: Path,
    range_decimation_factor: int,
    azimuth_decimation_factor: int,
):
    """Write a slant dem product based on the ellipsoid"""

    slant_dem_pf = create_product_folder(slant_dem_product, overwrite_ok=True)

    reference_metadata = create_new_metadata()

    product_raster_info = product_metadata.get_raster_info()

    raster_info = metadata.RasterInfo(
        lines=product_raster_info.lines // azimuth_decimation_factor,
        samples=product_raster_info.samples // range_decimation_factor,
        celltype=metadata.ECellType.float64,
        filename=None,
        header_offset_bytes=0,
        row_prefix_bytes=0,
        byteorder="LITTLEENDIAN",
        invalid_value=None,
        format_type=None,
    )
    raster_info.set_lines_axis(
        product_raster_info.lines_start,
        product_raster_info.lines_start_unit,
        product_raster_info.lines_step * azimuth_decimation_factor,
        product_raster_info.lines_step_unit,
    )
    raster_info.set_samples_axis(
        product_raster_info.samples_start,
        product_raster_info.samples_start_unit,
        product_raster_info.samples_step * range_decimation_factor,
        product_raster_info.samples_step_unit,
    )

    reference_metadata.insert_element(raster_info)
    reference_metadata.insert_element(product_metadata.get_swath_info())

    geocoding_side = product_metadata.get_dataset_info().side_looking
    assert geocoding_side is not None

    gso = create_general_sar_orbit(product_metadata.get_state_vectors(), ignore_anx_after_orbit_start=True)

    azimuth_axis = raster_info.lines_start + np.arange(raster_info.lines) * raster_info.lines_step
    range_axis = raster_info.samples_start + np.arange(raster_info.samples) * raster_info.samples_step

    positions = gso.get_position(azimuth_axis).T
    velocities = gso.get_velocity(azimuth_axis).T

    ground_points = direct_geocoding_monostatic(
        sensor_positions=positions,
        sensor_velocities=velocities,
        range_times=range_axis,
        geocoding_side=geocoding_side.value,
        geodetic_altitude=0.0,
        frequencies_doppler_centroid=0.0,
        wavelength=1.0,
    )

    coordinates = ground_points.transpose((2, 0, 1))

    look_angles = np.empty((azimuth_axis.size, range_axis.size))
    incidence_angles = np.empty((azimuth_axis.size, range_axis.size))
    for azimuth_index, pos in enumerate(positions):
        look_angles[azimuth_index, :] = compute_look_angles(pos, -pos, ground_points[azimuth_index])
        incidence_angles[azimuth_index, :] = compute_incidence_angles(pos, ground_points[azimuth_index])

    data = [
        coordinates[0],
        coordinates[1],
        coordinates[2],
        look_angles,
        incidence_angles,
    ]
    for raster_index in range(5):
        channel_id = raster_index + 1
        raster_file = slant_dem_pf.get_channel_data(channel_id)
        raster_info.file_name = raster_file.name
        write_raster_with_raster_info(
            raster_file,
            data[raster_index],
            raster_info,
        )

        metadata_file = slant_dem_pf.get_channel_metadata(channel_id)
        write_metadata(reference_metadata, metadata_file)
