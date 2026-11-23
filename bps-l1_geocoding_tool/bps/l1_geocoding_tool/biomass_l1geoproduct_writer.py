# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""BIOMASS L0 product writer"""

from datetime import datetime
from pathlib import Path

import numpy as np
from bps.transcoder.sarproduct.sarproduct import SARProduct
from netCDF4 import Dataset


def write_l1_product_geocoded(biomass_l1_product_geocoded: SARProduct, output_path: Path):
    """Write L1 product geocoded to NetCDF file"""

    output_product_path = output_path / Path(biomass_l1_product_geocoded.name + ".nc")

    # =====================================================
    # Create file
    # =====================================================

    nc = Dataset(output_product_path, "w", format="NETCDF4")
    fcomplex = np.dtype([("real", np.float32), ("imag", np.float32)])
    fcomplex_t = nc.createCompoundType(fcomplex, "fcomplex")

    # =====================================================
    # GLOBAL ATTRIBUTES
    # =====================================================

    nc.description = "BIOMASS Geocoded L1 Product"

    nc.product_name = biomass_l1_product_geocoded.name
    nc.mission = biomass_l1_product_geocoded.mission
    nc.product_type = biomass_l1_product_geocoded.type
    nc.swath = biomass_l1_product_geocoded.swath_list[0]

    nc.file_originator = "ARESYS"
    nc.file_origination_date = datetime.now().isoformat()

    nc.nodata_value = np.float32(-9999.0)
    reference_azimuth_time = biomass_l1_product_geocoded.general_sar_orbit[0].reference_time
    nc.reference_azimuth_time = reference_azimuth_time.isoformat(timespec="microseconds")[:-1]

    # =====================================================
    # GROUP 1 — SAR DATA
    # =====================================================

    sar_data_group = nc.createGroup("sar_data")
    sar_data_group.description = "SAR data"

    for channel in range(biomass_l1_product_geocoded.channels):
        polarization = biomass_l1_product_geocoded.polarization_list[channel]
        pol_group = sar_data_group.createGroup(polarization.replace("/", ""))
        pol_group.description = f"SAR data ({polarization} polarization)"

        ri = biomass_l1_product_geocoded.raster_info_list[channel]
        pol_group.createDimension("latitude", ri.lines)
        pol_group.createDimension("longitude", ri.samples)

        latitude = pol_group.createVariable("latitude", np.float32, ("latitude",))
        latitude.units = "deg"
        latitude[:] = ri.lines_start + np.arange(ri.lines) * ri.lines_step

        longitude = pol_group.createVariable("longitude", np.float32, ("longitude",))
        longitude.units = "deg"
        longitude[:] = ri.samples_start + np.arange(ri.samples) * ri.samples_step

        sar_data = biomass_l1_product_geocoded.data_list[channel]
        if np.any(np.iscomplex(sar_data)):
            data = pol_group.createVariable(
                "data",
                fcomplex_t,
                ("latitude", "longitude"),
                # zlib=True,
                # complevel=4,
                # shuffle=True,
                # chunksizes=(256, sar_data.range_time.size),
            )
            data_temp = np.empty(sar_data.shape, fcomplex)
            data_temp["real"] = sar_data.real
            data_temp["imag"] = sar_data.imag
            data[:, :] = data_temp
        else:
            data = pol_group.createVariable(
                "data",
                np.float32,
                ("latitude", "longitude"),
                # zlib=True,
                # complevel=4,
                # shuffle=True,
                # chunksizes=(256, sar_data.range_time.size),
            )
            data[:, :] = sar_data

    # =====================================================
    # GROUP 2 — PLATFORM DATA (Orbit + Attitude)
    # =====================================================

    platform_data_group = nc.createGroup("platform_data")
    platform_data_group.description = "Platform data"

    gso = biomass_l1_product_geocoded.general_sar_orbit[0]
    platform_data_group.createDimension("azimuth_time", gso.position_vector.shape[0])
    platform_data_group.createDimension("state_vector_number", gso.position_vector.shape[1])

    azimuth_time = platform_data_group.createVariable("azimuth_time", np.float32, ("azimuth_time",))
    azimuth_time.units = "s"
    azimuth_time[:] = (
        gso.reference_time + np.arange(gso.position_vector.shape[0]) * gso.time_step - reference_azimuth_time
    )

    position = platform_data_group.createVariable(
        "position",
        np.float32,
        ("azimuth_time", "state_vector_number"),
        # zlib=True,
        # complevel=4,
        # shuffle=True,
        # chunksizes=(256, sar_data.range_time.size),
    )
    position.units = "m"
    position[:, :] = biomass_l1_product_geocoded.general_sar_orbit[0].position_vector

    velocity = platform_data_group.createVariable(
        "velocity",
        np.float32,
        ("azimuth_time", "state_vector_number"),
        # zlib=True,
        # complevel=4,
        # shuffle=True,
        # chunksizes=(256, sar_data.range_time.size),
    )
    velocity.units = "m/s"
    velocity[:, :] = biomass_l1_product_geocoded.general_sar_orbit[0].velocity_vector

    # =====================================================
    # Close file
    # =====================================================

    nc.close()
