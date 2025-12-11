# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to handle different
----------------------------------
"""

from pathlib import Path

from bps.common import bps_logger
from bps.l1_processor.processor_interface.aux_pp1 import (
    AuxProcessingParametersL1,
    GeneralConf,
)
from bps.l1_processor.processor_interface.joborder_l1 import L1JobOrder

_DEM_DB_SUBDIR = {
    GeneralConf.EarthModel.COPERNICUS_DEM: "copernicus",
    GeneralConf.EarthModel.SRTM: "srtm",
}


def select_dem(job_order: L1JobOrder, aux_pp1: AuxProcessingParametersL1) -> Path | None:
    """Select the proper DEM folder, fallback on wgs84 if needed"""
    if aux_pp1.general.requested_height_model == GeneralConf.EarthModel.ELLIPSOID:
        bps_logger.info("Requested processing on WGS84 ellipsoid")
        return None

    dem_db = None
    main_dem_db_dir = job_order.dem_database_entry_point
    if main_dem_db_dir is None or not main_dem_db_dir.exists():
        if job_order.dem_database_entry_point is None:
            bps_logger.warning("DEM database not specified in the JobOrder")
        else:
            bps_logger.warning(f"DEM database not found: {main_dem_db_dir}")
    else:
        assert main_dem_db_dir is not None
        specific_dem_dir_name = _DEM_DB_SUBDIR[aux_pp1.general.requested_height_model]
        specific_dem_dir = main_dem_db_dir.joinpath(specific_dem_dir_name)

        if not specific_dem_dir.exists():
            bps_logger.warning(f"Specific DEM folder not found in DEM database: {specific_dem_dir}")
        else:
            versions = aux_pp1.general.height_model_version.split()
            assert len(versions) >= 1
            height_model_version = versions[0]

            dem_db_candidate = specific_dem_dir.joinpath(height_model_version)
            if not dem_db_candidate.exists():
                bps_logger.warning(f"Specific DEM version not found in DEM database: {dem_db_candidate}")
            else:
                dem_db = dem_db_candidate

    if dem_db is None:
        bps_logger.warning("Height model set to 'WGS84'")
        aux_pp1.general.height_model = GeneralConf.EarthModel.ELLIPSOID
        aux_pp1.general.height_model_version = ""
    else:
        bps_logger.info(f"Found DEM folder: {dem_db}")

    return dem_db
