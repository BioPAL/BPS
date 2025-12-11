# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Scene center computation
------------------------
"""

from pathlib import Path

import numpy as np
from arepytools.geometry.conversions import xyz2llh
from arepytools.geometry.generalsarorbit import create_general_sar_orbit
from arepytools.io import open_product_folder, read_metadata
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import bps_logger


def compute_scene_center(
    raw_product: Path,
    azimuth_interval: tuple[PreciseDateTime, PreciseDateTime] | None,
) -> tuple[PreciseDateTime, tuple[float, float, float]]:
    """Compute scene center considering azimuth cut and chirp length

    Parameters
    ----------
    raw_product : Path
        raw product
    azimuth_interval : Optional[Tuple[PreciseDateTime, PreciseDateTime]]
        optional azimuth interval cut

    Returns
    -------
    Tuple[PreciseDateTime, Tuple[float, float, flow]]
        azimuth center and corresponding point in lat lon h coordinates (deg)
    """
    log_prefix = "IRI20 wrapper - input computation -"

    pf = open_product_folder(raw_product)
    metadata = read_metadata(pf.get_channel_metadata(pf.get_channels_list()[0]))
    reference_raster_info = metadata.get_raster_info()
    reference_state_vectors = metadata.get_state_vectors()

    side_looking = metadata.get_dataset_info().side_looking
    assert side_looking is not None

    pulse_length = metadata.get_pulse().pulse_length
    assert isinstance(pulse_length, float)

    if azimuth_interval is not None:
        az_start, az_stop = azimuth_interval
    else:
        az_start = reference_raster_info.lines_start
        assert isinstance(az_start, PreciseDateTime)
        az_stop = az_start + reference_raster_info.lines_step * reference_raster_info.lines

    rg_start = reference_raster_info.samples_start
    rg_stop = rg_start + reference_raster_info.samples_step * reference_raster_info.samples - pulse_length

    bps_logger.debug("%s azimuth start          : %s", log_prefix, az_start)
    bps_logger.debug("%s azimuth stop           : %s", log_prefix, az_stop)
    bps_logger.debug("%s range start            : %s", log_prefix, rg_start)
    bps_logger.debug("%s range stop             : %s", log_prefix, rg_stop)

    azimuth_center = az_start + (az_stop - az_start) / 2.0
    range_center = rg_start + (rg_stop - rg_start) / 2.0

    gso = create_general_sar_orbit(reference_state_vectors, ignore_anx_after_orbit_start=True)
    central_point = gso.sat2earth(azimuth_center, range_center, side_looking.value)
    central_point_llh = xyz2llh(central_point).squeeze()
    central_point_llh[0:2] = np.rad2deg(central_point_llh[0:2])

    bps_logger.debug("%s central point llh [deg]: %s", log_prefix, central_point_llh)
    bps_logger.debug("%s central time           : %s", log_prefix, azimuth_center)

    return azimuth_center, tuple(central_point_llh)
