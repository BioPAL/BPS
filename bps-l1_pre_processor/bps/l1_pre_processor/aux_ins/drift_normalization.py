# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Drift normalization
---------------------------------------
"""

from pathlib import Path

from arepytools.io import (
    iter_channels,
    open_product_folder,
    read_raster_with_raster_info,
    write_raster_with_raster_info,
)
from bps.common import AcquisitionMode, Polarization, bps_logger
from bps.l1_pre_processor.aux_ins.aux_ins import AuxInsParameters


def retrieve_reference_drifts(
    aux_ins_params: AuxInsParameters, acquisition_mode: AcquisitionMode
) -> dict[Polarization, complex]:
    """Retrieve reference drift from aux ins parameters"""
    params = aux_ins_params.parameters[acquisition_mode]

    return {
        polarization: int_cal_params.reference_drift for polarization, int_cal_params in params.int_cal_params.items()
    }


def normalize_drift_product(drift: Path, reference_drifts: dict[Polarization, complex]):
    """Normalize drift values (inplace)"""
    per_line_normalization = open_product_folder(drift)

    for channel_index, channel_metadata in iter_channels(per_line_normalization):
        channel_swath_info = channel_metadata.get_swath_info()
        channel_raster_info = channel_metadata.get_raster_info()

        polarization = Polarization(channel_swath_info.polarization.name.upper())
        reference_drift = reference_drifts[polarization]
        raster_file = per_line_normalization.get_channel_data(channel_index)
        data = read_raster_with_raster_info(raster_file=raster_file, raster_info=channel_raster_info)
        channel_normalized_values = data / reference_drift

        bps_logger.debug(
            "Drift values for channel %s normalized with %s",
            polarization.value,
            str(reference_drift),
        )

        # overwriting channels
        write_raster_with_raster_info(
            raster_file=raster_file,
            data=channel_normalized_values,
            raster_info=channel_raster_info,
        )
