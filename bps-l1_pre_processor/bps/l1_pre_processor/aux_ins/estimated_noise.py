# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Estimated Noise product utilities
---------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    open_product_folder,
    read_metadata,
    write_metadata,
    write_raster_with_raster_info,
)
from arepytools.io.metadata import (
    AcquisitionTimeLine,
    EPolarization,
    RasterInfo,
    SwathInfo,
)
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import AcquisitionMode, Polarization, bps_logger
from bps.l1_pre_processor.aux_ins.aux_ins import AuxInsParameters


def retrieve_noise_power(aux_ins: AuxInsParameters, acquisition_mode: AcquisitionMode) -> dict[Polarization, complex]:
    """Retrieve noise power from aux ins parameters"""
    params = aux_ins.parameters[acquisition_mode]

    return {
        polarization: complex(int_cal_params.noise_power, 0)
        for polarization, int_cal_params in params.int_cal_params.items()
    }


def _translate_polarization(polarization: Polarization) -> EPolarization:
    """Convert to ArePyTools pol"""
    if polarization == Polarization.HH:
        return EPolarization.hh
    if polarization == Polarization.VV:
        return EPolarization.vv
    if polarization == Polarization.VH:
        return EPolarization.vh
    if polarization == Polarization.HV:
        return EPolarization.hv

    raise RuntimeError(f"Unknown pol: {polarization}")


_DEFAULT_DATE = PreciseDateTime.from_numeric_datetime(year=2020)


def write_estimated_noise_product(
    reference_product: Path,
    output_product: Path,
    default_values: dict[Polarization, complex],
):
    """Write a default estimated noise product"""
    ref_product = open_product_folder(reference_product)
    ref_channel = read_metadata(ref_product.get_channel_metadata(ref_product.get_channels_list()[0]))
    ref_raster_info = ref_channel.get_raster_info()
    ref_swath_info = ref_channel.get_swath_info()

    # Times
    preamble_time = ref_raster_info.lines_start
    postamble_time = ref_raster_info.lines_start + ref_raster_info.lines_step * ref_raster_info.lines

    # Output acquisition time line
    acq_time_line = AcquisitionTimeLine()
    acq_time_line.noise_packet = (
        preamble_time - ref_raster_info.lines_start,  # type: ignore
        postamble_time - ref_raster_info.lines_start,  # type: ignore
    )

    noise_product = create_product_folder(output_product, overwrite_ok=True)

    for channel_index, (pol, noise) in enumerate(default_values.items()):
        bps_logger.debug(
            "Writing default estimated noise: %s for polarization: %s",
            str(noise),
            pol.value,
        )
        noise_metadata = create_new_metadata()

        swath_info = SwathInfo(
            swath_i=ref_swath_info.swath,
            polarization_i=_translate_polarization(pol),
            acquisition_prf_i=0.0,
        )
        swath_info.acquisition_start_time = _DEFAULT_DATE

        noise_raster_info = RasterInfo(
            lines=2,
            samples=ref_raster_info.samples,
            celltype="FLOAT_COMPLEX",
            filename=noise_product.get_channel_data(channel_index + 1).name,
        )
        noise_raster_info.set_lines_axis(_DEFAULT_DATE, "Utc", 0, "s")
        noise_raster_info.set_samples_axis(0, "s", ref_raster_info.samples_step, "s")
        noise_metadata.insert_element(noise_raster_info)

        noise_metadata.insert_element(swath_info)
        noise_metadata.insert_element(acq_time_line)
        data = noise + np.zeros((2, ref_raster_info.samples))

        write_raster_with_raster_info(
            raster_file=noise_product.get_channel_data(channel_index + 1),
            data=data,
            raster_info=noise_raster_info,
        )
        write_metadata(
            metadata_obj=noise_metadata,
            metadata_file=noise_product.get_channel_metadata(channel_index + 1),
        )
