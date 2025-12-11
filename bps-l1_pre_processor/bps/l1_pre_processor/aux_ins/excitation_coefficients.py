# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Excitation coefficient product utilities
----------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    iter_channels,
    open_product_folder,
    write_metadata,
    write_raster_with_raster_info,
)
from arepytools.io.metadata import EPolarization, RasterInfo, SwathInfo
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common import AcquisitionMode, Polarization, bps_logger
from bps.l1_pre_processor.aux_ins.aux_ins import AuxInsParameters


def retrieve_tx_power_tracking(
    aux_ins: AuxInsParameters, acquisition_mode: AcquisitionMode
) -> dict[Polarization, complex]:
    """Retrieve TX power tracking values from aux ins parameters"""
    params = aux_ins.parameters[acquisition_mode]

    return {
        polarization: int_cal_params.tx_power_tracking for polarization, int_cal_params in params.int_cal_params.items()
    }


def _translate_copolarization(polarization: EPolarization) -> Polarization:
    """Convert an ArePyTools copol into a bps copol"""
    if polarization == EPolarization.hh:
        return Polarization.HH
    if polarization == EPolarization.vv:
        return Polarization.VV

    raise RuntimeError(f"Unknown co-pol: {polarization}")


_DEFAULT_DATE = PreciseDateTime.from_numeric_datetime(year=2020)


def write_excitation_coefficient_product(
    reference_product: Path,
    output_product: Path,
    excitation_coefficients: dict[Polarization, complex],
) -> None:
    """Write a default excitation coefficient products.

    This product has 8 channels.

    channel_idx doubletID   TX/RX   polarization
    1           D1          TX      H
    2           D2          TX      H
    3           D1          TX      V
    4           D2          TX      V
    5           D1          RX      H
    6           D2          RX      H
    7           D1          RX      V
    8           D2          RX      V
    """
    ref_product = open_product_folder(reference_product)
    coefficients_product = create_product_folder(output_product, overwrite_ok=True)

    for _, reference_metadata in iter_channels(ref_product, polarization=[EPolarization.hh, EPolarization.vv]):
        raster_info = reference_metadata.get_raster_info()
        swath_info = reference_metadata.get_swath_info()
        polarization = _translate_copolarization(swath_info.polarization)
        coefficient = excitation_coefficients[polarization]

        channels = {1, 2, 5, 6} if polarization == Polarization.HH else {3, 4, 7, 8}

        coeff_raster_info = RasterInfo(
            lines=raster_info.lines,
            samples=1,
            celltype="FLOAT_COMPLEX",
            filename="",
        )
        coeff_raster_info.set_lines_axis(
            raster_info.lines_start,
            raster_info.lines_start_unit,
            raster_info.lines_step,
            raster_info.lines_step_unit,
        )
        coeff_raster_info.set_samples_axis(
            0.0,
            raster_info.samples_start_unit,
            0.0,
            raster_info.samples_step_unit,
        )

        coeff_swath_info = SwathInfo(
            swath_i=swath_info.swath,
            polarization_i=swath_info.polarization,
            acquisition_prf_i=0.0,
        )
        coeff_swath_info.acquisition_start_time = _DEFAULT_DATE

        for channel in channels:
            doublet = "D1" if channel % 2 == 1 else "D2"
            role = "TX" if channel < 5 else "RX"
            pol = "H" if polarization == Polarization.HH else "V"

            bps_logger.debug(f"Writing excitation coefficients for {role} {doublet} {pol}: {coefficient}")

            meta = create_new_metadata()
            coeff_raster_info.file_name = coefficients_product.get_channel_data(channel).name
            meta.insert_element(coeff_raster_info)
            meta.insert_element(coeff_swath_info)

            write_metadata(
                metadata_obj=meta,
                metadata_file=coefficients_product.get_channel_metadata(channel),
            )

            data = coefficient + np.zeros((raster_info.lines, 1))
            write_raster_with_raster_info(
                raster_file=coefficients_product.get_channel_data(channel),
                data=data,
                raster_info=coeff_raster_info,
            )
