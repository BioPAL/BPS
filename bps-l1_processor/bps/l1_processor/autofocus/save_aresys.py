# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l., DLR, Deimos Space
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

from copy import copy
from pathlib import Path

import numpy as np
from arepytools import io
from arepytools.io import metadata

elements_to_copy = [
    "DataSetInfo",
    "SwathInfo",
    "SamplingConstants",
    "StateVectors",
    "AttitudeInfo",
    "Pulse",
    "GroundCornerPoints",
    "DopplerCentroidVector",
    "DopplerRateVector",
    "TopsAzimuthModulationRateVector",
    "SlantToGroundVector",
    "GroundToSlantVector",
    "SlantToIncidenceVector",
    "SlantToElevationVector",
    "AntennaInfo",
    "DataStatistics",
    "CoregPolyVector",
]


def copy_metadata(src_channel: metadata.MetaData, dst_channel: metadata.MetaData):
    """Copy metadata elements from one channel to another."""
    for element_name in elements_to_copy:
        element = src_channel.get_metadata_channels(0).get_element(element_name)
        if element is None:
            continue

        dst_channel.get_metadata_channels(0).insert_element(copy(element))


def create_raster_info_from_reference(
    reference_raster_info: metadata.RasterInfo, celltype: str, filename: str, lines: int
) -> metadata.RasterInfo:
    """Create a new RasterInfo object based on the reference RasterInfo."""
    output_raster_info = metadata.RasterInfo(
        lines=lines, samples=reference_raster_info.samples, celltype=celltype, filename=filename
    )
    output_raster_info.set_lines_axis(
        reference_raster_info.lines_start,
        reference_raster_info.lines_start_unit,
        reference_raster_info.lines_step,
        reference_raster_info.lines_step_unit,
    )
    output_raster_info.set_samples_axis(
        reference_raster_info.samples_start,
        reference_raster_info.samples_start_unit,
        reference_raster_info.samples_step,
        reference_raster_info.samples_step_unit,
    )
    return output_raster_info


def create_burst_info_from_reference(reference_burst_info: metadata.BurstInfo) -> metadata.BurstInfo:
    """Create a new BurstInfo object based on the reference BurstInfo."""
    output_burst_info = metadata.BurstInfo()
    number_of_bursts = reference_burst_info.get_number_of_bursts()

    for burst_index in range(number_of_bursts):
        burst = reference_burst_info.get_burst(burst_index)
        output_burst_info.add_burst(burst.range_start_time, burst.azimuth_start_time, burst.lines)

    return output_burst_info


def create_metadata_from_reference(
    reference_metadata: metadata.MetaData, raster_info: metadata.RasterInfo, burst_info: metadata.BurstInfo
) -> metadata.MetaData:
    """Create a new MetaData object based on the reference MetaData."""
    output_meta = io.create_new_metadata()
    output_meta.insert_element(raster_info)
    output_meta.insert_element(burst_info)
    copy_metadata(reference_metadata, output_meta)
    return output_meta


def save_SLC_aresys(pf_path_in: Path | str, pf_path_out: Path | str, data: np.ndarray):
    """Save SLC data to a new product folder."""

    pf_in = io.open_product_folder(pf_path_in)
    pf_out = io.create_product_folder(pf_path_out, overwrite_ok=True)

    for ii in range(data.shape[2]):
        channel_index = ii + 1

        input_meta = io.read_metadata(pf_in.get_channel_metadata(channel_index))

        output_metadata_file = pf_out.get_channel_metadata(channel_index)
        output_raster_file = pf_out.get_channel_data(channel_index)

        bi_in = input_meta.get_burst_info()

        bi_out = create_burst_info_from_reference(bi_in)

        number_of_bursts = bi_in.get_number_of_bursts()
        lines = number_of_bursts * bi_in.lines_per_burst

        ri_out = create_raster_info_from_reference(
            input_meta.get_raster_info(), celltype="FLOAT_COMPLEX", filename=output_raster_file.name, lines=lines
        )

        output_meta = create_metadata_from_reference(input_meta, ri_out, bi_out)

        io.write_metadata(output_meta, output_metadata_file)
        io.write_raster_with_raster_info(output_raster_file, data[..., ii], output_meta.get_raster_info())


def save_ph_screen_aresys(pf_path_out: Path | str, data: np.ndarray, saveAllScreens: bool = True):
    pf_out = io.create_product_folder(pf_path_out, overwrite_ok=True)

    lines, samples, Niter = data.shape
    if saveAllScreens:
        for ii in range(Niter):
            output_channel = ii + 1
            raster_file = pf_out.get_channel_data(output_channel)
            ri_out = metadata.RasterInfo(lines=lines, samples=samples, celltype="FLOAT32", filename=raster_file.name)

            output_meta = io.create_new_metadata()
            output_meta.insert_element(ri_out)

            io.write_metadata(output_meta, pf_out.get_channel_metadata(output_channel))
            io.write_raster_with_raster_info(raster_file, data[..., ii], output_meta.get_raster_info())
    else:
        raster_file = pf_out.get_channel_data(1)
        ri_out = metadata.RasterInfo(lines=lines, samples=samples, celltype="FLOAT32", filename=raster_file.name)

        output_meta = io.create_new_metadata()
        output_meta.insert_element(ri_out)

        io.write_metadata(output_meta, pf_out.get_channel_metadata(1))
        io.write_raster_with_raster_info(raster_file, data[..., -1], output_meta.get_raster_info())


def save_ph_acc_aresys(
    pf_path_in: Path | str,
    pf_path_out: Path | str,
    ph_data: np.ndarray,
    acc_data: np.ndarray,
    saveAllScreens: bool = True,
):
    pf_in = io.open_product_folder(pf_path_in)
    pf_out = io.create_product_folder(pf_path_out, overwrite_ok=True)

    input_meta = io.read_metadata(pf_in.get_channel_metadata(1))
    ri_in = input_meta.get_raster_info()
    bi_in = input_meta.get_burst_info()

    bi_out = create_burst_info_from_reference(bi_in)

    lines, samples, Niter = ph_data.shape
    number_of_bursts = bi_in.get_number_of_bursts()
    lines = number_of_bursts * bi_in.lines_per_burst

    if saveAllScreens:
        for ii in range(Niter):
            output_channel = ii + 1
            raster_file = pf_out.get_channel_data(output_channel)
            ri_out = create_raster_info_from_reference(
                ri_in, celltype="FLOAT32", filename=raster_file.name, lines=lines
            )

            output_meta = create_metadata_from_reference(input_meta, ri_out, bi_out)

            io.write_metadata(output_meta, pf_out.get_channel_metadata(output_channel))
            io.write_raster_with_raster_info(raster_file, ph_data[..., ii], output_meta.get_raster_info())

        for ii in range(Niter):
            output_channel = ii + Niter + 1

            raster_file = pf_out.get_channel_data(output_channel)

            ri_out = create_raster_info_from_reference(
                ri_in, celltype="FLOAT32", filename=raster_file.name, lines=lines
            )

            output_meta = create_metadata_from_reference(input_meta, ri_out, bi_out)

            io.write_metadata(output_meta, pf_out.get_channel_metadata(output_channel))
            io.write_raster_with_raster_info(raster_file, ph_data[..., ii], output_meta.get_raster_info())

    else:
        aux = np.empty((lines, samples, 2), "float32")
        aux[..., 0] = ph_data[..., -1]
        aux[..., 1] = acc_data[..., -1]

        for ii in range(2):
            output_channel = ii + 1
            raster_file = pf_out.get_channel_data(output_channel)
            ri_out = create_raster_info_from_reference(
                ri_in, celltype="FLOAT32", filename=raster_file.name, lines=lines
            )

            output_meta = create_metadata_from_reference(input_meta, ri_out, bi_out)

            io.write_metadata(output_meta, pf_out.get_channel_metadata(output_channel))
            io.write_raster_with_raster_info(raster_file, aux[..., ii], output_meta.get_raster_info())


def save_fr_ph_screen_aresys(pf_path_out: Path | str, data: np.ndarray):
    pf_out = io.create_product_folder(pf_path_out, overwrite_ok=True)

    raster_file = pf_out.get_channel_data(1)
    lines, samples = data.shape

    ri_out = metadata.RasterInfo(lines=lines, samples=samples, celltype="FLOAT32", filename=raster_file.name)

    output_meta = io.create_new_metadata()
    output_meta.insert_element(ri_out)
    io.write_metadata(output_meta, pf_out.get_channel_metadata(1))
    io.write_raster_with_raster_info(raster_file, data, output_meta.get_raster_info())
