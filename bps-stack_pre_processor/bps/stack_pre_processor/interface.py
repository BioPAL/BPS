# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
The Stack Pre-Processor Interface
---------------------------------
"""

from pathlib import Path

import numpy.typing as npt
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    write_metadata,
    write_raster_with_raster_info,
)
from arepytools.io.metadata import ECellType, RasterInfo
from bps.stack_pre_processor.core.utils import StackPreProcessorRuntimeError


def write_pre_processor_dem_product(
    dem_product_xyz: tuple[npt.NDArray[float], ...],
    num_data_channels: int,
    raster_info_list: list[RasterInfo],
    output_path: Path,
):
    """
    Write the DEM data as required by the coregistrator.

    Parameters
    ----------
    dem_product_xyz: tuple[npt.NDArray[float], ...]
        The DEM data listed as ECEF X/Y/Z coordinates.

    num_data_channels: int
        Number of channels in the original L1a data.

    raster_info_list: list[RasterInfo]
        The raster info.

    output_path: Path
        The destination path.

    Raises
    ------
    StackPreProcessorRuntimeError

    """
    if len(dem_product_xyz) != 3:
        raise StackPreProcessorRuntimeError("DEM must have x/y/z components")

    pf_xyz = create_product_folder(output_path, overwrite_ok=False)

    for channel, raster_info in enumerate(raster_info_list * 3):
        new_raster_info = RasterInfo(
            lines=raster_info.lines,
            samples=raster_info.samples,
            celltype=ECellType.float64,
            filename=pf_xyz.get_channel_data(channel).name,
        )
        new_raster_info.set_lines_axis(
            raster_info_list[channel % num_data_channels].lines_start,
            raster_info_list[channel % num_data_channels].lines_start_unit,
            raster_info_list[channel % num_data_channels].lines_step,
            raster_info_list[channel % num_data_channels].lines_step_unit,
        )
        new_raster_info.set_samples_axis(
            raster_info_list[channel % num_data_channels].samples_start,
            raster_info_list[channel % num_data_channels].samples_start_unit,
            raster_info_list[channel % num_data_channels].samples_step,
            raster_info_list[channel % num_data_channels].samples_step_unit,
        )

        metadata = create_new_metadata(num_metadata_channels=1)
        metadata.insert_element(new_raster_info)

        write_metadata(
            metadata_obj=metadata,
            metadata_file=pf_xyz.get_channel_metadata(channel),
        )
        write_raster_with_raster_info(
            raster_file=pf_xyz.get_channel_data(channel),
            data=dem_product_xyz[channel % 3],
            raster_info=new_raster_info,
        )
