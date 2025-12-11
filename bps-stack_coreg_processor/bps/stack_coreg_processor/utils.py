# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Stack Coreg Processor Utility
-----------------------------
"""

from pathlib import Path
from shutil import rmtree

import numpy.typing as npt
from arepytools.io import (
    create_new_metadata,
    create_product_folder,
    write_metadata,
    write_raster_with_raster_info,
)
from arepytools.io.metadata import ECellType, RasterInfo
from arepytools.io.productfolder2 import is_product_folder
from bps.common import bps_logger


class StackCoregProcessorRuntimeError(RuntimeError):
    """Handle the errors of the coregistrator."""


def write_product_folder(
    data: npt.NDArray,
    output_pf_path: Path,
    *,
    cell_type: ECellType = ECellType.float32,
    data_name: str | None = None,
):
    """Write a minimal single-channel product folder."""
    if output_pf_path.exists():
        bps_logger.debug(f"Removing product folder {output_pf_path}")
        rmtree(output_pf_path, ignore_errors=True)

    pf = create_product_folder(output_pf_path)

    raster_info = RasterInfo(
        lines=data.shape[0],
        samples=data.shape[1],
        celltype=cell_type,
        filename=pf.get_channel_data(0).name,
    )

    metadata = create_new_metadata(num_metadata_channels=1)
    metadata.insert_element(raster_info)

    try:
        write_metadata(
            metadata_obj=metadata,
            metadata_file=pf.get_channel_metadata(0),
        )
        write_raster_with_raster_info(
            raster_file=pf.get_channel_data(0),
            data=data,
            raster_info=raster_info,
        )
        # pylint: disable-next=broad-exception-caught
    except Exception as err:
        bps_logger.error("Cannot export intermediate product to %s", output_pf_path)
        raise StackCoregProcessorRuntimeError(err) from err

    if not is_product_folder(output_pf_path):
        raise StackCoregProcessorRuntimeError(
            "Could not cache data{} to {}".format(
                f" '{data_name}'" if data_name is not None else "",
                output_pf_path,
            )
        )
